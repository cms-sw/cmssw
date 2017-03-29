//-------------------------------------------------
//
//   Class: L1MuBMSectorReceiver
//
//   Description: Sector Receiver
//
//
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSectorReceiver.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSectorProcessor.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMDataBuffer.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTrackSegLoc.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTrackSegPhi.h"

#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

using namespace std;

// --------------------------------
//       class L1MuBMSectorReceiver
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuBMSectorReceiver::L1MuBMSectorReceiver(L1MuBMSectorProcessor& sp, edm::ConsumesCollector && iC) :
       m_sp(sp),
       m_DTDigiToken(iC.consumes<L1MuDTChambPhContainer>(L1MuBMTFConfig::getBMDigiInputTag())){

}


//--------------
// Destructor --
//--------------
L1MuBMSectorReceiver::~L1MuBMSectorReceiver() {

//  reset();

}


//--------------
// Operations --
//--------------

//
// receive track segment data from the BBMX and CSC chamber triggers
//
void L1MuBMSectorReceiver::run(int bx, const edm::Event& e, const edm::EventSetup& c) {

  c.get< L1MuDTTFParametersRcd >().get( pars );
  //c.get< L1MuDTTFMasksRcd >().get( msks );

  const L1TMuonBarrelParamsRcd& bmtfParamsRcd = c.get<L1TMuonBarrelParamsRcd>();
  bmtfParamsRcd.get(bmtfParamsHandle);
  const L1TMuonBarrelParams& bmtfParams = *bmtfParamsHandle.product();
  msks =  bmtfParams.l1mudttfmasks;
  //msks.print();

  // get track segments from BBMX chamber trigger
  receiveBBMXData(bx, e, c);

  // get track segments from CSC chamber trigger
  //if ( L1MuBMTFConfig::overlap() && m_sp.ovl() ) {
    //receiveCSCData(bx, e, c);
  //}

}


//
// clear
//
void L1MuBMSectorReceiver::reset() {

}


//
// receive track segment data from the BBMX chamber trigger
//
void L1MuBMSectorReceiver::receiveBBMXData(int bx, const edm::Event& e, const edm::EventSetup& c) {
  edm::Handle<L1MuDTChambPhContainer> dttrig;
  //e.getByLabel(L1MuBMTFConfig::getBMDigiInputTag(),dttrig);
  e.getByToken(m_DTDigiToken,dttrig);
  L1MuDTChambPhDigi const* ts=0;

  // const int bx_offset = dttrig->correctBX();
  int bx_offset=0;
  bx = bx + bx_offset;
  // get BBMX phi track segments
  int address = 0;
  for ( int station = 1; station <= 4; station++ ) {
    int max_address = (station == 1) ? 2 : 12;
    for (int reladr =0; reladr < max_address; reladr++) {
      address++;
      //if ( m_sp.ovl() && (reladr/2)%2 != 0 ) continue;
      int wheel  = address2wheel(reladr);
      int sector = address2sector(reladr);
      //if ( (wheel==2 || wheel==-2) && station==1 ) continue;

      if ( reladr%2 == 0 ) ts = dttrig->chPhiSegm1(wheel,station,sector,bx);
      if ( reladr%2 == 1 ) ts = dttrig->chPhiSegm2(wheel,station,sector,bx-1);
      if ( ts ) {
        int phi  = ts->phi();
//        int phib = ts->phiB();
        int phib = 0;
        if(station!=3) phib = ts->phiB();

        int qual = ts->code();
        bool tag = (reladr%2 == 1) ? true : false;

        int lwheel = m_sp.id().wheel();
        lwheel = abs(lwheel)/lwheel*(abs(wheel)+1);

        if ( station == 1 ) {
          if ( msks.get_inrec_chdis_st1(lwheel, sector) ) continue;
          if ( qual < pars->get_inrec_qual_st1(lwheel, sector) ) continue;
        }
        else if ( station == 2 ) {
          if ( msks.get_inrec_chdis_st2(lwheel, sector) ) continue;
          if ( qual < pars->get_inrec_qual_st2(lwheel, sector) ) continue;
          }
        else if ( station == 3 ) {
          if ( msks.get_inrec_chdis_st3(lwheel, sector) ) continue;
          if ( qual < pars->get_inrec_qual_st3(lwheel, sector) ) continue;
        }
        else if ( station == 4 ) {
          if ( msks.get_inrec_chdis_st4(lwheel, sector) ) continue;
          if ( qual < pars->get_inrec_qual_st4(lwheel, sector) ) continue;
        }

        if ( reladr/2 == 1 && qual < pars->get_soc_stdis_n(m_sp.id().wheel(), m_sp.id().sector())  ) continue;
        if ( reladr/2 == 2 && qual < pars->get_soc_stdis_wl(m_sp.id().wheel(), m_sp.id().sector()) ) continue;
        if ( reladr/2 == 3 && qual < pars->get_soc_stdis_zl(m_sp.id().wheel(), m_sp.id().sector()) ) continue;
        if ( reladr/2 == 4 && qual < pars->get_soc_stdis_wr(m_sp.id().wheel(), m_sp.id().sector()) ) continue;
        if ( reladr/2 == 5 && qual < pars->get_soc_stdis_zr(m_sp.id().wheel(), m_sp.id().sector()) ) continue;

        //
        // out-of-time TS filter (compare TS at +-1 bx)
        //
        bool skipTS = false;

        bool nbx_del = pars->get_soc_nbx_del(m_sp.id().wheel(), m_sp.id().sector());
        if ( L1MuBMTFConfig::getTSOutOfTimeFilter() || nbx_del ) {

          int sh_phi = 12 - L1MuBMTFConfig::getNbitsExtPhi();
          int tolerance = L1MuBMTFConfig::getTSOutOfTimeWindow();

          L1MuDTChambPhDigi const * tsPreviousBX_1 = dttrig->chPhiSegm1(wheel,station,sector,bx-1);
          if ( tsPreviousBX_1 ) {
            int phiBX  = tsPreviousBX_1->phi();
            int qualBX = tsPreviousBX_1->code();
            if ( abs( (phi >> sh_phi) - (phiBX >> sh_phi) ) <= tolerance &&
                 qualBX > qual ) skipTS = true;
          }

          L1MuDTChambPhDigi const * tsPreviousBX_2 = dttrig->chPhiSegm2(wheel,station,sector,bx-1);
          if ( tsPreviousBX_2 ) {
            int phiBX  = tsPreviousBX_2->phi();
            int qualBX = tsPreviousBX_2->code();
            if ( abs( (phi >> sh_phi) - (phiBX >> sh_phi) ) <= tolerance &&
                 qualBX > qual ) skipTS = true;
          }

          L1MuDTChambPhDigi const * tsNextBX_1 = dttrig->chPhiSegm1(wheel,station,sector,bx+1);
          if ( tsNextBX_1 ) {
            int phiBX  = tsNextBX_1->phi();
            int qualBX = tsNextBX_1->code();
            if ( abs( (phi >> sh_phi) - (phiBX >> sh_phi) ) <= tolerance &&
                 qualBX > qual ) skipTS = true;
          }

          L1MuDTChambPhDigi const * tsNextBX_2 = dttrig->chPhiSegm2(wheel,station,sector,bx+1);
          if ( tsNextBX_2 ) {
            int phiBX  = tsNextBX_2->phi();
            int qualBX = tsNextBX_2->code();
            if ( abs( (phi >> sh_phi) - (phiBX >> sh_phi) ) <= tolerance &&
                 qualBX > qual ) skipTS = true;
          }

        }

        if ( !skipTS ) {

           /* if(reladr%2 == 0) {
                L1MuBMTrackSegPhi tmpts(wheel,sector,station,phi,phib,
                                  static_cast<L1MuBMTrackSegPhi::TSQuality>(qual),
                                  tag,bx-bx_offset);
                m_sp.data()->addTSphi(address-1,tmpts);
            }
            if(reladr%2 == 1) {
                L1MuBMTrackSegPhi tmpts(wheel,sector,station,phi,phib,
                                  static_cast<L1MuBMTrackSegPhi::TSQuality>(qual),
                                  tag,bx+1);
                m_sp.data()->addTSphi(address-1,tmpts);
            }*/
          L1MuBMTrackSegPhi tmpts(wheel,sector,station,phi,phib,
                                  static_cast<L1MuBMTrackSegPhi::TSQuality>(qual),
                                  tag,bx-bx_offset);
          m_sp.data()->addTSphi(address-1,tmpts);
        }
      }

    }
  }

}


//
// receive track segment data from CSC chamber trigger
//
/*
void L1MuBMSectorReceiver::receiveCSCData(int bx, const edm::Event& e, const edm::EventSetup& c) {

  if ( (L1MuBMTFConfig::getCSCTrSInputTag()).label() == "none" ) return;

  if ( bx < -6 || bx > 6 ) return;

  edm::Handle<CSCTriggerContainer<csctf::TrackStub> > csctrig;
  e.getByLabel(L1MuBMTFConfig::getCSCTrSInputTag(),csctrig);

  const int bxCSC = 6;

  vector<csctf::TrackStub> csc_list;
  vector<csctf::TrackStub>::const_iterator csc_iter;

  int station = 1; // only ME13
  int wheel = m_sp.id().wheel();
  int side = ( wheel == 3 ) ? 1 : 2;
  int sector = m_sp.id().sector();
  int csc_sector = ( sector == 0 ) ? 6 : (sector+1)/2;
  int subsector = ( sector%2 == 0 ) ? 2 : 1;

  csc_list = csctrig->get(side,station,csc_sector,subsector,bxCSC+bx);
  int ncsc = 0;
  for ( csc_iter = csc_list.begin(); csc_iter != csc_list.end(); csc_iter++ ) {
    bool etaFlag = ( csc_iter->etaPacked() > 17 );
    int qualCSC = csc_iter->getQuality();

    // convert CSC quality code to BBMX quality code
    unsigned int qual = 7;
    if ( qualCSC ==  2 ) qual = 0;
    if ( qualCSC ==  6 ) qual = 1;
    if ( qualCSC ==  7 ) qual = 2;
    if ( qualCSC ==  8 ) qual = 2;
    if ( qualCSC ==  9 ) qual = 3;
    if ( qualCSC == 10 ) qual = 3;
    if ( qualCSC == 11 ) qual = 4;
    if ( qualCSC == 12 ) qual = 5;
    if ( qualCSC == 13 ) qual = 5;
    if ( qualCSC == 14 ) qual = 6;
    if ( qualCSC == 15 ) qual = 6;
    if ( qual == 7 ) continue;

    // convert CSC phi to BBMX phi
    int phi = csc_iter->phiPacked();
    if ( phi > 2047 ) phi -= 4096;
    if ( phi < -2048 || phi > 2047 ) continue;

    if ( msks.get_inrec_chdis_csc(m_sp.id().wheel(), m_sp.id().sector()) ) continue;
    if ( qual < pars->get_soc_qual_csc(m_sp.id().wheel(), m_sp.id().sector()) ) continue;
    if ( pars->get_soc_csc_etacanc(m_sp.id().wheel(), m_sp.id().sector()) && etaFlag ) continue;
    if ( L1MuBMTFConfig::getEtaCanc() && etaFlag ) continue;

    if ( ncsc < 2 ) {
      int address = 16 + ncsc;
      bool tag = (ncsc == 1 ) ? true : false;
      L1MuBMTrackSegPhi tmpts(wheel,sector,station+2,phi,0,
                              static_cast<L1MuBMTrackSegPhi::TSQuality>(qual),
                              tag,bx,etaFlag);
      m_sp.data()->addTSphi(address,tmpts);
      ncsc++;
    }
    //    else cout << "too many CSC track segments!" << endl;
  }

}

*/

//
// find the right sector for a given address
//
int L1MuBMSectorReceiver::address2sector(int adr) const {

  int sector = m_sp.id().sector();

  if ( adr >= 4 && adr <= 7  ) sector = (sector+13)%12; // +1
  if ( adr >= 8 && adr <= 11 ) sector = (sector+11)%12; // -1

  return sector;

}


//
// find the right wheel for a given address
//
int L1MuBMSectorReceiver::address2wheel(int adr) const {

  int wheel = m_sp.id().locwheel();

  // for 2, 3, 6, 7, 10, 11
  if ( (adr/2)%2 == 1 ) wheel = m_sp.id().wheel();

  return wheel;

}
