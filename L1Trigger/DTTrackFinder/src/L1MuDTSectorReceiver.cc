//-------------------------------------------------
//
//   Class: L1MuDTSectorReceiver
//
//   Description: Sector Receiver 
//
//
//   $Date: 2006/07/26 10:31:00 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------
using namespace std;

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorReceiver.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTDataBuffer.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegPhi.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"

// --------------------------------
//       class L1MuDTSectorReceiver
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuDTSectorReceiver::L1MuDTSectorReceiver(L1MuDTSectorProcessor& sp) : 
        m_sp(sp) {

}


//--------------
// Destructor --
//--------------
L1MuDTSectorReceiver::~L1MuDTSectorReceiver() { 

//  reset();
  
}


//--------------
// Operations --
//--------------

//
// receive track segment data from the DTBX and CSC chamber triggers
//
void L1MuDTSectorReceiver::run(int bx, const edm::Event& e) {

  // get track segments from DTBX chamber trigger
  receiveDTBXData(bx, e);
  
  // get track segments from CSC chamber trigger
  if ( L1MuDTTFConfig::overlap() && m_sp.ovl() ) { 
    receiveCSCData(bx, e);
  }

}


//
// clear
//
void L1MuDTSectorReceiver::reset() {

}


//
// receive track segment data from the DTBX chamber trigger
//
void L1MuDTSectorReceiver::receiveDTBXData(int bx, const edm::Event& e) {

  edm::Handle<L1MuDTChambPhContainer> dttrig;
  e.getByType(dttrig);

  L1MuDTChambPhDigi* ts=0;

  // const int bx_offset = dttrig->correctBX();
  int bx_offset=16;
  bx = bx + bx_offset;

  // get DTBX phi track segments  
  int address = 0;
  for ( int station = 1; station <= 4; station++ ) {
    int max_address = (station == 1) ? 2 : 12;
    for (int reladr =0; reladr < max_address; reladr++) {
      address++;
      if ( m_sp.ovl() && (reladr/2)%2 != 0 ) continue;
      int wheel  = address2wheel(reladr);
      int sector = address2sector(reladr);     
      if ( reladr%2 == 0 ) ts = dttrig->chPhiSegm1(wheel,station,sector,bx);
      if ( reladr%2 == 1 ) ts = dttrig->chPhiSegm2(wheel,station,sector,bx);
      if ( ts ) {
        int phi  = ts->phi();
        int phib = ts->phiB();
        int qual = ts->code();
        if (qual < 2) qual = 2;
        bool tag = (reladr%2 == 1) ? true : false;
        //
        // out-of-time TS filter (compare TS at +-1 bx)
        // 
        bool skipTS = false;

        if ( L1MuDTTFConfig::getTSOutOfTimeFilter() ) {
 
          int sh_phi = 12 - L1MuDTTFConfig::getNbitsExtPhi();
          int tolerance = L1MuDTTFConfig::getTSOutOfTimeWindow();

          L1MuDTChambPhDigi* tsPreviousBX_1 = dttrig->chPhiSegm1(wheel,station,sector,bx-1);
          if ( tsPreviousBX_1 ) {
            int phiBX  = tsPreviousBX_1->phi();
            int qualBX = tsPreviousBX_1->code();
            if ( abs( (phi >> sh_phi) - (phiBX >> sh_phi) ) <= tolerance &&
                 qualBX > qual ) skipTS = true;
          }
          
          L1MuDTChambPhDigi* tsPreviousBX_2 = dttrig->chPhiSegm2(wheel,station,sector,bx-1);
          if ( tsPreviousBX_2 ) {
            int phiBX  = tsPreviousBX_2->phi();
            int qualBX = tsPreviousBX_2->code();
            if ( abs( (phi >> sh_phi) - (phiBX >> sh_phi) ) <= tolerance &&
                 qualBX > qual ) skipTS = true;
          }
     
          L1MuDTChambPhDigi* tsNextBX_1 = dttrig->chPhiSegm1(wheel,station,sector,bx+1);
          if ( tsNextBX_1 ) {
            int phiBX  = tsNextBX_1->phi();
            int qualBX = tsNextBX_1->code();
            if ( abs( (phi >> sh_phi) - (phiBX >> sh_phi) ) <= tolerance &&
                 qualBX > qual ) skipTS = true;
          }

          L1MuDTChambPhDigi* tsNextBX_2 = dttrig->chPhiSegm2(wheel,station,sector,bx+1);
          if ( tsNextBX_2 ) {
            int phiBX  = tsNextBX_2->phi();
            int qualBX = tsNextBX_2->code();
            if ( abs( (phi >> sh_phi) - (phiBX >> sh_phi) ) <= tolerance &&
                 qualBX > qual ) skipTS = true;
          }

        }

        if ( !skipTS ) { 
          L1MuDTTrackSegPhi tmpts(wheel,sector,station,phi,phib,
                                  static_cast<L1MuDTTrackSegPhi::TSQuality>(qual),
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
void L1MuDTSectorReceiver::receiveCSCData(int bx, const edm::Event& e) {
  
  if ( bx < CSCConstants::MIN_BUNCH || bx > CSCConstants::MAX_BUNCH ) return;

  edm::Handle<CSCTriggerContainer<csctf::TrackStub> > csctrig;
  e.getByType(csctrig);

  const int bxCSC = CSCConstants::TIME_OFFSET;
  
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
    if ( csc_iter->etaPacked() > 17 ) continue;
    bool etaFlag = ( csc_iter->etaPacked() > 17 ); 
    int phiCSC = csc_iter->phiPacked();
    int qualCSC = csc_iter->getQuality();
           
    // convert CSC quality code to DTBX quality code
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

    if ( subsector == 1 && phiCSC >= 2048 ) continue;
    if ( subsector == 2 && phiCSC <  2048 ) continue;
        
    // convert CSC phi to DTBX phi
    double dphi = ((double) phiCSC) - 2048.*16./31.;
    if ( sector%2 == 0 ) dphi = dphi  - 2048.*30./ 31.;
    dphi = dphi*62.*M_PI/180.;
    int phi = static_cast<int>(floor( dphi ));
    if ( phi < -2048 || phi > 2047 ) continue; 

    if ( ncsc < 2 ) {
      int address = 16 + ncsc;
      bool tag = (ncsc == 1 ) ? true : false;
      L1MuDTTrackSegPhi tmpts(wheel,sector,station+2,phi,0,
                              static_cast<L1MuDTTrackSegPhi::TSQuality>(qual),
                              tag,bx,etaFlag);
      m_sp.data()->addTSphi(address,tmpts);
      ncsc++;
    }
    else cout << "too many CSC track segments!" << endl;
  }  

}



//
// find the right sector for a given address
//
int L1MuDTSectorReceiver::address2sector(int adr) const {

  int sector = m_sp.id().sector();

  if ( adr >= 4 && adr <= 7  ) sector = (sector+13)%12; // +1
  if ( adr >= 8 && adr <= 11 ) sector = (sector+11)%12; // -1

  return sector;

}


//
// find the right wheel for a given address
//
int L1MuDTSectorReceiver::address2wheel(int adr) const {

  int wheel = m_sp.id().locwheel();

  // for 2, 3, 6, 7, 10, 11
  if ( (adr/2)%2 == 1 ) wheel = m_sp.id().wheel();

  return wheel;

}
