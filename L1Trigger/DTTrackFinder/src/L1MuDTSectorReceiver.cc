//-------------------------------------------------
//
//   Class: L1MuDTSectorReceiver
//
//   Description: Sector Receiver 
//
//
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $
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
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

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
    receiveCSCData(bx);
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
  int bx_offset=0;
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
void L1MuDTSectorReceiver::receiveCSCData(int bx) {
  
  return;
  /*
  if ( bx < -6 || bx > 6 ) return;
  
  L1MuCSCPrimitiveSetup* csc_setup = Singleton<L1MuCSCPrimitiveSetup>::instance();
  L1MuCSCPrimitiveGenerator* csctrig = csc_setup->PrimitiveGenerator();

  bool cscTwentyDegree = L1MuCSCSetup::twentyDegree();
  const int bxCSC = L1MuCSCSetup::currentBx();
  
  vector<L1MuCSCTrackStub> csc_list;
  vector<L1MuCSCTrackStub>::const_iterator csc_iter;  
  
  int station = 1; // only ME13
  int wheel = m_sp.id().wheel();
  int side = ( wheel == 3 ) ? 1 : 2;
  int my_sector = m_sp.id().sector();
  for ( int offset = 0; offset < 3; offset++ ) {
    int sector = my_sector;
    if (offset == 1) sector = (my_sector+13)%12; // +1
    if (offset == 2) sector = (my_sector+11)%12; // -1
    int csc_sector = ( sector == 0 ) ? 6 : (sector+1)/2;
    int sub = ( sector%2 == 0 ) ? 2 : 1;
    int subsector = sub;
    int sub20from = 1;
    int sub20to = 1;
    if ( cscTwentyDegree ) {
      sub20from = ( subsector == 1 ) ? 1 : 2;
      sub20to   = ( subsector == 1 ) ? 2 : 3;
    }
    for ( int sub20 = sub20from; sub20 <= sub20to; sub20++ ) {
      if ( cscTwentyDegree ) {
        csc_list = csctrig->trackStubList(side,station,csc_sector,sub20,bxCSC+bx);
      } else {
        csc_list = csctrig->trackStubList(side,station,csc_sector,subsector,bxCSC+bx);
      }
      int ncsc = 0;
      for ( csc_iter = csc_list.begin(); csc_iter != csc_list.end(); csc_iter++ ) {
        if ( !(*csc_iter).interestingToBarrel() ) continue;
        bool etaFlag = (*csc_iter).stubInEndcap(); 
        unsigned int qualCSC = (*csc_iter).quality();
        int phi  = (*csc_iter).phi();
        int phib = (*csc_iter).phiBend();
          
        // convert CSC quality code to DTBX quality code
        unsigned int qual = 7;
        if ( qualCSC > 2 ) {
          qual = qualCSC-3;
        }
        if ( qualCSC == 10 ) {
           qual = 6;
        }
        
        if ( sub == 1 && phi >= 2048 ) continue;
        if ( sub == 2 && phi <  2048 ) continue;
        
        // convert CSC phi to DTBX phi
        double gphi = (*csc_iter).phiValue(); // now 0 - 2pi
        double phi0 = sector * M_PI/6. ;
        double dphi = fmod(gphi-phi0 + 3*M_PI, 2*M_PI) - M_PI;
        phi = static_cast<int> ( dphi*4096. );
       
        if ( ncsc < 2 ) {
          int address = 4 + station*12 + offset*4 + ncsc;
          bool tag = (ncsc == 1 ) ? true : false;
          L1MuDTTrackSegPhi tmpts(wheel,sector,station+2,phi,phib,
                                  static_cast<L1MuDTTrackSegPhi::TSQuality>(qual),
                                  tag,bx,etaFlag);
          m_sp.data()->addTSphi(address,tmpts);
          ncsc++;
        }
        else cout << "too many CSC track segments!" << endl;
      }  
    }
  }
  */
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
