//-------------------------------------------------
//
//   Class: L1MuDTMuonSorter
//
//   Description: DT Muon Sorter
//
//
//   $Date: 2008/02/18 17:38:04 $
//   $Revision: 1.4 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTMuonSorter.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTWedgeSorter.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackFinder.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrack.h"

using namespace std;

// --------------------------------
//       class L1MuDTMuonSorter
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTMuonSorter::L1MuDTMuonSorter(const L1MuDTTrackFinder& tf) :
      m_tf(tf), m_TrackCands() {

  m_TrackCands.reserve(4);
  
}


//--------------
// Destructor --
//--------------

L1MuDTMuonSorter::~L1MuDTMuonSorter() {

}


//--------------
// Operations --
//--------------

//
// run DT Muon Sorter 
//
void L1MuDTMuonSorter::run() {

  // get track candidates from Wedge Sorters
  vector<L1MuDTTrack*> mycands;
  mycands.reserve(24);
  
  for ( int wedge = 0; wedge < 12; wedge++ ) {
    vector<const L1MuDTTrack*> wscand = m_tf.ws(wedge)->tracks();
    vector<const L1MuDTTrack*>::iterator iter = wscand.begin();
    while ( iter != wscand.end() ) {
      if ( *iter && !(*iter)->empty() ) 
        mycands.push_back(const_cast<L1MuDTTrack*>(*iter) );
      iter++;
    }
  }

  // print input data
  if ( L1MuDTTFConfig::Debug(4) ) {
    cout << "DT Muon Sorter input: "
         << mycands.size() << endl;
    vector<L1MuDTTrack*>::const_iterator iter;
    for ( iter = mycands.begin(); iter != mycands.end(); iter++ ) {
      if (*iter ) (*iter)->print();
    }
  }
 
  // run Cancel Out Logic
  runCOL(mycands);

  // remove disabled candidates
  vector<L1MuDTTrack*>::iterator it = mycands.begin();
  while ( it != mycands.end() ) {
    if ( *it && (*it)->empty() ) {
      mycands.erase(it);
      it = mycands.begin(); continue;
    }
    it++;
  } 

  // sort pt and quality
  stable_sort( mycands.begin(), mycands.end(), L1MuDTTrack::Rank() );

  // copy the best 4 candidates
  int number_of_tracks = 0;
  vector<L1MuDTTrack*>::const_iterator iter1 = mycands.begin();
  while ( iter1 != mycands.end() ) {
    if ( *iter1 && number_of_tracks < 4 ) {
      m_TrackCands.push_back(*iter1);
      number_of_tracks++;
    }
    iter1++;
  }  

}


//
// reset DT Muon Sorter
//
void L1MuDTMuonSorter::reset() {

  m_TrackCands.clear();
  vector<const L1MuDTTrack*>::iterator iter;
  for ( iter = m_TrackCands.begin(); iter != m_TrackCands.end(); iter++ ) {
    *iter = 0;
  }

}


//
// print candidates found in the DT Muon Sorter
//
void L1MuDTMuonSorter::print() const {

  cout << endl;
  cout << "Muon candidates found by the barrel MTTF : "
       << numberOfTracks() << endl;
  vector<const L1MuDTTrack*>::const_iterator iter = m_TrackCands.begin();
  while ( iter != m_TrackCands.end() ) {
    if ( *iter ) cout << *(*iter) << endl;
    iter++;
  }
  cout << endl;

}


//
// Cancel Out Logic for DT Muon Sorter
//
void L1MuDTMuonSorter::runCOL(vector<L1MuDTTrack*>& cands) const {

  // compare candidates which were found in nearby sectors and wheels
  // if 2 candidates have at least one track segment in common
  // disable the one with lower quality
  // compare addresses from stations 2, 3 and 4

  typedef vector<L1MuDTTrack*>::iterator TI;
  for ( TI iter1 = cands.begin(); iter1 != cands.end(); iter1++ ) {
    if ( *iter1 == 0 ) continue;
    if ( (*iter1)->empty() ) continue;
    L1MuDTSecProcId sp1 = (*iter1)->spid();
    int qual1 = (*iter1)->quality();
    for ( TI iter2 = cands.begin(); iter2 != cands.end(); iter2++ ) {
      if ( *iter2 == 0 ) continue;
      if ( *iter1 == *iter2 ) continue; 
      if ( (*iter2)->empty() ) continue;
      L1MuDTSecProcId sp2 = (*iter2)->spid();
      int qual2 = (*iter2)->quality();
      if (sp1 == sp2 ) continue;
      int topology = neighbour(sp1,sp2);
      if ( topology == -1 ) continue;     
      int countTS = 0;
      for ( int stat = 2; stat <= 4; stat++ ) {
        int adr1 = (*iter1)->address(stat);
        int adr2 = (*iter2)->address(stat);
        if ( adr1 == 15 || adr2 == 15 ) continue;
        switch ( topology ) {
          case 1 : {
                    if ( adr1 > 7 ) continue;
                    if ( adr2 > 3 && adr2 < 8 ) continue;
                    int adr_shift = ( adr2 > 7 ) ? -8 : 4;
                    if ( adr1 == adr2+adr_shift ) countTS++;
                    break;
                   }
          case 2 : {
                    if ( adr2 > 7 ) continue;
                    if ( adr1 > 3 && adr1 < 8 ) continue;
                    int adr_shift = ( adr2 > 3 ) ? -4 : 8;
                    if ( adr1 == adr2+adr_shift ) countTS++;
                    break;
                   }
          case 3 : {
                    if ( ( adr1 == 6 && adr2 == 0 ) ||
                         ( adr1 == 7 && adr2 == 1 ) ||
                         ( adr1 == 2 && adr2 == 8 ) ||
                         ( adr1 == 3 && adr2 == 9 ) ) countTS++;
                    break;
                   }
          case 4 : {
                    if ( ( adr1 == 2  && adr2 == 4 ) ||
                         ( adr1 == 3  && adr2 == 5 ) ||
                         ( adr1 == 10 && adr2 == 0 ) ||
                         ( adr1 == 11 && adr2 == 1 ) ) countTS++;
                    break;
                   }
          case 5 : {
                    if ( ( adr1 == 0 && adr2 == 8 ) ||
                         ( adr1 == 1 && adr2 == 9 ) ||
                         ( adr1 == 4 && adr2 == 0 ) ||
                         ( adr1 == 5 && adr2 == 1 ) ) countTS++;
                    break;
                   }
          case 6 : {
                    if ( ( adr1 == 0 && adr2 == 4 ) ||
                         ( adr1 == 1 && adr2 == 5 ) ||
                         ( adr1 == 8 && adr2 == 0 ) ||
                         ( adr1 == 9 && adr2 == 1 ) ) countTS++;
                    break;
                   }                   
          default : break;
        }   
      }
      if (  countTS > 0 ) {
        if ( qual1 < qual2 ) {
          if ( L1MuDTTFConfig::Debug(5) ) { 
            cout << "Muon Sorter cancel : "; (*iter1)->print();
          }
          (*iter1)->disable();
          break;
        }
        else {
          if ( L1MuDTTFConfig::Debug(5) ) {
            cout << "Muon Sorter cancel : "; (*iter2)->print();
          }
         (*iter2)->disable();
        }
      }
    }
  }
  
  
  // if two candidates have exactly the same phi and eta values
  // remove the one with lower rank

  for ( TI iter1 = cands.begin(); iter1 != cands.end(); iter1++ ) {
    if ( *iter1 == 0 ) continue;
    if ( (*iter1)->empty() ) continue;
    int phi1 = (*iter1)->phi();
    int pt1 = (*iter1)->pt();
    int qual1 = (*iter1)->quality();
    for ( TI iter2 = cands.begin(); iter2 != cands.end(); iter2++ ) {
      if ( *iter2 == 0 ) continue;
      if ( *iter1 == *iter2 ) continue; 
      if ( (*iter2)->empty() ) continue;
      int phi2 = (*iter2)->phi();
      int pt2 = (*iter2)->pt();
      int qual2 = (*iter2)->quality();
      int w1 = (*iter1)->getStartTSphi().wheel();
      int w2 = (*iter2)->getStartTSphi().wheel();
      int phidiff = (phi2 - phi1)%144;
      if ( phidiff >= 72 ) phidiff -= 144;
      if ( phidiff < -72 ) phidiff += 144;
      if ( abs(phidiff) < 2 && (w1 == w2) ) {
        int rank1 = 10 * pt1 + qual1;
        int rank2 = 10 * pt2 + qual2;
        if ( L1MuDTTFConfig::Debug(5) ) { 
          cout << "========================================" << endl;
          cout << " special cancellation : " << endl;
          (*iter1)->print(); if ( rank1 <  rank2 ) cout << "cancelled" << endl;
          (*iter2)->print(); if ( rank1 >= rank2 ) cout << "cancelled" << endl;
          cout << "========================================" << endl;
        }      
        if ( rank1 >= rank2 ) (*iter2)->disable();
        if ( rank1 <  rank2 ) { (*iter1)->disable(); break; }       
      }
    }
  }  

}


//
// find out if two Sector Processors are neighbours
//
int L1MuDTMuonSorter::neighbour(const L1MuDTSecProcId& spid1, 
                                const L1MuDTSecProcId& spid2) {

  // definition of valid topologies:

  //              E T A
  //        -----------------
  //   +
  //        ---               ---
  //   |   | 2 |             | 2 |
  // P |   |___|             |___|
  //   |    ---    ---    ---        ---
  // H |   | 1 |  | 1 |  | 1 |      | 1 |
  //   |   |___|  |___|  |___|      |___|
  // I |           ---                   ---
  //   |          | 2 |                 | 2 |
  //              |___|                 |___|
  //   -
  // result: 1      2        3          4         5    6  otherwise : -1

  int topology = -1;

  int sector1 = spid1.sector();
  int wheel1  = spid1.wheel();

  int sector2 = spid2.sector();
  int wheel2  = spid2.wheel();

  int sectordiff = (sector2 - sector1)%12;
  if ( sectordiff >= 6 ) sectordiff -= 12;
  if ( sectordiff < -6 ) sectordiff += 12;

  if ( abs(sectordiff) == 1 ) {

    if ( wheel1 == wheel2 ) topology = (sectordiff > 0) ? 1 : 2;
    if ( wheel1 == +1 && wheel2 == -1 )  topology = (sectordiff > 0) ? 5 : 6; 
    if ( ( wheel1 == -2 && wheel2 == -3 ) ||
         ( wheel1 == -1 && wheel2 == -2 ) ||
         ( wheel1 == +1 && wheel2 == +2 ) ||
         ( wheel1 == +2 && wheel2 == +3 ) ) topology = (sectordiff > 0) ? 3 : 4;
 
  }     
       
  return topology;     

}
