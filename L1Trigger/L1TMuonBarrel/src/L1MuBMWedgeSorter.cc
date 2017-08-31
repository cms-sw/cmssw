//-------------------------------------------------
//
//   Class: L1MuBMWedgeSorter
//
//   Description: Wedge Sorter
//                find the 2 highest rank candidates per wedge
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

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMWedgeSorter.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrackFinder.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrack.h"
#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMSecProcId.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSectorProcessor.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMEtaProcessor.h"

using namespace std;

// --------------------------------
//       class L1MuBMWedgeSorter
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMWedgeSorter::L1MuBMWedgeSorter(const L1MuBMTrackFinder& tf, int id) :
      m_tf(tf), m_wsid(id), m_TrackCands(3) {

  m_TrackCands.reserve(3);

}


//--------------
// Destructor --
//--------------

L1MuBMWedgeSorter::~L1MuBMWedgeSorter() {

}


//--------------
// Operations --
//--------------

//
// run Wedge Sorter
//
void L1MuBMWedgeSorter::run() {

  // get track candidates from Sector Processors
  vector<L1MuBMTrack*> wedgecands;
  wedgecands.reserve(12);

  int sector = m_wsid;
  for ( int wheel = -3; wheel <= 3; wheel++ ) {
    if ( wheel == 0 ) continue;
    L1MuBMSecProcId tmpspid(wheel,sector);
    for ( int number = 0; number < 2; number++ ) {
      const L1MuBMTrack* cand = m_tf.sp(tmpspid)->track(number);
      if ( cand && !cand->empty() ) {
        // remove tracks which where found in wheel 0 and
        // which didn't cross wheel boundaries (SP -1)
        bool reject = false;
        if ( wheel == -1 ) {
          reject = true;
          for ( int stat = 2; stat <= 4; stat++ ) {
            int adr = cand->address(stat);
            // check addresses : 0,1,4,5,8,9 (own wheel)
            if ( adr != 15 ) reject &= ( (adr/2)%2 == 0 );
          }
        }
        if ( !reject ) wedgecands.push_back(const_cast<L1MuBMTrack*>(cand));
      }
    }
  }

  // print input data
  if ( L1MuBMTFConfig::Debug(5) ) {
    cout << "Wedge Sorter " << m_wsid << " input: "
         << wedgecands.size() << endl;
    vector<L1MuBMTrack*>::const_iterator iter;
    for ( iter = wedgecands.begin(); iter != wedgecands.end(); iter++ ) {
      if (*iter ) (*iter)->print();
    }
  }

  // print input data
  runCOL(wedgecands);

  // remove disabled candidates
  vector<L1MuBMTrack*>::iterator it = wedgecands.begin();
  while ( it != wedgecands.end() ) {
    if ( *it && (*it)->empty() ) {
      wedgecands.erase(it);
      it = wedgecands.begin(); continue;
    }
    it++;
  }

  // sort candidates by pt and quality and copy the 2 best candidates
  partial_sort_copy( wedgecands.begin(), wedgecands.end(),
                     m_TrackCands.begin(), m_TrackCands.end(),
                     L1MuBMTrack::Rank() );

  if ( L1MuBMTFConfig::Debug(4) ) {
    cout << "Wedge Sorter " << m_wsid << " output: " << endl;
    vector<const L1MuBMTrack*>::const_iterator iter;
    for ( iter  = m_TrackCands.begin();
          iter != m_TrackCands.end(); iter++ ) {
      if (*iter) (*iter)->print();
    }
  }

}


//
// reset Wedge Sorter
//
void L1MuBMWedgeSorter::reset() {

  vector<const L1MuBMTrack*>::iterator iter;
  for ( iter = m_TrackCands.begin(); iter != m_TrackCands.end(); iter++ ) {
    *iter = nullptr;
  }

}


//
// print candidates found in  Wedge Sorter
//
void L1MuBMWedgeSorter::print() const {

  if ( anyTrack() ) {
    cout << "Muon candidates found in Wedge Sorter " << m_wsid << " : " << endl;
    vector<const L1MuBMTrack*>::const_iterator iter = m_TrackCands.begin();
    while ( iter != m_TrackCands.end() ) {
      if ( *iter ) cout << *(*iter) << " found in "
                        << (*iter)->spid() << endl;
      iter++;
    }
  }

}


//
// are there any muon candidates?
//
bool L1MuBMWedgeSorter::anyTrack() const {

  vector<const L1MuBMTrack*>::const_iterator iter = m_TrackCands.begin();
  while ( iter != m_TrackCands.end() ) {
    if ( *iter && !(*iter)->empty() ) return true;
    iter++;
  }

  return false;

}


//
// Cancel Out Logic for Wedge Muon Sorter
//
void L1MuBMWedgeSorter::runCOL(vector<L1MuBMTrack*>& cands) const {

  // compare candidates which were found in nearby wheels:
  // if 2 candidates have at least one track segment in common
  // disable the one with lower quality;
  // compare addresses from stations 2, 3 and 4

  typedef vector<L1MuBMTrack*>::iterator TI;
  for ( TI iter1 = cands.begin(); iter1 != cands.end(); iter1++ ) {
    if ( *iter1 == nullptr ) continue;
    if ( (*iter1)->empty() ) continue;
    L1MuBMSecProcId sp1 = (*iter1)->spid();
    int qual1 = (*iter1)->quality();
    for ( TI iter2 = cands.begin(); iter2 != cands.end(); iter2++ ) {
      if ( *iter2 == nullptr ) continue;
      if ( *iter1 == *iter2 ) continue;
      if ( (*iter2)->empty() ) continue;
      L1MuBMSecProcId sp2 = (*iter2)->spid();
      int qual2 = (*iter2)->quality();
      if ( sp1 == sp2 ) continue;
      if ( !neighbour(sp1,sp2) ) continue;
      int adr_shift = ( sp2.wheel() == -1 ) ? 0 : 2;
      int countTS = 0;
      for ( int stat = 2; stat <= 4; stat++ ) {
        int adr1 = (*iter1)->address(stat);
        int adr2 = (*iter2)->address(stat);
        if ( adr1 == 15 || adr2 == 15 ) continue;
        if ( (adr2/2)%2 == 1 ) continue;
        if ( adr1 == adr2 + adr_shift ) countTS++;
      }
      if (  countTS > 0 ) {
        if ( qual1 < qual2 ) {
          if ( L1MuBMTFConfig::Debug(5) ) {
            cout << "Wedge Sorter cancel : "; (*iter1)->print();
          }
          (*iter1)->disable();
          break;
        }
        else {
          if ( L1MuBMTFConfig::Debug(5) ) {
            cout << "Wedge Sorter cancel : "; (*iter2)->print();
          }
         (*iter2)->disable();
        }
      }
    }
  }

}


//
// find out if two Sector Processors are neighbours in the same wedge
//
bool L1MuBMWedgeSorter::neighbour(const L1MuBMSecProcId& spid1,
                                  const L1MuBMSecProcId& spid2) {

  // neighbour definition:
  // wheel 1 :  -2,  -1,  +1,  +1,  +2
  // wheel 2 :  -3,  -2,  -1,  +2,  +3

  bool neigh = false;

  int sector1 = spid1.sector();
  int wheel1  = spid1.wheel();

  int sector2 = spid2.sector();
  int wheel2  = spid2.wheel();

  if ( sector1 == sector2 ) {

    if ( ( wheel1 == -2 && wheel2 == -3 ) ||
         ( wheel1 == -1 && wheel2 == -2 ) ||
         ( wheel1 == +1 && wheel2 == -1 ) ||
         ( wheel1 == +1 && wheel2 == +2 ) ||
         ( wheel1 == +2 && wheel2 == +3 ) ) neigh = true;

  }

  return neigh;

}
