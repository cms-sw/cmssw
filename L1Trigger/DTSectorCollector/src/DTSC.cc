//-------------------------------------------------
//
//   Class: DTSC.cpp
//
//   Description: Implementation of DTSectColl trigger algorithm
//
//
//   Author List:
//   S. Marcellini
//   Modifications:
//   11/11/06 C. Battilana : theta cand added
//   12/12/06 C. Battilana : _stat added
//   09/01/07 C. Battilana : updated to local conf
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSC.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigSectColl.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectColl.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhCand.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThCand.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <algorithm>

//----------------
// Constructors --
//----------------

DTSC::DTSC(int istat) : _ignoreSecondTrack(0), _stat(istat) {
  // reserve the appropriate amount of space for vectors
  // test _incand[0].reserve(DTConfigSectColl::NTSMSC);
  // test_incand[1].reserve(DTConfigSectColl::NTSMSC);
  // test _outcand.reserve(2);
}

//--------------
// Destructor --
//--------------
DTSC::~DTSC() { clear(); }

//--------------
// Operations --
//--------------

void DTSC::clear() {
  _ignoreSecondTrack = 0;

  for (int itk = 0; itk <= 1; itk++) {
    _incand_ph[itk].clear();
  }

  _outcand_ph.clear();
  _cand_th.clear();
}

//
void DTSC::run() {
  if (config()->debug()) {
    std::cout << "DTSC::run: Processing DTSectColl: ";
    std::cout << nFirstTPh() << " first & " << nSecondTPh() << " second Phi tracks ";
    std::cout << " - " << nCandTh() << " Theta tracks" << std::endl;
  }

  if (nFirstTPh() < 1)
    return;  // skip if no first tracks
             //
             // SORT 1
             //

  // debugging
  if (config()->debug()) {
    std::cout << "Vector of first Phi tracks in DTSectColl: " << std::endl;
    std::vector<DTSectCollPhCand*>::const_iterator p;
    for (p = _incand_ph[0].begin(); p != _incand_ph[0].end(); p++) {
      (*p)->print();
    }
  }
  // end debugging

  DTSectCollPhCand* first = DTSectCollsort1();
  if (config()->debug()) {
    std::cout << "SC: DTSC::run: first Phi track is = " << first << std::endl;
  }
  if (first != nullptr) {
    _outcand_ph.push_back(first);
  }
  if (nSecondTPh() < 1)
    return;  // skip if no second tracks

  //
  // SORT 2
  //

  // debugging
  if (config()->debug()) {
    std::vector<DTSectCollPhCand*>::const_iterator p;
    std::cout << "Vector of second Phi tracks in DTSectColl: " << std::endl;
    for (p = _incand_ph[1].begin(); p != _incand_ph[1].end(); p++) {
      (*p)->print();
    }
  }
  // end debugging

  DTSectCollPhCand* second = DTSectCollsort2();
  if (second != nullptr) {
    _outcand_ph.push_back(second);
  }
}

DTSectCollPhCand* DTSC::DTSectCollsort1() {
  // Do a sort 1
  DTSectCollPhCand* best = nullptr;
  DTSectCollPhCand* carry = nullptr;
  std::vector<DTSectCollPhCand*>::iterator p;
  for (p = _incand_ph[0].begin(); p != _incand_ph[0].end(); p++) {
    DTSectCollPhCand* curr = (*p);

    curr->setBitsSectColl();  // SM sector collector set bits in dataword to make SC sorting

    // NO Carry in Sector Collector sorting in default
    if (config()->SCGetCarryFlag(_stat)) {  // get carry

      if (best == nullptr) {
        best = curr;
      } else if ((*curr) < (*best)) {
        carry = best;
        best = curr;
      } else if (carry == nullptr) {
        carry = curr;
      } else if ((*curr) < (*carry)) {
        carry = curr;
      }

    } else if (config()->SCGetCarryFlag(_stat) == 0) {  // no carry (default)
      if (best == nullptr) {
        best = curr;
      } else if ((*curr) < (*best)) {
        best = curr;
      }
    }

    if (carry != nullptr && config()->SCGetCarryFlag(_stat)) {  // reassign carry to sort 2 candidates
      carry->setSecondTrack();                                  // change value of 1st/2nd track bit
      _incand_ph[1].push_back(carry);                           // add to list of 2nd track
    }
  }

  return best;
}

DTSectCollPhCand* DTSC::DTSectCollsort2() {
  // Check if there are second tracks

  if (nTracksPh() < 1) {
    std::cout << "DTSC::DTSectCollsort2: called with no first Phi track.";
    std::cout << " empty pointer returned!" << std::endl;
    return nullptr;
  }
  // If a first track at the following BX is present, ignore second tracks of any kind
  if (_ignoreSecondTrack) {
    for (std::vector<DTSectCollPhCand*>::iterator p = _incand_ph[1].begin(); p != _incand_ph[1].end(); p++) {
    }
    return nullptr;
  }

  // If no first tracks at the following BX, do a sort 2
  //  DTSectCollCand* best=getTrack(1);  ! not needed as lons as there is no comparison with best in sort 2
  DTSectCollPhCand* second = nullptr;
  std::vector<DTSectCollPhCand*>::iterator p;
  for (p = _incand_ph[1].begin(); p != _incand_ph[1].end(); p++) {
    DTSectCollPhCand* curr = (*p);
    curr->setBitsSectColl();  // SM sector collector set bits in dataword to make SC sorting

    if (second == nullptr) {
      second = curr;
    } else if ((*curr) < (*second)) {
      second = curr;
    }
  }

  return second;
}

void DTSC::addPhCand(DTSectCollPhCand* cand) { _incand_ph[(1 - cand->isFirst())].push_back(cand); }

void DTSC::addThCand(DTSectCollThCand* cand) { _cand_th.push_back(cand); }

unsigned DTSC::nCandPh(int ifs) const {
  if (ifs < 1 || ifs > 2) {
    std::cout << "DTSC::nCandPh: wrong track number: " << ifs;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }
  return _incand_ph[ifs - 1].size();
}

unsigned DTSC::nCandTh() const { return _cand_th.size(); }

DTSectCollPhCand* DTSC::getDTSectCollPhCand(int ifs, unsigned n) const {
  if (ifs < 1 || ifs > 2) {
    std::cout << "DTSC::getDTSectCollPhCand: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return nullptr;
  }
  if (n < 1 || n > nCandPh(ifs)) {
    std::cout << "DTSC::getDTSectCollPhCand: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return nullptr;
  }

  std::vector<DTSectCollPhCand*>::const_iterator p = _incand_ph[ifs - 1].begin() + n - 1;
  return (*p);
}

DTSectCollThCand* DTSC::getDTSectCollThCand(unsigned n) const {
  if (n < 1 || n > nCandTh()) {
    std::cout << "DTSC::getDTSectCollThCand: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return nullptr;
  }

  std::vector<DTSectCollThCand*>::const_iterator p = _cand_th.begin() + n - 1;
  return (*p);
}

void DTSC::addDTSectCollPhCand(DTSectCollPhCand* cand) {
  int ifs = (cand->isFirst()) ? 0 : 1;

  _incand_ph[ifs].push_back(cand);
}

DTSectCollPhCand* DTSC::getTrackPh(int n) const {
  if (n < 1 || n > nTracksPh()) {
    std::cout << "DTSC::getTrackPh: requested track not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return nullptr;
  }

  std::vector<DTSectCollPhCand*>::const_iterator p = _outcand_ph.begin() + n - 1;

  return (*p);
}

DTSectCollThCand* DTSC::getTrackTh(int n) const {
  if (n < 1 || n > nTracksTh()) {
    std::cout << "DTSC::getTrackTh: requested track not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return nullptr;
  }

  std::vector<DTSectCollThCand*>::const_iterator p = _cand_th.begin() + n - 1;

  return (*p);
}
