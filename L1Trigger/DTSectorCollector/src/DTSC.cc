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
//
//
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSC.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTUtilities/interface/DTConfig.h"
//#include "Trigger/DTTriggerServerPhi/interface/DTTSPhi.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectColl.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <algorithm>

//----------------
// Constructors --
//----------------

DTSC::DTSC(DTConfig* config) : _config(config), _ignoreSecondTrack(0) {

  // reserve the appropriate amount of space for vectors
  // test _incand[0].reserve(DTConfig::NTSMSC);
  // test_incand[1].reserve(DTConfig::NTSMSC);
  // test _outcand.reserve(2);
 
}


//--------------
// Destructor --
//--------------
DTSC::~DTSC() {

  clear();

}


//--------------
// Operations --
//--------------

void
DTSC::clear() {

  _ignoreSecondTrack=0;

  for(int itk=0;itk<=1;itk++){
 
    _incand[itk].clear();
  }

  _outcand.clear();

}


// 
void
DTSC::run() {

  if(config()->debug()>2){
    std::cout << "DTSC::run: Processing DTSectColl: ";
    std::cout << nFirstT() << " first & " << nSecondT() << " second tracks" << std::endl;
  }

  if(nFirstT()<1)return; // skip if no first tracks
  //
  // SORT 1
  //

  // debugging
    if(config()->debug()>2){
     std::cout << "Vector of first tracks in DTSectColl: " << std::endl;
    std::vector<DTSectCollCand*>::const_iterator p;
    for(p=_incand[0].begin(); p!=_incand[0].end(); p++) {
           (*p)->print();
    }
   }
  // end debugging
 
  DTSectCollCand* first=DTSectCollsort1();
  if(config()->debug()>2){
    std::cout << "SC: DTSC::run: first track is = " << first << std::endl;
  }
  if(first!=0) {
    _outcand.push_back(first); 

  }
  if(nSecondT()<1)return; // skip if no second tracks

  //
  // SORT 2
  //

  // debugging
  if(config()->debug()>2){
    std::vector<DTSectCollCand*>::const_iterator p;
    std::cout << "Vector of second tracks in DTSectColl: " << std::endl;
    for(p=_incand[1].begin(); p!=_incand[1].end(); p++) {
       (*p)->print();
    }
  }
  // end debugging

  DTSectCollCand* second=DTSectCollsort2();
  if(second!=0) {
    _outcand.push_back(second); 
  }
  
}


DTSectCollCand*
DTSC::DTSectCollsort1() {

  // Do a sort 1
  DTSectCollCand* best=0;
  DTSectCollCand* carry=0;
  std::vector<DTSectCollCand*>::iterator p;
  for(p=_incand[0].begin(); p!=_incand[0].end(); p++) {
    DTSectCollCand* curr=(*p);

    curr->setBitsSectColl();    // SM sector collector set bits in dataword to make SC sorting
    
    // NO Carry in Sector Collector sorting in default 
    if(config()->SCGetCarryFlag()==1) {  // get carry

      if(best==0){
	best=curr;
      } 
      else if((*curr)<(*best)){
	carry=best;
	best=curr;
      } 
      else if(carry==0){
	carry=curr;
      } 
      else if((*curr)<(*carry)){
	carry=curr;
      } 

    }
    else if(config()->SCGetCarryFlag()==0){ // no carry (default)
      if(best==0){
	best=curr;
      } 
      else if((*curr)<(*best)){
	
	best=curr;
      } 
      
    }
    
    if(carry!=0 && config()->SCGetCarryFlag()==1) { // reassign carry to sort 2 candidates
      carry->setSecondTrack(); // change value of 1st/2nd track bit
      _incand[1].push_back(carry); // add to list of 2nd track
 
    }
  } 
  
  return best;

}


DTSectCollCand*
DTSC::DTSectCollsort2() {

  // Check if there are second tracks

  if(nTracks()<1){
    std::cout << "DTSC::DTSectCollsort2: called with no first track.";
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  // If a first track at the following BX is present, ignore second tracks of any kind
  if(_ignoreSecondTrack){

    for(std::vector<DTSectCollCand*>::iterator p=_incand[1].begin(); p!=_incand[1].end(); p++) {

    }
    return 0;
  }

  // If no first tracks at the following BX, do a sort 2
  //  DTSectCollCand* best=getTrack(1);  ! not needed as lons as there is no comparison with best in sort 2
  DTSectCollCand* second=0;
  std::vector<DTSectCollCand*>::iterator p;
  for(p=_incand[1].begin(); p!=_incand[1].end(); p++) {
    DTSectCollCand* curr=(*p);
    curr->setBitsSectColl();    // SM sector collector set bits in dataword to make SC sorting
    
    if(second==0){
      second=curr;
    } 
    else if((*curr)<(*second)){
      second=curr;
    } 
    
  }

  return second;

}


void
DTSC::addCand(DTSectCollCand* cand) {

  _incand[(1-cand->isFirst())].push_back(cand); 

}


unsigned
DTSC::nCand(int ifs) const {

  if(ifs<1||ifs>2){
    std::cout << "DTSC::nCand: wrong track number: " << ifs;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }
  return _incand[ifs-1].size();

}


DTSectCollCand*
DTSC::getDTSectCollCand(int ifs, unsigned n) const {

  if(ifs<1||ifs>2){
    std::cout << "DTSC::getDTSectCollCand: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>nCand(ifs)) {
    std::cout << "DTSC::getDTSectCollCand: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  std::vector<DTSectCollCand*>::const_iterator p = _incand[ifs-1].begin()+n-1;
  return (*p);

}


void
DTSC::addDTSectCollCand(DTSectCollCand* cand) {

  int ifs = (cand->isFirst()) ? 0 : 1;
 
  _incand[ifs].push_back(cand); 

}


const DTTracoTrigData*
DTSC::getTracoT(int ifs, unsigned n) const {

  if(ifs<1||ifs>2){
    std::cout << "DTSC::getTracoT: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>nCand(ifs)) {
    std::cout << "DTSC::getTracoT: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  return getDTSectCollCand(ifs, n)->tracoTr();

}


DTSectCollCand*
DTSC::getTrack(int n) const {

  if(n<1 || n>nTracks()) {
    std::cout << "DTSC::getTrack: requested track not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  std::vector<DTSectCollCand*>::const_iterator p = _outcand.begin()+n-1;

  return (*p);

}
