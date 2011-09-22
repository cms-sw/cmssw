//-------------------------------------------------
//
//   Class: DTTSM.cpp
//
//   Description: Implementation of DTTSM trigger algorithm
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   S. Marcellini, D. Bonacorsi 
//   04/01/2007 : C.Battilana local config update
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSM.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSPhi.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSCand.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <algorithm>

//----------------
// Constructors --
//----------------
// DBSM-doubleTSM
DTTSM::DTTSM(int n) : _n(n), _ignoreSecondTrack(0) {
  // reserve the appropriate amount of space for vectors
  _incand[0].reserve(DTConfigTSPhi::NTSSTSM);
  _incand[1].reserve(DTConfigTSPhi::NTSSTSM);
  _outcand.reserve(2);
}


//--------------
// Destructor --
//--------------
DTTSM::~DTTSM(){
  clear();
}


//--------------
// Operations --
//--------------

void
DTTSM::clear() {
  _ignoreSecondTrack=0;
  for(int itk=0;itk<=1;itk++){
    // content of _incand is deleted by DTTSPhi
    _incand[itk].clear();
  }
  // content of _outcand is deleted by DTTSPhi
  _outcand.clear(); 
}

void
DTTSM::run(int bkmod) {
  
  if(config()->debug()){
    std::cout << "DTTSM::run: Processing DTTSM: ";
    std::cout << nFirstT() << " first & " << nSecondT() << " second tracks" << std::endl;
  }
  
  if(nFirstT()<1)return; // skip if no first tracks
  //
  // SORT 1
  //
  
  // debugging
  if(config()->debug()){
    std::cout << "Vector of first tracks in DTTSM: " << std::endl;
    std::vector<DTTSCand*>::const_iterator p;
    for(p=_incand[0].begin(); p!=_incand[0].end(); p++) {
      (*p)->print();
    }
  }
  // end debugging
  
  
  DTTSCand* first=sortTSM1(bkmod);
  if(first!=0) {
    _outcand.push_back(first); 
  }
  if(nSecondT()<1)return; // skip if no second tracks

  //
  // SORT 2
  //
  
  // debugging
  if(config()->debug()){
    std::vector<DTTSCand*>::const_iterator p;
    std::cout << "Vector of second tracks (including carry) in DTTSM: " << std::endl;
    for(p=_incand[1].begin(); p!=_incand[1].end(); p++) {
      (*p)->print();
    }
  }
  // end debugging
  
  DTTSCand* second=sortTSM2(bkmod);
  if(second!=0) {
    
    _outcand.push_back(second); 
  }
  
}

DTTSCand*
DTTSM::sortTSM1(int bkmod) {
  // Do a sort 1
  DTTSCand* best=0;
  DTTSCand* carry=0;
  std::vector<DTTSCand*>::iterator p;
  for(p=_incand[0].begin(); p!=_incand[0].end(); p++) {
    DTTSCand* curr=(*p);
    
    if ( bkmod == 1 ) { // NORMAL mode ---> sorting on dataword
      curr->setBitsTss(); // maybe not necessary, as they are the same as for TSS in the default
    } else if( bkmod == 0) { //  { // BACKUP mode ---> sorting on modified dataword
      curr->setBitsBkmod();
    } else {
      std::cout << "DTTSM::sortTSM1:  bkmod not properly assigned!" << std::endl;
    }
    
    
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
    } //else { }    
  }
  
  
  // Ghost 1 suppression: use carry only if not suppressed
  if(carry!=0) {   // A carry is present
    
    // Carry enabled if correlated and TRACO is next to best
    bool inner_or_corr;
    if(config()->TsmGhost1Corr()) {
      inner_or_corr=carry->isInner() || carry->isCorr();  
    } else {
      inner_or_corr=carry->isInner();
    }
    
    if(config()->TsmGhost1Flag()<2){ // Carry isn't always suppressed
      // check if adjacent DTTracoChips
      int adj = (carry->tssNumber()==best->tssNumber()+1 &&          //next DTTracoChip
		 best->TcPos()==DTConfigTSPhi::NTCTSS && carry->TcPos()==1 ) ||
	(carry->tssNumber()==best->tssNumber()-1 &&         // prev DTTracoChip
	 best->TcPos()==1 && carry->TcPos()==DTConfigTSPhi::NTCTSS ) ||
	(carry->tssNumber()==best->tssNumber() &&           // same DTTracoChip
	 abs(carry->TcPos()-best->TcPos())==1                     );   
      
      if(config()->TsmGhost1Flag()==0 ||          // Carry always enabled
	 //       carry->isInner() ||                      // Carry is inner
	 inner_or_corr ||                         // Carry is inner or corr
	 !adj                          ) {        // Carry not adj. to best   
	// add carry to second tracks to for sort 2
	carry->setSecondTrack(); // change value of first/second track bit
	// NEW DESIGN: DTTSM is not configurable!
	// carry->setBitsTsm();     // set quality bits as for second tracks
	_incand[1].push_back(carry); // add to list of second tracks
      }
    }
  }
  // best->print();
  return best;
}




DTTSCand*
DTTSM::sortTSM2(int bkmod) {

  // If second tracks are always suppressed skip processing
  if(config()->TsmGhost2Flag()==3)return 0;
  
  // Check if there are second tracks
  if(nTracks()<1){
    std::cout << "DTTSM::sortTSM2: called with no first track.";
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  
  // If a first track at the following BX is present, ignore second tracks of any kind
  if(_ignoreSecondTrack){
    std::vector<DTTSCand*>::iterator p;
    for(p=_incand[1].begin(); p!=_incand[1].end(); p++) {
      if((*p)->isCarry()) return (*p);
    }
    return 0;
  }
  
  // If no first tracks at the following BX, do a sort 2
  DTTSCand* best=getTrack(1);
  DTTSCand* second=0;
  std::vector<DTTSCand*>::iterator p;
  for(p=_incand[1].begin(); p!=_incand[1].end(); p++) {
    DTTSCand* curr=(*p);
    // ghost 2 suppression: skip track if suppressed
    // this is not needed if config of DTTSM == config of DTTSS
    
    bool inner_or_corr;
    if(config()->TsmGhost2Corr())
      {inner_or_corr=curr->isInner() || curr->isCorr();
      }
    else
      {inner_or_corr=curr->isInner();
      }
    
    if(config()->TsmGhost2Flag()!=0){     // 2nd tracks not always enabled
      if(
	 //!curr->isInner() &&                // outer track
         !inner_or_corr &&                    // outer and not corr
         (curr->tssNumber()==best->tssNumber() &&
          curr->TcPos()==best->TcPos()) ) { // same correlator of 1st track
        if(config()->TsmGhost2Flag()==2 ||    // do not look to corr bit of 1st
           ( (!best->isCorr() ) &&  config()->TsmGhost2Flag()!=4 ) || // skip if best is not corr
           ( (!best->isCorr() ) &&  best->isInner() && config()->TsmGhost2Flag()==4) )   // skip only if best is inner and not corr
	  {                 
	    continue;                             // skip track
	  }
      }
    }
    
    // added DBSM
    // SM double TSM    if ( bkmod == 1 ) { // NORMAL mode ---> sorting with <
    if ( bkmod == 1 ) { // NORMAL mode ---> sorting on dataword
      curr->setBitsTss(); // maybe not necessary, as they are the same as for TSS in the default
    } else if( bkmod == 0) { //  { // BACKUP mode ---> sorting on modified dataword
      curr->setBitsBkmod();
    } else {
      std::cout << " DTTSM::sortTSM2 bkmod not properly assigned!" << std::endl;
    }
    
    
    
    // added DBSM
    // SM double TSM    if ( bkmod == 1 ) { // NORMAL mode ---> sorting with <
    
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
DTTSM::addCand(DTTSCand* cand) {
  // NEW DESIGN: DTTSM is not configurable!
  //  cand->resetCarry(); // reset carry information
  //  cand->setBitsTsm(); // set quality bits for DTTSM sorting
  _incand[(1-cand->isFirst())].push_back(cand); 
}

unsigned
DTTSM::nCand(int ifs) const {
  if(ifs<1||ifs>2){
    std::cout << "DTTSM::nCand: wrong track number: " << ifs;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }

  return _incand[ifs-1].size();
}

DTTSCand*
DTTSM::getDTTSCand(int ifs, unsigned n) const {
  if(ifs<1||ifs>2){
    std::cout << "DTTSM::getDTTSCand: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>nCand(ifs)) {
    std::cout << "DTTSM::getDTTSCand: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  std::vector<DTTSCand*>::const_iterator p = _incand[ifs-1].begin()+n-1;
  return (*p);
}

const DTTracoTrigData*
DTTSM::getTracoT(int ifs, unsigned n) const {
  if(ifs<1||ifs>2){
    std::cout << "DTTSM::getTracoT: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>nCand(ifs)) {
    std::cout << "DTTSM::getTracoT: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  return getDTTSCand(ifs, n)->tracoTr();
}

DTTSCand*
DTTSM::getTrack(int n) const {

  if(n<1 || n>nTracks()) {
    std::cout << "DTTSM::getTrack: requested track not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  std::vector<DTTSCand*>::const_iterator p = _outcand.begin()+n-1;
  return (*p);
}






