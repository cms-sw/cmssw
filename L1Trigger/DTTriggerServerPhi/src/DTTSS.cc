//-------------------------------------------------
//
//   Class: DTTSS.cpp
//
//   Description: Implementation of DTTSS trigger algorithm
//
//
//   Author List:
//   C. Grandi
//   Modifications:
//   04/01/2007 : C. Battilana local config update
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSS.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSCand.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSPhi.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <algorithm>

//----------------
// Constructors --
//----------------
DTTSS::DTTSS(int n) : _n(n), _ignoreSecondTrack(0) {

  // reserve the appropriate amount of space for vectors
  //_tctrig[0].reserve(DTConfigTSPhi::NTCTSS);
  //_tctrig[1].reserve(DTConfigTSPhi::NTCTSS);
  //_outcand.reserve(2);
  _logWord1 = "1/----";
  _logWord2 = "2/----";

}


//--------------
// Destructor --
//--------------
DTTSS::~DTTSS(){
  clear();
}


//--------------
// Operations --
//--------------

void
DTTSS::clear() {
  _ignoreSecondTrack=0;
  for(int itk=0;itk<=1;itk++){
    // content of _tctrig is deleted in the DTTSPhi
    _tctrig[itk].clear();
  }
  // content of _outcand is deleted in the DTTSPhi
  _outcand.clear(); 

  // log words
  _logWord1 = "1/----";
  _logWord2 = "2/----";
}

void
DTTSS::run() {

  if(config()->debug()){
    std::cout << "DTTSS::run: Processing DTTSS number " << _n << " : ";
    std::cout << nFirstT() << " first & " << nSecondT() << " second tracks" << std::endl;
  }

  if(nFirstT()<1)return; // skip if no first tracks
  //
  // SORT 1
  //
  // debugging
  if(config()->debug()){
    std::cout << "Vector of first tracks in DTTSS: " << std::endl;
    std::vector<DTTSCand*>::const_iterator p;
    for(p=_tctrig[0].begin(); p!=_tctrig[0].end(); p++) {
      (*p)->print();
    }
  }
  // end debugging

  DTTSCand* first=sortTSS1();
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
    std::cout << "Vector of second tracks (including carry) in DTTSS: " << std::endl;
    for(p=_tctrig[1].begin(); p!=_tctrig[1].end(); p++) {
      (*p)->print();
    }
  }
  // end debugging
  
  DTTSCand* second=sortTSS2();
  if(second!=0) {
    _outcand.push_back(second); 
  }

}

DTTSCand*
DTTSS::sortTSS1() {

  // Do a sort 1
  DTTSCand* best=0;
  DTTSCand* carry=0;
  std::vector<DTTSCand*>::iterator p;
  for(p=_tctrig[0].begin(); p!=_tctrig[0].end(); p++) {
    DTTSCand* curr= (*p) ? (*p) : 0;
    // SM sector collector Set bits for tss
    curr->setBitsTss(); 
    if(curr->dataword()==0x1ff)continue;   
    _logWord1[1+curr->TcPos()] = (curr->isFirst()) ? '1' : '2';
//     std::cout << "Running TSS: --->curr->dataword() sort 1 " << curr->dataword()  << std::endl;
    if(best==0){
      best=curr;
    } 
    else if((*curr)<=(*best)){
      carry=best;
      best=curr;
    } 
    else if(carry==0){
      carry=curr;
    } 
    else if((*curr)<=(*carry)){
      carry=curr;
    }
  }

  // Ghost 1 suppression: use carry only if not suppressed

  if(carry!=0) { // A carry is present

  // Carry enabled if correlated and TRACO is next to best
    bool inner_or_corr;
    if(config()->TssGhost1Corr())
      {inner_or_corr=carry->isInner() || carry->isCorr();

      }
    else 
      {inner_or_corr=carry->isInner(); 

    }
    if(config()->TssGhost1Flag()<2 && (       // Carry isn't always suppressed
       config()->TssGhost1Flag()==0 ||        // Carry always enabled
       //       carry->isInner() ||                    // Carry is inner
       inner_or_corr ||                       // carry is inner or corr
       abs(carry->TcPos()-best->TcPos())!=1)  // Carry not adj. to best   
                                             ) {
       // add carry to second tracks for sort 2
      carry->setSecondTrack(); // change value of first/second track bit
      carry->setBitsTss();     // set quality bits as for second tracks
      _tctrig[1].push_back(carry); // add to list of second tracks
    } else {
      _logWord1[1+carry->TcPos()] = 'g';
    }
  }

  /*
  if(carry!=0 && config()->TssGhost1Flag()<2){ // Carry isn't always suppressed
    if(config()->TssGhost1Flag()==0 ||           // Carry always enabled
       carry->isInner() ||                       // Carry is inner
       abs(carry->TcPos()-best->TcPos())!=1) {   // Carry not adj. to best   
       // add carry to second tracks to for sort 2
      carry->setSecondTrack(); // change value of first/second track bit
      carry->setBitsTss();     // set quality bits as for second tracks
      _tctrig[1].push_back(carry); // add to list of second tracks
    }
  }
  */
  //std::cout << " best TSS sort 1 = " << best->dataword() << std::endl;
  //std::cout << " SM end of TSS sort 1: best = " <<  std::endl;
  //best->print();
  
  return best;

}

DTTSCand*
DTTSS::sortTSS2() {

  // Check if there are second tracks
  if(nTracks()<1){
    std::cout << "DTTSS::DTTSSsort2: called with no first track.";
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  if(_ignoreSecondTrack){
    // At the time being if a a first track at the following BX is present,
    // the carry is thrown
    //    std::vector<DTTSCand*>::iterator p;
    //    for(p=_tctrig[1].begin(); p!=_tctrig[1].end(); p++) {
    //      if((*p)->isCarry()) return (*p);
    //    }
    // Fill log word
    std::vector<DTTSCand*>::iterator p;
    for(p=_tctrig[1].begin(); p!=_tctrig[1].end(); p++)
      if(!(*p)->isCarry())_logWord2[1+(*p)->TcPos()] = 'o'; // out of time
    return 0;
  }

  // If second tracks are always suppressed skip processing
  if(config()->TssGhost2Flag()==3) {
    // Fill log word
    std::vector<DTTSCand*>::iterator p;
    for(p=_tctrig[1].begin(); p!=_tctrig[1].end(); p++)
      _logWord2[1+(*p)->TcPos()] = 'G';
    return 0;
  }

  // If no first tracks at the following BX, do a sort 2
  DTTSCand* best=getTrack(1);
  DTTSCand* second=0;
  std::vector<DTTSCand*>::iterator p;
  for(p=_tctrig[1].begin(); p!=_tctrig[1].end(); p++) {
    DTTSCand* curr=(*p);
    // SM sector collector set bits for tss
        curr->setBitsTss(); 
//     std::cout << "Running TSS sort 2: --- curr->dataword() "  << curr->dataword()  << std::endl;
    if(!curr->isCarry()) {
      _logWord2[1+curr->TcPos()] = (curr->isFirst()) ? '1' : '2';
      if(curr->dataword()==0x1ff)continue;
    }
    // ghost 2 suppression: skip track if suppressed
    if(config()->TssGhost2Flag()!=0){    // 2nd tracks not always enabled

       bool inner_or_corr;
       if(config()->TssGhost2Corr())
          {inner_or_corr=curr->isInner() || curr->isCorr();

          }
       else
          {inner_or_corr=curr->isInner();

	  }

      if(
	 //!curr->isInner() &&               // outer track 
         !inner_or_corr &&                    // outer and not corr
         curr->TcPos()==best->TcPos()) {   // same correlator of 1st track
        if(config()->TssGhost2Flag()==2 ||   // do not look to corr bit of 1st
           ( (!best->isCorr() ) && config()->TssGhost2Flag()!=4)  || // skip if best is not corr
	   ( (!best->isCorr() ) && best->isInner() && config()->TssGhost2Flag()==4) )   // skip only if best is inner and not corr
 {                
	  _logWord2[1+curr->TcPos()] = 'G';
// 	  std::cout << " skip sort 2 in TSS" << std::endl;
          continue;                            // skip track
        }
      }
    }
    if(second==0){
      second=curr;
    } 
    else if((*curr)<=(*second)){
      second=curr;
    } 
  }
 //  if(!second==0) {std::cout << " best sort 2 = " << second->dataword() << std::endl;
//   std::cout << " SM end of TSS sort 2: second = " <<  std::endl;
//   second->print(); }

  return second;

}

void
DTTSS::addDTTSCand(DTTSCand* cand) {
  int ifs = (cand->isFirst()) ? 0 : 1;
  //  std::cout << "SM DTTSS::addDTTSCand ifs = " << ifs << std::endl;
  _tctrig[ifs].push_back(cand); 
}

unsigned
DTTSS::nTracoT(int ifs) const {
  if(ifs<1||ifs>2){
    std::cout << "DTTSS::nTracoT: wrong track number: " << ifs;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }
  return _tctrig[ifs-1].size();
}

DTTSCand*
DTTSS::getDTTSCand(int ifs, unsigned n) const {
  if(ifs<1||ifs>2){
    std::cout << "DTTSS::getDTTSCand: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>nTracoT(ifs)) {
    std::cout << "DTTSS::getDTTSCand: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  std::vector<DTTSCand*>::const_iterator p=_tctrig[ifs-1].begin()+n-1;
  return (*p);
}

const DTTracoTrigData*
DTTSS::getTracoT(int ifs, unsigned n) const {
  if(ifs<1||ifs>2){
    std::cout << "DTTSS::getTracoT: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>nTracoT(ifs)) {
    std::cout << "DTTSS::getTracoT: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  return getDTTSCand(ifs, n)->tracoTr();
}

DTTSCand*
DTTSS::getTrack(int n) const {
  if(n<1 || n>nTracks()) {
    std::cout << "DTTSS::getTrack: requested track not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  std::vector<DTTSCand*>::const_iterator p = _outcand.begin()+n-1;
  return (*p);
}

DTTSCand*
DTTSS::getCarry() const {
  std::vector<DTTSCand*>::const_iterator p;
  for(p=_tctrig[1].begin(); p!=_tctrig[1].end(); p++)
    if((*p)->isCarry()) return (*p);
  return 0;
}

std::string
DTTSS::logWord(int n) const {
  std::string lw = "";
  switch (n) {
  case 1:
    lw = _logWord1; break;
  case 2:
    lw = _logWord2; break;
  default:
    std::cout << "DTTSS::logWord: requested track not present: " << n;
    std::cout << " empty string returned!" << std::endl;
  }
  return lw;
}





