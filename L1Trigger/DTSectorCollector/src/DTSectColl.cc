//-------------------------------------------------
//
//   Class: DTSectColl.cpp
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
#include "L1Trigger/DTSectorCollector/interface/DTSectColl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTUtilities/interface/DTConfig.h"
#include "L1Trigger/DTSectorCollector/interface/DTSC.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollCand.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSPhi.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSCTrigUnit.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <algorithm>

//----------------
// Constructors --
//----------------

DTSectColl::DTSectColl(DTConfig* config) :
  _config(config) {
  
  // create Sector Collectors
  for(int is=0;is<DTConfig::NSTEPL-DTConfig::NSTEPF+1;is++) {
    _tsc[is] = new DTSC(_config);
  }

}

//--------------
// Destructor --
//--------------
DTSectColl::~DTSectColl(){

  for(int is=0;is<DTConfig::NSTEPL-DTConfig::NSTEPF+1;is++){
    delete _tsc[is];
  }

  //SM: new  clear();
  //localClear();

}


//--------------
// Operations --
//--------------

void
DTSectColl::localClear() {

  // clear all sector collectors
  for(int is=0;is<DTConfig::NSTEPL-DTConfig::NSTEPF+1;is++) {
    _tsc[is]->clear();
  }

}


void
DTSectColl::addTU(DTSCTrigUnit* tru, int flag) {

  // add a Trigger Unit to the Sector Collector

  if(flag==2) { 
       _tsphi1 = tru->TSPhTrigs();    // these are the "normal" stations
       _tsphi2 = 0;}
  else if (flag==0){ 
    _tsphi1 = tru->TSPhTrigs(); }
  else if(flag==1) {    
    _tsphi2 = tru->TSPhTrigs();        // these are the "double" stations
  }

}


void
DTSectColl::loadSectColl() {

  localClear();
  
  std::vector<DTChambPhSegm>::const_iterator p;
  std::vector<DTChambPhSegm>::const_iterator p1end=_tsphi1->end();

  for(p=_tsphi1->begin();p!=p1end;p++){
    int step = p->step();
    int fs = (p->isFirst()) ? 1 : 2 ;

    // load trigger
    addTSPhi(step,  &(*p), fs );
  }

  if(!(_tsphi2==0)){  // only for double stations
    std::vector<DTChambPhSegm>::const_iterator p2end=_tsphi2->end();

    for(p=_tsphi2->begin();p!=p2end;p++){
      int step = p->step();
      int fs = (p->isFirst()) ? 1 : 2 ;
      
      // load trigger
      addTSPhi(step, &(*p), fs );
    }
  }

}


void
DTSectColl::addTSPhi(int step, const DTChambPhSegm* tsmsegm, int ifs) {

  if(step<DTConfig::NSTEPF||step>DTConfig::NSTEPL){
    std::cout << "DTSectColl::addTSPhi: step out of range: " << step;
    std::cout << " trigger not added!" << std::endl;
    return;
  }
  
  // Check that a trigger is present, by its code
  if(tsmsegm->oldCode()==0) {
    std::cout << "DTSectColl::loadSectColl -->  code = 0 ! ";
    std::cout << " trigger not added!" << std::endl;
    return;
  }
   
  DTSC* tsc = getDTSC(step);
  
  DTSectCollCand* cand = new DTSectCollCand(tsc, tsmsegm, ifs);
   
  _tstrig[step-DTConfig::NSTEPF].push_back(cand);
  
  tsc->addDTSectCollCand(cand);
  
  // Debugging...
  if(config()->debug()>2){
    std::cout << "DTSectColl::addTSPhi at step " << step; 
    if(ifs==1) {
      std::cout << " (first track)" << std::endl;
    } else {
      std::cout << " (second track)" << std::endl;
    }
  }
  // end debugging
  
}


DTSC*
DTSectColl::getDTSC(int step) const {

  if(step<DTConfig::NSTEPF||step>DTConfig::NSTEPL){
    std::cout << "DTSectColl::getDTSectColl: step out of range: " << step;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  return _tsc[step-DTConfig::NSTEPF];  

}


void
DTSectColl::runSectColl() {

  for(int is=DTConfig::NSTEPF;is<DTConfig::NSTEPL+1;is++) {
 
    if(_tsc[is-DTConfig::NSTEPF]->nFirstT()>0) {
           
      _tsc[is-DTConfig::NSTEPF]->run();
      
      if(_tsc[is-DTConfig::NSTEPF]->nTracks()>0) {
	
	_cache.push_back(DTChambPhSegm(_tsc[is-DTConfig::NSTEPF]->getTrack(1)->tsTr()->ChamberId(),is,_tsc[is-DTConfig::NSTEPF]->getTrack(1)->tsTr()->tracoTrig(),1));
		
	if(_tsc[is-DTConfig::NSTEPF]->nTracks()>1) {
	  
	  _cache.push_back(DTChambPhSegm(_tsc[is-DTConfig::NSTEPF]->getTrack(2)->tsTr()->ChamberId(),is,_tsc[is-DTConfig::NSTEPF]->getTrack(2)->tsTr()->tracoTrig(),2));
	  	  
	}
      }
    }
  }
  // Sector collector section end
  
  // debugging...
  if(config()->debug()>0){
    if(_cache.size()>0){
      std::cout << "====================================================" << std::endl;
      std::cout << "                  Sect Coll segments                      " << std::endl;
      std::vector<DTChambPhSegm>::const_iterator p;
      for(p=_cache.begin();p<_cache.end();p++) {
  	p->print();
      }
      std::cout << "====================================================" << std::endl;
    }
  }
  //  end debugging
    
}


DTSectCollCand*
DTSectColl::getDTSectCollCand(int ifs, unsigned n) const {

  if(ifs<1||ifs>2){
    std::cout << "DTSectColl::getDTSectCollCand: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>nCand(ifs)) {
    std::cout << "DTSectColl::getDTSectCollCand: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  std::vector<DTSectCollCand*>::const_iterator p = _incand[ifs-1].begin()+n-1;
  return (*p);

}


DTSectCollCand*
DTSectColl::getTrack(int n) const {

  if(n<1 || n>nTracks()) {
    std::cout << "DTSectColl::getTrack: requested track not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  std::vector<DTSectCollCand*>::const_iterator p = _outcand.begin()+n-1;
  return (*p);

}


unsigned
DTSectColl::nCand(int ifs) const {

  if(ifs<1||ifs>2){
    std::cout << "DTSectColl::nCand: wrong track number: " << ifs;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }

  return _incand[ifs-1].size();

}


int 
DTSectColl::nSegm(int step) {

  int n=0;
  std::vector<DTChambPhSegm>::const_iterator p;
  for(p=begin(); p<end(); p++) {   
    if(p->step()==step)n++;  
  } 

  return n;

}


const DTChambPhSegm*
DTSectColl::segment(int step, unsigned n) {

  std::vector<DTChambPhSegm>::const_iterator p; 
  for(p=begin();p<end();p++){
    if(p->step()==step&&((n==1&&p->isFirst())||(n==2&&!p->isFirst())))
      return &(*p); 
  }

  return 0;

}
