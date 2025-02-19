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
//   11/11/06 C. Battilana : CoarseSync and Theta included 
//   11/12/06 C. Battilana : New Sector Collector Definition
//   09/01/07 C. Battilana : moved to local conf
//   mar07 - S. Vanini : parameters from DTConfigManager 
//
//
//--------------------------------------------------


//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSectColl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigSectColl.h"
#include "L1Trigger/DTSectorCollector/interface/DTSC.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThCand.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhCand.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSPhi.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTTSTheta.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSCTrigUnit.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <algorithm>

//----------------
// Constructors --
//----------------

DTSectColl::DTSectColl(DTSectCollId id) : _sectcollid(id){

  //_config = _conf_manager->getDTConfigSectColl(_sectcollid);
  
  // create SC Chips
  for(int istat=0;istat<4;istat++){
    for(int istep=0;istep<DTConfigSectColl::NSTEPL-DTConfigSectColl::NSTEPF+1;istep++) {
      _tsc[istep][istat] = new DTSC(istat+1);
    }
  }
  for (int istat=0;istat<5;istat++) _tsphi[istat]=0;
  for (int istat=0;istat<3;istat++) _tstheta[istat]=0;

}

//--------------
// Destructor --
//--------------
DTSectColl::~DTSectColl(){

  localClear();

  for(int istat=0;istat<4;istat++){
    for(int istep=0;istep<DTConfigSectColl::NSTEPL-DTConfigSectColl::NSTEPF+1;istep++){
      delete _tsc[istep][istat];
    }
  }

}


//--------------
// Operations --
//--------------

void
DTSectColl::localClear() {

  // clear all sector collectors
  for(int istat=0;istat<4;istat++){
    for(int istep=0;istep<DTConfigSectColl::NSTEPL-DTConfigSectColl::NSTEPF+1;istep++) {
      _tsc[istep][istat]->clear();
    }
  }
  
  for(int iph=0;iph<2;++iph){
    std::vector<DTSectCollPhCand*>::const_iterator phbi = _incand_ph[iph].begin();
    std::vector<DTSectCollPhCand*>::const_iterator phei = _incand_ph[iph].end();
    for ( std::vector<DTSectCollPhCand*>::const_iterator iphit = phbi;iphit!= phei;++iphit)
      delete (*iphit);
    _incand_ph[iph].clear();
  }
  
  _outcand_ph.clear();
  
  std::vector<DTSectCollThCand*>::const_iterator thb = _incand_th.begin();
  std::vector<DTSectCollThCand*>::const_iterator the = _incand_th.end();
  for ( std::vector<DTSectCollThCand*>::const_iterator ithit = thb;ithit!= the;++ithit)
    delete (*ithit);
  _incand_th.clear();
  
  _outcand_th.clear();
  
}

void
DTSectColl::setConfig (const DTConfigManager *conf){

  _config = conf->getDTConfigSectColl(_sectcollid);

  for(int istat=0;istat<4;istat++){
    for(int istep=0;istep<DTConfigSectColl::NSTEPL-DTConfigSectColl::NSTEPF+1;istep++){
      _tsc[istep][istat]->setConfig(config());
    }
  }

}


void
DTSectColl::addTU(DTSCTrigUnit* tru /*, int flag*/) {

  int stat = tru->station();
  int sect = tru->sector();
  switch (sect){
  case 13:
    stat = 5;
    sect = 4;
    break;
  case 14:
    stat = 5;
    sect = 10;
    break;
  }
   
  if (_sectcollid!=DTSectCollId() &&
      _sectcollid!=DTSectCollId(tru->wheel(),sect)){
    std::cout << "DTSectColl::addTU: Trying to add tru " << tru
	      << " into SectColl " << _sectcollid
	      << " Skipping insertion" << std::endl;
    return;
  }
  
  if (stat<1 || stat >5) {
    std::cout << "DTSectColl::addTU: Wrong station number Skipping insertion" << std::endl;
    return;
  }

  _tsphi[stat-1] = tru->TSPhTrigs();
  if (stat<4) _tstheta[stat-1]=tru->TSThTrigs();
  
  if (_sectcollid==DTSectCollId())
    _sectcollid=DTSectCollId(tru->wheel(),sect);

  // add a Trigger Unit to the Sector Collector
//   if(flag==2) { 
//        _tsphi1 = tru->TSPhTrigs();    // these are the "normal" stations
//        _tsphi2 = 0;
//        _tstheta = tru->TSThTrigs();
//   }
//   else if (flag==0){ 
//     _tsphi1 = tru->TSPhTrigs();
//     _tstheta = 0;
//   }
//   else if(flag==1) {    
//     _tsphi2 = tru->TSPhTrigs();        // these are the "double" stations
//     _tstheta = 0;
//   }
//   // generates SectColl Id from tsphi Id
//   if (flag==2 || flag==0){
//     int sect  = tru->sector();
//     if (sect == 13) sect=4;
//     if (sect == 14) sect=10;
//     _sectcollid=DTSectCollId(tru->wheel(),tru->station(),sect);
//   }

}


void
DTSectColl::loadSectColl() {

  localClear();
  
  std::vector<DTChambPhSegm>::const_iterator p;
  std::vector<DTChambPhSegm>::const_iterator pend;

  for(int istat=1;istat<5;istat++){
    pend=_tsphi[istat-1]->end();
    for(p=_tsphi[istat-1]->begin();p!=pend;p++){
      int step = p->step();
      int fs = (p->isFirst()) ? 1 : 2 ;
      // load trigger
      addTSPhi(step, &(*p), fs, istat);
    }
  }

  if(!(_tsphi[4]==0)){  // only for double stations
    pend=_tsphi[4]->end();
    for(p=_tsphi[4]->begin();p!=pend;p++){
      int step = p->step();
      int fs = (p->isFirst()) ? 1 : 2 ;
      // load trigger
      addTSPhi(step, &(*p), fs ,4);
    }
  }
  std::vector<DTChambThSegm>::const_iterator pth;
  std::vector<DTChambThSegm>::const_iterator pthend;
  
  for(int istat=1;istat<4;istat++){
    pthend=_tstheta[istat-1]->end();
    for(pth=_tstheta[istat-1]->begin();pth!=pthend;pth++){
      int step = pth->step();      
      // load trigger
      addTSTheta(step,  &(*pth), istat);
    }
  }

}


void
DTSectColl::addTSPhi(int step, const DTChambPhSegm* tsmsegm, int ifs, int istat) {

  if(step<DTConfigSectColl::NSTEPF||step>DTConfigSectColl::NSTEPL){
    std::cout << "DTSectColl::addTSPhi: step out of range: " << step;
    std::cout << " trigger not added!" << std::endl;
    return;
  }

  if(istat<1 || istat>4){
    std::cout << "DTSectColl::addTSPhi: station out of SC range: " << istat;
    std::cout << " trigger not added!" << std::endl;
    return;
  }
  
  // Check that a trigger is present, by its code
  if(tsmsegm->oldCode()==0) {
    std::cout << "DTSectColl::addTSPhi -->  code = 0 ! ";
    std::cout << " trigger not added!" << std::endl;
    return;
  }
   
  DTSC* tsc = getDTSC(step,istat);
  
  DTSectCollPhCand* cand = new DTSectCollPhCand(tsc, tsmsegm, ifs);
   
  bool fs = (ifs==1);
  _incand_ph[fs].push_back(cand);
  
  tsc->addDTSectCollPhCand(cand);
  
  // Debugging...
  if(config()->debug()){
    std::cout << "DTSectColl::addTSPhi at step " << step; 
    std::cout << " in SC station " << istat;
    if(ifs==1) {
      std::cout << " (first track)" << std::endl;
    } else {
      std::cout << " (second track)" << std::endl;
    }
  }

}

void
DTSectColl::addTSTheta(int step, const DTChambThSegm* tstsegm, int istat) {

  if(step<DTConfigSectColl::NSTEPF||step>DTConfigSectColl::NSTEPL){
    std::cout << "DTSectColl::addTSTheta: step out of range: " << step;
    std::cout << " trigger not added!" << std::endl;
    return;
  }

  if(istat<1 || istat>5){
    std::cout << "DTSectColl::addTSTheta: station out of SC range: " << istat;
    std::cout << " trigger not added!" << std::endl;
    return;
  }

  // Check if a trigger is present in theta
  bool is_empty=0;
  for (int i=0;i<7;i++) if (tstsegm->position(i)==1){
      is_empty = false;
      break;
    }
  if (is_empty==true) {
    std::cout << "DTSectColl::addTSTheta --> no position bit equal to 1 ! ";
    std::cout << " trigger not added!" << std::endl;
    return;
  }
  
   

  DTSC* tsc = getDTSC(step,istat);
  
  DTSectCollThCand* cand = new DTSectCollThCand(tsc, tstsegm);
   
  _incand_th.push_back(cand);

  tsc->addThCand(cand);
  
  // Debugging...
  if(config()->debug()){
    std::cout << "DTSectColl::addTSTheta at step " << step << std::endl;
  }
  
}


DTSC*
DTSectColl::getDTSC(int step, int istat) const {

  if(step<DTConfigSectColl::NSTEPF||step>DTConfigSectColl::NSTEPL){
    std::cout << "DTSectColl::getDTSC: step out of range: " << step;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  if(istat<1 || istat>4){
    std::cout << "DTSectColl::getDTSC: station out of SC range: " << istat;
    std::cout << " emty pointer returned!" << std::endl;
    return 0;
  }

  return _tsc[step-DTConfigSectColl::NSTEPF][istat-1];  

}


void
DTSectColl::runSectColl() {

  for(int istat=0;istat<4;istat++){
    for(int istep=DTConfigSectColl::NSTEPF;istep<DTConfigSectColl::NSTEPL+1;istep++) {
      
      if(_tsc[istep-DTConfigSectColl::NSTEPF][istat]->nFirstTPh()>0 || _tsc[istep-DTConfigSectColl::NSTEPF][istat]->nCandTh()>0 ) {
           
	_tsc[istep-DTConfigSectColl::NSTEPF][istat]->run();
	
	if(_tsc[istep-DTConfigSectColl::NSTEPF][istat]->nTracksPh()>0) {
	
	  DTSectCollPhCand *cand = _tsc[istep-DTConfigSectColl::NSTEPF][istat]->getTrackPh(1);
	  DTSCPhCache::_cache.push_back(DTSectCollPhSegm(SectCollId(),istep+cand->CoarseSync(),cand->tsTr(),1));
	  _outcand_ph.push_back(cand);
	  
	  if(_tsc[istep-DTConfigSectColl::NSTEPF][istat]->nTracksPh()>1) {
	    
	    DTSectCollPhCand *cand = _tsc[istep-DTConfigSectColl::NSTEPF][istat]->getTrackPh(2);
	    DTSCPhCache::_cache.push_back(DTSectCollPhSegm(SectCollId(),istep+cand->CoarseSync(),cand->tsTr(),2)); 
	    _outcand_ph.push_back(cand);
	  }
	}
	if(_tsc[istep-DTConfigSectColl::NSTEPF][istat]->nTracksTh()>0) {
	  
	  DTSectCollThCand *cand = _tsc[istep-DTConfigSectColl::NSTEPF][istat]->getTrackTh(1);
	  DTSCThCache::_cache.push_back(DTSectCollThSegm(SectCollId(),istep+cand->CoarseSync(),cand->tsTr()));
	  _outcand_th.push_back(cand); // CB getTrackTh non dovrebbe prendere argomenti modificala!

	}
      }
    }
  }
 
  // debugging...
  if(config()->debug()){
    if( DTSCPhCache::_cache.size()>0 || DTSCThCache::_cache.size()>0){
      std::cout << "====================================================" << std::endl;
      std::cout << "                  Sect Coll segments                      " << std::endl;
	if (DTSCPhCache::_cache.size()>0){
	std:: cout << "                  ***Phi Segments***                      " << std:: endl;
	std::vector<DTSectCollPhSegm>::const_iterator pph;
	for(pph=DTSCPhCache::_cache.begin();pph<DTSCPhCache::_cache.end();pph++) {
	  pph->print();
	}
      }
      if (DTSCThCache::_cache.size()>0){
	std:: cout << "                  **Theta Segments**                      " << std:: endl;
	std::vector<DTSectCollThSegm>::const_iterator pth;
	for(pth=DTSCThCache::_cache.begin();pth<DTSCThCache::_cache.end();pth++) {
	  pth->print();
	}
      }
      std::cout << "====================================================" << std::endl;
    }
  }
  //  end debugging
  
}


DTSectCollPhCand*
DTSectColl::getDTSectCollPhCand(int ifs, unsigned n) const {

  if(ifs<1||ifs>2){
    std::cout << "DTSectColl::getDTSectCollPhCand: wrong track number: " << ifs;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }
  if(n<1 || n>nCandPh(ifs)) {
    std::cout << "DTSectColl::getDTSectCollPhCand: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  std::vector<DTSectCollPhCand*>::const_iterator p = _incand_ph[ifs-1].begin()+n-1;
  return (*p);

}

DTSectCollThCand*
DTSectColl::getDTSectCollThCand(unsigned n) const {

  if(n<1 || n>nCandTh()) {
    std::cout << "DTSectColl::getDTSectCollThCand: requested trigger not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  std::vector<DTSectCollThCand*>::const_iterator p = _incand_th.begin()+n-1;
  return (*p);

}


DTSectCollPhCand*
DTSectColl::getTrackPh(int n) const {

  if(n<1 || n>nTracksPh()) {
    std::cout << "DTSectColl::getTrackPh: requested track not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  std::vector<DTSectCollPhCand*>::const_iterator p = _outcand_ph.begin()+n-1;
  return (*p);

}

DTSectCollThCand*
DTSectColl::getTrackTh(int n) const {

  if(n<1 || n>nTracksTh()) {
    std::cout << "DTSectColl::getTrackTh: requested track not present: " << n;
    std::cout << " empty pointer returned!" << std::endl;
    return 0;
  }

  std::vector<DTSectCollThCand*>::const_iterator p = _outcand_th.begin()+n-1;
  return (*p);

}


unsigned
DTSectColl::nCandPh(int ifs) const {

  if(ifs<1||ifs>2){
    std::cout << "DTSectColl::nCandPh: wrong track number: " << ifs;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }

  return _incand_ph[ifs-1].size();

}

unsigned
DTSectColl::nCandTh() const {

  return _incand_th.size();

}

int 
DTSectColl::nSegmPh(int step) {

  int n=0;
  std::vector<DTSectCollPhSegm>::const_iterator p;
   std::vector<DTSectCollPhSegm>::const_iterator endp = DTSCPhCache::end();
  for(p=DTSCPhCache::begin(); p<endp; p++) {   
    if(p->step()==step)n++;  
  } 

  return n;

}

int 
DTSectColl::nSegmTh(int step) {

  int n=0;
  std::vector<DTSectCollThSegm>::const_iterator p;
  std::vector<DTSectCollThSegm>::const_iterator endp = DTSCThCache::end();
  for(p=DTSCThCache::begin(); p>endp; p++) {   
    if(p->step()==step)n++;  
  } 

  return n;

}


const DTSectCollPhSegm*
DTSectColl::SectCollPhSegment(int step, unsigned n) {

  std::vector<DTSectCollPhSegm>::const_iterator p;
  std::vector<DTSectCollPhSegm>::const_iterator endp = DTSCPhCache::end(); 
  for(p=DTSCPhCache::begin();p<endp;p++){
    if(p->step()==step&&((n==1&&p->isFirst())||(n==2&&!p->isFirst())))
      return &(*p); 
  }

  return 0;

}

const DTSectCollThSegm*
DTSectColl::SectCollThSegment(int step) {

  std::vector<DTSectCollThSegm>::const_iterator p;
 std::vector<DTSectCollThSegm>::const_iterator endp = DTSCThCache::end();
  for(p=DTSCThCache::begin();p<endp;p++){
    if(p->step()==step)
      return &(*p); 
  }

  return 0;

}
