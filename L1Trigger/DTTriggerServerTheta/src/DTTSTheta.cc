//-------------------------------------------------
//
//   Class: L1MuDTTSTheta.cpp
//
//   Description: Implementation of TS Theta trigger algorithm
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   III/2005 : Sara Vanini NEWGEO update
//   I/2007 : Carlo Battilana Config class update
//   mar07 - S. Vanini : parameters from DTConfigManager 
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTriggerServerTheta/interface/DTTSTheta.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"
#include "L1Trigger/DTBti/interface/DTBtiCard.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------
// Constructors --
//----------------
DTTSTheta::DTTSTheta(DTTrigGeom* geom, DTBtiCard* bticard) : 
  DTGeomSupplier(geom),  _bticard(bticard) {

  for(int i=0;i<DTConfigTSTheta::NSTEPL-DTConfigTSTheta::NSTEPF+1;i++){
    _trig[i].zero();
    _Htrig[i].zero();
    _ntrig[i] = 0;
    _nHtrig[i] = 0;
  }

}


//--------------
// Destructor --
//--------------
DTTSTheta::~DTTSTheta(){
  //delete _config;
}


//--------------
// Operations --
//--------------

void
DTTSTheta::localClear() {
  for(int is=0;is<DTConfigTSTheta::NSTEPL-DTConfigTSTheta::NSTEPF+1;is++){
    // clear input bit masks
    _trig[is].zero();
    _Htrig[is].zero();
    _ntrig[is] = 0;
    _nHtrig[is] = 0;
  }
}

void
DTTSTheta::setConfig(const DTConfigManager *conf){
  
	DTChamberId sid = ChamberId();
	_config = conf->getDTConfigTSTheta(sid);

}

void 
DTTSTheta::loadDTTSTheta() {
  localClear();
  if(station()==4)return;
 
  if(config()->debug()){
    std::cout << "DTTSTheta::loadDTTSTheta called for wheel=" << wheel() ;
    std::cout <<                                ", station=" << station();
    std::cout <<                                ", sector="  << sector() << std::endl;
  }

  // loop on all BTI triggers
  std::vector<DTBtiTrigData>::const_iterator p;
  std::vector<DTBtiTrigData>::const_iterator pend=_bticard->end();
  for(p=_bticard->begin();p!=pend;p++){
    // Look only to BTIs in SL 2
    if (p->btiSL() == 2) {
      // BTI number
      int step = p->step();
      add_btiT( step, &(*p) );
    }
  }

}

void 
DTTSTheta::add_btiT(int step, const DTBtiTrigData* btitrig){
  int n = btitrig->btiNumber();

/*
OBSOLETE! in NEWGEO the bti number order is the correct one!
  // check where the BTI is, and reverse the order in stat 1 and 2 and 3 (only for some sectors)

  if( (wheel()==0 && (sector()==1 ||  sector()==4 || sector()==5 ||
		      sector()==8 || sector()==9 || sector()==12))
      || wheel()==-1 
      || wheel()==-2)
    {n=DTConfigTSTheta::NCELLTH + 1 - n; }
  
*/

  if( n<1 || n>geom()->nCell(2) ) {
    std::cout << "DTTSTheta::add_btiT: BTI out of range: " << n;
    std::cout << " trigger not added!" << std::endl;
    return;
  }
  if(step<DTConfigTSTheta::NSTEPF||step>DTConfigTSTheta::NSTEPL){
    std::cout << "DTTSTheta::add_btiT: step out of range: " << step;
    std::cout << " trigger not added!" << std::endl;
    return;
  }
  _trig[step-DTConfigTSTheta::NSTEPF].set(n-1);
  _ntrig[step-DTConfigTSTheta::NSTEPF]++;

  if(btitrig->code()==8){
    _Htrig[step-DTConfigTSTheta::NSTEPF].set(n-1);
    _nHtrig[step-DTConfigTSTheta::NSTEPF]++;
  }

  if(config()->debug()){
    std::cout << "BTI Trigger added at step " << step;
    std::cout << " to DTTSTheta at position " << n <<  std::endl;
  }
  return;
}

void
DTTSTheta::runDTTSTheta() {
  // Just make a DTChambThSegm for each step and store it
  for(int is=DTConfigTSTheta::NSTEPF;is<DTConfigTSTheta::NSTEPL+1;is++) {
    if(_ntrig[is-DTConfigTSTheta::NSTEPF]>0) {
      int i=0;
      int code[8];
      int pos[8];
      int qual[8];
      for(i=0;i<8;i++) {
	//@@ MULT not implemented:
	pos[i]=btiMask(is)->byte(i).any();
	qual[i]=btiQual(is)->byte(i).any();
	code[i]=pos[i]+qual[i];
      }

      // SM .OR. response of BTI number 57 in previous group of 8 BTIs 

      if(pos[7] > pos[6])    pos[6]=pos[7];
      if(qual[7] > qual[6])   qual[6]=qual[7]; 
      if(code[7] > code[6])   code[6]=code[7]; 

      
      if(config()->debug()){
      std::cout  << " wheel = " << wheel() << " station = " << station() << " sector = " << sector() << std::endl;
	std::cout << " pos :  ";
	for(i=0;i<8;i++) {
	  std::cout << pos[i] << " ";
	}
	std::cout << std::endl;
	std::cout << " qual :  ";
	for(i=0;i<8;i++) {
	  std::cout << qual[i] << " ";
	}
	std::cout << std::endl;
	std::cout << " code :  ";
	for(i=0;i<8;i++) {
	  std::cout << code[i] << " ";

	}
	std::cout << std::endl;
	std::cout << std::endl;
      }

      _cache.push_back(DTChambThSegm(ChamberId(),is,pos,qual));
    }
  }

  // debugging...
  if(config()->debug()){
    if(_cache.size()>0){
      std::cout << "====================================================" << std::endl;
      std::cout << "                 Theta segments                     " << std::endl;
      std::vector<DTChambThSegm>::const_iterator p;
      for(p=_cache.begin();p<_cache.end();p++) {
	p->print();
      }
      std::cout << "====================================================" << std::endl;
    }
  }
  // end debugging
  
}

int 
DTTSTheta::nSegm(int step) {
  int n=0;
  std::vector<DTChambThSegm>::const_iterator p;
  for(p=begin(); p<end(); p++) {
    if(p->step()==step)n++;
  }
  return n;
}

const DTChambThSegm*
DTTSTheta::segment(int step, unsigned n) {
  std::vector<DTChambThSegm>::const_iterator p;
  for(p=begin();p<end();p++){
    if(p->step()==step&&n==1)
      return &(*p);
  }
  return 0;
}

int 
DTTSTheta::nTrig(int step) {
  if(step<DTConfigTSTheta::NSTEPF||step>DTConfigTSTheta::NSTEPL){
    std::cout << "DTTSTheta::nTrig: step out of range " << step;
    std::cout << " 0 returned" << std::endl;
    return 0;
  }
  if(size()>0) return _ntrig[step-DTConfigTSTheta::NSTEPF];
  return 0;
}

int 
DTTSTheta::nHTrig(int step) { 
  if(step<DTConfigTSTheta::NSTEPF||step>DTConfigTSTheta::NSTEPL){
    std::cout << "DTTSTheta::nHTrig: step out of range " << step;
    std::cout << " 0 returned" << std::endl;
    return 0;
  }
  if(size()>0) return _nHtrig[step-DTConfigTSTheta::NSTEPF]; 
  return 0;
}

BitArray<DTConfigTSTheta::NCELLTH>*
DTTSTheta::btiMask(int step) const {
  if(step<DTConfigTSTheta::NSTEPF||step>DTConfigTSTheta::NSTEPL){
    std::cout << "DTTSTheta::btiMask: step out of range " << step;
    std::cout << " empty pointer returned" << std::endl;
    return 0;
  }
  return (BitArray<DTConfigTSTheta::NCELLTH>*)&_trig[step-DTConfigTSTheta::NSTEPF]; 
}

BitArray<DTConfigTSTheta::NCELLTH>*
DTTSTheta::btiQual(int step) const {
  if(step<DTConfigTSTheta::NSTEPF||step>DTConfigTSTheta::NSTEPL){
    std::cout << "DTTSTheta::btiQual: step out of range " << step;
    std::cout << " empty pointer returned" << std::endl;
    return 0;
  }
  return (BitArray<DTConfigTSTheta::NCELLTH>*)&_Htrig[step-DTConfigTSTheta::NSTEPF]; 
}

LocalPoint 
DTTSTheta::localPosition(const DTTrigData* tr) const {
  //const DTChambThSegm* trig = dynamic_cast<const DTChambThSegm*>(tr);
  //@@ Not implemented yet
  return LocalPoint(0,0,0);
}

LocalVector 
DTTSTheta::localDirection(const DTTrigData* tr) const  {
  //const DTChambThSegm* trig = dynamic_cast<const DTChambThSegm*>(tr);
  //@@ Not implemented yet
  return LocalVector(0,0,0);
}

void
DTTSTheta::print(const DTTrigData* trig) const {
  trig->print();
  //@@ coordinate printing not implemented yet
  //@@ rermove this method as soon as the local coordinates are meaningful
  
}
