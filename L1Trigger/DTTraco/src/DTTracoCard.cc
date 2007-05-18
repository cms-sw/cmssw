//-------------------------------------------------
//
//   Class: DTTracoCard
//
//   Description: Contains active DTTracoChips
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   X/03 Sara Vanini
//   22/VI/04 SV: last trigger code update
//   13/XII/04 SV: Zotto's traco acceptance routine implemented
//   V/05 SV: NEWGEO
//   9/V/05 SV: mt ports ing K units, bug fixed  
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTraco/interface/DTTracoCard.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "L1Trigger/DTTraco/interface/DTTracoChip.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrig.h"
#include "L1Trigger/DTBti/interface/DTBtiCard.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTTSTheta.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <cmath>
#include <utility>  
#include <iomanip>
#include <math.h>

//----------------
// Constructors --
//----------------

DTTracoCard::DTTracoCard(DTTrigGeom* geo, DTBtiCard* bticard,
  DTTSTheta* tstheta, edm::ParameterSet& traco_pset) : DTGeomSupplier(geo) , 
  _bticard(bticard), _tstheta(tstheta) { 

  // get DTConfigTraco configuration
  _configTraco = new DTConfigTraco(traco_pset);


  // Set K acceptances of DTTracoChip MT ports: Ktraco = Xinner - Xouter 
  float h = geom()->cellH();
  float pitch = geom()->cellPitch();
  float distsl = geom()->distSL();
  float K0 = config()->BTIC();
  float shiftSL = geom()->phiSLOffset() / pitch * K0;

  // mt  ports from orca geometry: this is always the case with new DTConfig
  //if(config()->trigSetupGeom() != 1){
  {
    // Master Plane
    int i = 0;
    for(i=0;i<DTConfig::NBTITC;i++){
      float Xin_min     =  (i + DTConfig::NBTITC) * K0 + shiftSL;
      float Xin_max     =  Xin_min + K0;
      float Xout_min    =  0;
      float Xout_max    =  3 * DTConfig::NBTITC * K0;
      _PSIMAX[i]  =  int( 2.*h/distsl * (Xin_max - Xout_min) + K0 + 1.01 );
      _PSIMIN[i]  =  int( 2.*h/distsl * (Xin_min - Xout_max) + K0 );
    }

    // Slave Plane
    for(i=0;i<3*DTConfig::NBTITC;i++){
      float Xin_min     =  (DTConfig::NBTITC) * K0 + shiftSL;
      float Xin_max     =  2. * DTConfig::NBTITC * K0 + shiftSL;
      float Xout_min    =  i * K0;
      float Xout_max    =  Xout_min + K0;
      _PSIMAX[DTConfig::NBTITC+i]  =  int( 2.*h/distsl * (Xin_max - Xout_min) + K0 + 1.01 );
      _PSIMIN[DTConfig::NBTITC+i]  =  int( 2.*h/distsl * (Xin_min - Xout_max) + K0 );
    }
  }

/* this is obsolete with new DTConfig
  if(config()->trigSetupGeom()==1){
    //SV TB2003: acceptance from LH,LL,CH,CL,RH,RL parameters...
    //bti 1,2,3,4
    int supl=1;
    int cell=1+16;
    for(int n=0;n<4;n++){
      _PSIMIN[n]=config()->CL_bti(cell,supl);
      _PSIMAX[n]=config()->CH_bti(cell,supl);
      cell++;
    }
    supl=3;
    cell=1+16;
    for(int n=4;n<8;n++){
      _PSIMIN[n]=config()->RL_bti(cell,supl);
      _PSIMAX[n]=config()->RH_bti(cell,supl);
      cell++;
    }
    cell=5+16;
    for(int n=8;n<12;n++){
      _PSIMIN[n]=config()->CL_bti(cell,supl);
      _PSIMAX[n]=config()->CH_bti(cell,supl);
      cell++;
    }
    cell=9+16;
    for(int n=12;n<16;n++){
      _PSIMIN[n]=config()->LL_bti(cell,supl);
      _PSIMAX[n]=config()->LH_bti(cell,supl);
      cell++;
    }
  }  
*/

  // debugging
  if(config()->debug()==4){
    //if(wheel()==2&&station()==3&&sector()==1){ // only 1 chamber
      std::cout << "Acceptance of mt ports for offset (cell unit) " 
           << geom()->phiSLOffset() / pitch << std::endl;
      for(int i=0;i<4*DTConfig::NBTITC;i++){
        std::cout << "Port " << i+1 << " : ";
        std::cout << _PSIMIN[i] << " --> " << _PSIMAX[i] << std::endl;
      }
    //}
  }// end debugging

}

//--------------
// Destructor --
//--------------

DTTracoCard::~DTTracoCard(){

  localClear();
  delete _configTraco;

}

//--------------
// Operations --
//--------------

void
DTTracoCard::clearCache(){

  TRACOCache::clearCache();
  localClear();

}

void
DTTracoCard::localClear(){
  // Clear the map
  for(TRACO_iter p=_tracomap.begin();p!=_tracomap.end();p++){
    delete (*p).second;
  }
  _tracomap.clear();
}

void 
DTTracoCard::loadTRACO() {
 
  localClear();

  if(config()->debug()==4){
    std::cout << "DTTracoCard::loadTRACO called for wheel=" << wheel() ;
    std::cout <<                                ", station=" << station();
    std::cout <<                                ", sector="  << sector() << std::endl;
  }

  int maxtc = int(ceil( float(geom()->nCell(1)) / float(DTConfig::NBTITC) ));

  // loop on all BTI triggers
  std::vector<DTBtiTrigData>::const_iterator p;
  std::vector<DTBtiTrigData>::const_iterator pend=_bticard->end();
  for(p=_bticard->begin();p!=pend;p++){
    if(config()->debug()>1){
      std::cout << "Found bti trigger: ";
      (*p).print();
    }
    // BTI number
    int nbti = (*p).btiNumber();
    int nsl  = (*p).btiSL(); 
    int step = (*p).step();

    // assign BTI to TRACO
    int ntc = static_cast<int>((nbti-1)/DTConfig::NBTITC)+1;
    if( ntc<1 || ntc>maxtc ) 
      continue;

    // position inside cor.: 
    int pos = nbti-(ntc-1)*DTConfig::NBTITC;

    // store trigger in TRACO. Create TRACO if it doesn't exist
    // SV tb2003 : check if traco is connected!

    // Load master TRACO plane
    if( nsl==1 ) {
      if( /*config()->usedTraco(ntc)==1 &&*/ ( ntc>0 && ntc<=maxtc ) )
        activeGetTRACO(ntc)->add_btiT( step, pos, &(*p) );
      else{
        if(config()->debug()==4)
          std::cout << "ATTENTION: traco " << ntc << " is disconnected!" << std::endl;
      }  
    } 

    // Load slave TRACO plane
    if( nsl==3 ) {
      // 3 TRACO's
      for(int tci=-1;tci<=1;tci++) {
        if( /*config()->usedTraco(ntc+tci)==1 &&*/ ( (ntc+tci)>0 && (ntc+tci)<=maxtc ) )
          activeGetTRACO(ntc+tci)->add_btiT( step, pos+8-4*tci, &(*p) );
        else{
          if(config()->debug()==4)
            std::cout << "ATTENTION: traco " << ntc+tci << " is disconnected!" << std::endl;
        }
      } 
    }

  }//end loop on bti trigs
}

void 
DTTracoCard::runTRACO() {

  if(config()->debug()==4){
    std::cout << "DTTracoCard:runTRACO called for wheel=" << wheel() ;
    std::cout <<                               ", station=" << station();
    std::cout <<                               ", sector="  << sector();
    std::cout << ", " << _tracomap.size() << " TRACOs with signal" << std::endl;
  }

  // run TRACO algorithm on all non-empty TRACO
  if(_tracomap.size()>0){

    if(config()->debug()>0){
      std::cout << "====================================================" << std::endl;
      std::cout << "              TRACO triggers                        " << std::endl; 
    }


    TRACO_iter ptraco;
    for(ptraco=_tracomap.begin(); ptraco!=_tracomap.end(); ptraco++) {
      DTTracoChip* traco = (*ptraco).second;
      traco->run();
      for(int step=DTConfig::NSTEPF; step<=DTConfig::NSTEPL; step++){
        if( traco->nTrig(step)>0 ){ 
          _cache.push_back( traco->triggerData(step,1) );
          /*
          std::cout<<"first bti sl3: "<<geom()->localPosition(DTBtiId(wheel(),station(),sector(),3,1))<<std::endl;
          std::cout<<"traco pos: " << geom()->localPosition((traco->triggerData(step,1).parentId()))<<std::endl; 
          traco->triggerData(step,1).print();
          std::cout<<"pos: " << localPosition(&(traco->triggerData(step,1))) << std::endl;
          std::cout<<"dir: " << localDirection(&(traco->triggerData(step,1))) << std::endl;
          std::cout << std::endl;
          */
        }  
        // Store second track only if no first track at previous BX
	  if( traco->nTrig(step)>1 && traco->useSecondTrack(step) ){
	    _cache.push_back( traco->triggerData(step,2) );
          /*
          std::cout<<"first bti sl3: "<<geom()->localPosition(DTBtiId(wheel(),station(),sector(),3,1))<<std::endl;
          std::cout<<"traco pos: " << geom()->localPosition((traco->triggerData(step,2).parentId()))<<std::endl; 
          traco->triggerData(step,2).print();
          std::cout<<"pos: " << localPosition(&(traco->triggerData(step,2))) << std::endl;
          std::cout<<"dir: " << localDirection(&(traco->triggerData(step,2))) << std::endl;
          std::cout << std::endl;
          */
        }
      }
    }
    if(config()->debug()>0)
      std::cout << "====================================================" << std::endl;
  }
}

DTTracoChip*
DTTracoCard::activeGetTRACO(int n) {

  DTTracoChip* traco=0;
  TRACO_iter ptraco = _tracomap.find(n);
  if( ptraco!=_tracomap.end() ) {
    traco=(*ptraco).second;
  } else {
    traco = new DTTracoChip(this,n,_configTraco);
    _tracomap[n]=traco;
  }
  return traco;
}

DTTracoChip*
DTTracoCard::getTRACO(int n) const {
  TRACO_const_iter ptraco = _tracomap.find(n);
  if( ptraco==_tracomap.end() ) return 0;
  return (*ptraco).second;
}

std::vector<DTTracoChip*> 
DTTracoCard::tracoList() {

  std::vector<DTTracoChip*> blist;

  if(size()<1)return blist;

  for(TRACO_const_iter p=_tracomap.begin();p!=_tracomap.end();p++){
    blist.push_back((*p).second);
  }
  return blist;

}

DTTracoTrig*
DTTracoCard::storeTrigger(DTTracoTrigData td) {
  DTTracoId tracoid = td.parentId();
  if(!(tracoid.wheel()==wheel() &&
       tracoid.sector()==sector() &&
       tracoid.station()==station()) ) return 0;
  std::cout << "DTTracoChip::trigger: trigger not belonging to this card! ";
  std::cout << "card=(" << wheel() << "," << station() << "," << sector() << ") ";
  std::cout << "trig=(" << tracoid.wheel() << "," << tracoid.station() << "," 
       << tracoid.sector() << ")";
  // get the appropriate Traco
  DTTracoChip* traco = activeGetTRACO(tracoid.traco());
  // create a new trigger in the Traco
  DTTracoTrig* trig = new DTTracoTrig(traco,td);
  // add the trigger to the Traco
  traco->addTrig(td.step(),trig);
  // return the trigger
  return trig;
}

/*
LocalPoint 
DTTracoCard::localPosition(const DTTrigData* tr) const {
std::cout<<"oldgeo";

  //@@ patch for Sun 4.2 compiler
  DTTracoTrigData* trig = dynamic_cast<DTTracoTrigData*>(const_cast<DTTrigData*>(tr));
  //  const DTTracoTrigData* trig = dynamic_cast<const DTTracoTrigData*>(tr);
  if(!trig) {
    std::cout << "DTTracoCard::localPosition called with wrong argument!" << std::endl;
    return LocalPoint(0,0,0);
  }
  float x = geom()->localPosition(trig->parentId()).x();
  float y = geom()->localPosition(trig->parentId()).y();
  float z = geom()->localPosition(trig->parentId()).z();

  x += geom()->cellPitch() * ( (float)trig->X() / (float)(config()->BTIC())
                              - 1.5 * (float)(DTConfig::NBTITC) );
  // If not correlated get the position of the SL instead of the chamber center
  if       (trig->posIn()==0 ) {
    z -= 0.5 * geom()->distSL(); // no inner ==> only outer
  } else if(trig->posOut()==0) {
    z += 0.5 * geom()->distSL(); // no outer ==> only inner
  }
  return LocalPoint(x,y,z);
}
*/

LocalPoint 
DTTracoCard::localPosition(const DTTrigData* tr) const {
  //NEWGEO
  DTTracoTrigData* trig = dynamic_cast<DTTracoTrigData*>(const_cast<DTTrigData*>(tr));
  if(!trig) {
    std::cout << "DTTracoCard::localPosition called with wrong argument!" << std::endl;
    return LocalPoint(0,0,0);
  }
  float x = geom()->localPosition(trig->parentId()).x();
  float y = geom()->localPosition(trig->parentId()).y();
  float z = geom()->localPosition(trig->parentId()).z();

  float trig_pos = geom()->cellPitch() * ( (float)trig->X() / (float)(config()->BTIC()));

//  10/7/06 May be not needed anymore in new geometry 
//   if(geom()->posFE(1)==1)
//   trig_pos = -trig_pos;

  x += trig_pos;

  // If not correlated get the position of the SL instead of the chamber center
  // z axis toward vertex
  if(trig->posIn()==0 ) {
    z -= 0.5 * geom()->distSL(); // no inner ==> only outer
  } 
  else if(trig->posOut()==0) {
    z += 0.5 * geom()->distSL(); // no outer ==> only inner
  }
  return LocalPoint(x,y,z);
}

/* OLDGEO
LocalVector 
DTTracoCard::localDirection(const DTTrigData* tr) const {
  //@@ patch for Sun 4.2 compiler
  DTTracoTrigData* trig = dynamic_cast<DTTracoTrigData*>(const_cast<DTTrigData*>(tr));
  //  const DTTracoTrigData* trig = dynamic_cast<const DTTracoTrigData*>(tr);
  if(!trig) {
    std::cout << "DTtracoCard::localDirection called with wrong argument!" << std::endl;
    return LocalVector(0,0,0);
  }
  float r,x,y,z;
  x = -(float)trig->K() * geom()->cellPitch() /
                      (float)(config()->BTIC());
  y = 0;
  z = -geom()->distSL();
  r = sqrt(x*x+z*z);
  x /= r;
  z /= r;
  return LocalVector(x,y,z);
}
*/

LocalVector 
DTTracoCard::localDirection(const DTTrigData* tr) const {
  //NEWGEO
  DTTracoTrigData* trig = dynamic_cast<DTTracoTrigData*>(const_cast<DTTrigData*>(tr));
  if(!trig) {
    std::cout << "DTtracoCard::localDirection called with wrong argument!" << std::endl;
    return LocalVector(0,0,0);
  }

  //FE position
  //int FE = geom()->posFE(3);

  float psi = atan((float)(trig->K())*geom()->cellPitch()
                   /( geom()->distSL() * config()->BTIC()) );

  if(config()->debug()==4)
    std::cout << "K " << trig->K() << " == psi " << psi << " in FE frame " << std::endl;
    
  // (xd,yd,zd) in chamber frame
  float xd=-sin(psi);
  float yd=0;
  float zd=-cos(psi);

  // 10/07/06 Not needed anymore (chages in geometry)
  // if(FE==1){//FE in negative y
  //    xd = - xd;
  //}

 
  if(config()->debug()==4)
    std::cout << "Direction in chamber frame is (" << xd << "," << yd << "," << zd << ")" << std::endl;
 
  return LocalVector(xd,yd,zd);
}

