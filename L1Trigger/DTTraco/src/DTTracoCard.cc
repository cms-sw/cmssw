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
//   30/III/07 SV : config with DTConfigManager every single chip 
//   2/XI/09 SV : bti acceptance windows included
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTraco/interface/DTTracoCard.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
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
  DTTSTheta* tstheta) : DTGeomSupplier(geo) , 
  _bticard(bticard), _tstheta(tstheta) { 

}

//--------------
// Destructor --
//--------------

DTTracoCard::~DTTracoCard(){

localClear();

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
DTTracoCard::setConfig(const DTConfigManager *conf){
  // get traco configuration map  
  DTChamberId sid = ChamberId();
  _conf_traco_map = conf->getDTConfigTracoMap(sid);	
  _debug = conf->getDTTPGDebug();

  // get bti acceptance flag
  _flag_acc = conf->useAcceptParam();

  // get lut computation flag
  _lut_from_db = conf->lutFromDB();

  // get lut configuration for this chamber
  // 100511 SV only if luts are read from OMDS
  if(_lut_from_db)
    _conf_luts = conf->getDTConfigLUTs(sid);

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

  if(debug()){
    std::cout << "DTTracoCard::loadTRACO called for wheel=" << wheel() ;
    std::cout <<                                ", station=" << station();
    std::cout <<                                ", sector="  << sector() << std::endl;
  }

  int maxtc = int(ceil( float(geom()->nCell(1)) / float(DTConfig::NBTITC) ));

  // loop on all BTI triggers
  std::vector<DTBtiTrigData>::const_iterator p;
  std::vector<DTBtiTrigData>::const_iterator pend=_bticard->end();
  for(p=_bticard->begin();p!=pend;p++){
    if(debug()){
      std::cout << "Found bti trigger: ";
      (*p).print();
    }

    // BTI data
    int nbti 	= (*p).btiNumber();
    int nsl  	= (*p).btiSL(); 
    int step 	= (*p).step();
    int K 	= (*p).K();
    DTBtiId id_bti = (*p).parentId();

    DTConfigBti* conf_bti = _bticard->config_bti( id_bti ); 
    int LL 	= conf_bti->LL();
    int LH 	= conf_bti->LH();
    int CL 	= conf_bti->CL();
    int CH 	= conf_bti->CH();
    int RL 	= conf_bti->RL();
    int RH 	= conf_bti->RH();
/*
    if(debug())
      std::cout << "Bti trigger acceptances: \n" 
		<< " LL " << LL << ", LH " << LH << "\n"
		<< " CL " << CL << ", CH " << CH << "\n"
		<< " RL " << RL << ", RH " << RH << std::endl;
*/
    // assign BTI to TRACO (central TRACO for sl=3); ntc=1,2...maxtc
    int ntc = static_cast<int>((nbti-1)/DTConfig::NBTITC)+1;
    if( ntc<1 || ntc>maxtc ) 
      continue;
    
    if(debug())
      std::cout << "Bti trigger assigned to traco " << ntc << " (maxtc " << maxtc << ")" << std::endl;

    // TRACO information
    DTTracoId tracoid = DTTracoId(wheel(),station(),sector(),ntc);
     
    // position inside TRACO: 
    int pos = nbti-(ntc-1)*DTConfig::NBTITC;

    // store trigger in TRACO. Create TRACO if it doesn't exist
    // SV tb2003 : check if traco is connected!

    // SV 091103 : add bti trigger filtering in acceptance windows
    // if flag is useAcceptParam() = true

    // Load master TRACO plane
    if( nsl==1 ) {
      if( !_flag_acc || (K>=CL && K<=CH) )
        activeGetTRACO(ntc)->add_btiT( step, pos, &(*p) );
      else
        if(debug())
          std::cout 	<< "ATTENTION: in TRACO n. " << ntc 
			<< " bti pos " << pos << " trigger K= " << K 
			<< " outside acceptance " << CL << "<K<" << CH << std::endl;
    } 

    // Load slave TRACO plane
    if( nsl==3 ) {
      // 3 TRACO's
      //for(int tci=-1;tci<=1;tci++) {
      //  if( (ntc+tci)>0 && (ntc+tci)<=maxtc )
      //    activeGetTRACO(ntc+tci)->add_btiT( step, pos+8-4*tci, &(*p) );
      //  else{
      //    if(debug())
      //      std::cout << "ATTENTION: traco " << ntc+tci << " is disconnected!" << std::endl;
      //  }

      // Left Traco
      if( (ntc-1)>0 && (ntc-1)<=maxtc ) {
	if( !_flag_acc || (K>=LL && K<=LH) ) {
	  activeGetTRACO(ntc-1)->add_btiT( step, pos+8-4*(-1), &(*p) );
	} else {
	  if(debug()) {
	    std::cout 	<< "ATTENTION: in TRACO n. " << ntc-1
			<< " bti pos " << pos+8-4*(-1) << " trigger K= " << K 
			<< " outside acceptance " << LL << "<K<" << LH << std::endl;
	  }
	}
      }

      // Central Traco
      if( (ntc)>0 && (ntc)<=maxtc ) {
	if( !_flag_acc || (K>=CL && K<=CH) ) {
	  activeGetTRACO(ntc)->add_btiT( step, pos+8-4*(0), &(*p) );
	} else {
	  if(debug())
	    std::cout 	<< "ATTENTION: in TRACO n. " << ntc 
			<< " bti pos " << pos+8-4*(0) << " trigger K= " << K 
			<< " outside acceptance " << CL << "<K<" << CH << std::endl;
	}
      }
      
      // Right Traco
      if( (ntc+1)>0 && (ntc+1)<=maxtc ) {
	if( !_flag_acc || (K>=RL && K<=RH) ) {
	  activeGetTRACO(ntc+1)->add_btiT( step, pos+8-4*(+1), &(*p) );
	} else {
	  if(debug())
	    std::cout 	<< "ATTENTION: in TRACO n. " << ntc+1 
			<< " bti pos " << pos+8-4*(+1) << " trigger K= " << K 
			<< " outside acceptance " << RL << "<K<" << RH << std::endl;
	}
      }
    }

    // Filter Theta BTIs -> this is done in DTBtiChip 

  }//end loop on bti trigs
}

void 
DTTracoCard::runTRACO() {

  if(debug()){
    std::cout << "DTTracoCard:runTRACO called for wheel=" << wheel() ;
    std::cout <<                               ", station=" << station();
    std::cout <<                               ", sector="  << sector();
    std::cout << ", " << _tracomap.size() << " TRACOs with signal" << std::endl;
  }

  // run TRACO algorithm on all non-empty TRACO
  if(_tracomap.size()>0){

    if(debug()){
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
    if(debug())
      std::cout << "====================================================" << std::endl;
  }
}

DTTracoChip*
DTTracoCard::activeGetTRACO(int n) {

  // the traco identifier
  DTChamberId sid = geom()->statId();
  DTTracoId _id = DTTracoId(sid,n);
 
  DTTracoChip* traco = 0;
  TRACO_iter ptraco = _tracomap.find(n);
  if( ptraco!=_tracomap.end() ) {
    traco=(*ptraco).second;
  } else {
    traco = new DTTracoChip(this,n,config_traco(_id));
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

  float trig_pos = geom()->cellPitch() * ( (float)trig->X() / (float)(config_traco(trig->parentId())->BTIC()));

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
                      (float)(config_traco(trig->parentId())->BTIC());
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
                   /( geom()->distSL() * config_traco(trig->parentId())->BTIC()) );

  if(config_traco(trig->parentId())->debug()==4)
    std::cout << "K " << trig->K() << " == psi " << psi << " in FE frame " << std::endl;
    
  // (xd,yd,zd) in chamber frame
  float xd=-sin(psi);
  float yd=0;
  float zd=-cos(psi);

  // 10/07/06 Not needed anymore (chages in geometry)
  // if(FE==1){//FE in negative y
  //    xd = - xd;
  //}

 
  if(config_traco(trig->parentId())->debug()==4)
    std::cout << "Direction in chamber frame is (" << xd << "," << yd << "," << zd << ")" << std::endl;
 
  return LocalVector(xd,yd,zd);
}

DTConfigTraco* 
DTTracoCard::config_traco(const DTTracoId& tracoid) const
{
  //loop on map to find traco
  ConfTracoMap::const_iterator titer = _conf_traco_map.find(tracoid);
  if (titer == _conf_traco_map.end()){
    std::cout << "DTTracoCard::config_traco : TRACO (" << tracoid.wheel()
	      << "," << tracoid.sector()
	      << "," << tracoid.station()
	      << "," << tracoid.traco()
	      << ") not found, return 0" << std::endl;
    return 0;
  }

  return const_cast<DTConfigTraco*>(&(*titer).second);
} 

