//-------------------------------------------------
//
//   Class: DTTrig
//
//   Description: Steering routine for L1 trigger simulation in
//                a muon barrel station
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   S Vanini, S. Marcellini, D. Bonacorsi
//
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTrigger/interface/DTTrig.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"

//SV to access geometry from TestBeams, TB2003 setup...
//#include "TestBeams/DTBXSetUp/interface/DTBXSetUp.h"
//#include "Utilities/Notification/interface/Singleton.h"
//#include "Muon/MBDetector/interface/MuBarDetectorMap.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "Utilities/Notification/interface/Singleton.h"
//#include "Muon/MBSetup/interface/MuBarrelSetup.h"
//#include "Muon/MBDetector/interface/MuBarDetectorMap.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"

//----------------
// Constructors --
//----------------
DTTrig::DTTrig() {


  // Set configuration parameters
  _config = new DTConfig();

  if(config()->debug()>3){
    std::cout << std::endl;
    std::cout << "**** Initialization of DTTrigger ****" << std::endl;
    std::cout << std::endl;
  }

  for(int iwh=-2;iwh<=2;iwh++){ 
    for(int ist=1;ist<=4;ist++){ 
      for(int ise=1;ise<=12;ise++){
	DTChamberId chid(iwh,ist,ise);
	// create varous config files
	Conf_iterator cit = _localconf.find(chid);
	if ( cit != _localconf.end()) {
	  std::cout << "DTTrig::init: Local Config File already exists" << std::endl;
	  continue;
	}
	_localconf[chid] = _config;
      }
    }
  }
  for(int iwh=-2;iwh<=2;iwh++){  
    for(int ise=13;ise<=14;ise++){
      int ist=4;
      DTChamberId chid(iwh,ist,ise);
      // create varous config files
      Conf_iterator cit = _localconf.find(chid);
      if ( cit != _localconf.end()) {
	std::cout << "DTTrig::init: Local Config File already exists" << std::endl;
	continue;
      }
      _localconf[chid] = _config;
    }
  }
  

}


DTTrig::DTTrig(const edm::ParameterSet& pset, std::string mysync) {


  // Set configuration parameters
  _config = new DTConfig();

  if(config()->debug()>3){
    std::cout << std::endl;
    std::cout << "**** Initialization of DTTrigger ****" << std::endl;
    std::cout << std::endl;
  }

  for(int iwh=-2;iwh<=2;iwh++){ 
    for(int ist=1;ist<=4;ist++){ 
      for(int ise=1;ise<=12;ise++){
	DTChamberId chid(iwh,ist,ise);
	// create varous config files
	Conf_iterator cit = _localconf.find(chid);
	if ( cit != _localconf.end()) {
	  std::cout << "DTTrig::init: Local Config File already exists" << std::endl;
	  continue;
	}
	DTConfig *conf = new DTConfig();
	std::stringstream os;
	os << "wh" << chid.wheel()
	   << "se" << chid.sector()
	   << "st" << chid.station();
	double ftdelay = pset.getParameter<double>(os.str());
	conf->setParamValue("BTI setup time","psetdelay",ftdelay*32./25.);
	//conf->setParam("BTI setup time","psetdelay");
	conf->setParam("Programmable Dealy",mysync);
	_localconf[chid] = conf;
      }
    }
  }
  for(int iwh=-2;iwh<=2;iwh++){  
    for(int ise=13;ise<=14;ise++){
      int ist=4;
      DTChamberId chid(iwh,ist,ise);
      // create varous config files
      Conf_iterator cit = _localconf.find(chid);
      if ( cit != _localconf.end()) {
	std::cout << "DTTrig::init: Local Config File already exists" << std::endl;
	continue;
      }
      DTConfig *conf = new DTConfig();
      std::stringstream os;
      os << "wh" << chid.wheel()
	 << "se" << chid.sector()
	 << "st" << chid.station();
      double ftdelay = pset.getParameter<double>(os.str());
      conf->setParamValue("BTI setup time","psetdelay",ftdelay*32./25.);
      //conf->setParam("BTI Fine sync delay","psetdelay");
      conf->setParam("Programmable Dealy",mysync);
      _localconf[chid] = conf;
    }
  }
  
}


//--------------
// Destructor --
//--------------
DTTrig::~DTTrig(){
  delete _config;
  clear();
}

//--------------
// Operations --
//--------------
struct TmpSort : 
  public std::binary_function< const DTChamber*, 
  const DTChamber*, bool> {
    public:
      bool operator()(const DTChamber* c1, 
                      const DTChamber* c2) const {
        return c1->id() < c2->id(); }
};

void 
//DTTrig::createTUs(TBSetUp* run){   //SV: for TestBeams setup
// DTTrig::createTUs(G3SetUp* run){
DTTrig::createTUs(const edm::EventSetup& iSetup ){
  
  // build up Sector Collectors and then
  // build the trrigger units (one for each chamber)
  //     for(int iwh=1;iwh<=5;iwh++){ 
  for(int iwh=-2;iwh<=2;iwh++){ 
    for(int ist=1;ist<=4;ist++){    
      for(int ise=1;ise<=12;ise++){ 
	if(config()->debug()>3){
	  std::cout << "calling sectcollid wh st sc " << iwh << " " <<ist << " " << ise << std::endl;}
	DTSectCollId scid(iwh, ist, ise);
	SC_iterator it =  _cache1.find(scid);
	if ( it != _cache1.end()) {
	  std::cout << "DTTrig::createTUs: Sector Collector unit already exists"<<std::endl;
	  continue;
	}    
	// add a sector collector to the map
	DTSectColl* sc = new DTSectColl( config());
	if(config()->debug()>3){
	  std::cout << " DTTrig::createTUs new SC sc = " << sc << " at scid.station() " << scid.station() 
		    << " at scid.sector() " << scid.sector() << " at scid.wheel() " << scid.wheel()   << std::endl;}
	_cache1[scid] = sc; 
	
      }
    }
  }
  
  // ---------------  
  
  // build the trigger units (one for each chamber)
//   DTrelSetup* muon_setup = Singleton<MuBarrelSetup>::instance();
//   const DTDetectorMap& map = muon_setup->map();
//   const std::vector<DTChamber*> chamsTmp = map.chambers();
//   std::vector<DTChamber*> chams(chamsTmp);
//   sort(chams.begin(), chams.end(), TmpSort());
//   for ( std::vector<DTChamber*>::const_iterator ich = chams.begin();
//         ich!=chams.end(); ich++ ) { 
  //Fino qua dovrebbe diventare...
  edm::ESHandle<DTGeometry>pDD;
  iSetup.get<MuonGeometryRecord>().get(pDD);
  //iEvent.getByLabel("srivi_qui_il_label",pDD);
  for (std::vector<DTChamber*>::const_iterator ich=pDD->chambers().begin(); ich!=pDD->chambers().end();ich++){
       
    DTChamber* chamb = (*ich);
    DTChamberId chid = chamb->id();
    TU_iterator it = _cache.find(chid);
    if ( it != _cache.end()) {
      std::cout << "DTTrig::init: Trigger unit already exists" << std::endl;
      continue;
    }    
    Conf_iterator cit = _localconf.find(chid);
      if ( cit == _localconf.end()) {
	std::cout << "DTTrig::init: Local Config File already existsnot found using default congig" << std::endl;
	DTSCTrigUnit* tru = new DTSCTrigUnit(chamb,config());
	_cache[chid] = tru;
	continue;
      }
    // add a trigger unit to the map with a link to the station
    DTSCTrigUnit* tru = new DTSCTrigUnit(chamb,(*cit).second);
    _cache[chid] = tru;
    
    //----------- add TU to corresponding SC
    // returning correspondent SC id
    int flag = 0;
    DTSectCollId scid;
    if(chid.sector()==13) { 
      flag = 1; 
      scid = DTSectCollId(chid.wheel(), chid.station(), 4);}
    else if(chid.sector()==14)  {
      flag = 1; 
      scid = DTSectCollId(chid.wheel(), chid.station(), 10);}
    else if((chid.sector()==4 ||chid.sector()==10) && (chid.station()==4)) { flag = 0; 
    scid = DTSectCollId(chid.wheel(), chid.station(), chid.sector() );}
    else  { flag = 2; 
    scid = DTSectCollId(chid.wheel(), chid.station(), chid.sector() );}
    
    SC_iterator it1 =  _cache1.find(scid);

    if ( it1 != _cache1.end()) {
      
      DTSectColl* sc = (*it1).second;
      if(config()->debug()>3) {
      std::cout << "DTTrig::init:  adding TU in SC << at station() " << scid.station() << 
	" sector = " << scid.sector() << " wheel = " << scid.wheel() << std::endl;}
      sc->addTU(tru,flag);    
    }
    else {
      std::cout << "DTTrig::createTUs: Trigger Unit not in the map: ";
    }

  }
  /*
  //SV for TestBeam 2003: Chambers from DTBXSetp ddd geometry
  // loop over chambers, superlayers and layers
  if(config()->debug()>3)
    std::cout << "DTTrig::createTUs" << std::endl;
  DTBXSetUp* setup = Singleton<DTBXSetUp>::instance();
  DTDetectorMap* detMap = setup->chamberMap();
  std::vector<DTChamber*> chambers = detMap->chambers();
  DTDetectorMap::ChamIter stat;
  for(stat = chambers.begin(); stat != chambers.end(); stat++){
    DTChamberId idc = (*stat)->id();
    TU_iterator it = _cache.find(idc);
    if ( it != _cache.end()) {
      std::cout << "DTTrig::init: Trigger unit already exists";
      continue;
    }
    // add a trigger unit to the map with a link to the station
    if(config()->debug()>3)
      std::cout << "Creating Trigger Unit and adding to the cache" << std::endl;
    DTSCTrigUnit* tru = new DTSCTrigUnit((*stat), config());
    _cache[idc] = tru;
  }//end loop on chambers
*/

}


void 
DTTrig::triggerReco(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  DTDigiMap digiMap;
  //Sort digis by chamber so they can be used by BTIs
  edm::Handle<DTDigiCollection> dtDigis;
  iEvent.getByLabel("muonDTDigis", dtDigis);   
  DTDigiCollection::DigiRangeIterator detUnitIt;
  
  for (detUnitIt=dtDigis->begin();
       detUnitIt!=dtDigis->end();
       ++detUnitIt){
    const DTLayerId& layId = (*detUnitIt).first;
    const DTChamberId chambId=layId.superlayerId().chamberId();
    const DTDigiCollection::Range& range = (*detUnitIt).second;
    //    DTDigiCollection tmpDTDigiColl;
    //tmpDTDigiColl.put(range,layId); 
    digiMap[chambId].put(range,layId); 
//     const DTDigiCollection::Range& range = (*detUnitIt).second;
//     for (DTDigiCollection::const_iterator digiIt = range.first;
// 	 digiIt!=range.second;
// 	 ++digiIt){
//       DTDigiCollection tmp;
      
//       digiMap[chambId].push_back((*digiIt));
//     }
  }
  
  if(config()->debug()>2){
    std::cout << "----------DTDigis ordered by chamber:" << std::endl;
    for (DTDigiMap_const_iterator digiMapIt=digiMap.begin();
	 digiMapIt!=digiMap.end();
	 digiMapIt++){
      DTChamberId chambId = (*digiMapIt).first;
      DTDigiCollection digis = (*digiMapIt).second;
      std::cout << "Chamber id   " << chambId << std::endl;
      DTDigiCollection::DigiRangeIterator RangeIt;
      for (RangeIt=digis.begin();
	   RangeIt!=digis.end();
	   RangeIt++){
	std::cout << "Digi's layer   " << (*RangeIt).first << std::endl;
	const DTDigiCollection::Range& range = (*RangeIt).second;
	for (DTDigiCollection::const_iterator digiIt = range.first;
	     digiIt!=range.second;
	     ++digiIt){
	  std::cout << "Digi's data   " << (*digiIt) << std::endl; 
	}
	
	
      }
    }
  }

  //Run reconstruct for single trigger subsystem (Bti, Traco TS)
  for (TU_iterator it=_cache.begin();it!=_cache.end();it++){
    //DTDigiCollection dummydigicoll;
    DTSCTrigUnit* thisTU=(*it).second;
    DTChamberId chid=thisTU->statId();
    DTDigiMap_iterator dmit = digiMap.find(chid);
    if (dmit !=digiMap.end()) {thisTU->BtiTrigs()->reconstruct((*dmit).second);}
    else {thisTU->BtiTrigs()->clearCache();}
    thisTU->TSThTrigs()->reconstruct();
    thisTU->TracoTrigs()->reconstruct();
    thisTU->TSPhTrigs()->reconstruct();
  }
  //Run reconstruct for Sector Collector
  for (SC_iterator it=_cache1.begin();it!=_cache1.end();it++){
    (*it).second->reconstruct();
  }
}

void
DTTrig::clear() {
  // Delete the map
  for (TU_iterator it=_cache.begin();it!=_cache.end();it++){
    // Delete all the trigger units 
    delete (*it).second;
  }
  _cache.clear(); 

  for (SC_iterator it=_cache1.begin();it!=_cache1.end();it++){
    // Delete all the Sector Collectors
    delete (*it).second;
  }
  _cache1.clear();

}

DTSCTrigUnit*
DTTrig::trigUnit(DTChamberId chid) {
  /*check();*/  return constTrigUnit(chid);

}



DTSCTrigUnit*
DTTrig::constTrigUnit(DTChamberId chid) const {
//    std::cout << " SC: running DTTrig::constTrigUnit(DTChamberId chid)" << std::endl;
  TU_const_iterator it = _cache.find(chid);
  if ( it == _cache.end()) {
    std::cout << "DTTrig::trigUnit: Trigger Unit not in the map: ";
    std::cout << " wheel=" << chid.wheel() ;
    std::cout << ", station=" << chid.station();
    std::cout << ", sector=" << chid.sector();
    std::cout << std::endl;
    return 0;
  }

  return (*it).second;
}

DTSectColl*
DTTrig::SCUnit(DTSectCollId scid) const {
SC_const_iterator it = _cache1.find(scid);
  if ( it == _cache1.end()) {
    std::cout << "DTTrig::SCUnit: Trigger Unit not in the map: ";
    std::cout << " wheel=" << scid.wheel() ;
    std::cout << ", station=" << scid.station();
    std::cout << ", sector=" << scid.sector();
    std::cout << std::endl;
    return 0;
  }

  return (*it).second;
}

DTSCTrigUnit*
DTTrig::trigUnit(int wheel, int stat, int sect) {     //, int flag) {
//    std::cout << " SC: running DTTrig::trigUnit(int wheel, int stat, int sect, int flag " << "wheel, stat, sect, flag " << wheel << " " << stat << " " << sect << " " << flag << std::endl;
  /*check();*/ return constTrigUnit(wheel, stat, sect); //, flag);
}

DTSectColl*
DTTrig::SCUnit(int wheel, int stat, int sect) const {     //, int flag) {
//    std::cout << " SC: running DTTrig::SCUnit(int wheel, int stat, int sect, int flag " << "wheel, stat, sect, flag " << wheel << " " << stat << " " << sect << " " << flag << std::endl;

  /*check();*/ 
  //  wheel+=3;
  sect++;
return SCUnit(DTSectCollId(wheel,stat,sect));
}


DTSCTrigUnit*
DTTrig::constTrigUnit(int wheel, int stat, int sect) const {  // ,int flag) const {
  ///sm  wheel+=3; // offset 3 for wheel number ([-2,2] --> [1,5])
  sect++;   // offset 1 for sector number ([0,11] --> [1,12])
  // 90 and 270 deg sectors in MB4 are split up into 2 chambers
  // if flag!=0 the 'twin' chamber is passed
  //  std::cout << "SC: running DTTrig::constTrigUnit(int wheel, int stat, int sect, int flag) " << std::endl;;
  //  if(stat==4&&(sect==4||sect==10)&&flag!=0) sect = (sect+74)/6; 

  return constTrigUnit(DTChamberId(wheel,stat,sect));

}


DTChambPhSegm* 
DTTrig::chPhiSegm1(DTSCTrigUnit* unit, int step) {
  if(unit==0)return 0;
  if(unit->nPhiSegm(step)<1)return 0;
  return const_cast<DTChambPhSegm*>(unit->phiSegment(step,1));
}

DTChambPhSegm* 
DTTrig::chPhiSegm2(DTSCTrigUnit* unit, int step) {
  if(unit==0)return 0;
  if(unit->nPhiSegm(step)<2)return 0;
  return const_cast<DTChambPhSegm*>(unit->phiSegment(step,2));
}

DTChambThSegm* 
DTTrig::chThetaSegm(DTSCTrigUnit* unit, int step) {
  if(unit==0)return 0;
  if(unit->nThetaSegm(step)<1)return 0;
  return const_cast<DTChambThSegm*>(unit->thetaSegment(step,1));
}

DTChambPhSegm* 
DTTrig::chPhiSegm1(DTChamberId sid, int step) {
  return chPhiSegm1(trigUnit(sid),step);
}

DTChambPhSegm* 
DTTrig::chPhiSegm2(DTChamberId sid, int step) {
  return chPhiSegm2(trigUnit(sid),step);
}

DTChambThSegm* 
DTTrig::chThetaSegm(DTChamberId sid, int step) {
  if(sid.station()==4)return 0;
  return chThetaSegm(trigUnit(sid),step);
}

DTChambPhSegm*
DTTrig::chPhiSegm1(int wheel, int stat, int sect, int step) {
  // return chPhiSegm1(trigUnit(wheel,stat,sect),step);
  // to make it transparent to the outside world
    return chSectCollSegm1(wheel,stat,sect,step);

}

DTChambPhSegm* 
DTTrig::chPhiSegm2(int wheel, int stat, int sect, int step) {
  //  if(stat==4&&(sect==3||sect==9)) {
  // if hrizontal chambers of MB4 get first track of twin chamber (flag=1)
  //   return chPhiSegm1(trigUnit(wheel,stat,sect,1),step);
  //  } else {
  //  return chPhiSegm2(trigUnit(wheel,stat,sect),step);
  // to make it transparent to the outside world
  return chSectCollSegm2(wheel,stat,sect,step);
  //}
}

DTChambThSegm* 
DTTrig::chThetaSegm(int wheel, int stat, int sect, int step) {
  if(stat==4)return 0;
  return chThetaSegm(trigUnit(wheel,stat,sect),step);
}

// SM sector collector section
DTChambPhSegm* 
DTTrig::chSectCollSegm1(DTSectColl* unit, int step) {

  if(unit==0)return 0;
   if(unit->nSegm(step)<1)return 0;
  //  std::cout << " SC: running DTTrig::chSectCollSegm(DTSectColl* unit, int step) BEFORE END" << std::endl;
  return const_cast<DTChambPhSegm*>(unit->SectCollSegment(step,1));
}

DTChambPhSegm* 
DTTrig::chSectCollSegm2(DTSectColl* unit, int step) {
  if(unit==0)return 0;
    if(unit->nSegm(step)<2)return 0;
  return const_cast<DTChambPhSegm*>(unit->SectCollSegment(step,2));
}

//DTChambPhSegm* 
//DTTrig::chSectCollSegm1(DTChamberId sid, int step) {
//  return chSectCollSegm1(trigUnit(sid),step);
//}

//DTChambPhSegm* 
//DTTrig::chSectCollSegm2(DTChamberId sid, int step) {
//  return chSectCollSegm2(trigUnit(sid),step);
//}



DTChambPhSegm*
DTTrig::chSectCollSegm1(int wheel, int stat, int sect, int step) {

  return chSectCollSegm1(SCUnit(wheel,stat,sect),step);
}

DTChambPhSegm* 
DTTrig::chSectCollSegm2(int wheel, int stat, int sect, int step) {
  //  if(stat==4&&(sect==3||sect==9)) {
    // if hrizontal chambers of MB4 get first track of twin chamber (flag=1)
  //return chSectCollSegm1(trigUnit(wheel,stat,sect,1),step);
  //} else {
    return chSectCollSegm2(SCUnit(wheel,stat,sect),step);
    //}
}


  // end SM


void 
DTTrig::dumpGeom() {
  /*check();*/
  for (TU_const_iterator it=_cache.begin();it!=_cache.end();it++){
    ((*it).second)->dumpGeom();
  }
}

std::vector<DTBtiTrigData> 
DTTrig::BtiTrigs() {
  /*check();*/
  std::vector<DTBtiTrigData> trigs;
  TU_iterator ptu;
  for(ptu=_cache.begin();ptu!=_cache.end();ptu++) {
    DTSCTrigUnit* tu = (*ptu).second;
    std::vector<DTBtiTrigData>::const_iterator p; //p=0;
    std::vector<DTBtiTrigData>::const_iterator peb=tu->BtiTrigs()->end();
    for(p=tu->BtiTrigs()->begin();p!=peb;p++){
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTTracoTrigData> 
DTTrig::TracoTrigs()  {
  std::vector<DTTracoTrigData> trigs;
  TU_iterator ptu;
  /*check();*/
  for(ptu=_cache.begin();ptu!=_cache.end();ptu++) {
    DTSCTrigUnit* tu = (*ptu).second;
    std::vector<DTTracoTrigData>::const_iterator p; //p=0;
    std::vector<DTTracoTrigData>::const_iterator peb=tu->TracoTrigs()->end();
    for(p=tu->TracoTrigs()->begin();p!=peb;p++){
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTChambPhSegm> 
DTTrig::TSPhTrigs()  {
  /*check();*/
  std::vector<DTChambPhSegm> trigs;
  TU_iterator ptu;
  for(ptu=_cache.begin();ptu!=_cache.end();ptu++) {
    DTSCTrigUnit* tu = (*ptu).second;
    std::vector<DTChambPhSegm>::const_iterator p; //p=0;
    std::vector<DTChambPhSegm>::const_iterator peb=tu->TSPhTrigs()->end();
    for(p=tu->TSPhTrigs()->begin();p!=peb;p++){
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTChambThSegm> 
DTTrig::TSThTrigs()  {
  /*check();*/
  std::vector<DTChambThSegm> trigs;
  TU_iterator ptu;
  for(ptu=_cache.begin();ptu!=_cache.end();ptu++) {
    DTSCTrigUnit* tu = (*ptu).second;
    std::vector<DTChambThSegm>::const_iterator p; //p=0;
    std::vector<DTChambThSegm>::const_iterator peb=tu->TSThTrigs()->end();
    for(p=tu->TSThTrigs()->begin();p!=peb;p++){
      trigs.push_back(*p);
    }
  }
  return trigs;
}

std::vector<DTChambPhSegm> 
DTTrig::SCTrigs()  {
  /*check();*/
  std::vector<DTChambPhSegm> trigs;
  //  TU_iterator ptu;
  SC_iterator psc;
  for(psc=_cache1.begin();psc!=_cache1.end();psc++) {
    //    DTSCTrigUnit* tu = (*ptu).second;
    //
    // old SMDB:    
    //      DTSectColl* tu = (*ptu).second;
    //      std::vector<DTChambPhSegm>::const_iterator p=0;
    //      std::vector<DTChambPhSegm>::const_iterator peb=tu->SCTrigs()->end();
    //      for(p=tu->SCTrigs()->begin();p!=peb;p++){
    //        trigs.push_back(*p);
    //      } 

    DTSectColl* sc = (*psc).second;
    std::vector<DTChambPhSegm>::const_iterator p; //p=0;
    std::vector<DTChambPhSegm>::const_iterator peb=sc->end();
    for(p=sc->begin();p!=peb;p++){
      trigs.push_back(*p);
    }

  }
  return trigs;
}







