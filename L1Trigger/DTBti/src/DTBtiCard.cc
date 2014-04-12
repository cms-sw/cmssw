//-------------------------------------------------
//
//   Class: DTBtiCard
//
//   Description: Contains active DTBtiChips
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   S. Vanini
//   22/VI/04 SV: last trigger code update - digi offset subtracted from digi time
//   III/05 SV  : NEWGEO : trigger in new geometry ! 
//   05/II/07 SV : new DTConfig update 
//   30/III/07 SV : config with DTConfigManager every single chip
//----------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTBti/interface/DTBtiCard.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "L1Trigger/DTBti/interface/DTBtiChip.h"
#include "L1Trigger/DTBti/interface/DTBtiTrig.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <cmath>
#include <utility>  
#include <vector>


using namespace edm;                            


//----------------
// Constructors --
//----------------

DTBtiCard::DTBtiCard(DTTrigGeom *geom) : 
  DTGeomSupplier(geom) {

        //_configBti = new DTConfigBti(bti_pset);
	//_configBti->print();

	//DTChamberId sid = ChamberId();
	//_conf_bti_map = conf_manager->getDTConfigBtiMap(sid);	
	//_debug = conf_manager->getDTTPGDebug();

	//_finedelay   = conf_manager->getDTConfigTrigUnit(sid)->MCSetupTime();
  	//_MCdelay     = conf_manager->getDTConfigTrigUnit(sid)->MCDigiOffset();

	
	
}

//--------------
// Destructor --
//--------------

DTBtiCard::~DTBtiCard(){

  localClear();
  //delete _conf_Bti;
  
}

//--------------
// Operations --
//--------------

void
DTBtiCard::clearCache(){

  BTICache::clearCache();
  localClear();
  
}

void
DTBtiCard::setConfig(const DTConfigManager *conf){
  
  DTChamberId sid = ChamberId();
  _conf_bti_map = conf->getDTConfigBtiMap(sid);	
  _debug = conf->getDTTPGDebug();
  _pedestals = conf->getDTConfigPedestals();

  // get bti acceptance flag
  _flag_acc = conf->useAcceptParam();

}


void
DTBtiCard::localClear(){
// Clear the maps
  for(int i=0; i<3; i++){
    for(BTI_iter p=_btimap[i].begin();p!=_btimap[i].end();p++){
      delete (*p).second;
    }
    _btimap[i].clear();
  }

  //clear digis
  std::vector<DTDigi*>::const_iterator p;
  for(p=_digis.begin();p<_digis.end();p++)
    delete (*p);
  _digis.clear();
}

// void 
// //DTBtiCard::loadBTI() {
// DTBtiCard::loadBTI(const DTDigiCollection dtDigis) {

//   localClear();
 
//    int ndigis = stat()->ndigis();


// // /*
// //   //SV version for tb2003 data
// //   // loop over chambers, superlayers and layers
// //   DTBXSetUp* setup = Singleton<DTBXSetUp>::instance();
// //   DTDetectorMap* detMap = setup->chamberMap();
// //   std::vector<DTChamber*> chambers = detMap->chambers();
// //   DTDetectorMap::ChamIter stat;
// //   for ( stat = chambers.begin(); stat != chambers.end(); stat++ ) {
// // */

// //   // Get DTDigi's from DTChamber  (stat is DTChamber*) 
// //   // Loop over all Digis and make a local copy
// //   // Store a pointer to them in the relevant DTBtiChip's

//   if(ndigis){

//     if(debug()){
//       std::cout << "====================================================" << std::endl;
//       std::cout <<  ndigis << " DIGIS in  wheel " << wheel() << 
//               " station " << station() <<
//               " sector "  << sector() << std::endl; 
//     }


//     std::vector<DTDigi> digi_container = stat()->digis();
//     std::vector<DTDigi>::const_iterator       digi_iter;

//     for(digi_iter=digi_container.begin(); digi_iter<digi_container.end(); digi_iter++){

//     /*Testbeam 2004 specific code
//     //digi.time in ns; digi.countsTDC in tdcunits
//     //float tdrift = (*digi_iter).time()-1.0*DTDigiParameters::offset;
//     //TB2004 version: no offset subtraction because digi are already t0 off!!!!
//     //tdrift must be float, otherwise DTDigi constructor assumes tdcunits!
//     //add chamber specific delay
//     int tdctime = (*digi_iter).countsTDC();
//     int mcdelay = (stat()->id().station()==1) ? config()->SetupTimeMB1() : config()->SetupTimeMB
// 3();
//     tdctime += mcdelay;
//     */
//       float tdrift = (*digi_iter).time()-1.0*DTDigiParameters::offset;
//       DTWireId idwire( wheel(), station(), sector(), (*digi_iter).slayer(), 
//                         (*digi_iter).layer(), (*digi_iter).wire() );


// /*
//   //NEWGEO TEST: sw testpulse for testing new geometry in wire 1 of each sl
//   for(int l=0;l<12;l++){
//     int sl = int(l/4.) + 1;
//     if(station()!=4 || sl!=2){
//     int lay = int(fmod(l,4.)) + 1;
//     int wire = 2*sector();
//     if(geom()->posFE(sl)==0 || station()==1 || sl==2)
//       wire += lay==4? 1:2;
//     else if(geom()->posFE(sl)==1) 
//       wire += lay==3? 3:2;

//     //int wire = 1;
//     //tmax=400ns, straight tracks in middle : 200ns (float for ns, int for tdc_counts)
//     //float tdrift = 187.5;
//     float tdrift = 200.0;
//     DTWireId idwire( wheel(), station(), sector(), sl, lay, wire);
// */

//       DTDigi* pdigi = new DTDigi(idwire,tdrift);
//       _digis.push_back(const_cast<DTDigi*>(pdigi));

//       if(debug())
//         pdigi->print();

//       int sln = pdigi->slayer();
//       int layn = pdigi->layer();
//       int tube = pdigi->wire();

//       // map in FE channel number:
//       int nch = geom()->mapTubeInFEch(sln,layn,tube);

//       // assign ch numbers to btis: depends on layer staggering!
//       int nbti = nch + geom()->layerFEStaggering(sln,layn);

//       if(debug()){
//         std::cout << "FE ch " << nch;
//         std::cout << " of bti " << nbti;
//         std::cout << " staggering " <<  geom()->layerFEStaggering(sln,layn);
//         std::cout << " now nbti is " << nch;
//         std::cout << " SL phi offset " << geom()->phiSLOffset() << std::endl; 
//       }  

//       switch(layn) {
//       case 1:
//         if(activeGetBTI(sln, nbti  )) activeGetBTI(sln, nbti  )->add_digi(4, pdigi);
//         if(activeGetBTI(sln, nbti-1)) activeGetBTI(sln, nbti-1)->add_digi(8, pdigi);
//       break;
//       case 2:
//         if(activeGetBTI(sln, nbti  )) activeGetBTI(sln, nbti  )->add_digi(2, pdigi);
//         if(activeGetBTI(sln, nbti-1)) activeGetBTI(sln, nbti-1)->add_digi(6, pdigi);
//       break;
//       case 3:
//         if(activeGetBTI(sln, nbti  )) activeGetBTI(sln, nbti  )->add_digi(3, pdigi);
//         if(activeGetBTI(sln, nbti-1)) activeGetBTI(sln, nbti-1)->add_digi(7, pdigi);
//       break;
//       case 4:
//         if(activeGetBTI(sln, nbti  )) activeGetBTI(sln, nbti  )->add_digi(1, pdigi);
//         if(activeGetBTI(sln, nbti-1)) activeGetBTI(sln, nbti-1)->add_digi(5, pdigi);
//         if(activeGetBTI(sln, nbti-2)) activeGetBTI(sln, nbti-2)->add_digi(9, pdigi);
//       break;
//       }
//     }//end loop on digi

//     if(debug())
//       std::cout << "====================================================" << std::endl;

//   }//end if(ndigis)
// }

void 
DTBtiCard::loadBTI(const DTDigiCollection dtDigis) {
  
  localClear();

  if(debug()){
    std::cout << "DTBtiCard::loadBTI called for wheel=" << wheel() ;
    std::cout << ", station=" << station();
    std::cout << ", sector="  << sector() << std::endl;
  }
  
  DTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=dtDigis.begin();
       detUnitIt!=dtDigis.end();
       ++detUnitIt){
    
    const DTLayerId& id = (*detUnitIt).first;
    const DTDigiCollection::Range& range = (*detUnitIt).second;
    
    // DTLayerId print-out
    if(debug()){
      std::cout<<"--------------"<<std::endl;
      std::cout<<"id: "<<id;
    }
    
    // Loop over the digis of this DetUnit
    for (DTDigiCollection::const_iterator digiIt = range.first;
	 digiIt!=range.second;
	 ++digiIt){

      int tube = (*digiIt).wire();
      const DTWireId tubeid(id,tube);
      float tdrift = (*digiIt).time() - _pedestals->getOffset(tubeid);

      if ( debug() ){  
	std::cout << " digi time : " << (*digiIt).time();
	std::cout << " pedestal offset : " << _pedestals->getOffset(tubeid) << std::endl;
	std::cout << " drift time after subtraction : " << tdrift << std::endl;
      }
	

      if(tdrift<500 && tdrift>-500){
	if(debug()) 
		(*digiIt).print();

	// get bti number for access to the configuration for this bti chip : SV
        // FIX: for the moment take first bti configuration!!!!! ok fixed
	DTSuperLayerId slnum = id.superlayerId();
	//DTChamberId dtcham =id.chamberId();
	int sln = slnum.superlayer();
	int layn = id.layer();
	//int tube = (*digiIt).wire();

        // map in FE channel number: SL theta tubes are numbered inversely w.r.t. hardware setup in new geometry 19/06/06
        // assign ch numbers to btis: in new geometry does not depend on layer staggering anymore! Same staggering anywhere.
	//int nch = geom()->mapTubeInFEch(sln,layn,tube);
	//	int nbti = nch + geom()->layerFEStaggering(sln,layn);
	
	int nch = geom()->mapTubeInFEch(sln,layn,tube);
	int nbti = nch ; 

	if(debug()){
	  std::cout << "FE ch " << nch;
	  std::cout << " of bti " << nbti;
//	  std::cout << " staggering " <<  geom()->layerFEStaggering(sln,layn);
	  std::cout << " now nbti is " << nch <<  std::endl;
//	  std::cout << " SL phi offset " << geom()->phiSLOffset() << std::endl; 
	}
        
  	// FIXED get configuration for the nbti chip Identifier, and from it MCdelay + finedelay
  	//DTChamberId sid = geom()->statId();
  	//DTBtiId _id = DTBtiId(sid, sln, nbti); 

	DTDigi* pdigi = new DTDigi((*digiIt).wire(),tdrift);
	_digis.push_back(pdigi);
	
	
	switch(layn) {
	case 1:
	  if(activeGetBTI(sln, nbti  )) activeGetBTI(sln, nbti  )->add_digi(4, pdigi);
	  if(activeGetBTI(sln, nbti-1)) activeGetBTI(sln, nbti-1)->add_digi(8, pdigi);
	  break;
	case 2:
	  if(activeGetBTI(sln, nbti  )) activeGetBTI(sln, nbti  )->add_digi(2, pdigi);
	  if(activeGetBTI(sln, nbti-1)) activeGetBTI(sln, nbti-1)->add_digi(6, pdigi);
	  break;
	case 3:
	  if(activeGetBTI(sln, nbti  )) activeGetBTI(sln, nbti  )->add_digi(3, pdigi);
	  if(activeGetBTI(sln, nbti-1)) activeGetBTI(sln, nbti-1)->add_digi(7, pdigi);
	  break;
	case 4:
	  if(activeGetBTI(sln, nbti  )) activeGetBTI(sln, nbti  )->add_digi(1, pdigi);
	  if(activeGetBTI(sln, nbti-1)) activeGetBTI(sln, nbti-1)->add_digi(5, pdigi);
	  if(activeGetBTI(sln, nbti-2)) activeGetBTI(sln, nbti-2)->add_digi(9, pdigi);
	  break;
	}
      }//end loop on digi
      //    }//end loop on lay
      //  }//end loop on sl
    }
  }
}
    
void 
DTBtiCard::runBTI() {

  int nbtisig = _btimap[0].size()+_btimap[1].size()+_btimap[2].size();

  if(debug()){
    std::cout << "DTBtiCard::runBTI called for wheel=" << wheel() ;
    std::cout <<                               ", station=" << station();
    std::cout <<                               ", sector="  << sector();
    std::cout << ", " << nbtisig
         << " BTIs with signal" << std::endl;
  }

  if( nbtisig){

    if(debug()){
      std::cout << "====================================================" << std::endl;
      std::cout << "              BTI triggers                          " << std::endl; 
    }


    // run DTBtiChip algorithm on all non-empty BTI
    for(int sl=1;sl<=3;sl++){
      if(_btimap[sl-1].size()>0){
        BTI_iter pbti;
        for(pbti=_btimap[sl-1].begin(); pbti!=_btimap[sl-1].end(); pbti++) {
	    DTBtiChip* bti = (*pbti).second; 
          if(bti->number()<200)
            bti->run();
          for(int step=DTConfig::NSTEPF; step<=DTConfig::NSTEPL; step++){
            if(bti->nTrig(step)>0) {
              _cache.push_back( bti->triggerData(step,1) );
              //localPosition(&(bti->triggerData(step,1))); 
              //localDirection(&(bti->triggerData(step,1))); 
             }
	    }
        }
      }
    }

    if(debug())
      std::cout << "====================================================" << std::endl;
  }//end if(nbtisig)
}

DTBtiChip* 
DTBtiCard::activeGetBTI(int sl, int n){

  DTBtiChip* bti=0;
  //check if BTi is out of range before all
  if( n<1 || n>geom()->nCell(sl) ){
    if(debug()){
      std::cout << "DTBtiCard::activeGetBTI :";
      std::cout << " invalid bti number: " << n;
      std::cout << " not in range [1," << geom()->nCell(sl) << "]";
      std::cout << " dummy BTI returned!" << std::endl;
    }
    sl = 1;
    n = 999;
    return bti;
  }

  // get configuration for this chip Identifier
  DTChamberId sid = geom()->statId();
  DTBtiId _id = DTBtiId(sid, sl, n); 

  //debug this chip
  int _debug_bti = config_bti(_id)->debug();

  if(_debug_bti==3){
    std::cout << "DTBtiCard::activeGetBTI :";
    std::cout << " bti number: " << n << std::endl;
  }

  if( sl<1 || sl>3){
    if(_debug_bti>1){
      std::cout << "DTBtiCard::activeGetBTI :";
      std::cout << " invalid superlayer number: " << sl;
      std::cout << " dummy BTI returned!" << std::endl;
    }
    sl = 1;
    n = 999;
    return bti; 
  }

  BTI_iter pbti = _btimap[sl-1].find(n);
  if( pbti!=_btimap[sl-1].end() ) {
    bti = (*pbti).second;
  } else {
    bti = new DTBtiChip(this, geom(),sl,n, config_bti(_id));
    _btimap[sl-1][n]=bti;
  }
  return bti;
}

DTBtiChip*
DTBtiCard::getBTI(int sl, int n) const {
  if(sl<1||sl>3||n==999){
    std::cout << "DTBtiCard::getBTI :";
    std::cout << " invalid superlayer number: " << sl;
    std::cout << " 0 returned!" << std::endl;
    return 0;
  }
  BTI_const_iter pbti = _btimap[sl-1].find(n);
  if( pbti==_btimap[sl-1].end() ){ 
    return 0;
  }
  return (*pbti).second;
}

std::vector<DTBtiChip*> 
DTBtiCard::btiList(int sl) {

  std::vector<DTBtiChip*> blist;

  if(size()<1)return blist;

  if(sl<1||sl>3){
    std::cout << "DTBtiCard::btiList :";
    std::cout << " invalid superlayer number: " << sl;
    std::cout << " empty list returned" << std::endl;
    return blist;
  }

  for(BTI_const_iter p=_btimap[sl-1].begin();p!=_btimap[sl-1].end();p++){
    blist.push_back((*p).second);
  }
  return blist;

}

DTBtiTrig*
DTBtiCard::storeTrigger(DTBtiTrigData td) {
  DTBtiId btiid = td.parentId();
  if(!(btiid.wheel()==wheel() &&
       btiid.sector()==sector() &&
       btiid.station()==station()) ) return 0;
  std::cout << "DTBtiCard::trigger: trigger not belonging to this card! ";
  std::cout << "card=(" << wheel() << "," << station() << "," << sector() << ") ";
  std::cout << "trig=(" << btiid.wheel() << "," << btiid.station() << "," 
       << btiid.sector() << ")";
  // get the appropriate BTI
  DTBtiChip* bti = activeGetBTI(btiid.superlayer(), btiid.bti());
  // create a new trigger in the BTI
  DTBtiTrig* trig = new DTBtiTrig(bti,td);
  // add the trigger to the BTI
  bti->addTrig(td.step(),trig);
  // return the trigger
  return trig;
}

/*
LocalPoint 
DTBtiCard::localPosition(const DTTrigData* tr) const {
//OBSOLETE!!!!!
  //@@ patch for Sun 4.2 compiler
  DTBtiTrigData* trig = dynamic_cast<DTBtiTrigData*>(const_cast<DTTrigData*>(tr));
  //  const DTBtiTrigData* trig = dynamic_cast<const DTBtiTrigData*>(tr);
  if(!trig) {
    std::cout << "DTBtiCard::localPosition called with wrong argument!" << std::endl;
    return LocalPoint(0,0,0);
  }
  float x = geom()->localPosition(trig->parentId()).x();
  float y = geom()->localPosition(trig->parentId()).y();
  float z = geom()->localPosition(trig->parentId()).z();
  
  //FE position
  int FE = geom()->posFE(trig->parentId().superlayer());

  //trigger position in the BTI frame
  float xt = 0;
  float yt = 0;
  float xtrig = (float)trig->X() * geom()->cellPitch() / (config()->ST());
  if(trig->btiSL()==2) 
    yt = - xtrig;
  else
    xt = + xtrig;

  if(FE==1){//FE in negative y
      xt = - xt;
      yt = - yt;
  }

  std::cout << "DTBtiCard::localPosition of BTI in ("<<x<<","<<y<<","<<z<<")"<<std::endl;
  std::cout << " called for trig "; 
  trig->print(); 
  std::cout << "in Local Point ("<<x+xt<<","<<y+yt<<","<<z<<")"<<std::endl;

  return LocalPoint(x+xt,y+yt,z);
}
*/

LocalPoint 
DTBtiCard::localPosition(const DTTrigData* tr) const {
//NEWGEO!!!
 const DTBtiTrigData* trig = dynamic_cast<const DTBtiTrigData*>(tr);
 if(!trig) {
    std::cout << "DTBtiCard::localPosition called with wrong argument!" << std::endl;
    return LocalPoint(0,0,0);
  }

  //BTI position in chamber frame
  float x = geom()->localPosition(trig->parentId()).x();
  float y = geom()->localPosition(trig->parentId()).y();
  float z = geom()->localPosition(trig->parentId()).z();
  
  //FE position
  //int FE = geom()->posFE(trig->parentId().superlayer());

  //trigger position in the BTI frame
  float xt = 0;
  float yt = 0;

  DTBtiId _btiid = trig->parentId();
  float xtrig = (float)trig->X() * geom()->cellPitch() / (config_bti(_btiid)->ST());
  if(trig->btiSL()==2) 
    yt = - xtrig;
  else
    xt = + xtrig;

//   17/07/06 Should be obsolete in CMSSW
//   if(FE==1){//FE in negative y
//       xt = - xt;
//       yt = - yt;
//   }



  if(debug()){
    std::cout << "DTBtiCard::localPosition of BTI in ("<<x<<","<<y<<","<<z<<")"<<std::endl;
    std::cout << " called for trig "; 
    trig->print(); 
    std::cout << "in Local Point ("<<x+xt<<","<<y+yt<<","<<z<<")"<<std::endl;
  }

  return LocalPoint(x+xt,y+yt,z);
}


LocalVector
DTBtiCard::localDirection(const DTTrigData* tr) const {
//NEWGEO
 const DTBtiTrigData* trig = dynamic_cast<const DTBtiTrigData*>(tr);
 //this method is only for check purpose
  if(!trig) {
    std::cout << "DTBtiCard::localDirection called with wrong argument!" << std::endl;
    return LocalVector(0,0,0);
  }

  //FE position
  //int FE = geom()->posFE(trig->parentId().superlayer());

  // psi in BTI frame
  DTBtiId _btiid = trig->parentId();
  float psi = atan((float)(trig->K() - config_bti(_btiid)->ST())*geom()->cellPitch()
                   /(2*geom()->cellH() * config_bti(_btiid)->ST()));
  // (xd,yd,zd) in chamber frame
  float xd=0;
  float yd=0;
  float zd=-cos(psi);
  if(trig->btiSL()==2) 
    yd = sin(psi);
  else
    xd = -sin(psi);

// 17/07/06 Ths should be obsolete in CMSSW
//   if(FE==1){//FE in negative y
//       xd = - xd;
//       yd = - yd;
//  }

  if(debug()){
    //BTI position in chamber frame
    float xb = geom()->localPosition(trig->parentId()).x();
    float yb = geom()->localPosition(trig->parentId()).y();
    float zb = geom()->localPosition(trig->parentId()).z();
  
    std::cout << "DTBtiCard::localDirection of BTI in ("<<xb<<","<<yb<<","<<zb<<")"<<std::endl;
    std::cout << " called for trig "; 
    trig->print(); 
    std::cout << "in Local Direction ("<<xd<<","<<yd<<","<<zd<<")"<<std::endl;
  }

  return LocalVector(xd,yd,zd);
}

/*

LocalVector
DTBtiCard::localDirection(const DTTrigData* tr) const {
//OBSOLETE!!!!
  //@@ patch for Sun 4.2 compiler
  const DTBtiTrigData* trig = dynamic_cast<const DTBtiTrigData*>(const_cast<DTTrigData*>(tr));
  //  const DTBtiTrigData* trig = dynamic_cast<const DTBtiTrigData*>(tr);
  if(!trig) {
    std::cout << "DTBtiCard::localDirection called with wrong argument!" << std::endl;
    return LocalVector(0,0,0);
  }
  float x1,r,x,y,z;
  x1 = -(float)trig->K() * geom()->cellPitch() /
                      (float)(config()->ST());
  z = - 2 * geom()->cellH();
  r = sqrt(x1*x1+z*z);
  if(trig->btiSL()!=2) {
    x = x1/r;
    y = 0;
  } else {
    x = 0;
    y = x1/r;
  }
  z /= r;
  return LocalVector(x,y,z);
}
*/

DTConfigBti* 
DTBtiCard::config_bti(DTBtiId& btiid) const 
{
  //loop on map to find bti
  ConfBtiMap::const_iterator biter = _conf_bti_map.find(btiid);
  if (biter == _conf_bti_map.end()){
    std::cout << "DTBtiCard::config_bti : BTI (" << btiid.wheel()
	      << "," << btiid.sector()
	      << "," << btiid.station()
	      << "," << btiid.superlayer()
	      << "," << btiid.bti()
	      << ") not found, return 0" << std::endl;
    return 0;
  }

  return const_cast<DTConfigBti*>(&(*biter).second);

}
