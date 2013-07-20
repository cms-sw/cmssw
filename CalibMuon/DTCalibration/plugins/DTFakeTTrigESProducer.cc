/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/09 22:44:10 $
 *  $Revision: 1.3 $
 *  \author S. Bolognesi - INFN Torino
 */

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "CalibMuon/DTCalibration/plugins/DTFakeTTrigESProducer.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

DTFakeTTrigESProducer::DTFakeTTrigESProducer(const edm::ParameterSet& pset)
{
  //framework
  setWhatProduced(this,&DTFakeTTrigESProducer::produce);
  findingRecord<DTTtrigRcd>();
  
  //read constant value for ttrig from cfg
  tMean = pset.getParameter<double>("tMean");
  sigma = pset.getParameter<double>("sigma");
  kFact = pset.getParameter<double>("kFactor");
}


DTFakeTTrigESProducer::~DTFakeTTrigESProducer(){
}

// ------------ method called to produce the data  ------------
DTTtrig* DTFakeTTrigESProducer::produce(const DTTtrigRcd& iRecord){

  DTTtrig* tTrigMap = new DTTtrig();

  for (int wheel=-2; wheel<3; wheel++){
    for(int station=1; station<5; station++){
      for(int sector=1; sector<13; sector++){
	for(int superlayer=1; superlayer<4; superlayer++){
	  if(superlayer==2 && station==4) continue;
	  DTSuperLayerId slId(DTChamberId(wheel, station, sector),superlayer);
	  tTrigMap->set(slId, tMean, sigma, kFact, DTTimeUnits::ns);
	}
      }
    }
  }

   for (int wheel=-2; wheel<3; wheel++){
     for(int superlayer=1; superlayer<4; superlayer++){
       if(superlayer==2) continue;
       DTSuperLayerId slId(DTChamberId(wheel, 4, 13),superlayer);
	 tTrigMap->set(slId, tMean, sigma, kFact, DTTimeUnits::ns);
     }  
   }

   for (int wheel=-2; wheel<3; wheel++){
     for(int superlayer=1; superlayer<4; superlayer++){
       if(superlayer==2) continue;
       DTSuperLayerId slId(DTChamberId(wheel, 4, 14),superlayer);
	 tTrigMap->set(slId, tMean, sigma, kFact, DTTimeUnits::ns);
     }  
   }
   
   return tTrigMap;
}

 void DTFakeTTrigESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
