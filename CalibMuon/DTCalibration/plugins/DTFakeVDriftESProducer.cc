/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/03 08:34:49 $
 *  $Revision: 1.2 $
 *  \author S. Bolognesi - INFN Torino
 */

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "CalibMuon/DTCalibration/plugins/DTFakeVDriftESProducer.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

DTFakeVDriftESProducer::DTFakeVDriftESProducer(const edm::ParameterSet& pset)
{
  //framework
  setWhatProduced(this,&DTFakeVDriftESProducer::produce);
  findingRecord<DTMtimeRcd>();
  
  //read constant value for ttrig from cfg
  vDrift = pset.getParameter<double>("vDrift");
  reso = pset.getParameter<double>("reso");
}


DTFakeVDriftESProducer::~DTFakeVDriftESProducer(){
}

// ------------ method called to produce the data  ------------
DTMtime* DTFakeVDriftESProducer::produce(const DTMtimeRcd& iRecord){

  DTMtime* mTimerMap = new DTMtime();

  for (int wheel=-2; wheel<3; wheel++){
    for(int station=1; station<5; station++){
      for(int sector=1; sector<13; sector++){
	for(int superlayer=1; superlayer<4; superlayer++){
	  if(superlayer==2 && station==4) continue;
	  DTSuperLayerId slId(DTChamberId(wheel, station, sector),superlayer);
	  mTimerMap->set(slId, vDrift, reso, DTVelocityUnits::cm_per_ns);
	}
      }
    }
  }

   for (int wheel=-2; wheel<3; wheel++){
     for(int superlayer=1; superlayer<4; superlayer++){
       if(superlayer==2) continue;
       DTSuperLayerId slId(DTChamberId(wheel, 4, 13),superlayer);
	 mTimerMap->set(slId, vDrift, reso, DTVelocityUnits::cm_per_ns);
     }  
   }

   for (int wheel=-2; wheel<3; wheel++){
     for(int superlayer=1; superlayer<4; superlayer++){
       if(superlayer==2) continue;
       DTSuperLayerId slId(DTChamberId(wheel, 4, 14),superlayer);
	 mTimerMap->set(slId, vDrift, reso, DTVelocityUnits::cm_per_ns);
     }  
   }
   
   return mTimerMap;
}

 void DTFakeVDriftESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
