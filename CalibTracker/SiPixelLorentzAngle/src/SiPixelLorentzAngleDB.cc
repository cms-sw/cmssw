#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CalibTracker/SiPixelLorentzAngle/interface/SiPixelLorentzAngleDB.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CLHEP/Random/RandGauss.h"
#include "CondTools/SiPixel/test/SiPixelCondObjBuilder.h"

using namespace std;

  //Constructor

SiPixelLorentzAngleDB::SiPixelLorentzAngleDB(edm::ParameterSet const& conf) : 
  conf_(conf){
//   if(conf_.getParameter<bool>("DoCalibration")) siStripLorentzAngleAlgorithm_=new SiStripLorentzAngleAlgorithm(conf);
//   else siStripLorentzAngleAlgorithm_=0;
}

  //BeginJob

void SiPixelLorentzAngleDB::beginJob(const edm::EventSetup& c){
  
	SiPixelLorentzAngle* LorentzAngle = new SiPixelLorentzAngle();
	
	edm::ESHandle<TrackerGeometry> pDD;
	c.get<TrackerDigiGeometryRecord>().get( pDD );
	edm::LogInfo("SiPixelLorentzAngle") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
	
  	float langle = 0.424;
	for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
    
		if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
			uint32_t detid=((*it)->geographicalId()).rawId();
			if ( ! LorentzAngle->putLorentzAngle(detid,langle) )
					edm::LogError("SiPixelLorentzAngleDB")<<"[SiPixelLorentzAngleDB::analyze] detid already exists"<<std::endl;
			}
			
		}      
  	

	edm::Service<cond::service::PoolDBOutputService> mydbservice;
	if( mydbservice.isAvailable() ){
		try{
			if( mydbservice->isNewTagRequest("SiPixelLorentzAngleRcd") ){
				mydbservice->createNewIOV<SiPixelLorentzAngle>(LorentzAngle,mydbservice->endOfTime(),"SiPixelLorentzAngleRcd");
			} else {
				mydbservice->appendSinceTime<SiPixelLorentzAngle>(LorentzAngle,mydbservice->currentTime(),"SiPixelLorentzAngleRcd");
			}
		}catch(const cond::Exception& er){
			edm::LogError("SiPixelLorentzAngleDB")<<er.what()<<std::endl;
		}catch(const std::exception& er){
			edm::LogError("SiPixelLorentzAngleDB")<<"caught std::exception "<<er.what()<<std::endl;
		}catch(...){
			edm::LogError("SiPixelLorentzAngleDB")<<"Funny error"<<std::endl;
		}
	}else{
		edm::LogError("SiPixelLorentzAngleDB")<<"Service is unavailable"<<std::endl;
	}
   
}
// Virtual destructor needed.

SiPixelLorentzAngleDB::~SiPixelLorentzAngleDB() {  

}  

// Analyzer: Functions that gets called by framework every event

void SiPixelLorentzAngleDB::analyze(const edm::Event& e, const edm::EventSetup& es)
{


}

void SiPixelLorentzAngleDB::endJob(){


}
