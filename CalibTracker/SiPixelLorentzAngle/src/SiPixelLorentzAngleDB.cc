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


#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

using namespace std;
using namespace edm;

  //Constructor

SiPixelLorentzAngleDB::SiPixelLorentzAngleDB(edm::ParameterSet const& conf) : 
  conf_(conf){
  	magneticField_ = conf_.getParameter<double>("magneticField");
	recordName_ = conf_.getUntrackedParameter<std::string>("record","SiPixelLorentzAngleRcd");
	bPixLorentzAnglePerTesla_ = (float)conf_.getParameter<double>("bPixLorentzAnglePerTesla");
	fPixLorentzAnglePerTesla_ = (float)conf_.getParameter<double>("fPixLorentzAnglePerTesla");
	useFile_ = conf_.getParameter<bool>("useFile");		
	fileName_ = conf_.getParameter<string>("fileName");

}

  //BeginJob

void SiPixelLorentzAngleDB::beginJob(){
  
}
// Virtual destructor needed.

SiPixelLorentzAngleDB::~SiPixelLorentzAngleDB() {  

}  

// Analyzer: Functions that gets called by framework every event

void SiPixelLorentzAngleDB::analyze(const edm::Event& e, const edm::EventSetup& es)
{

	SiPixelLorentzAngle* LorentzAngle = new SiPixelLorentzAngle();
	   
	
	edm::ESHandle<TrackerGeometry> pDD;
	es.get<TrackerDigiGeometryRecord>().get( pDD );
	edm::LogInfo("SiPixelLorentzAngle") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
	
	for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
    
		if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
			DetId detid=(*it)->geographicalId();
			
			// fill bpix values for LA 
			if(detid.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
				
				if(!useFile_){
					if ( ! LorentzAngle->putLorentzAngle(detid.rawId(),bPixLorentzAnglePerTesla_) )
					edm::LogError("SiPixelLorentzAngleDB")<<"[SiPixelLorentzAngleDB::analyze] detid already exists"<<std::endl;
				} else {
					cout << "method for reading file not implemented yet" << endl;
				}
				
			
			// fill bpix values for LA 
			} else if(detid.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
				
				if ( ! LorentzAngle->putLorentzAngle(detid.rawId(),fPixLorentzAnglePerTesla_) )
					edm::LogError("SiPixelLorentzAngleDB")<<"[SiPixelLorentzAngleDB::analyze] detid already exists"<<std::endl;
			} else {
				edm::LogError("SiPixelLorentzAngleDB")<<"[SiPixelLorentzAngleDB::analyze] detid is Pixel but neither bpix nor fpix"<<std::endl;
			}
		
		}
			
	}      
  	

	edm::Service<cond::service::PoolDBOutputService> mydbservice;
	if( mydbservice.isAvailable() ){
		try{
			if( mydbservice->isNewTagRequest(recordName_) ){
				mydbservice->createNewIOV<SiPixelLorentzAngle>(LorentzAngle,
									       mydbservice->beginOfTime(),
									       mydbservice->endOfTime(),
									       recordName_);
			} else {
				mydbservice->appendSinceTime<SiPixelLorentzAngle>(LorentzAngle,
										  mydbservice->currentTime(),
										  recordName_);
			}
		}catch(const cond::Exception& er){
			edm::LogError("SiPixelLorentzAngleDB")<<er.what()<<std::endl;
		}catch(const std::exception& er){
			edm::LogError("SiPixelLorentzAngleDB")<<"caught std::exception "<<er.what()<<std::endl;
		}
	}else{
		edm::LogError("SiPixelLorentzAngleDB")<<"Service is unavailable"<<std::endl;
	}
   

}

void SiPixelLorentzAngleDB::endJob(){


}
