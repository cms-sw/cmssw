// -*- C++ -*-
//
// Package:    SiPixelFakeLorentzAngleESSource
// Class:      SiPixelFakeLorentzAngleESSource
// 
/**\class SiPixelFakeLorentzAngleESSource SiPixelFakeLorentzAngleESSource.h CalibTracker/SiPixelESProducer/src/SiPixelFakeLorentzAngleESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lotte Wilke
//         Created:  Jan 31 2008
//
//

// user include files

#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeLorentzAngleESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// constructors and destructor
//
SiPixelFakeLorentzAngleESSource::SiPixelFakeLorentzAngleESSource(const edm::ParameterSet& conf_) : fp_(conf_.getParameter<edm::FileInPath>("file")),
	tanLorentzAnglePerTesla_(conf_.getParameter<double>("tanLorentzAnglePerTesla"))
{
	edm::LogInfo("SiPixelFakeLorentzAngleESSource::SiPixelFakeLorentzAngleESSource");
	//the following line is needed to tell the framework what
	// data is being produced
	setWhatProduced(this);
	findingRecord<SiPixelLorentzAngleRcd>();
}

SiPixelFakeLorentzAngleESSource::~SiPixelFakeLorentzAngleESSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

std::auto_ptr<SiPixelLorentzAngle> SiPixelFakeLorentzAngleESSource::produce(const SiPixelLorentzAngleRcd & )
{

	using namespace edm::es;
	unsigned int nmodules = 0;
	SiPixelLorentzAngle * obj = new SiPixelLorentzAngle();
	SiPixelDetInfoFileReader reader(fp_.fullPath());
	const std::vector<uint32_t> DetIds = reader.getAllDetIds();
	
	// Loop over detectors
	for(std::vector<uint32_t>::const_iterator detit = DetIds.begin(); detit!=DetIds.end(); detit++) {
		nmodules++;
		float langle =  tanLorentzAnglePerTesla_;
		//std::cout << "detid " << (*detit) << std::endl;
	
		if( !obj->putLorentzAngle(*detit,langle) ) edm::LogError("SiPixelFakeLorentzAngleESSource")<<"[SiPixelFakeLorentzAngleESSource::produce] detid already exists"<<std::endl;
	}
	
	//std::cout << "Modules = " << nmodules << std::endl;

	return std::auto_ptr<SiPixelLorentzAngle>(obj);
	

}

void SiPixelFakeLorentzAngleESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
