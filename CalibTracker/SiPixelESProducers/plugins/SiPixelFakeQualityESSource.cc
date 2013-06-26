// -*- C++ -*-
//
// Package:    SiPixelFakeQualityESSource
// Class:      SiPixelFakeQualityESSource
// 
/**\class SiPixelFakeQualityESSource SiPixelFakeQualityESSource.h CalibTracker/SiPixelESProducer/src/SiPixelFakeQualityESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Bernadette Heyburn
//         Created:  Oct 21 2008
//
//

// user include files

#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeQualityESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// constructors and destructor
//
SiPixelFakeQualityESSource::SiPixelFakeQualityESSource(const edm::ParameterSet& conf_) : fp_(conf_.getParameter<edm::FileInPath>("file"))
{
	edm::LogInfo("SiPixelFakeQualityESSource::SiPixelFakeQualityESSource");
	//the following line is needed to tell the framework what
	// data is being produced
	setWhatProduced(this);
	findingRecord<SiPixelQualityFromDbRcd>();
}

SiPixelFakeQualityESSource::~SiPixelFakeQualityESSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

std::auto_ptr<SiPixelQuality> SiPixelFakeQualityESSource::produce(const SiPixelQualityFromDbRcd & )
{


      ///////////////////////////////////////////////////////
      //  errortype "whole" = int 0 in DB  BadRocs = 65535 //
      //  errortype "tbmA" = int 1 in DB  BadRocs = 255    //
      //  errortype "tbmB" = int 2 in DB  Bad Rocs = 65280 //
      //  errortype "none" = int 3 in DB                   //
      ///////////////////////////////////////////////////////
  
    SiPixelQuality * obj = new SiPixelQuality();

    SiPixelQuality::disabledModuleType BadModule;
    BadModule.DetID = 1; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);

    return std::auto_ptr<SiPixelQuality>(obj);

}

void SiPixelFakeQualityESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
