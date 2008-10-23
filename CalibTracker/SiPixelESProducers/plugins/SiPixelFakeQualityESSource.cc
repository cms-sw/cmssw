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
	findingRecord<SiPixelQualityRcd>();
}

SiPixelFakeQualityESSource::~SiPixelFakeQualityESSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

std::auto_ptr<SiPixelQuality> SiPixelFakeQualityESSource::produce(const SiPixelQualityRcd & )
{


      ///////////////////////////////////////////////////////
      //  errortype "whole" = int 0 in DB  BadRocs = 65535 //
      //  errortype "tbmA" = int 1 in DB  BadRocs = 255    //
      //  errortype "tbmB" = int 2 in DB  Bad Rocs = 65280 //
      //  errortype "none" = int 3 in DB                   //
      ///////////////////////////////////////////////////////
  
    SiPixelQuality * obj = new SiPixelQuality();

    SiPixelQuality::disabledModuleType BadModule;
    BadModule.DetID = 302197784; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 302195232; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 302123296; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 302127136; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 302125076; BadModule.errorType = 1; BadModule.BadRocs = 255;   obj->addDisabledModule(BadModule);
    BadModule.DetID = 302126364; BadModule.errorType = 2; BadModule.BadRocs = 65280; obj->addDisabledModule(BadModule);
    BadModule.DetID = 302188552; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 302121992; BadModule.errorType = 1; BadModule.BadRocs = 255;   obj->addDisabledModule(BadModule);
    BadModule.DetID = 302126596; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344074500; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344074504; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344074508; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344074512; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344074756; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344074760; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344074764; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344075524; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344075528; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344075532; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344075536; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344075780; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344075784; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344075788; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344076548; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344076552; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344076556; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344076556; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344076560; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344076804; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344076808; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344076812; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344005128; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344020236; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344020240; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344020488; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344020492; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344019212; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344019216; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344019464; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344019468; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344018188; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344018192; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344018440; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344018444; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344014340; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344014344; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
    BadModule.DetID = 344014348; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);

    return std::auto_ptr<SiPixelQuality>(obj);

}

void SiPixelFakeQualityESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
