// -*- C++ -*-
//
// Package:    SiPixelFakeGainESSource
// Class:      SiPixelFakeGainESSource
// 
/**\class SiPixelFakeGainESSource SiPixelFakeGainESSource.h CalibTracker/SiPixelESProducer/src/SiPixelFakeGainESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Fri Apr 27 12:31:25 CEST 2007
// $Id: SiPixelFakeGainESSource.cc,v 1.4 2007/07/28 09:15:17 elmer Exp $
//
//



// user include files

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// constructors and destructor
//
SiPixelFakeGainESSource::SiPixelFakeGainESSource(const edm::ParameterSet& conf_){

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  fp_ = conf_.getParameter<edm::FileInPath>("file");
}

SiPixelFakeGainESSource::~SiPixelFakeGainESSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

boost::shared_ptr<SiPixelGainCalibration> SiPixelFakeGainESSource::produce(const SiPixelGainCalibrationRcd & iRecord)
{

   using namespace edm::es;
   SiPixelGainCalibration * obj = new SiPixelGainCalibration();
   SiPixelDetInfoFileReader reader(fp_.fullPath());
   const std::vector<uint32_t> DetIds = reader.getAllDetIds();

   for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++) {
     
     const std::pair<int, int> & detUnitDimensions = reader.getDetUnitDimensions(*detit);

   }

   // 
   return boost::shared_ptr<SiPixelGainCalibration>(obj);


}

void SiPixelFakeGainESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
