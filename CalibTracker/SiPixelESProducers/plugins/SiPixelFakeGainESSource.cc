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
// $Id: SiPixelFakeGainESSource.cc,v 1.6 2008/02/11 15:23:38 friis Exp $
//
//

// user include files

#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// constructors and destructor
//
SiPixelFakeGainESSource::SiPixelFakeGainESSource(const edm::ParameterSet& conf_) :
  fp_(conf_.getParameter<edm::FileInPath>("file"))
{
 edm::LogInfo("SiPixelFakeGainESSource::SiPixelFakeGainESSource");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelGainCalibrationRcd>();
}

SiPixelFakeGainESSource::~SiPixelFakeGainESSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

std::auto_ptr<SiPixelGainCalibration> SiPixelFakeGainESSource::produce(const SiPixelGainCalibrationRcd & )
{

   using namespace edm::es;
   unsigned int nmodules = 0;
   uint32_t nchannels = 0;
   SiPixelGainCalibration * obj = new SiPixelGainCalibration(25.,30., 2.,3.);
   SiPixelDetInfoFileReader reader(fp_.fullPath());
   const std::vector<uint32_t> DetIds = reader.getAllDetIds();

   // Loop over detectors
   for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++) {
     nmodules++;
     std::vector<char> theSiPixelGainCalibration;
     const std::pair<int, int> & detUnitDimensions = reader.getDetUnitDimensions(*detit);

     // Loop over columns and rows
     for(int i=0; i<detUnitDimensions.first; i++) {
       for(int j=0; j<detUnitDimensions.second; j++) {
	 nchannels++;
	 float gain =  2.8;
	 float ped  = 28.2;	 
	 obj->setData(ped, gain , theSiPixelGainCalibration);	 
       }
     }

     //std::cout << "detid " << (*detit) << std::endl;

     SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(),theSiPixelGainCalibration.end());
     if( !obj->put(*detit,range,detUnitDimensions.first) )
       edm::LogError("SiPixelFakeGainESSource")<<"[SiPixelFakeGainESSource::produce] detid already exists"<<std::endl;
   }

   //std::cout << "Modules = " << nmodules << " Channels " << nchannels << std::endl;
   

   // 
   return std::auto_ptr<SiPixelGainCalibration>(obj);


}

void SiPixelFakeGainESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
