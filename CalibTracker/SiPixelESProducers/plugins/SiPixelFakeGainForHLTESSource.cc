// -*- C++ -*-
//
// Package:    SiPixelFakeGainForHLTESSource
// Class:      SiPixelFakeGainForHLTESSource
// 
/**\class SiPixelFakeGainForHLTESSource SiPixelFakeGainForHLTESSource.h CalibTracker/SiPixelESProducer/src/SiPixelFakeGainForHLTESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Fri Apr 27 12:31:25 CEST 2007
// $Id: SiPixelFakeGainForHLTESSource.cc,v 1.5 2008/01/22 19:15:07 muzaffar Exp $
//
//

// user include files

#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainForHLTESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// constructors and destructor
//
SiPixelFakeGainForHLTESSource::SiPixelFakeGainForHLTESSource(const edm::ParameterSet& conf_) :
  fp_(conf_.getParameter<edm::FileInPath>("file"))
{
 edm::LogInfo("SiPixelFakeGainForHLTESSource::SiPixelFakeGainForHLTESSource");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelGainCalibrationForHLTRcd>();
}

SiPixelFakeGainForHLTESSource::~SiPixelFakeGainForHLTESSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

std::auto_ptr<SiPixelGainCalibrationForHLT> SiPixelFakeGainForHLTESSource::produce(const SiPixelGainCalibrationForHLTRcd & )
{

   using namespace edm::es;
   unsigned int nmodules = 0;
   uint32_t nchannels = 0;
   SiPixelGainCalibrationForHLT * obj = new SiPixelGainCalibrationForHLT(25.,30., 2.,3.);
   SiPixelDetInfoFileReader reader(fp_.fullPath());
   const std::vector<uint32_t> DetIds = reader.getAllDetIds();

   // Loop over detectors
   for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++) {
     nmodules++;
     std::vector<char> theSiPixelGainCalibration;
     const std::pair<int, int> & detUnitDimensions = reader.getDetUnitDimensions(*detit);

     // Loop over columns and rows
     for(int i=0; i<detUnitDimensions.first; i++) {
       float totalGain  = 0.0;
       float totalPed   = 0.0; 
       for(int j=0; j<detUnitDimensions.second; j++) {
         //this innerloop is unnecessary but is left as an example in case someone wishes to provide gain/ped distributions etc
	 nchannels++;
         totalGain      += 2.8;
         totalPed       += 28.2;
       }
       float gain       = totalGain/(float)detUnitDimensions.second;
       float ped        = totalPed/(float)detUnitDimensions.second;
       obj->setData(ped, gain , theSiPixelGainCalibration);	 
     }

     //std::cout << "detid " << (*detit) << std::endl;

     SiPixelGainCalibrationForHLT::Range range(theSiPixelGainCalibration.begin(),theSiPixelGainCalibration.end());
     if( !obj->put(*detit,range) )
       edm::LogError("SiPixelFakeGainForHLTESSource")<<"[SiPixelFakeGainForHLTESSource::produce] detid already exists"<<std::endl;
   }

   //std::cout << "Modules = " << nmodules << " Channels " << nchannels << std::endl;
   

   // 
   return std::auto_ptr<SiPixelGainCalibrationForHLT>(obj);


}

void SiPixelFakeGainForHLTESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
