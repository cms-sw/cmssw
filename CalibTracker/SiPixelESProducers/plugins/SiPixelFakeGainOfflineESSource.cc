// -*- C++ -*-
//
// Package:    SiPixelFakeGainOfflineESSource
// Class:      SiPixelFakeGainOfflineESSource
// 
/**\class SiPixelFakeGainOfflineESSource SiPixelFakeGainOfflineESSource.h CalibTracker/SiPixelESProducer/src/SiPixelFakeGainOfflineESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Fri Apr 27 12:31:25 CEST 2007
// $Id: SiPixelFakeGainOfflineESSource.cc,v 1.2 2008/04/22 12:36:49 friis Exp $
//
//

// user include files

#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGainOfflineESSource.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// constructors and destructor
//
SiPixelFakeGainOfflineESSource::SiPixelFakeGainOfflineESSource(const edm::ParameterSet& conf_) :
  fp_(conf_.getParameter<edm::FileInPath>("file"))
{
 edm::LogInfo("SiPixelFakeGainOfflineESSource::SiPixelFakeGainOfflineESSource");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelGainCalibrationOfflineRcd>();
}

SiPixelFakeGainOfflineESSource::~SiPixelFakeGainOfflineESSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

std::auto_ptr<SiPixelGainCalibrationOffline> SiPixelFakeGainOfflineESSource::produce(const SiPixelGainCalibrationOfflineRcd & )
{

   using namespace edm::es;
   unsigned int nmodules = 0;
   uint32_t nchannels = 0;
   SiPixelGainCalibrationOffline * obj = new SiPixelGainCalibrationOffline(25.,30., 2.,3.);
   SiPixelDetInfoFileReader reader(fp_.fullPath());
   const std::vector<uint32_t> DetIds = reader.getAllDetIds();

   // Loop over detectors
   for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++) {
     nmodules++;
     std::vector<char> theSiPixelGainCalibrationOffline;
     const std::pair<int, int> & detUnitDimensions = reader.getDetUnitDimensions(*detit);

     // Loop over columns and rows
     for(int i=0; i<detUnitDimensions.first; i++) {
       float totalGain    = 0.0;
       float totalEntries = 0.0;
       for(int j=0; j<detUnitDimensions.second; j++) {
	 nchannels++;
         totalGain  += 2.8;
	 float ped  = 28.2;	 
         totalEntries += 1.0;
	 obj->setDataPedestal(ped, theSiPixelGainCalibrationOffline);	 
         if ((j + 1) % 80 == 0) //compute the gain average after each ROC
         {
            float gain = totalGain/totalEntries;
            obj->setDataGain(gain, 80, theSiPixelGainCalibrationOffline);
            totalGain    = 0;
            totalEntries = 0.0;
         }
       }
     }

     //std::cout << "detid " << (*detit) << std::endl;

     SiPixelGainCalibrationOffline::Range range(theSiPixelGainCalibrationOffline.begin(),theSiPixelGainCalibrationOffline.end());
     // the 80 in the line below represents the number of columns averaged over.  
     if( !obj->put(*detit,range,detUnitDimensions.first) )
       edm::LogError("SiPixelFakeGainOfflineESSource")<<"[SiPixelFakeGainOfflineESSource::produce] detid already exists"<<std::endl;
   }

   //std::cout << "Modules = " << nmodules << " Channels " << nchannels << std::endl;
   

   // 
   return std::auto_ptr<SiPixelGainCalibrationOffline>(obj);


}

void SiPixelFakeGainOfflineESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
