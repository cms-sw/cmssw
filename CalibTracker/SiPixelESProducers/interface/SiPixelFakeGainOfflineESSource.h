#ifndef CalibTracker_SiPixelGainESProducers_SiPixelFakeGainOfflineESSource_h
#define CalibTracker_SiPixelGainESProducers_SiPixelFakeGainOfflineESSource_h
// -*- C++ -*-
//
// Package:    SiPixelFakeGainESSource
// Class:      SiPixelFakeGainOfflineESSource
// 
/**\class SiPixelFakeGainOfflineESSource SiPixelFakeGainOfflineESSource.h CalibTracker/SiPixelGainESProducer/src/SiPixelFakeGainOfflineESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Tue 8 12:31:25 CEST 2007
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineRcd.h"
//
// class decleration
//

class SiPixelFakeGainOfflineESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeGainOfflineESSource(const edm::ParameterSet &);
  ~SiPixelFakeGainOfflineESSource() override;
  
  //      typedef edm::ESProducts<> ReturnType;
  
  virtual std::unique_ptr<SiPixelGainCalibrationOffline>  produce(const SiPixelGainCalibrationOfflineRcd &);
  
 protected:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& ) override;
  
  
 private:
  
  edm::FileInPath fp_;

};
#endif
