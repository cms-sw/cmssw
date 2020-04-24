#ifndef CalibTracker_SiPixelGainESProducers_SiPixelFakeGainESSource_h
#define CalibTracker_SiPixelGainESProducers_SiPixelFakeGainESSource_h
// -*- C++ -*-
//
// Package:    SiPixelFakeGainESSource
// Class:      SiPixelFakeGainESSource
// 
/**\class SiPixelFakeGainESSource SiPixelFakeGainESSource.h CalibTracker/SiPixelGainESProducer/src/SiPixelFakeGainESSource.cc

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
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"
//
// class decleration
//

class SiPixelFakeGainESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeGainESSource(const edm::ParameterSet &);
  ~SiPixelFakeGainESSource() override;
  
  //      typedef edm::ESProducts<> ReturnType;
  
  virtual std::unique_ptr<SiPixelGainCalibration>  produce(const SiPixelGainCalibrationRcd &);
  
 protected:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& ) override;
  
  
 private:
  
  edm::FileInPath fp_;

};
#endif
