#ifndef CalibTracker_SiPixelESProducers_SiPixelFakeQualityESSource_h
#define CalibTracker_SiPixelESProducers_SiPixelFakeQualityESSource_h
// -*- C++ -*-
//
// Package:    SiPixelFakeQualityESSource
// Class:      SiPixelFakeQualityESSource
// 
/**\class SiPixelFakeQualityESSource SiPixelFakeQualityESSource.h CalibTracker/SiPixelGainESProducer/src/SiPixelFakeQualityESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Bernadette Heyburn
//         Created:  Oct. 21st, 2008
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
// #include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"
//
// class decleration
//

class SiPixelFakeQualityESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeQualityESSource(const edm::ParameterSet &);
  ~SiPixelFakeQualityESSource();
  
  //      typedef edm::ESProducts<> ReturnType;
  
  virtual std::auto_ptr<SiPixelQuality>  produce(const SiPixelQualityFromDbRcd &);
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
  
 private:
  
  edm::FileInPath fp_;

};
#endif
