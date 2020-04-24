#ifndef CalibTracker_SiPixelESProducers_SiPixelFakeLorentzAngleESSource_h
#define CalibTracker_SiPixelESProducers_SiPixelFakeLorentzAngleESSource_h
// -*- C++ -*-
//
// Package:    SiPixelFakeLorentzAngleESSource
// Class:      SiPixelFakeLorentzAngleESSource
// 
/**\class SiPixelFakeLorentzAngleESSource SiPixelFakeLorentzAngleESSource.h CalibTracker/SiPixelGainESProducer/src/SiPixelFakeLorentzAngleESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lotte Wilke
//         Created:  Jan. 31st, 2008
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
// #include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"
//
// class decleration
//

class SiPixelFakeLorentzAngleESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeLorentzAngleESSource(const edm::ParameterSet &);
  ~SiPixelFakeLorentzAngleESSource() override;
  
  //      typedef edm::ESProducts<> ReturnType;
  
  virtual std::unique_ptr<SiPixelLorentzAngle>  produce(const SiPixelLorentzAngleRcd &);
  
 protected:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& ) override;
  
  
 private:
  
  edm::FileInPath fp_;

};
#endif
