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
// $Id: SiPixelFakeGainESSource.h,v 1.4 2008/02/11 15:23:28 friis Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

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
  ~SiPixelFakeGainESSource();
  
  //      typedef edm::ESProducts<> ReturnType;
  
  virtual std::auto_ptr<SiPixelGainCalibration>  produce(const SiPixelGainCalibrationRcd &);
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
  
 private:
  
  edm::FileInPath fp_;

};
#endif
