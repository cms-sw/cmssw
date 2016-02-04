#ifndef CalibTracker_SiPixelGainForHLTESProducers_SiPixelFakeGainForHLTESSource_h
#define CalibTracker_SiPixelGainForHLTESProducers_SiPixelFakeGainForHLTESSource_h
// -*- C++ -*-
//
// Package:    SiPixelFakeGainForHLTESSource
// Class:      SiPixelFakeGainForHLTESSource
// 
/**\class SiPixelFakeGainForHLTESSource SiPixelFakeGainForHLTESSource.h CalibTracker/SiPixelGainForHLTESProducer/src/SiPixelFakeGainForHLTESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Tue 8 12:31:25 CEST 2007
// $Id: SiPixelFakeGainForHLTESSource.h,v 1.1 2008/02/11 15:23:29 friis Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
//
// class decleration
//

class SiPixelFakeGainForHLTESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelFakeGainForHLTESSource(const edm::ParameterSet &);
  ~SiPixelFakeGainForHLTESSource();
  
  //      typedef edm::ESProducts<> ReturnType;
  
  virtual std::auto_ptr<SiPixelGainCalibrationForHLT>  produce(const SiPixelGainCalibrationForHLTRcd &);
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
  
 private:
  
  edm::FileInPath fp_;

};
#endif
