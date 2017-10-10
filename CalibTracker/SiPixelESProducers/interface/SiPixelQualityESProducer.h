#ifndef CalibTracker_SiPixelESProducers_SiPixelQualityESProducer_h
#define CalibTracker_SiPixelESProducers_SiPixelQualityESProducer_h
// -*- C++ -*-
//
// Package:    SiPixelQualityESProducer
// Class:      SiPixelQualityESProducer
// 
/**\class SiPixelQualityESProducer SiPixelQualityESProducer.h CalibTracker/SiPixelESProducer/src/SiPixelQualityESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gemma Tinti
//         Created:  Jan 13 2011
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

//
// class decleration
//

class SiPixelQualityESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {

 public:
  SiPixelQualityESProducer(const edm::ParameterSet & iConfig);
  ~SiPixelQualityESProducer() override;
  
  
  /* virtual*/ std::unique_ptr<SiPixelQuality> produce(const SiPixelQualityRcd & iRecord) ;
  
protected:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& ) override;
  
  
 private:
  
  edm::FileInPath fp_;
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toGet;


};
#endif
