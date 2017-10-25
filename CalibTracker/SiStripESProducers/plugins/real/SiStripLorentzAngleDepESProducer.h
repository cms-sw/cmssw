#ifndef CalibTracker_SiStripESProducers_SiStripLorentzAngleDepESProducer
#define CalibTracker_SiStripESProducers_SiStripLorentzAngleDepESProducer

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripLorentzAngleDepESProducer : public edm::ESProducer {
 public:
  SiStripLorentzAngleDepESProducer(const edm::ParameterSet&);
  ~SiStripLorentzAngleDepESProducer() override{};
  
  std::shared_ptr<SiStripLorentzAngle> produce(const SiStripLorentzAngleDepRcd&);
   
 private:

  edm::ParameterSet pset_; 
  //edm::FileInPath fp_;
  //bool MergeList_; 

 // typedef std::vector< edm::ParameterSet > Parameters;
  edm::ParameterSet getLatency;
  edm::ParameterSet getPeak;
  edm::ParameterSet getDeconv;

  std::shared_ptr<SiStripLorentzAngle> siStripLA_;

};

#endif

