#ifndef CalibTracker_SiStripESProducers_SiStripDelayESProducer
#define CalibTracker_SiStripESProducers_SiStripDelayESProducer

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDelay.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripDelayESProducer : public edm::ESProducer {
 public:
  SiStripDelayESProducer(const edm::ParameterSet&);
  ~SiStripDelayESProducer(){};
  
  boost::shared_ptr<SiStripDelay> produce(const SiStripDelayRcd&);
   
 private:

  edm::ParameterSet pset_; 
  edm::FileInPath fp_;
  bool MergeList_; 

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toGet;

  boost::shared_ptr<SiStripDelay> delay;
};

#endif
