#ifndef CalibTracker_SiStripESProducers_SiStripBackPlaneCorrectionDepESProducer
#define CalibTracker_SiStripESProducers_SiStripBackPlaneCorrectionDepESProducer

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
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripBackPlaneCorrectionDepESProducer : public edm::ESProducer {
 public:
  SiStripBackPlaneCorrectionDepESProducer(const edm::ParameterSet&);
  ~SiStripBackPlaneCorrectionDepESProducer(){};
  
  boost::shared_ptr<SiStripBackPlaneCorrection> produce(const SiStripBackPlaneCorrectionDepRcd&);
   
 private:

  edm::ParameterSet pset_; 
  //edm::FileInPath fp_;
  //bool MergeList_; 

 // typedef std::vector< edm::ParameterSet > Parameters;
  edm::ParameterSet getLatency;
  edm::ParameterSet getPeak;
  edm::ParameterSet getDeconv;

  boost::shared_ptr<SiStripBackPlaneCorrection> siStripBPC_;

};

#endif

