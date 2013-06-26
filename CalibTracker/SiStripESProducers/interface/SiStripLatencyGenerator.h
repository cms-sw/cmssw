#ifndef CalibTracker_SiStripESProducers_SiStripLatencyGenerator_H
#define CalibTracker_SiStripESProducers_SiStripLatencyGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include <string>

/**
 * Fake generator for latency and mode values stored in the SiStripLatency object. <br>
 */

class SiStripLatencyGenerator : public SiStripCondObjBuilderBase<SiStripLatency> {
 public:

  explicit SiStripLatencyGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripLatencyGenerator();

  void getObj(SiStripLatency* & obj){createObject(); obj=obj_;}

 private:

  void createObject();

};

#endif 
