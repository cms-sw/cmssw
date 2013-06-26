#ifndef CalibTracker_SiStripESProducers_SiStripBaseDelayGenerator_H
#define CalibTracker_SiStripESProducers_SiStripBaseDelayGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"
#include <string>

/**
 * Fake generator for base delay values stored in the SiStripBaseDelay object. <br>
 */

class SiStripBaseDelayGenerator : public SiStripCondObjBuilderBase<SiStripBaseDelay> {
 public:

  explicit SiStripBaseDelayGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripBaseDelayGenerator();

  void getObj(SiStripBaseDelay* & obj){createObject(); obj=obj_;}

 private:

  void createObject();

};

#endif 
