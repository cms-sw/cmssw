#ifndef CalibTracker_SiStripESProducers_SiStripThresholdGenerator_H
#define CalibTracker_SiStripESProducers_SiStripThresholdGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include <string>

class SiStripThresholdGenerator : public SiStripCondObjBuilderBase<SiStripThreshold> {
 public:

  explicit SiStripThresholdGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripThresholdGenerator();
  
  void getObj(SiStripThreshold* & obj){createObject(); obj=obj_;}

 private:
  
  void createObject();

  
};

#endif 
