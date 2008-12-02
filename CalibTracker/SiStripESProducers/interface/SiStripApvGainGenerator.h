#ifndef CalibTracker_SiStripESProducers_SiStripApvGainGenerator_H
#define CalibTracker_SiStripESProducers_SiStripApvGainGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <string>

class SiStripApvGainGenerator : public SiStripCondObjBuilderBase<SiStripApvGain> {
 public:

  explicit SiStripApvGainGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripApvGainGenerator();
  
  void getObj(SiStripApvGain* & obj){createObject(); obj=obj_;}

 private:
  
  void createObject();

  
};

#endif 
