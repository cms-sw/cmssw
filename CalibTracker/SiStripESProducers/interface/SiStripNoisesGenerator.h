#ifndef CalibTracker_SiStripESProducers_SiStripNoisesGenerator_H
#define CalibTracker_SiStripESProducers_SiStripNoisesGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include <string>

class SiStripNoisesGenerator : public SiStripCondObjBuilderBase<SiStripNoises> {
 public:

  explicit SiStripNoisesGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripNoisesGenerator();
  
  void getObj(SiStripNoises* & obj){createObject(); obj=obj_;}

 private:
  
  void createObject();

  
};

#endif 
