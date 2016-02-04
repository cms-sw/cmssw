#ifndef CalibTracker_SiStripESProducers_SiStripApvGainGenerator_H
#define CalibTracker_SiStripESProducers_SiStripApvGainGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <string>

/**
 * Two ways to generate the gain:
 * - default: geneartes gain = MeanGain (from cfg) the same for all strips
 * - gaussian: generates gain with a gaussian distribution centered at MeanGain and with sigma = SigmaGain (from cfg)
 */

class SiStripApvGainGenerator : public SiStripCondObjBuilderBase<SiStripApvGain> {
 public:

  explicit SiStripApvGainGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripApvGainGenerator();
  
  void getObj(SiStripApvGain* & obj){createObject(); obj=obj_;}

 private:
  
  void createObject();

  
};

#endif 
