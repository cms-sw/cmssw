#ifndef CalibTracker_SiStripESProducers_SiStripLorentzAngleGenerator_H
#define CalibTracker_SiStripESProducers_SiStripLorentzAngleGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include <string>

class SiStripLorentzAngleGenerator : public SiStripCondObjBuilderBase<SiStripLorentzAngle> {
 public:

  explicit SiStripLorentzAngleGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripLorentzAngleGenerator();
  
  void getObj(SiStripLorentzAngle* & obj){createObject(); obj=obj_;}

 private:
  
  void createObject();

  
};

#endif 
