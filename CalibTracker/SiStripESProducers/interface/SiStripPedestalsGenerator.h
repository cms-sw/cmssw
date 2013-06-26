#ifndef CalibTracker_SiStripESProducers_SiStripPedestalsGenerator_H
#define CalibTracker_SiStripESProducers_SiStripPedestalsGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include <string>

class SiStripPedestalsGenerator : public SiStripCondObjBuilderBase<SiStripPedestals> {
 public:

  explicit SiStripPedestalsGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripPedestalsGenerator();
  
  void getObj(SiStripPedestals* & obj){createObject(); obj=obj_;}

 private:
  
  void createObject();

  
};

#endif 
