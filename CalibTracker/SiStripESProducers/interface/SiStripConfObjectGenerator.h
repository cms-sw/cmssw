#ifndef CalibTracker_SiStripESProducers_SiStripConfObjectGenerator_H
#define CalibTracker_SiStripESProducers_SiStripConfObjectGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

/**
 * Fake generator for configuration parameters stored in the SiStripConfObject object. <br>
 */

class SiStripConfObjectGenerator : public SiStripCondObjBuilderBase<SiStripConfObject> {
 public:

  explicit SiStripConfObjectGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripConfObjectGenerator();

  void getObj(SiStripConfObject* & obj){createObject(); obj=obj_;}

 private:

  void createObject();

  std::vector<edm::ParameterSet> parameters_;
};

#endif 
