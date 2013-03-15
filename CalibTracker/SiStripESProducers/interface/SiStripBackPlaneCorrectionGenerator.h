
#ifndef CalibTracker_SiStripESProducers_SiStripBackPlaneCorrectionGenerator_H
#define CalibTracker_SiStripESProducers_SiStripBackPlaneCorrectionGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include <string>

/**
 * Generator of the ideal/fake conditions for the BackPlaneCorrection.<br>
 * It receives input values for each module geometry type and it creates a per detIt reccord. <br>
 */

class SiStripBackPlaneCorrectionGenerator : public SiStripCondObjBuilderBase<SiStripBackPlaneCorrection> {
 public:

  explicit SiStripBackPlaneCorrectionGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripBackPlaneCorrectionGenerator();
  
  void getObj(SiStripBackPlaneCorrection* & obj){createObject(); obj=obj_;}

 private:

  void createObject();
};

#endif 
