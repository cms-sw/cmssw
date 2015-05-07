
#ifndef CalibTracker_SiStripESProducers_SiStripBackPlaneCorrectionGenerator_H
#define CalibTracker_SiStripESProducers_SiStripBackPlaneCorrectionGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripDepCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <string>

/**
 * Generator of the ideal/fake conditions for the BackPlaneCorrection.<br>
 * It receives input values for each module geometry type and it creates a per detIt reccord. <br>
 */

class SiStripBackPlaneCorrectionGenerator : public SiStripDepCondObjBuilderBase<SiStripBackPlaneCorrection,TrackerTopology> {
 public:

  explicit SiStripBackPlaneCorrectionGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripBackPlaneCorrectionGenerator();
  
  void getObj(SiStripBackPlaneCorrection* & obj, const TrackerTopology* tTopo){obj=createObject(tTopo);}

 private:

  SiStripBackPlaneCorrection*  createObject(const TrackerTopology* tTopo);
};

#endif 
