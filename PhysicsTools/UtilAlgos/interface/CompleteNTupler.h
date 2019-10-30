#include "FWCore/Framework/interface/ProducesCollector.h"
#include "PhysicsTools/UtilAlgos/interface/NTupler.h"

#include "PhysicsTools/UtilAlgos/interface/StringBasedNTupler.h"
#include "PhysicsTools/UtilAlgos/interface/VariableNTupler.h"
//#include "PhysicsTools/UtilAlgos/interface/AdHocNTupler.h"

class CompleteNTupler : public NTupler {
public:
  CompleteNTupler(const edm::ParameterSet& iConfig) {
    sN = new StringBasedNTupler(iConfig);
    if (iConfig.exists("variablesPSet"))
      if (!iConfig.getParameter<edm::ParameterSet>("variablesPSet").empty())
        vN = new VariableNTupler(iConfig);
      else
        vN = nullptr;
    else
      vN = nullptr;

    /*    if (iConfig.exists("AdHocNPSet"))
      if (!iConfig.getParameter<edm::ParameterSet>("AdHocNPSet").empty())
	aN = new AdHocNTupler(iConfig);
      else aN=0;
    else
      aN=0;
    */
  }

  uint registerleaves(edm::ProducesCollector producesCollector) override {
    uint nLeaves = 0;
    nLeaves += sN->registerleaves(producesCollector);
    if (vN)
      nLeaves += vN->registerleaves(producesCollector);
    //    if (aN)
    //      nLeaves+=aN->registerleaves(producesCollector);
    return nLeaves;
  }
  void fill(edm::Event& iEvent) override {
    sN->fill(iEvent);
    if (vN)
      vN->fill(iEvent);
    //    if (aN)
    //      aN->fill(iEvent);

    sN->callBack();
    if (vN)
      vN->callBack();
    //    if (aN)
    //      aN->callBack();
  }

private:
  StringBasedNTupler* sN;
  VariableNTupler* vN;
  //  AdHocNTupler * aN;
};
