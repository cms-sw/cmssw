// Authors: Felice Pantaleo, Marco Rovere
// Emails: felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 06/2019

#ifndef RecoHGCal_TICL_SeedingRegionAlgoBase_H__
#define RecoHGCal_TICL_SeedingRegionAlgoBase_H__

#include <memory>
#include <vector>
#include "DataFormats/HGCalReco/interface/Common.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace ticl {
  class SeedingRegionAlgoBase {
  public:
    SeedingRegionAlgoBase(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")) {}
    virtual ~SeedingRegionAlgoBase(){};

    virtual void initialize(const edm::EventSetup& es) = 0;

    virtual void makeRegions(const edm::Event& ev,
                             const edm::EventSetup& es,
                             std::vector<TICLSeedingRegion>& result) = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); }

  protected:
    int algo_verbosity_;
    int algoId_;
  };
}  // namespace ticl

#endif
