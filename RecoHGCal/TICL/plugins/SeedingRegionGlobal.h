// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

#ifndef __RecoHGCal_TICL_SeedingRegionGlobal_H__
#define __RecoHGCal_TICL_SeedingRegionGlobal_H__
#include <memory>  // unique_ptr
#include <string>
#include "RecoHGCal/TICL/plugins/SeedingRegionAlgoBase.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace ticl {
  class SeedingRegionGlobal final : public SeedingRegionAlgoBase {
  public:
    SeedingRegionGlobal(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes);
    ~SeedingRegionGlobal() override;

    void initialize(const edm::EventSetup& es) override{};

    void makeRegions(const edm::Event& ev, const edm::EventSetup& es, std::vector<TICLSeedingRegion>& result) override;
  };
}  // namespace ticl
#endif
