#ifndef TECClusterFilter_H
#define TECClusterFilter_H

// -*- C++ -*-
//
// Package:     SiStripChannelChargeFilter
// Class  :     TECClusterFilter
//
//
// Original Author: sfricke

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace cms {
  class TECClusterFilter : public edm::stream::EDFilter<> {
  public:
    TECClusterFilter(const edm::ParameterSet& ps);
    ~TECClusterFilter() override = default;
    bool filter(edm::Event& e, edm::EventSetup const& c) override;

  private:
    std::string clusterProducer;
    unsigned int ChargeThresholdTEC;
    unsigned int minNrOfTECClusters;
    std::vector<uint32_t> ModulesToBeExcluded;
  };
}  // namespace cms
#endif
