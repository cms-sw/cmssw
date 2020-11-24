#ifndef ClusterMTCCFilter_H
#define ClusterMTCCFilter_H

// -*- C++ -*-
//
// Package:     SiStripChannelChargeFilter
// Class  :     ClusterMTCCFilter
//
//
// Original Author:  dkcira

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class TrackerTopology;
namespace cms {
  class ClusterMTCCFilter : public edm::EDFilter {
  public:
    ClusterMTCCFilter(const edm::ParameterSet& ps);
    ~ClusterMTCCFilter() override {}
    bool filter(edm::Event& e, edm::EventSetup const& c) override;

  private:
    std::string clusterProducer;
    unsigned int ChargeThresholdTIB;
    unsigned int ChargeThresholdTOB;
    unsigned int ChargeThresholdTEC;
    unsigned int MinClustersDiffComponents;
    std::vector<uint32_t> ModulesToBeExcluded;
    // unsigned int: generalized_layer: 10*subdetId + layer
    std::map<unsigned int, std::vector<SiStripCluster> > clusters_in_subcomponents;
    edm::ESGetToken<TrackerTopology, IdealGeometryRecord> tTopoToken_;
  };
}  // namespace cms
#endif
