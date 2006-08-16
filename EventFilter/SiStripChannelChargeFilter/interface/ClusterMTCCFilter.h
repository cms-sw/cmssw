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

namespace cms
{
 class ClusterMTCCFilter : public edm::EDFilter {
  public:
    ClusterMTCCFilter(const edm::ParameterSet& ps);
    virtual ~ClusterMTCCFilter() {}
    virtual bool filter(edm::Event & e, edm::EventSetup const& c);

  private:
   std::string clusterProducer;
   uint ChargeThresholdTIB;
   uint ChargeThresholdTOB;
   uint ChargeThresholdTEC;
   uint MinClustersDiffComponents;
   // uint: generalized_layer: 10*subdetId + layer
   std::map<uint,std::vector<SiStripCluster> > clusters_in_subcomponents;
  };
}
#endif
