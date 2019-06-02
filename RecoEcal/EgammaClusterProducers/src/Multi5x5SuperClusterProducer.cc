// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// Reconstruction Classes
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/Multi5x5SuperClusterProducer.h"

Multi5x5SuperClusterProducer::Multi5x5SuperClusterProducer(const edm::ParameterSet& ps)
    : barrelEtaSearchRoad_{static_cast<float>(ps.getParameter<double>("barrelEtaSearchRoad"))},
      barrelPhiSearchRoad_{static_cast<float>(ps.getParameter<double>("barrelPhiSearchRoad"))},
      endcapEtaSearchRoad_{static_cast<float>(ps.getParameter<double>("endcapEtaSearchRoad"))},
      endcapPhiSearchRoad_{static_cast<float>(ps.getParameter<double>("endcapPhiSearchRoad"))},
      seedTransverseEnergyThreshold_{static_cast<float>(ps.getParameter<double>("seedTransverseEnergyThreshold"))},
      doBarrel_{ps.getParameter<bool>("doBarrel")},
      doEndcaps_{ps.getParameter<bool>("doEndcaps")},
      totalE{0.},
      noSuperClusters{0} {
  if (doEndcaps_) {
    eeClustersToken_ = consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("endcapClusterTag"));
  }
  if (doBarrel_) {
    ebClustersToken_ = consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("barrelClusterTag"));
  }

  const edm::ParameterSet bremRecoveryPset = ps.getParameter<edm::ParameterSet>("bremRecoveryPset");
  bool dynamicPhiRoad = ps.getParameter<bool>("dynamicPhiRoad");

  bremAlgo_p = std::make_unique<Multi5x5BremRecoveryClusterAlgo>(bremRecoveryPset,
                                                                 barrelEtaSearchRoad_,
                                                                 barrelPhiSearchRoad_,
                                                                 endcapEtaSearchRoad_,
                                                                 endcapPhiSearchRoad_,
                                                                 dynamicPhiRoad,
                                                                 seedTransverseEnergyThreshold_);

  if (doEndcaps_) {
    endcapPutToken_ =
        produces<reco::SuperClusterCollection>(ps.getParameter<std::string>("endcapSuperclusterCollection"));
  }
  if (doBarrel_) {
    barrelPutToken_ =
        produces<reco::SuperClusterCollection>(ps.getParameter<std::string>("barrelSuperclusterCollection"));
  }
}

void Multi5x5SuperClusterProducer::endStream() {
  double averEnergy = 0.;
  std::ostringstream str;
  str << "Multi5x5SuperClusterProducer::endJob()\n"
      << "  total # reconstructed super clusters: " << noSuperClusters << "\n"
      << "  total energy of all clusters: " << totalE << "\n";
  if (noSuperClusters > 0) {
    averEnergy = totalE / noSuperClusters;
    str << "  average SuperCluster energy = " << averEnergy << "\n";
  }
  edm::LogInfo("Multi5x5SuperClusterProducerInfo") << str.str() << "\n";
}

void Multi5x5SuperClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  if (doEndcaps_)
    produceSuperclustersForECALPart(evt, eeClustersToken_, endcapPutToken_);

  if (doBarrel_)
    produceSuperclustersForECALPart(evt, ebClustersToken_, barrelPutToken_);
}

void Multi5x5SuperClusterProducer::produceSuperclustersForECALPart(
    edm::Event& evt,
    const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken,
    const edm::EDPutTokenT<reco::SuperClusterCollection>& putToken) {
  // get the cluster collection out and turn it to a BasicClusterRefVector:
  reco::CaloClusterPtrVector clusterPtrVector_p = getClusterPtrVector(evt, clustersToken);

  // run the brem recovery and get the SC collection
  reco::SuperClusterCollection superclusters_ap(bremAlgo_p->makeSuperClusters(clusterPtrVector_p));

  // count the total energy and the number of superclusters
  for (auto const& sc : superclusters_ap) {
    totalE += sc.energy();
    noSuperClusters++;
  }

  // put the SC collection in the event
  evt.emplace(putToken, std::move(superclusters_ap));
}

reco::CaloClusterPtrVector Multi5x5SuperClusterProducer::getClusterPtrVector(
    edm::Event& evt, const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken) const {
  reco::CaloClusterPtrVector clusterPtrVector_p;
  edm::Handle<reco::BasicClusterCollection> bccHandle;
  evt.getByToken(clustersToken, bccHandle);

  const reco::BasicClusterCollection* clusterCollection_p = bccHandle.product();
  clusterPtrVector_p.reserve(clusterCollection_p->size());
  for (unsigned int i = 0; i < clusterCollection_p->size(); i++) {
    clusterPtrVector_p.push_back(reco::CaloClusterPtr(bccHandle, i));
  }
  return clusterPtrVector_p;
}
