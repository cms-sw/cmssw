#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5BremRecoveryClusterAlgo.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

class Multi5x5SuperClusterProducer : public edm::stream::EDProducer<> {
public:
  Multi5x5SuperClusterProducer(const edm::ParameterSet& ps);

  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

private:
  edm::EDGetTokenT<reco::BasicClusterCollection> eeClustersToken_;
  edm::EDGetTokenT<reco::BasicClusterCollection> ebClustersToken_;
  edm::EDPutTokenT<reco::SuperClusterCollection> endcapPutToken_;
  edm::EDPutTokenT<reco::SuperClusterCollection> barrelPutToken_;

  const float barrelEtaSearchRoad_;
  const float barrelPhiSearchRoad_;
  const float endcapEtaSearchRoad_;
  const float endcapPhiSearchRoad_;
  const float seedTransverseEnergyThreshold_;

  const bool doBarrel_;
  const bool doEndcaps_;

  std::unique_ptr<Multi5x5BremRecoveryClusterAlgo> bremAlgo_p;

  double totalE;
  int noSuperClusters;

  reco::CaloClusterPtrVector getClusterPtrVector(
      edm::Event& evt, const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken) const;

  void produceSuperclustersForECALPart(edm::Event& evt,
                                       const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken,
                                       const edm::EDPutTokenT<reco::SuperClusterCollection>& putToken);

  void outputValidationInfo(reco::SuperClusterCollection& superclusterCollection);
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Multi5x5SuperClusterProducer);

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
