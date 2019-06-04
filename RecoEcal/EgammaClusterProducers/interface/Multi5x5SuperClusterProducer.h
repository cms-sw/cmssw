#ifndef RecoEcal_EgammaClusterProducers_Multi5x5SuperClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_Multi5x5SuperClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5BremRecoveryClusterAlgo.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
//

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

#endif
