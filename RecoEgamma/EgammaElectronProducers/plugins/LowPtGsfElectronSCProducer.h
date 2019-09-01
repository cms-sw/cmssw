#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSCProducer_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSCProducer_h

#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class LowPtGsfElectronSCProducer : public edm::stream::EDProducer<> {
public:
  explicit LowPtGsfElectronSCProducer(const edm::ParameterSet&);

  ~LowPtGsfElectronSCProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  reco::PFClusterRef closestCluster(const reco::PFTrajectoryPoint& point,
                                    const edm::Handle<reco::PFClusterCollection>& clusters,
                                    std::vector<int>& matched);

  const edm::EDGetTokenT<reco::GsfPFRecTrackCollection> gsfPfRecTracks_;
  const edm::EDGetTokenT<reco::PFClusterCollection> ecalClusters_;
  const double dr2_;
};

#endif  // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSCProducer_h
