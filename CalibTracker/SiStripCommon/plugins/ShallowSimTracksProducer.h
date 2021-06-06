#ifndef SHALLOW_SIMTRACKS_PRODUCER
#define SHALLOW_SIMTRACKS_PRODUCER

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

class ShallowSimTracksProducer : public edm::stream::EDProducer<> {
public:
  explicit ShallowSimTracksProducer(const edm::ParameterSet&);

private:
  const std::string Prefix;
  const std::string Suffix;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_token_;
  const edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> associator_token_;
  const edm::EDGetTokenT<edm::View<reco::Track> > tracks_token_;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
};
#endif
