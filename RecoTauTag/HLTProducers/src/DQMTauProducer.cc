#include "RecoTauTag/HLTProducers/interface/DQMTauProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
//
// class decleration
//

DQMTauProducer::DQMTauProducer(const edm::ParameterSet& iConfig) {
  trackIsolatedJets_ =
      consumes<reco::IsolatedTauTagInfoCollection>(iConfig.getParameter<edm::InputTag>("TrackIsoJets"));
  matchingCone_ = iConfig.getParameter<double>("MatchingCone");
  signalCone_ = iConfig.getParameter<double>("SignalCone");
  ptMin_ = iConfig.getParameter<double>("MinPtTracks");

  isolationCone_ = iConfig.getParameter<double>("IsolationCone");
  produces<reco::HLTTauCollection>();
}

DQMTauProducer::~DQMTauProducer() {}

void DQMTauProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iES) const {
  using namespace reco;
  using namespace edm;
  using namespace std;

  auto jetCollection = std::make_unique<HLTTauCollection>();

  edm::Handle<IsolatedTauTagInfoCollection> tauL25Jets;
  iEvent.getByToken(trackIsolatedJets_, tauL25Jets);

  IsolatedTauTagInfoCollection tau = *(tauL25Jets.product());

  float eta_, phi_, pt_;
  float ptLeadTk = 0.;
  int trackIsolation = 1000.;
  int nTracks = 1000.;

  for (unsigned int i = 0; i < tau.size(); i++) {
    JetTracksAssociationRef jetTracks = tau[i].jtaRef();
    math::XYZVector jetDir(jetTracks->first->px(), jetTracks->first->py(), jetTracks->first->pz());
    eta_ = jetDir.eta();
    phi_ = jetDir.phi();
    pt_ = jetTracks->first->pt();

    const TrackRef leadTk = tau[i].leadingSignalTrack(jetDir, matchingCone_, 1.);
    if (!leadTk) {
    } else {
      trackIsolation = (int)tau[i].discriminator(jetDir, matchingCone_, signalCone_, isolationCone_, 1., 1., 0);
      ptLeadTk = (*leadTk).pt();
      nTracks = (tau[i].tracksInCone((*leadTk).momentum(), isolationCone_, ptMin_)).size() -
                (tau[i].tracksInCone((*leadTk).momentum(), signalCone_, ptMin_)).size();
    }
    HLTTau pippo(eta_, phi_, pt_, -1, trackIsolation, ptLeadTk, trackIsolation, ptLeadTk);
    pippo.setNL25TrackIsolation(nTracks);
    pippo.setNL3TrackIsolation(nTracks);
    jetCollection->push_back(pippo);
  }

  iEvent.put(std::move(jetCollection));
}
