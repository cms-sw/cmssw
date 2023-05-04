
#include "RecoTauTag/HLTProducers/interface/L2TauPixelTrackMatch.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaPhi.h"

L2TauPixelTrackMatch::L2TauPixelTrackMatch(const edm::ParameterSet& conf) {
  m_jetSrc = consumes<trigger::TriggerFilterObjectWithRefs>(conf.getParameter<edm::InputTag>("JetSrc"));
  m_jetMinPt = conf.getParameter<double>("JetMinPt");
  m_jetMaxEta = conf.getParameter<double>("JetMaxEta");
  //m_jetMinN	     = conf.getParameter<int>("JetMinN");
  m_trackSrc = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackSrc"));
  m_trackMinPt = conf.getParameter<double>("TrackMinPt");
  m_deltaR = conf.getParameter<double>("deltaR");
  m_beamSpotTag = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("BeamSpotSrc"));

  produces<reco::CaloJetCollection>();
}

L2TauPixelTrackMatch::~L2TauPixelTrackMatch() {}

void L2TauPixelTrackMatch::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const {
  using namespace std;
  using namespace reco;

  // *** Pick up beam spot ***

  // use beam spot for vertex x,y
  edm::Handle<BeamSpot> bsHandle;
  ev.getByToken(m_beamSpotTag, bsHandle);
  const reco::BeamSpot& bs = *bsHandle;
  math::XYZPoint beam_spot(bs.x0(), bs.y0(), bs.z0());

  // *** Pick up pixel tracks ***

  edm::Handle<TrackCollection> tracksHandle;
  ev.getByToken(m_trackSrc, tracksHandle);

  // *** Pick up L2 tau jets that were previously selected by some other filter ***

  // first, get L2 object refs by the label
  edm::Handle<trigger::TriggerFilterObjectWithRefs> jetsHandle;
  ev.getByToken(m_jetSrc, jetsHandle);

  // now we can get pre-selected L2 tau jets
  std::vector<CaloJetRef> tau_jets;
  jetsHandle->getObjects(trigger::TriggerTau, tau_jets);
  const size_t n_jets = tau_jets.size();

  // *** Selects interesting tracks ***

  vector<TinyTrack> good_tracks;
  for (TrackCollection::const_iterator itrk = tracksHandle->begin(); itrk != tracksHandle->end(); ++itrk) {
    if (itrk->pt() < m_trackMinPt)
      continue;
    if (std::abs(itrk->eta()) > m_jetMaxEta + m_deltaR)
      continue;

    TinyTrack trk;
    trk.pt = itrk->pt();
    trk.phi = itrk->phi();
    trk.eta = itrk->eta();
    double dz = itrk->dz(beam_spot);
    trk.vtx = math::XYZPoint(bs.x(dz), bs.y(dz), dz);
    good_tracks.push_back(trk);
  }

  // *** Match tau jets to intertesting tracks  and assign them new vertices ***

  // the new product
  std::unique_ptr<CaloJetCollection> new_tau_jets(new CaloJetCollection);
  if (!good_tracks.empty())
    for (size_t i = 0; i < n_jets; ++i) {
      reco::CaloJetRef jet = tau_jets[i];
      if (jet->pt() < m_jetMinPt || std::abs(jet->eta()) > m_jetMaxEta)
        continue;

      size_t n0 = new_tau_jets->size();

      for (vector<TinyTrack>::const_iterator itrk = good_tracks.begin(); itrk != good_tracks.end(); ++itrk) {
        math::XYZTLorentzVector new_jet_dir = Jet::physicsP4(itrk->vtx, *jet, itrk->vtx);
        float dphi = reco::deltaPhi(new_jet_dir.phi(), itrk->phi);
        float deta = new_jet_dir.eta() - itrk->eta;

        if (dphi * dphi + deta * deta > m_deltaR * m_deltaR)
          continue;

        // create a jet copy and assign a new vertex to it
        CaloJet new_jet = *jet;
        new_jet.setVertex(itrk->vtx);
        new_tau_jets->push_back(new_jet);
      }
      ///if (jet_with_vertices.size()) new_tau_jets->push_back(jet_with_vertices);
    }

  // store the result
  ev.put(std::move(new_tau_jets));
}
