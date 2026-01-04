/**  \class TrackingAssocValueMapsProducer
 *
 *  \brief Produces ValueMaps with tracking particle association info for tracks
 *
 *  This producer creates ValueMaps for a collection of reco::Track containing
 *  information about their association to TrackingParticles:
 *  - whether the track is matched to a TP
 *  - whether the track is a duplicate (i.e. the matched TP is also matched to other tracks)
 *  - the PDG ID and charge of the matched TP
 *  - optionally, the kinematic variables (pt, eta, phi) of the matched TP
 *  The association is performed using a TrackToTrackingParticleAssociator.
 *  The TrackingParticles considered for the association can be filtered
 *  using a set of selection criteria provided via a ParameterSet.
 *
 *   \author Luca Ferragina (INFN BO), 2025
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

class TrackingAssocValueMapsProducer : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit TrackingAssocValueMapsProducer(const edm::ParameterSet&);
  ~TrackingAssocValueMapsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<edm::View<reco::Track>> tracksToken_;
  const edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociatorToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> tpToken_;

  edm::ParameterSet tpSet_;
  TrackingParticleSelector tpSelector_;

  bool storeTPKinematics_;
};

namespace {
  template <typename T>
  void fillAndPut(edm::Event& evt,
                  const edm::Handle<edm::View<reco::Track>>& tracksH,
                  std::unique_ptr<edm::ValueMap<T>> product,
                  const std::vector<T>& values,
                  const std::string& label) {
    typename edm::ValueMap<T>::Filler filler(*product);
    filler.insert(tracksH, values.begin(), values.end());
    filler.fill();
    evt.put(std::move(product), label);
  }
}  // namespace

TrackingAssocValueMapsProducer::TrackingAssocValueMapsProducer(const edm::ParameterSet& cfg)
    : tracksToken_(consumes<edm::View<reco::Track>>(cfg.getParameter<edm::InputTag>("trackCollection"))),
      trackAssociatorToken_(
          consumes<reco::TrackToTrackingParticleAssociator>(cfg.getParameter<edm::InputTag>("associator"))),
      tpToken_(consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticles"))),
      tpSet_(cfg.getParameter<edm::ParameterSet>("tpSelectorPSet")),
      storeTPKinematics_(cfg.getParameter<bool>("storeTPKinematics")) {
  tpSelector_ = TrackingParticleSelector(tpSet_.getParameter<double>("ptMin"),
                                         tpSet_.getParameter<double>("ptMax"),
                                         tpSet_.getParameter<double>("minRapidity"),
                                         tpSet_.getParameter<double>("maxRapidity"),
                                         tpSet_.getParameter<double>("tip"),
                                         tpSet_.getParameter<double>("lip"),
                                         tpSet_.getParameter<int>("minHit"),
                                         tpSet_.getParameter<bool>("signalOnly"),
                                         tpSet_.getParameter<bool>("intimeOnly"),
                                         tpSet_.getParameter<bool>("chargedOnly"),
                                         tpSet_.getParameter<bool>("stableOnly"),
                                         tpSet_.getParameter<std::vector<int>>("pdgId"),
                                         tpSet_.getParameter<bool>("invertRapidityCut"),
                                         tpSet_.getParameter<double>("minPhi"),
                                         tpSet_.getParameter<double>("maxPhi"));

  produces<edm::ValueMap<int>>("matched");
  produces<edm::ValueMap<int>>("duplicate");
  produces<edm::ValueMap<int>>("tpPdgId");
  produces<edm::ValueMap<int>>("tpCharge");
  if (storeTPKinematics_) {
    produces<edm::ValueMap<float>>("tpPt");
    produces<edm::ValueMap<float>>("tpEta");
    produces<edm::ValueMap<float>>("tpPhi");
  }
}

void TrackingAssocValueMapsProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<edm::View<reco::Track>> tracksH;
  iEvent.getByToken(tracksToken_, tracksH);

  edm::Handle<reco::TrackToTrackingParticleAssociator> associatorH;
  iEvent.getByToken(trackAssociatorToken_, associatorH);

  edm::Handle<TrackingParticleCollection> tpH;
  iEvent.getByToken(tpToken_, tpH);

  const size_t nTracks = tracksH->size();

  if (nTracks == 0 || !tracksH.isValid() || !associatorH.isValid() || !tpH.isValid()) {
    // No tracks or invalid handles, put empty ValueMaps and return
    fillAndPut<int>(iEvent, tracksH, std::make_unique<edm::ValueMap<int>>(), {}, "matched");
    fillAndPut<int>(iEvent, tracksH, std::make_unique<edm::ValueMap<int>>(), {}, "duplicate");
    fillAndPut<int>(iEvent, tracksH, std::make_unique<edm::ValueMap<int>>(), {}, "tpPdgId");
    fillAndPut<int>(iEvent, tracksH, std::make_unique<edm::ValueMap<int>>(), {}, "tpCharge");
    if (storeTPKinematics_) {
      fillAndPut<float>(iEvent, tracksH, std::make_unique<edm::ValueMap<float>>(), {}, "tpPt");
      fillAndPut<float>(iEvent, tracksH, std::make_unique<edm::ValueMap<float>>(), {}, "tpEta");
      fillAndPut<float>(iEvent, tracksH, std::make_unique<edm::ValueMap<float>>(), {}, "tpPhi");
    }
    return;
  }

  std::vector<int> matched(nTracks, 0);
  std::vector<int> duplicate(nTracks, 0);
  std::vector<int> tpPdgId(nTracks, 0);
  std::vector<int> tpCharge(nTracks, 0);

  std::vector<float> tpPt, tpEta, tpPhi;
  if (storeTPKinematics_) {
    tpPt.assign(nTracks, -1.f);
    tpEta.assign(nTracks, -10.f);
    tpPhi.assign(nTracks, -10.f);
  }

  TrackingParticleRefVector tpCollection;
  for (size_t i = 0, size = tpH->size(); i < size; ++i) {
    auto tp = TrackingParticleRef(tpH, i);
    if (tpSelector_(*tp)) {
      tpCollection.push_back(tp);
    }
  }

  edm::RefToBaseVector<reco::Track> trackRefs;
  for (edm::View<reco::Track>::size_type i = 0; i < nTracks; ++i) {
    trackRefs.push_back(tracksH->refAt(i));
  }

  reco::RecoToSimCollection const& recoToSimColl = associatorH->associateRecoToSim(trackRefs, tpCollection);
  reco::SimToRecoCollection const& simToRecoColl = associatorH->associateSimToReco(trackRefs, tpCollection);

  for (edm::View<reco::Track>::size_type i = 0; i < trackRefs.size(); ++i) {
    const auto& track = trackRefs[i];
    auto foundTp = recoToSimColl.find(track);
    if (foundTp != recoToSimColl.end()) {
      const auto& tp = foundTp->val;
      if (!tp.empty()) {
        matched[i] = 1;
        tpPdgId[i] = static_cast<int16_t>(tp[0].first->pdgId());
        tpCharge[i] = static_cast<int8_t>(tp[0].first->charge());
        if (storeTPKinematics_) {
          tpPt[i] = tp[0].first->pt();
          tpEta[i] = tp[0].first->eta();
          tpPhi[i] = tp[0].first->phi();
        }
      }
      if (simToRecoColl.find(tp[0].first) != simToRecoColl.end()) {
        if (simToRecoColl[tp[0].first].size() > 1) {
          duplicate[i] = 1;
        }
      }
    }
  }

  // Produce ValueMaps and store in the event
  fillAndPut<int>(iEvent, tracksH, std::make_unique<edm::ValueMap<int>>(), matched, "matched");
  fillAndPut<int>(iEvent, tracksH, std::make_unique<edm::ValueMap<int>>(), duplicate, "duplicate");
  fillAndPut<int>(iEvent, tracksH, std::make_unique<edm::ValueMap<int>>(), tpPdgId, "tpPdgId");
  fillAndPut<int>(iEvent, tracksH, std::make_unique<edm::ValueMap<int>>(), tpCharge, "tpCharge");
  if (storeTPKinematics_) {
    fillAndPut<float>(iEvent, tracksH, std::make_unique<edm::ValueMap<float>>(), tpPt, "tpPt");
    fillAndPut<float>(iEvent, tracksH, std::make_unique<edm::ValueMap<float>>(), tpEta, "tpEta");
    fillAndPut<float>(iEvent, tracksH, std::make_unique<edm::ValueMap<float>>(), tpPhi, "tpPhi");
  }
}

void TrackingAssocValueMapsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("trackCollection", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("associator", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));

  // TP Selector parameters
  edm::ParameterSetDescription tpSet;
  tpSet.add<double>("ptMin", 0.005);
  tpSet.add<double>("ptMax", 1e100);
  tpSet.add<double>("minRapidity", -2.5);
  tpSet.add<double>("maxRapidity", 2.5);
  tpSet.add<double>("tip", 60);
  tpSet.add<double>("lip", 30.0);
  tpSet.add<int>("minHit", 0);
  tpSet.add<bool>("signalOnly", false);
  tpSet.add<bool>("intimeOnly", true);
  tpSet.add<bool>("chargedOnly", true);
  tpSet.add<bool>("stableOnly", false);
  tpSet.add<std::vector<int>>("pdgId", {});
  tpSet.add<bool>("invertRapidityCut", false);
  tpSet.add<double>("minPhi", -3.2);
  tpSet.add<double>("maxPhi", 3.2);

  desc.add<edm::ParameterSetDescription>("tpSelectorPSet", tpSet);
  desc.add<bool>("storeTPKinematics", true);

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingAssocValueMapsProducer);
