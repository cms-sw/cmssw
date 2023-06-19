/** \class HLTDeDxFilter
*
*
*  \author Claude Nuttens
*
*/

#include "RecoTracker/DeDx/plugins/HLTDeDxFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
//#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

//
// constructors and destructor
//
HLTDeDxFilter::HLTDeDxFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  minDEDx_ = iConfig.getParameter<double>("minDEDx");
  minPT_ = iConfig.getParameter<double>("minPT");
  minNOM_ = iConfig.getParameter<double>("minNOM");
  maxETA_ = iConfig.getParameter<double>("maxETA");
  minNumValidHits_ = iConfig.getParameter<double>("minNumValidHits");
  maxNHitMissIn_ = iConfig.getParameter<double>("maxNHitMissIn");
  maxNHitMissMid_ = iConfig.getParameter<double>("maxNHitMissMid");
  maxRelTrkIsoDeltaRp3_ = iConfig.getParameter<double>("maxRelTrkIsoDeltaRp3");
  relTrkIsoDeltaRSize_ = iConfig.getParameter<double>("relTrkIsoDeltaRSize");
  maxAssocCaloE_ = iConfig.getParameter<double>("maxAssocCaloE");
  maxAssocCaloEDeltaRSize_ = iConfig.getParameter<double>("maxAssocCaloEDeltaRSize");
  inputTracksTag_ = iConfig.getParameter<edm::InputTag>("inputTracksTag");
  inputdedxTag_ = iConfig.getParameter<edm::InputTag>("inputDeDxTag");
  caloTowersTag_ = iConfig.getParameter<edm::InputTag>("caloTowersTag");

  if (maxAssocCaloE_ >= 0)
    caloTowersToken_ = consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("caloTowersTag"));
  inputTracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTracksTag"));
  inputdedxToken_ = consumes<edm::ValueMap<reco::DeDxData>>(iConfig.getParameter<edm::InputTag>("inputDeDxTag"));

  thisModuleTag_ = edm::InputTag(iConfig.getParameter<std::string>("@module_label"));

  //register your products
  produces<reco::RecoChargedCandidateCollection>();
}

HLTDeDxFilter::~HLTDeDxFilter() {}

void HLTDeDxFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("saveTags", false);
  desc.add<double>("minDEDx", 0.0);
  desc.add<double>("minPT", 0.0);
  desc.add<double>("minNOM", 0.0);
  desc.add<double>("maxETA", 5.5);
  desc.add<double>("minNumValidHits", 0);
  desc.add<double>("maxNHitMissIn", 99);
  desc.add<double>("maxNHitMissMid", 99);
  desc.add<double>("maxRelTrkIsoDeltaRp3", -1);
  desc.add<double>("relTrkIsoDeltaRSize", 0.3);
  desc.add<double>("maxAssocCaloE", -99);
  desc.add<double>("maxAssocCaloEDeltaRSize", 0.5);
  desc.add<edm::InputTag>("caloTowersTag", edm::InputTag("hltTowerMakerForAll"));
  desc.add<edm::InputTag>("inputTracksTag", edm::InputTag("hltL3Mouns"));
  desc.add<edm::InputTag>("inputDeDxTag", edm::InputTag("HLTdedxHarm2"));
  descriptions.add("hltDeDxFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLTDeDxFilter::hltFilter(edm::Event& iEvent,
                              const edm::EventSetup& iSetup,
                              trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  auto chargedCandidates = std::make_unique<std::vector<RecoChargedCandidate>>();

  ModuleDescription moduleDesc_;

  if (saveTags()) {
    filterproduct.addCollectionTag(thisModuleTag_);
    filterproduct.addCollectionTag(inputTracksTag_);
    filterproduct.addCollectionTag(inputdedxTag_);
  }

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(inputTracksToken_, trackCollectionHandle);
  const reco::TrackCollection& trackCollection = *trackCollectionHandle.product();

  edm::Handle<edm::ValueMap<reco::DeDxData>> dEdxTrackHandle;
  iEvent.getByToken(inputdedxToken_, dEdxTrackHandle);
  const edm::ValueMap<reco::DeDxData>& dEdxTrack = *dEdxTrackHandle.product();

  edm::Handle<CaloTowerCollection> caloTowersHandle;
  if (maxAssocCaloE_ >= 0)
    iEvent.getByToken(caloTowersToken_, caloTowersHandle);

  bool accept = false;

  // early return
  if (trackCollection.empty())
    return accept;

  //fill local arrays for eta, phi, and pt
  float eta[trackCollection.size()], phi[trackCollection.size()], pt[trackCollection.size()];
  for (unsigned int i = 0; i < trackCollection.size(); i++) {
    eta[i] = trackCollection[i].eta();
    phi[i] = trackCollection[i].phi();
    pt[i] = trackCollection[i].pt();
  }
  for (unsigned int i = 0; i < trackCollection.size(); i++) {
    reco::TrackRef track = reco::TrackRef(trackCollectionHandle, i);
    if (pt[i] > minPT_ && fabs(eta[i]) < maxETA_ && dEdxTrack[track].numberOfMeasurements() > minNOM_ &&
        dEdxTrack[track].dEdx() > minDEDx_) {
      if (track->numberOfValidHits() < minNumValidHits_)
        continue;
      if (track->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS) > maxNHitMissIn_)
        continue;
      if (track->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS) > maxNHitMissMid_)
        continue;
      if (saveTags()) {
        Particle::Charge q = track->charge();
        //SAVE DEDX INFORMATION AS IF IT WAS THE MASS OF THE PARTICLE
        Particle::LorentzVector p4(
            track->px(), track->py(), track->pz(), sqrt(pow(track->p(), 2) + pow(dEdxTrack[track].dEdx(), 2)));
        Particle::Point vtx(track->vx(), track->vy(), track->vz());
        //SAVE NOH, NOM, NOS INFORMATION AS IF IT WAS THE PDGID OF THE PARTICLE
        int Hits = ((dEdxTrack[track].numberOfSaturatedMeasurements() & 0xFF) << 16) |
                   ((dEdxTrack[track].numberOfMeasurements() & 0xFF) << 8) | (track->found() & 0xFF);
        RecoChargedCandidate cand(q, p4, vtx, Hits, 0);
        cand.setTrack(track);
        chargedCandidates->push_back(cand);
      }

      //calculate relative trk isolation only if parameter maxRelTrkIsoDeltaRp3 is greater than 0
      if (maxRelTrkIsoDeltaRp3_ >= 0) {
        auto ptCone = trackCollection[i].pt();
        for (unsigned int j = 0; j < trackCollection.size(); j++) {
          if (i == j)
            continue;  // do not compare track to itself
          auto trkDeltaR2 = deltaR2(eta[i], phi[i], eta[j], phi[j]);
          if (trkDeltaR2 < relTrkIsoDeltaRSize_ * relTrkIsoDeltaRSize_) {
            ptCone += pt[j];
          }
        }
        double relTrkIso = (ptCone - pt[i]) / (pt[i]);
        if (relTrkIso > maxRelTrkIsoDeltaRp3_)
          continue;
      }

      //calculate the calorimeter energy associated with the track if maxAssocCaloE_ >= 0
      if (maxAssocCaloE_ >= 0) {
        //Access info about Calo Towers
        double caloEMDeltaRp5 = 0;
        double caloHadDeltaRp5 = 0;
        const CaloTowerCollection& caloTower = *caloTowersHandle.product();
        for (CaloTowerCollection::const_iterator j = caloTower.begin(); j != caloTower.end(); j++) {
          auto caloDeltaR2 = deltaR2(eta[i], phi[i], j->eta(), j->phi());
          double Eem = j->emEnergy();
          double Ehad = j->hadEnergy();

          if (caloDeltaR2 < (maxAssocCaloEDeltaRSize_ * maxAssocCaloEDeltaRSize_)) {
            caloEMDeltaRp5 += Eem;
            caloHadDeltaRp5 += Ehad;
          }
        }
        if (caloEMDeltaRp5 + caloHadDeltaRp5 > maxAssocCaloE_)
          continue;
      }

      accept = true;
    }
  }

  // put filter object into the Event
  if (saveTags()) {
    auto const chargedCandidatesHandle = iEvent.put(std::move(chargedCandidates));
    for (unsigned int i = 0; i < chargedCandidatesHandle->size(); ++i) {
      filterproduct.addObject(trigger::TriggerMuon, reco::RecoChargedCandidateRef(chargedCandidatesHandle, i));
    }
  }

  return accept;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDeDxFilter);
