// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      TkAlV0sAnalyzer
//
/*
 *\class TkAlV0sAnalyzer TkAlV0sAnalyzer.cc Alignment/TkAlV0sAnalyzer/plugins/TkAlV0sAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 14 Dec 2023 15:10:34 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "TLorentzVector.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;

class TkAlV0sAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TkAlV0sAnalyzer(const edm::ParameterSet&);
  ~TkAlV0sAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  template <typename T, typename... Args>
  T* book(const Args&... args) const;
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<TrackCollection> tracksToken_;  //used to select what tracks to read from configuration file
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> vccToken_;
  edm::Service<TFileService> fs_;

  TH1F* h_diTrackMass;
  TH1F* h_V0Mass;
};

static constexpr double piMass2 = 0.13957018 * 0.13957018;

//
// constructors and destructor
//
TkAlV0sAnalyzer::TkAlV0sAnalyzer(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes<TrackCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
      vccToken_(consumes<reco::VertexCompositeCandidateCollection>(
          iConfig.getParameter<edm::InputTag>("vertexCompositeCandidates"))) {
  usesResource(TFileService::kSharedResource);
}

template <typename T, typename... Args>
T* TkAlV0sAnalyzer::book(const Args&... args) const {
  T* t = fs_->make<T>(args...);
  return t;
}

//
// member functions
//
// ------------ method called for each event  ------------
void TkAlV0sAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::vector<const reco::Track*> myTracks;

  edm::Handle<reco::VertexCompositeCandidateCollection> vccHandle;
  iEvent.getByToken(vccToken_, vccHandle);

  if (vccHandle->empty())
    return;

  reco::VertexCompositeCandidateCollection v0s = *vccHandle.product();

  for (const auto& track : iEvent.get(tracksToken_)) {
    myTracks.emplace_back(&track);
  }

  // exclude multiple candidates
  if (myTracks.size() != 2)
    return;

  for (const auto& v0 : v0s) {
    float mass = v0.mass();
    h_V0Mass->Fill(mass);

    for (size_t i = 0; i < v0.numberOfDaughters(); ++i) {
      //LogPrint("AlignmentTrackFromVertexCompositeCandidateSelector") << "daughter: " << i << std::endl;
      const reco::Candidate* daughter = v0.daughter(i);
      const reco::RecoChargedCandidate* chargedDaughter = dynamic_cast<const reco::RecoChargedCandidate*>(daughter);
      if (chargedDaughter) {
        //LogPrint("AlignmentTrackFromVertexCompositeCandidateSelector") << "charged daughter: " << i << std::endl;
        const reco::TrackRef trackRef = chargedDaughter->track();
        if (trackRef.isNonnull()) {
          // LogPrint("AlignmentTrackFromVertexCompositeCandidateSelector")
          // << "charged daughter has non-null trackref: " << i << std::endl;
        }
      }
    }
  }

  const auto& tplus = myTracks[0]->charge() > 0 ? myTracks[0] : myTracks[1];
  const auto& tminus = myTracks[0]->charge() < 0 ? myTracks[0] : myTracks[1];

  TLorentzVector p4_tplus(tplus->px(), tplus->py(), tplus->pz(), sqrt((tplus->p() * tplus->p()) + piMass2));
  TLorentzVector p4_tminus(tminus->px(), tminus->py(), tminus->pz(), sqrt((tminus->p() * tminus->p()) + piMass2));

  const auto& V0p4 = p4_tplus + p4_tminus;
  float track_invMass = V0p4.M();
  h_diTrackMass->Fill(track_invMass);
}

void TkAlV0sAnalyzer::beginJob() {
  h_diTrackMass = book<TH1F>("diTrackMass", "V0 mass from tracks in Event", 100, 0.400, 0.600);
  h_V0Mass = book<TH1F>("V0kMass", "Reconstructed V0 mass in Event", 100, 0.400, 0.600);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TkAlV0sAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertexCompositeCandidates", edm::InputTag("generalV0Candidates:Kshort"));
  desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("ALCARECOTkAlKShortTracks"));
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TkAlV0sAnalyzer);
