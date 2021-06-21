#ifndef HLTrigger_Muon_HLTTriMuonIsolation_h
#define HLTrigger_Muon_HLTTriMuonIsolation_h

#include <iostream>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

class HLTTriMuonIsolation : public edm::global::EDProducer<> {
public:
  explicit HLTTriMuonIsolation(const edm::ParameterSet& iConfig);
  ~HLTTriMuonIsolation() override = default;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> L3MuonsToken_;
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> AllMuonsToken_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> L3DiMuonsFilterToken_;
  const edm::EDGetTokenT<reco::TrackCollection> IsoTracksToken_;

  edm::Handle<reco::RecoChargedCandidateCollection> L3MuCands;
  edm::Handle<trigger::TriggerFilterObjectWithRefs> L3DiMuonsFilterCands;
  edm::Handle<reco::RecoChargedCandidateRef> PassedL3Muons;
  edm::Handle<reco::RecoChargedCandidateCollection> AllMuCands;
  edm::Handle<reco::TrackCollection> IsoTracks;

  template <typename T>
  static bool ptComparer(const T& cand_1, const T& cand_2) {
    return cand_1.pt() > cand_2.pt();
  }

  const double TwiceMuonMass_ = 2. * 0.1056583715;  // in GeV

  const double Muon1PtCut_;
  const double Muon2PtCut_;
  const double Muon3PtCut_;
  const double TriMuonPtCut_;
  const double TriMuonEtaCut_;
  const double ChargedRelIsoCut_;
  const double ChargedAbsIsoCut_;
  const double IsoConeSize_;
  const double MatchingConeSize_;
  const double MinTriMuonMass_;
  const double MaxTriMuonMass_;
  const double MaxTriMuonRadius_;
  const int TriMuonAbsCharge_;
  const double MaxDZ_;
  const bool EnableRelIso_;
  const bool EnableAbsIso_;
};

inline HLTTriMuonIsolation::HLTTriMuonIsolation(const edm::ParameterSet& iConfig)
    : L3MuonsToken_(consumes<reco::RecoChargedCandidateCollection>(iConfig.getParameter<edm::InputTag>("L3MuonsSrc"))),
      AllMuonsToken_(
          consumes<reco::RecoChargedCandidateCollection>(iConfig.getParameter<edm::InputTag>("AllMuonsSrc"))),
      L3DiMuonsFilterToken_(
          consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("L3DiMuonsFilterSrc"))),
      IsoTracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("IsoTracksSrc"))),
      Muon1PtCut_(iConfig.getParameter<double>("Muon1PtCut")),
      Muon2PtCut_(iConfig.getParameter<double>("Muon2PtCut")),
      Muon3PtCut_(iConfig.getParameter<double>("Muon3PtCut")),
      TriMuonPtCut_(iConfig.getParameter<double>("TriMuonPtCut")),
      TriMuonEtaCut_(iConfig.getParameter<double>("TriMuonEtaCut")),
      ChargedRelIsoCut_(iConfig.getParameter<double>("ChargedRelIsoCut")),
      ChargedAbsIsoCut_(iConfig.getParameter<double>("ChargedAbsIsoCut")),
      IsoConeSize_(iConfig.getParameter<double>("IsoConeSize")),
      MatchingConeSize_(iConfig.getParameter<double>("MatchingConeSize")),
      MinTriMuonMass_(iConfig.getParameter<double>("MinTriMuonMass")),
      MaxTriMuonMass_(iConfig.getParameter<double>("MaxTriMuonMass")),
      MaxTriMuonRadius_(iConfig.getParameter<double>("MaxTriMuonRadius")),
      TriMuonAbsCharge_(iConfig.getParameter<int>("TriMuonAbsCharge")),
      MaxDZ_(iConfig.getParameter<double>("MaxDZ")),
      EnableRelIso_(iConfig.getParameter<bool>("EnableRelIso")),
      EnableAbsIso_(iConfig.getParameter<bool>("EnableAbsIso")) {
  //register products
  produces<reco::CompositeCandidateCollection>("Taus");
  produces<reco::CompositeCandidateCollection>("SelectedTaus");
}

inline void HLTTriMuonIsolation::produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  std::unique_ptr<reco::CompositeCandidateCollection> Taus(new reco::CompositeCandidateCollection);
  std::unique_ptr<reco::CompositeCandidateCollection> SelectedTaus(new reco::CompositeCandidateCollection);

  // Get the L3 muon candidates
  edm::Handle<reco::RecoChargedCandidateCollection> L3MuCands;
  iEvent.getByToken(L3MuonsToken_, L3MuCands);

  // Get the L3 muon candidates that passed the filter
  edm::Handle<trigger::TriggerFilterObjectWithRefs> L3DiMuonsFilterCands;
  iEvent.getByToken(L3DiMuonsFilterToken_, L3DiMuonsFilterCands);

  std::vector<reco::RecoChargedCandidateRef> PassedL3Muons;
  L3DiMuonsFilterCands->getObjects(trigger::TriggerMuon, PassedL3Muons);

  // Get the Trk + L3 muon candidates (after merging)
  edm::Handle<reco::RecoChargedCandidateCollection> AllMuCands;
  iEvent.getByToken(AllMuonsToken_, AllMuCands);

  // Get iso tracks
  edm::Handle<reco::TrackCollection> IsoTracks;
  iEvent.getByToken(IsoTracksToken_, IsoTracks);

  if (AllMuCands->size() >= 3 && L3MuCands->size() >= 2) {
    // Create the 3-muon candidates
    // loop over L3/Trk muons and create all combinations
    auto AllMuCands_end = AllMuCands->end();
    for (auto i = AllMuCands->begin(); i != AllMuCands_end - 2; ++i) {
      // check that muon_i passes the previous filter
      bool passingPreviousFilter_1 = false;
      for (const auto& imu : PassedL3Muons) {
        if (reco::deltaR2(i->momentum(), imu->momentum()) < (MatchingConeSize_ * MatchingConeSize_))
          passingPreviousFilter_1 = true;
      }
      for (auto j = i + 1; j != AllMuCands_end - 1; ++j) {
        // check that muon_j passes the previous filter
        bool passingPreviousFilter_2 = false;
        for (const auto& jmu : PassedL3Muons) {
          if (reco::deltaR2(j->momentum(), jmu->momentum()) < (MatchingConeSize_ * MatchingConeSize_))
            passingPreviousFilter_2 = true;
        }
        // if, at this point, no muons passed the previous filter just skip to the next iteration
        if (!(passingPreviousFilter_1 || passingPreviousFilter_2))
          continue;
        for (auto k = j + 1; k != AllMuCands_end; ++k) {
          // check that muon_k passes the previous filter
          bool passingPreviousFilter_3 = false;
          for (const auto& kmu : PassedL3Muons) {
            if (reco::deltaR2(k->momentum(), kmu->momentum()) < (MatchingConeSize_ * MatchingConeSize_))
              passingPreviousFilter_3 = true;
          }
          // at least two muons must have passed the previous di-muon filter
          if (!((passingPreviousFilter_1 & passingPreviousFilter_2) ||
                (passingPreviousFilter_1 & passingPreviousFilter_3) ||
                (passingPreviousFilter_2 & passingPreviousFilter_3)))
            continue;

          // Create a composite candidate to be a tau
          reco::CompositeCandidate tau;

          // sort the muons by pt and add them to the tau
          reco::RecoChargedCandidateCollection daughters;
          daughters.reserve(3);

          daughters.push_back(*i);
          daughters.push_back(*j);
          daughters.push_back(*k);

          std::sort(daughters.begin(), daughters.end(), ptComparer<reco::RecoChargedCandidate>);

          // Muon kinematic selections
          if (daughters[0].pt() < Muon1PtCut_)
            continue;
          if (daughters[1].pt() < Muon2PtCut_)
            continue;
          if (daughters[2].pt() < Muon3PtCut_)
            continue;

          // assign the tau its daughters
          tau.addDaughter((daughters)[0], "Muon_1");
          tau.addDaughter((daughters)[1], "Muon_2");
          tau.addDaughter((daughters)[2], "Muon_3");

          // start building the tau
          int charge = daughters[0].charge() + daughters[1].charge() + daughters[2].charge();
          math::XYZTLorentzVectorD taup4 = daughters[0].p4() + daughters[1].p4() + daughters[2].p4();
          int tauPdgId = charge > 0 ? 15 : -15;

          tau.setP4(taup4);
          tau.setCharge(charge);
          tau.setPdgId(tauPdgId);
          tau.setVertex((daughters)[0].vertex());  // assign the leading muon vertex as tau vertex

          // the three muons must be close to each other in Z
          if (std::abs(tau.daughter(0)->vz() - tau.vz()) > MaxDZ_)
            continue;
          if (std::abs(tau.daughter(1)->vz() - tau.vz()) > MaxDZ_)
            continue;
          if (std::abs(tau.daughter(2)->vz() - tau.vz()) > MaxDZ_)
            continue;

          // require muons to be collimated
          bool collimated = true;
          for (auto const& idau : daughters) {
            if (reco::deltaR2(tau.p4(), idau.p4()) > MaxTriMuonRadius_ * MaxTriMuonRadius_) {
              collimated = false;
              break;
            }
          }

          if (!collimated)
            continue;

          // Tau kinematic selections
          if (tau.pt() < TriMuonPtCut_)
            continue;
          if (tau.mass() < MinTriMuonMass_)
            continue;
          if (tau.mass() > MaxTriMuonMass_)
            continue;
          if (std::abs(tau.eta()) > TriMuonEtaCut_)
            continue;

          // Tau charge selection
          if ((std::abs(tau.charge()) != TriMuonAbsCharge_) & (TriMuonAbsCharge_ >= 0))
            continue;

          // Sanity check against duplicates, di-muon masses must be > 2 * mass_mu
          if ((tau.daughter(0)->p4() + tau.daughter(1)->p4()).mass() < TwiceMuonMass_)
            continue;
          if ((tau.daughter(0)->p4() + tau.daughter(2)->p4()).mass() < TwiceMuonMass_)
            continue;
          if ((tau.daughter(1)->p4() + tau.daughter(2)->p4()).mass() < TwiceMuonMass_)
            continue;

          // a good tau, at last
          Taus->push_back(tau);
        }
      }
    }

    // Sort taus by pt
    std::sort(Taus->begin(), Taus->end(), ptComparer<reco::CompositeCandidate>);

    // Loop over taus and further select by isolation
    for (const auto& itau : *Taus) {
      // remove the candidate pt from the iso sum
      double sumPt = -itau.pt();

      // compute iso sum pT
      for (const auto& itrk : *IsoTracks) {
        if (reco::deltaR2(itrk.momentum(), itau.p4()) > IsoConeSize_ * IsoConeSize_)
          continue;
        if (std::abs(itrk.vz() - itau.vz()) > MaxDZ_)
          continue;
        sumPt += itrk.pt();
      }

      // apply the isolation cut
      double chAbsIsoCut = EnableAbsIso_ ? ChargedAbsIsoCut_ : std::numeric_limits<double>::infinity();
      double chRelIsoCut = EnableRelIso_ ? ChargedRelIsoCut_ * itau.pt() : std::numeric_limits<double>::infinity();

      if (!((sumPt < chAbsIsoCut) || (sumPt < chRelIsoCut)))
        continue;

      SelectedTaus->push_back(itau);
    }
  }

  // finally put the vector of 3-muon candidates in the event
  iEvent.put(std::move(Taus), "Taus");
  iEvent.put(std::move(SelectedTaus), "SelectedTaus");
}

inline void HLTTriMuonIsolation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L3MuonsSrc", edm::InputTag("hltIterL3FromL2MuonCandidates"));
  desc.add<edm::InputTag>("AllMuonsSrc", edm::InputTag("hltGlbTrkMuonCands"));
  desc.add<edm::InputTag>("L3DiMuonsFilterSrc", edm::InputTag("hltDiMuonForTau3MuDzFiltered0p3"));
  desc.add<edm::InputTag>("IsoTracksSrc", edm::InputTag("hltIter2L3FromL2MuonMerged"));
  desc.add<double>("Muon1PtCut", 5.);
  desc.add<double>("Muon2PtCut", 3.);
  desc.add<double>("Muon3PtCut", 0.);
  desc.add<double>("TriMuonPtCut", 8.);
  desc.add<double>("TriMuonEtaCut", 2.5);
  desc.add<double>("ChargedAbsIsoCut", 3.0);
  desc.add<double>("ChargedRelIsoCut", 0.1);
  desc.add<double>("IsoConeSize", 0.5);
  desc.add<double>("MatchingConeSize", 0.03);
  desc.add<double>("MinTriMuonMass", 0.5);
  desc.add<double>("MaxTriMuonMass", 2.8);
  desc.add<double>("MaxTriMuonRadius", 0.6);
  desc.add<int>("TriMuonAbsCharge", -1);
  desc.add<double>("MaxDZ", 0.3);
  desc.add<bool>("EnableRelIso", false);
  desc.add<bool>("EnableAbsIso", true);
  descriptions.add("hltTriMuonIsolationProducer", desc);
}

#endif
