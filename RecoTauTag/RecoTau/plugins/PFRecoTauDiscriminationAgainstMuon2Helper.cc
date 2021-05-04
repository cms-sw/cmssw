#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoTauTag/RecoTau/plugins/PFRecoTauDiscriminationAgainstMuon2Helper.h"
#include "RecoTauTag/RecoTau/interface/RecoTauMuonTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using reco::tau::format_vint;

PFRecoTauDiscriminationAgainstMuon2Helper::PFRecoTauDiscriminationAgainstMuon2Helper(
    const bool& verbosity,
    const std::string& moduleLabel,
    const bool srcMuons_label_empty,
    const double& minPtMatchedMuon,
    const double& dRmuonMatch,
    const bool& dRmuonMatchLimitedToJetArea,
    std::atomic<unsigned int>& numWarnings,
    const unsigned int& maxWarnings,
    const std::vector<int>& maskMatchesDT,
    const std::vector<int>& maskMatchesCSC,
    const std::vector<int>& maskMatchesRPC,
    const std::vector<int>& maskHitsDT,
    const std::vector<int>& maskHitsCSC,
    const std::vector<int>& maskHitsRPC,
    const edm::Handle<reco::MuonCollection>& muons,
    const reco::PFTauRef& pfTau,
    const reco::PFCandidatePtr& pfCand)
    : pfLeadChargedHadron_{pfCand} {
  if (verbosity) {
    edm::LogPrint("PFTauAgainstMuon2") << "<PFRecoTauDiscriminationAgainstMuon2Container::discriminate>:";
    edm::LogPrint("PFTauAgainstMuon2") << " moduleLabel = " << moduleLabel;
    edm::LogPrint("PFTauAgainstMuon2") << "tau #" << pfTau.key() << ": Pt = " << pfTau->pt()
                                       << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi();
  }

  std::vector<int> numMatchesDT(4);
  std::vector<int> numMatchesCSC(4);
  std::vector<int> numMatchesRPC(4);
  std::vector<int> numHitsDT(4);
  std::vector<int> numHitsCSC(4);
  std::vector<int> numHitsRPC(4);
  for (int iStation = 0; iStation < 4; ++iStation) {
    numMatchesDT[iStation] = 0;
    numMatchesCSC[iStation] = 0;
    numMatchesRPC[iStation] = 0;
    numHitsDT[iStation] = 0;
    numHitsCSC[iStation] = 0;
    numHitsRPC[iStation] = 0;
  }

  //pfLeadChargedHadron_ = pfTau->leadPFChargedHadrCand();
  if (pfLeadChargedHadron_.isNonnull()) {
    reco::MuonRef muonRef = pfLeadChargedHadron_->muonRef();
    if (muonRef.isNonnull()) {
      if (verbosity)
        edm::LogPrint("PFTauAgainstMuon2") << " has muonRef.";
      reco::tau::countMatches(*muonRef, numMatchesDT, numMatchesCSC, numMatchesRPC);
      reco::tau::countHits(*muonRef, numHitsDT, numHitsCSC, numHitsRPC);
    }
  }

  if (!srcMuons_label_empty) {
    size_t numMuons = muons->size();
    for (size_t idxMuon = 0; idxMuon < numMuons; ++idxMuon) {
      reco::MuonRef muon(muons, idxMuon);
      if (verbosity)
        edm::LogPrint("PFTauAgainstMuon2") << "muon #" << muon.key() << ": Pt = " << muon->pt()
                                           << ", eta = " << muon->eta() << ", phi = " << muon->phi();
      if (!(muon->pt() > minPtMatchedMuon)) {
        if (verbosity) {
          edm::LogPrint("PFTauAgainstMuon2") << " fails Pt cut --> skipping it.";
        }
        continue;
      }
      if (pfLeadChargedHadron_.isNonnull()) {
        reco::MuonRef muonRef = pfLeadChargedHadron_->muonRef();
        if (muonRef.isNonnull() && muon == pfLeadChargedHadron_->muonRef()) {
          if (verbosity) {
            edm::LogPrint("PFTauAgainstMuon2") << " matches muonRef of tau --> skipping it.";
          }
          continue;
        }
      }
      double dR = deltaR(muon->p4(), pfTau->p4());
      double dRmatch = dRmuonMatch;
      if (dRmuonMatchLimitedToJetArea) {
        double jetArea = 0.;
        if (pfTau->jetRef().isNonnull())
          jetArea = pfTau->jetRef()->jetArea();
        if (jetArea > 0.) {
          dRmatch = std::min(dRmatch, std::sqrt(jetArea / M_PI));
        } else {
          if (numWarnings < maxWarnings) {
            edm::LogInfo("PFRecoTauDiscriminationAgainstMuon2Container::discriminate")
                << "Jet associated to Tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta()
                << ", phi = " << pfTau->phi() << " has area = " << jetArea << " !!";
            ++numWarnings;
          }
          dRmatch = 0.1;
        }
      }
      if (dR < dRmatch) {
        if (verbosity)
          edm::LogPrint("PFTauAgainstMuon2") << " overlaps with tau, dR = " << dR;
        reco::tau::countMatches(*muon, numMatchesDT, numMatchesCSC, numMatchesRPC);
        reco::tau::countHits(*muon, numHitsDT, numHitsCSC, numHitsRPC);
      }
    }
  }

  for (int iStation = 0; iStation < 4; ++iStation) {
    if (numMatchesDT[iStation] > 0 && !maskMatchesDT[iStation])
      ++numStationsWithMatches_;
    if (numMatchesCSC[iStation] > 0 && !maskMatchesCSC[iStation])
      ++numStationsWithMatches_;
    if (numMatchesRPC[iStation] > 0 && !maskMatchesRPC[iStation])
      ++numStationsWithMatches_;
  }

  for (int iStation = 2; iStation < 4; ++iStation) {
    if (numHitsDT[iStation] > 0 && !maskHitsDT[iStation])
      ++numLast2StationsWithHits_;
    if (numHitsCSC[iStation] > 0 && !maskHitsCSC[iStation])
      ++numLast2StationsWithHits_;
    if (numHitsRPC[iStation] > 0 && !maskHitsRPC[iStation])
      ++numLast2StationsWithHits_;
  }

  if (verbosity) {
    edm::LogPrint("PFTauAgainstMuon2") << "numMatchesDT  = " << format_vint(numMatchesDT);
    edm::LogPrint("PFTauAgainstMuon2") << "numMatchesCSC = " << format_vint(numMatchesCSC);
    edm::LogPrint("PFTauAgainstMuon2") << "numMatchesRPC = " << format_vint(numMatchesRPC);
    edm::LogPrint("PFTauAgainstMuon2") << " --> numStationsWithMatches_ = " << numStationsWithMatches_;
    edm::LogPrint("PFTauAgainstMuon2") << "numHitsDT  = " << format_vint(numHitsDT);
    edm::LogPrint("PFTauAgainstMuon2") << "numHitsCSC = " << format_vint(numHitsCSC);
    edm::LogPrint("PFTauAgainstMuon2") << "numHitsRPC = " << format_vint(numHitsRPC);
    edm::LogPrint("PFTauAgainstMuon2") << " --> numLast2StationsWithHits_ = " << numLast2StationsWithHits_;
  }

  if (pfLeadChargedHadron_.isNonnull()) {
    energyECALplusHCAL_ = pfLeadChargedHadron_->ecalEnergy() + pfLeadChargedHadron_->hcalEnergy();
    if (verbosity) {
      if (pfLeadChargedHadron_->trackRef().isNonnull()) {
        edm::LogPrint("PFTauAgainstMuon2")
            << "decayMode = " << pfTau->decayMode() << ", energy(ECAL+HCAL) = " << energyECALplusHCAL_
            << ", leadPFChargedHadronP = " << pfLeadChargedHadron_->trackRef()->p();
      } else if (pfLeadChargedHadron_->gsfTrackRef().isNonnull()) {
        edm::LogPrint("PFTauAgainstMuon2")
            << "decayMode = " << pfTau->decayMode() << ", energy(ECAL+HCAL) = " << energyECALplusHCAL_
            << ", leadPFChargedHadronP = " << pfLeadChargedHadron_->gsfTrackRef()->p();
      }
    }
    if (pfLeadChargedHadron_->trackRef().isNonnull())
      leadTrack_ = pfLeadChargedHadron_->trackRef().get();
    else if (pfLeadChargedHadron_->gsfTrackRef().isNonnull())
      leadTrack_ = pfLeadChargedHadron_->gsfTrackRef().get();
  }
}

bool PFRecoTauDiscriminationAgainstMuon2Helper::eval(const PFRecoTauDiscriminationAgainstMuonConfigSet& config,
                                                     const reco::PFTauRef& pfTau) const {
  bool passesCaloMuonVeto = true;
  if (pfLeadChargedHadron_.isNonnull()) {
    if (pfTau->decayMode() == 0 && leadTrack_ && energyECALplusHCAL_ < (config.hop * leadTrack_->p()))
      passesCaloMuonVeto = false;
  }

  bool discriminatorValue = false;
  if (config.discriminatorOption == PFRecoTauDiscriminationAgainstMuonConfigSet::kLoose &&
      numStationsWithMatches_ <= config.maxNumberOfMatches)
    discriminatorValue = true;
  else if (config.discriminatorOption == PFRecoTauDiscriminationAgainstMuonConfigSet::kMedium &&
           numStationsWithMatches_ <= config.maxNumberOfMatches &&
           numLast2StationsWithHits_ <= config.maxNumberOfHitsLast2Stations)
    discriminatorValue = true;
  else if (config.discriminatorOption == PFRecoTauDiscriminationAgainstMuonConfigSet::kTight &&
           numStationsWithMatches_ <= config.maxNumberOfMatches &&
           numLast2StationsWithHits_ <= config.maxNumberOfHitsLast2Stations && passesCaloMuonVeto)
    discriminatorValue = true;
  else if (config.discriminatorOption == PFRecoTauDiscriminationAgainstMuonConfigSet::kCustom) {
    discriminatorValue = true;
    if (config.maxNumberOfMatches >= 0 && numStationsWithMatches_ > config.maxNumberOfMatches)
      discriminatorValue = false;
    if (config.maxNumberOfHitsLast2Stations >= 0 && numLast2StationsWithHits_ > config.maxNumberOfHitsLast2Stations)
      discriminatorValue = false;
    if (config.doCaloMuonVeto && !passesCaloMuonVeto)
      discriminatorValue = false;
  }
  return discriminatorValue;
}
