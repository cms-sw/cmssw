#ifndef DQMOFFLINE_L1TRIGGER_L1TPHASE2MUONOFFLINE_H
#define DQMOFFLINE_L1TRIGGER_L1TPHASE2MUONOFFLINE_H

/**
 * \file L1TPhase2MuonOffline.h
 *
 * \author S. Folgueras
*
 */

// DataFormats
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQMServices
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <memory>
#include "TRegexp.h"
#include <utility>
#include <vector>

class GenMuonGMTPair;

//
// DQM class declaration
//

class L1TPhase2MuonOffline : public DQMEDAnalyzer {
public:
  L1TPhase2MuonOffline(const edm::ParameterSet& ps);
  ~L1TPhase2MuonOffline() override = default;

  enum MuType { kSAMuon, kTkMuon, kNMuTypes };
  enum VarType { kPt, kEta, kPhi, kIso, kQual, kZ0, kD0, kNVarTypes };
  enum EffType { kEffPt, kEffPhi, kEffEta, kEffTypes };
  enum ResType { kResPt, kRes1OverPt, kResQOverPt, kResPhi, kResEta, kResCh, kNResTypes };
  enum EtaRegion { kEtaRegionAll, kEtaRegionBmtf, kEtaRegionOmtf, kEtaRegionEmtf, kNEtaRegions };
  enum QualLevel { kQualOpen, kQualDouble, kQualSingle, kNQualLevels };

protected:
  void dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void bookControlHistos(DQMStore::IBooker&, MuType type);
  void bookEfficiencyHistos(DQMStore::IBooker& ibooker, MuType type);
  void bookResolutionHistos(DQMStore::IBooker& ibooker, MuType type);
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup& iSetup) override;

  //Fill Histos
  void fillControlHistos();
  void fillEfficiencyHistos();
  void fillResolutionHistos();

private:
  // Cut and Matching
  void getMuonGmtPairs(edm::Handle<l1t::MuonBxCollection>& gmtCands);

  // Handles and Tokens
  edm::EDGetTokenT<l1t::SAMuonCollection> gmtMuonToken_;
  edm::EDGetTokenT<l1t::TrackerMuonCollection> gmtTkMuonToken_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticleToken_;

  edm::Handle<l1t::SAMuonCollection> gmtSAMuon_;
  edm::Handle<l1t::TrackerMuonCollection> gmtTkMuon_;
  edm::Handle<std::vector<reco::GenParticle>> genparticles_;

  //  PropagateToMuon muonpropagator_;

  // vectors of enum values to loop over (and store quantities)
  const std::vector<MuType> muonTypes_;
  const std::vector<EffType> effTypes_;
  const std::vector<ResType> resTypes_;
  //  const std::vector<VarType> varTypes_;
  const std::vector<EtaRegion> etaRegions_;
  const std::vector<QualLevel> qualLevels_;

  // maps with histogram name bits
  //  std::map<EffType, std::string> effNames_;
  //  std::map<EffType, std::string> effLabels_;
  std::map<ResType, std::string> resNames_;
  std::map<ResType, std::string> resLabels_;
  std::map<EtaRegion, std::string> etaNames_;
  std::map<QualLevel, std::string> qualNames_;
  std::map<MuType, std::string> muonNames_;

  // config params
  const std::string histFolder_;
  const std::vector<edm::ParameterSet> cutsVPSet_;

  const std::vector<double> effVsPtBins_;
  const std::vector<double> effVsPhiBins_;
  const std::vector<double> effVsEtaBins_;

  const double maxGmtMuonDR_;

  // Helper methods
  void matchMuonsToGen(std::vector<const reco::GenParticle*> genmus);
  std::vector<float> getHistBinsEff(EffType eff);
  std::tuple<int, double, double> getHistBinsRes(ResType res);

  MonitorElement* efficiencyNum_[kNMuTypes][kNEtaRegions][kNQualLevels][kEffTypes];
  MonitorElement* efficiencyDen_[kNMuTypes][kNEtaRegions][kNQualLevels][kEffTypes];
  MonitorElement* resolutionHistos_[kNMuTypes][kNEtaRegions][kNQualLevels][kNResTypes];
  MonitorElement* controlHistos_[kNMuTypes][kNVarTypes];

  // helper variables
  std::vector<GenMuonGMTPair> gmtSAMuonPairs_;
  std::vector<GenMuonGMTPair> gmtTkMuonPairs_;
  std::vector<std::pair<int, QualLevel>> cuts_;

  const float lsb_pt = Phase2L1GMT::LSBpt;
  const float lsb_phi = Phase2L1GMT::LSBphi;
  const float lsb_eta = Phase2L1GMT::LSBeta;
  const float lsb_z0 = Phase2L1GMT::LSBSAz0;
  const float lsb_d0 = Phase2L1GMT::LSBSAd0;
};

//
// helper class to manage GMT-GenMuon pairing
//
class GenMuonGMTPair {
public:
  GenMuonGMTPair(const reco::GenParticle* mu, const l1t::L1Candidate* gmtmu);
  GenMuonGMTPair(const GenMuonGMTPair& muongmtPair);
  ~GenMuonGMTPair(){};

  float dR2();
  float pt() const { return mu_->pt(); };
  float eta() const { return mu_->eta(); };
  float phi() const { return mu_->phi(); };
  int charge() const { return mu_->charge(); };

  // Now properties of the L1 candidate:
  float gmtPt() const { return gmtmu_ ? gmtmu_->pt() : -1.; };
  float gmtEta() const { return gmtmu_ ? gmtEta_ : -5.; };
  float gmtPhi() const { return gmtmu_ ? gmtPhi_ : -5.; };
  int gmtCharge() const { return gmtmu_ ? gmtmu_->charge() : -5; };
  int gmtQual() const { return gmtmu_ ? gmtmu_->hwQual() : -1; };

  L1TPhase2MuonOffline::EtaRegion etaRegion() const;
  double getDeltaVar(const L1TPhase2MuonOffline::ResType) const;
  double getVar(const L1TPhase2MuonOffline::EffType) const;

private:
  const reco::GenParticle* mu_;
  const l1t::L1Candidate* gmtmu_;

  // L1T muon eta and phi coordinates to be used
  // Can be the coordinates from the 2nd muon station or from the vertex
  float gmtEta_;
  float gmtPhi_;

  float muEta_;
  float muPhi_;
};

#endif
