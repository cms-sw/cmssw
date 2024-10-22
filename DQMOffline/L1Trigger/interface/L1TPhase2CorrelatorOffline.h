#ifndef DQMOFFLINE_L1TRIGGER_L1TPHASE2CORRELATOROFFLINE_H
#define DQMOFFLINE_L1TRIGGER_L1TPHASE2CORRELATOROFFLINE_H

// DataFormats
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

// FWCore
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

// DQMServices
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/L1Trigger/interface/HistDefinition.h"

// MagneticField
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// HLTrigger
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// L1Trigger
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TPFUtils.h"

// CommonTools
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <TRandom3.h>
#include <TTree.h>
#include <cstdint>

class L1TPhase2CorrelatorOffline : public DQMOneEDAnalyzer<> {
public:
  L1TPhase2CorrelatorOffline(const edm::ParameterSet& ps);
  ~L1TPhase2CorrelatorOffline() override;

  enum PlotConfig { resVsPt, resVsEta, ptDist, etaDist };

  static const std::map<std::string, unsigned int> PlotConfigNames;

protected:
  void dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void dqmEndRun(const edm::Run& run, const edm::EventSetup& iSetup) override;

  // Cut and Matching

private:
  void bookPhase2CorrelatorHistos(DQMStore::IBooker&);

  // other functions
  double Distance(const reco::Candidate& c1, const reco::Candidate& c2);
  double DistancePhi(const reco::Candidate& c1, const reco::Candidate& c2);
  double calcDeltaPhi(double phi1, double phi2);

  void computeResponseResolution();

  std::vector<float> getQuantile(float quant, TH2F* hist);
  void medianResponseCorrResolution(MonitorElement* in2D, MonitorElement* response, MonitorElement* resolution);

  struct SimpleObject {
    float pt, eta, phi;
    SimpleObject(float apt, float aneta, float aphi) : pt(apt), eta(aneta), phi(aphi) {}
    bool operator<(const SimpleObject& other) const { return eta < other.eta; }
    bool operator<(const float& other) const { return eta < other; }
  };
  class MultiCollection {
  public:
    MultiCollection(const edm::ParameterSet& iConfig, const std::string& name, edm::ConsumesCollector&& coll)
        : name_(name), prop_(false), sel_("") {
      if (name.find("Ecal") != std::string::npos)
        prop_ = true;
      else if (name.find("Hcal") != std::string::npos)
        prop_ = true;
      else if (name.find("Calo") != std::string::npos)
        prop_ = true;
      const std::vector<edm::InputTag>& tags = iConfig.getParameter<std::vector<edm::InputTag>>(name);
      for (const auto& tag : tags)
        tokens_.push_back(coll.consumes<reco::CandidateView>(tag));
      if (iConfig.existsAs<std::string>(name + "_sel")) {
        sel_ = StringCutObjectSelector<reco::Candidate>(iConfig.getParameter<std::string>(name + "_sel"), true);
      }
    }
    const std::string& name() const { return name_; }
    bool prop() const { return prop_; }
    void get(const edm::Event& iEvent) {
      edm::Handle<reco::CandidateView> handle;
      for (const auto& token : tokens_) {
        iEvent.getByToken(token, handle);
        for (const reco::Candidate& c : *handle) {
          if (sel_(c))
            objects_.emplace_back(c.pt(), c.eta(), c.phi());
        }
      }
      std::sort(objects_.begin(), objects_.end());
    }
    const std::vector<SimpleObject>& objects() const { return objects_; }
    void clear() { objects_.clear(); }

  private:
    std::string name_;
    bool prop_;
    std::vector<edm::EDGetTokenT<reco::CandidateView>> tokens_;
    StringCutObjectSelector<reco::Candidate> sel_;
    std::vector<SimpleObject> objects_;
  };
  class InCone {
  public:
    InCone(const std::vector<SimpleObject>& objects, float eta, float phi, float dr) {
      auto first =
          std::lower_bound(objects.begin(), objects.end(), eta - dr - 0.01f);  // small offset to avoid dealing with ==
      auto end = std::lower_bound(objects.begin(), objects.end(), eta + dr + 0.01f);
      float dr2 = dr * dr;
      sum04_ = 0;
      for (auto it = first; it < end; ++it) {
        float mydr2 = ::deltaR2(eta, phi, it->eta, it->phi);
        if (mydr2 < dr2)
          ptdr2_.emplace_back(it->pt, mydr2);
        if (mydr2 < 0.16f)
          sum04_ += it->pt;
      }
    }
    float sum(float dr = 0.4) const {
      if (dr == 0.4f)
        return sum04_;
      float dr2 = dr * dr;
      float mysum = 0;
      for (const auto& part : ptdr2_) {
        if (part.second < dr2)
          mysum += part.first;
      }
      return mysum;
    }
    int number(float dr, float threshold) const {
      float dr2 = dr * dr, absthreshold = sum() * threshold;
      int mysum = 0;
      for (const auto& part : ptdr2_) {
        if (part.second < dr2 && part.first > absthreshold)
          mysum++;
      }
      return mysum;
    }
    float mindr(float threshold) const {
      float best = 9999, absthreshold = sum() * threshold;
      for (const auto& part : ptdr2_) {
        if (part.second < best && part.first > absthreshold)
          best = part.second;
      }
      return std::sqrt(best);
    }
    float nearest() const {
      std::pair<float, float> best(0, 9999);
      for (const auto& part : ptdr2_) {
        if (part.second < best.second)
          best = part;
      }
      return best.first;
    }
    float max(float dr = 0.4) const {
      float best = 0, dr2 = dr * dr;
      for (const auto& part : ptdr2_) {
        if (part.first > best && part.second < dr2)
          best = part.first;
      }
      return best;
    }

  private:
    std::vector<std::pair<float, float>> ptdr2_;
    float sum04_;
  };

  // variables from config file
  edm::EDGetTokenT<std::vector<reco::GenJet>> genJetToken_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticleToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> BFieldTag_;
  edm::ParameterSet objs_;
  bool isParticleGun_;
  std::string histFolder_;
  std::string respresolFolder_;
  dqmoffline::l1t::HistDefinitions histDefinitions_;

  std::vector<edm::EDGetTokenT<std::vector<l1t::PFCandidate>>> phase2PFToken_;
  std::vector<edm::EDGetTokenT<std::vector<l1t::PFCandidate>>> phase2PuppiToken_;

  // config params
  struct McVars {
    float pt, pt02, eta, phi, iso02, iso04, iso08;
    int charge;
    float caloeta, calophi;
    int id;
    void fillP4(const reco::Candidate& cand) {
      pt = cand.pt();
      eta = cand.eta();
      phi = cand.phi();
      caloeta = eta;
      calophi = phi;
      charge = 0;
    }
    void fillPropagated(const reco::Candidate& cand, float bz) {
      if (cand.charge() != 0) {
        math::XYZTLorentzVector vertex(cand.vx(), cand.vy(), cand.vz(), 0.);
        auto caloetaphi = l1tpf::propagateToCalo(cand.p4(), vertex, cand.charge(), bz);
        caloeta = caloetaphi.first;
        calophi = caloetaphi.second;
      }
    }

  } mc_;
  struct RecoVars {
    float pt, pt02, pt08, ptbest, pthighest, mindr025;
    int n025, n010;
    void fill(const std::vector<SimpleObject>& objects, float eta, float phi) {
      InCone incone(objects, eta, phi, 0.8);
      pt = incone.sum();
      pt02 = incone.sum(0.2);
      pt08 = incone.sum(0.8);
      ptbest = incone.nearest();
      pthighest = incone.max();
      mindr025 = incone.mindr(0.25);
      n025 = incone.number(0.2, 0.25);
      n010 = incone.number(0.2, 0.10);
    }
    void clear() {
      pt = 0.;
      pt02 = 0.;
      pt08 = 0.;
      ptbest = 0.;
      pthighest = 0.;
      mindr025 = 0.;
      n025 = -1;
      n010 = -1;
    }
  };
  std::vector<std::pair<MultiCollection, RecoVars>> reco_;
  float bZ_;

  // Histograms
  MonitorElement* h_L1PF_pt_;
  MonitorElement* h_L1PF_eta_;
  MonitorElement* h_L1Puppi_pt_;
  MonitorElement* h_L1Puppi_eta_;

  MonitorElement* h_L1PF_pt_mu_;
  MonitorElement* h_L1PF_eta_mu_;
  MonitorElement* h_L1Puppi_pt_mu_;
  MonitorElement* h_L1Puppi_eta_mu_;

  MonitorElement* h_L1PF_pt_el_;
  MonitorElement* h_L1PF_eta_el_;
  MonitorElement* h_L1Puppi_pt_el_;
  MonitorElement* h_L1Puppi_eta_el_;

  MonitorElement* h_L1PF_pt_pho_;
  MonitorElement* h_L1PF_eta_pho_;
  MonitorElement* h_L1Puppi_pt_pho_;
  MonitorElement* h_L1Puppi_eta_pho_;

  MonitorElement* h_L1PF_pt_ch_;
  MonitorElement* h_L1PF_eta_ch_;
  MonitorElement* h_L1Puppi_pt_ch_;
  MonitorElement* h_L1Puppi_eta_ch_;

  MonitorElement* h_L1PF_pt_nh_;
  MonitorElement* h_L1PF_eta_nh_;
  MonitorElement* h_L1Puppi_pt_nh_;
  MonitorElement* h_L1Puppi_eta_nh_;

  MonitorElement* h_L1PF_part_ptratio_0p2_vs_pt_barrel_;
  MonitorElement* h_L1PF_part_ptratio_0p2_vs_pt_endcap_;
  MonitorElement* h_L1PF_part_ptratio_0p2_vs_pt_ecnotk_;
  MonitorElement* h_L1PF_part_ptratio_0p2_vs_pt_hf_;
  MonitorElement* h_L1PF_part_ptratio_0p2_vs_eta_;
  MonitorElement* h_L1Puppi_part_ptratio_0p2_vs_pt_barrel_;
  MonitorElement* h_L1Puppi_part_ptratio_0p2_vs_pt_endcap_;
  MonitorElement* h_L1Puppi_part_ptratio_0p2_vs_pt_ecnotk_;
  MonitorElement* h_L1Puppi_part_ptratio_0p2_vs_pt_hf_;
  MonitorElement* h_L1Puppi_part_ptratio_0p2_vs_eta_;
  MonitorElement* h_L1PF_jet_ptratio_vs_pt_barrel_;
  MonitorElement* h_L1PF_jet_ptratio_vs_pt_endcap_;
  MonitorElement* h_L1PF_jet_ptratio_vs_pt_ecnotk_;
  MonitorElement* h_L1PF_jet_ptratio_vs_pt_hf_;
  MonitorElement* h_L1PF_jet_ptratio_vs_eta_;
  MonitorElement* h_L1Puppi_jet_ptratio_vs_pt_barrel_;
  MonitorElement* h_L1Puppi_jet_ptratio_vs_pt_endcap_;
  MonitorElement* h_L1Puppi_jet_ptratio_vs_pt_ecnotk_;
  MonitorElement* h_L1Puppi_jet_ptratio_vs_pt_hf_;
  MonitorElement* h_L1Puppi_jet_ptratio_vs_eta_;

  MonitorElement* h_L1PF_part_response_0p2_pt_barrel_;
  MonitorElement* h_L1PF_part_response_0p2_pt_endcap_;
  MonitorElement* h_L1PF_part_response_0p2_pt_ecnotk_;
  MonitorElement* h_L1PF_part_response_0p2_pt_hf_;
  MonitorElement* h_L1PF_part_response_0p2_eta_;
  MonitorElement* h_L1Puppi_part_response_0p2_pt_barrel_;
  MonitorElement* h_L1Puppi_part_response_0p2_pt_endcap_;
  MonitorElement* h_L1Puppi_part_response_0p2_pt_ecnotk_;
  MonitorElement* h_L1Puppi_part_response_0p2_pt_hf_;
  MonitorElement* h_L1Puppi_part_response_0p2_eta_;
  MonitorElement* h_L1PF_jet_response_pt_barrel_;
  MonitorElement* h_L1PF_jet_response_pt_endcap_;
  MonitorElement* h_L1PF_jet_response_pt_ecnotk_;
  MonitorElement* h_L1PF_jet_response_pt_hf_;
  MonitorElement* h_L1PF_jet_response_eta_;
  MonitorElement* h_L1Puppi_jet_response_pt_barrel_;
  MonitorElement* h_L1Puppi_jet_response_pt_endcap_;
  MonitorElement* h_L1Puppi_jet_response_pt_ecnotk_;
  MonitorElement* h_L1Puppi_jet_response_pt_hf_;
  MonitorElement* h_L1Puppi_jet_response_eta_;

  MonitorElement* h_L1PF_part_resolution_0p2_pt_barrel_;
  MonitorElement* h_L1PF_part_resolution_0p2_pt_endcap_;
  MonitorElement* h_L1PF_part_resolution_0p2_pt_ecnotk_;
  MonitorElement* h_L1PF_part_resolution_0p2_pt_hf_;
  MonitorElement* h_L1Puppi_part_resolution_0p2_pt_barrel_;
  MonitorElement* h_L1Puppi_part_resolution_0p2_pt_endcap_;
  MonitorElement* h_L1Puppi_part_resolution_0p2_pt_ecnotk_;
  MonitorElement* h_L1Puppi_part_resolution_0p2_pt_hf_;
  MonitorElement* h_L1PF_jet_resolution_pt_barrel_;
  MonitorElement* h_L1PF_jet_resolution_pt_endcap_;
  MonitorElement* h_L1PF_jet_resolution_pt_ecnotk_;
  MonitorElement* h_L1PF_jet_resolution_pt_hf_;
  MonitorElement* h_L1Puppi_jet_resolution_pt_barrel_;
  MonitorElement* h_L1Puppi_jet_resolution_pt_endcap_;
  MonitorElement* h_L1Puppi_jet_resolution_pt_ecnotk_;
  MonitorElement* h_L1Puppi_jet_resolution_pt_hf_;
};

#endif
