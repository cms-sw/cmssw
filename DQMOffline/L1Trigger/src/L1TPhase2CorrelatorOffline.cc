/**
 *  @file     L1TPhase2CorrelatorOffline.cc
 *  @authors  Dylan Rankin (MIT)
 *  @date     20/10/2020
 *  @version  0.1
 *
 */

#include "DQMOffline/L1Trigger/interface/L1TPhase2CorrelatorOffline.h"

#include "TLorentzVector.h"
#include "TGraph.h"

#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace reco;
using namespace trigger;
using namespace edm;
using namespace std;

const std::map<std::string, unsigned int> L1TPhase2CorrelatorOffline::PlotConfigNames = {
    {"resVsPt", PlotConfig::resVsPt},
    {"resVsEta", PlotConfig::resVsEta},
    {"ptDist", PlotConfig::ptDist},
    {"etaDist", PlotConfig::etaDist}};

//
// -------------------------------------- Constructor --------------------------------------------
//
L1TPhase2CorrelatorOffline::L1TPhase2CorrelatorOffline(const edm::ParameterSet& ps)
    : genJetToken_(consumes<std::vector<reco::GenJet>>(ps.getUntrackedParameter<edm::InputTag>("genJetsInputTag"))),
      genParticleToken_(
          consumes<std::vector<reco::GenParticle>>(ps.getUntrackedParameter<edm::InputTag>("genParticlesInputTag"))),
      BFieldTag_{esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()},
      objs_(ps.getParameter<edm::ParameterSet>("objects")),
      isParticleGun_(ps.getParameter<bool>("isParticleGun")),
      histFolder_(ps.getParameter<std::string>("histFolder")),
      respresolFolder_(histFolder_ + "/respresol_raw"),
      histDefinitions_(dqmoffline::l1t::readHistDefinitions(ps.getParameterSet("histDefinitions"), PlotConfigNames)),
      h_L1PF_pt_(),
      h_L1PF_eta_(),
      h_L1Puppi_pt_(),
      h_L1Puppi_eta_(),
      h_L1PF_pt_mu_(),
      h_L1PF_eta_mu_(),
      h_L1Puppi_pt_mu_(),
      h_L1Puppi_eta_mu_(),
      h_L1PF_pt_el_(),
      h_L1PF_eta_el_(),
      h_L1Puppi_pt_el_(),
      h_L1Puppi_eta_el_(),
      h_L1PF_pt_pho_(),
      h_L1PF_eta_pho_(),
      h_L1Puppi_pt_pho_(),
      h_L1Puppi_eta_pho_(),
      h_L1PF_pt_ch_(),
      h_L1PF_eta_ch_(),
      h_L1Puppi_pt_ch_(),
      h_L1Puppi_eta_ch_(),
      h_L1PF_pt_nh_(),
      h_L1PF_eta_nh_(),
      h_L1Puppi_pt_nh_(),
      h_L1Puppi_eta_nh_(),
      h_L1PF_part_ptratio_0p2_vs_pt_barrel_(),
      h_L1PF_part_ptratio_0p2_vs_pt_endcap_(),
      h_L1PF_part_ptratio_0p2_vs_pt_ecnotk_(),
      h_L1PF_part_ptratio_0p2_vs_pt_hf_(),
      h_L1PF_part_ptratio_0p2_vs_eta_(),
      h_L1Puppi_part_ptratio_0p2_vs_pt_barrel_(),
      h_L1Puppi_part_ptratio_0p2_vs_pt_endcap_(),
      h_L1Puppi_part_ptratio_0p2_vs_pt_ecnotk_(),
      h_L1Puppi_part_ptratio_0p2_vs_pt_hf_(),
      h_L1Puppi_part_ptratio_0p2_vs_eta_(),
      h_L1PF_jet_ptratio_vs_pt_barrel_(),
      h_L1PF_jet_ptratio_vs_pt_endcap_(),
      h_L1PF_jet_ptratio_vs_pt_ecnotk_(),
      h_L1PF_jet_ptratio_vs_pt_hf_(),
      h_L1PF_jet_ptratio_vs_eta_(),
      h_L1Puppi_jet_ptratio_vs_pt_barrel_(),
      h_L1Puppi_jet_ptratio_vs_pt_endcap_(),
      h_L1Puppi_jet_ptratio_vs_pt_ecnotk_(),
      h_L1Puppi_jet_ptratio_vs_pt_hf_(),
      h_L1Puppi_jet_ptratio_vs_eta_() {
  edm::LogInfo("L1TPhase2CorrelatorOffline") << "Constructor "
                                             << "L1TPhase2CorrelatorOffline::L1TPhase2CorrelatorOffline " << std::endl;

  auto reconames = objs_.getParameterNamesForType<std::vector<edm::InputTag>>();
  for (const std::string& name : reconames) {
    reco_.emplace_back(L1TPhase2CorrelatorOffline::MultiCollection(objs_, name, consumesCollector()), RecoVars());
  }
  for (auto& obj : objs_.getParameter<std::vector<edm::InputTag>>("L1PF")) {
    phase2PFToken_.push_back(consumes<std::vector<l1t::PFCandidate>>(obj));
  }
  for (auto& obj : objs_.getParameter<std::vector<edm::InputTag>>("L1Puppi")) {
    phase2PuppiToken_.push_back(consumes<std::vector<l1t::PFCandidate>>(obj));
  }
}

//
// -- Destructor
//
L1TPhase2CorrelatorOffline::~L1TPhase2CorrelatorOffline() {
  edm::LogInfo("L1TPhase2CorrelatorOffline")
      << "Destructor L1TPhase2CorrelatorOffline::~L1TPhase2CorrelatorOffline " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void L1TPhase2CorrelatorOffline::dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  edm::LogInfo("L1TPhase2CorrelatorOffline") << "L1TPhase2CorrelatorOffline::beginRun" << std::endl;

  bZ_ = iSetup.getData(BFieldTag_).inTesla(GlobalPoint(0, 0, 0)).z();
}

//
// -------------------------------------- bookHistos --------------------------------------------
//
void L1TPhase2CorrelatorOffline::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  edm::LogInfo("L1TPhase2CorrelatorOffline") << "L1TPhase2CorrelatorOffline::bookHistograms" << std::endl;

  // book at beginRun
  bookPhase2CorrelatorHistos(ibooker);
}

//
// -------------------------------------- Analyze --------------------------------------------
//
void L1TPhase2CorrelatorOffline::analyze(edm::Event const& e, edm::EventSetup const& eSetup) {
  edm::Handle<std::vector<reco::GenJet>> genjets;
  edm::Handle<std::vector<reco::GenParticle>> genparticles;
  e.getByToken(genJetToken_, genjets);
  e.getByToken(genParticleToken_, genparticles);

  std::vector<const reco::GenParticle*> prompts, taus;
  for (const reco::GenParticle& gen : *genparticles) {
    if (isParticleGun_) {
      if (gen.statusFlags().isPrompt() == 1)
        prompts.push_back(&gen);
      continue;
    }
    if ((gen.isPromptFinalState() || gen.isDirectPromptTauDecayProductFinalState()) &&
        (std::abs(gen.pdgId()) == 11 || std::abs(gen.pdgId()) == 13) && gen.pt() > 5) {
      prompts.push_back(&gen);
    } else if (gen.isPromptFinalState() && std::abs(gen.pdgId()) == 22 && gen.pt() > 10) {
      prompts.push_back(&gen);
    } else if (abs(gen.pdgId()) == 15 && gen.isPromptDecayed()) {
      taus.push_back(&gen);
    }
  }

  for (auto& recopair : reco_) {
    recopair.first.get(e);
  }

  for (const reco::GenJet& j1 : *genjets) {
    bool ok = true;
    const reco::Candidate* match = nullptr;
    for (const reco::GenParticle* gp : prompts) {
      if (::deltaR2(*gp, j1) < 0.16f) {
        if (match != nullptr) {
          ok = false;
          break;
        } else {
          match = gp;
        }
      }
    }
    if (!ok)
      continue;
    if (!match) {
      // look for a tau
      for (const reco::GenParticle* gp : taus) {
        if (::deltaR2(*gp, j1) < 0.16f) {
          if (match != nullptr) {
            ok = false;
            break;
          } else {
            match = gp;
          }
        }
      }
      if (!ok)
        continue;
      if (match != nullptr && match->numberOfDaughters() == 2 &&
          std::abs(match->daughter(0)->pdgId()) + std::abs(match->daughter(1)->pdgId()) == 211 + 16) {
        // one-prong tau, consider it a pion
        match = (std::abs(match->daughter(0)->pdgId()) == 211 ? match->daughter(0) : match->daughter(1));
      }
    }
    if (match != nullptr) {
      if (std::abs(match->pdgId()) == 15) {
        reco::Particle::LorentzVector pvis;
        for (unsigned int i = 0, n = match->numberOfDaughters(); i < n; ++i) {
          const reco::Candidate* dau = match->daughter(i);
          if (std::abs(dau->pdgId()) == 12 || std::abs(dau->pdgId()) == 14 || std::abs(dau->pdgId()) == 16) {
            continue;
          }
          pvis += dau->p4();
        }
        mc_.pt = pvis.Pt();
        mc_.eta = pvis.Eta();
        mc_.phi = pvis.Phi();
      } else {
        mc_.fillP4(*match);
        mc_.fillPropagated(*match, bZ_);
      }
      mc_.id = std::abs(match->pdgId());
      mc_.iso04 = j1.pt() / mc_.pt - 1;
      mc_.iso02 = 0;
      for (const auto& dptr : j1.daughterPtrVector()) {
        if (::deltaR2(*dptr, *match) < 0.04f) {
          mc_.iso02 += dptr->pt();
        }
      }
      mc_.iso02 = mc_.iso02 / mc_.pt - 1;
    } else {
      if (j1.pt() < 20)
        continue;
      mc_.fillP4(j1);
      mc_.id = 0;
      mc_.iso02 = 0;
      mc_.iso04 = 0;
    }
    mc_.iso08 = mc_.iso04;
    for (const reco::GenJet& j2 : *genjets) {
      if (&j2 == &j1)
        continue;
      if (::deltaR2(j1, j2) < 0.64f)
        mc_.iso08 += j2.pt() / mc_.pt;
    }
    for (auto& recopair : reco_) {
      recopair.second.fill(recopair.first.objects(),
                           recopair.first.prop() ? mc_.caloeta : mc_.eta,
                           recopair.first.prop() ? mc_.calophi : mc_.phi);

      if (abs(mc_.id) != 0 && fabs(mc_.iso04) < 0.05) {
        if (recopair.first.name() == "L1PF") {
          if (fabs(mc_.eta) < 1.5) {
            h_L1PF_part_ptratio_0p2_vs_pt_barrel_->Fill(mc_.pt, recopair.second.pt02 / mc_.pt);
          } else if (fabs(mc_.eta) < 2.5) {
            h_L1PF_part_ptratio_0p2_vs_pt_endcap_->Fill(mc_.pt, recopair.second.pt02 / mc_.pt);
          } else if (fabs(mc_.eta) < 3.) {
            h_L1PF_part_ptratio_0p2_vs_pt_ecnotk_->Fill(mc_.pt, recopair.second.pt02 / mc_.pt);
          } else if (fabs(mc_.eta) < 5.) {
            h_L1PF_part_ptratio_0p2_vs_pt_hf_->Fill(mc_.pt, recopair.second.pt02 / mc_.pt);
          }
          h_L1PF_part_ptratio_0p2_vs_eta_->Fill(mc_.eta, recopair.second.pt02 / mc_.pt);
        }
        if (recopair.first.name() == "L1Puppi") {
          if (fabs(mc_.eta) < 1.5) {
            h_L1Puppi_part_ptratio_0p2_vs_pt_barrel_->Fill(mc_.pt, recopair.second.pt02 / mc_.pt);
          } else if (fabs(mc_.eta) < 2.5) {
            h_L1Puppi_part_ptratio_0p2_vs_pt_endcap_->Fill(mc_.pt, recopair.second.pt02 / mc_.pt);
          } else if (fabs(mc_.eta) < 3.) {
            h_L1Puppi_part_ptratio_0p2_vs_pt_ecnotk_->Fill(mc_.pt, recopair.second.pt02 / mc_.pt);
          } else if (fabs(mc_.eta) < 5.) {
            h_L1Puppi_part_ptratio_0p2_vs_pt_hf_->Fill(mc_.pt, recopair.second.pt02 / mc_.pt);
          }
          h_L1Puppi_part_ptratio_0p2_vs_eta_->Fill(mc_.eta, recopair.second.pt02 / mc_.pt);
        }
      }
      if (abs(mc_.id) == 0) {
        if (recopair.first.name() == "L1PF") {
          if (fabs(mc_.eta) < 1.5) {
            h_L1PF_jet_ptratio_vs_pt_barrel_->Fill(mc_.pt, recopair.second.pt / mc_.pt);
          } else if (fabs(mc_.eta) < 2.5) {
            h_L1PF_jet_ptratio_vs_pt_endcap_->Fill(mc_.pt, recopair.second.pt / mc_.pt);
          } else if (fabs(mc_.eta) < 3.) {
            h_L1PF_jet_ptratio_vs_pt_ecnotk_->Fill(mc_.pt, recopair.second.pt / mc_.pt);
          } else if (fabs(mc_.eta) < 5.) {
            h_L1PF_jet_ptratio_vs_pt_hf_->Fill(mc_.pt, recopair.second.pt / mc_.pt);
          }
          h_L1PF_jet_ptratio_vs_eta_->Fill(mc_.eta, recopair.second.pt / mc_.pt);
        }
        if (recopair.first.name() == "L1Puppi") {
          if (fabs(mc_.eta) < 1.5) {
            h_L1Puppi_jet_ptratio_vs_pt_barrel_->Fill(mc_.pt, recopair.second.pt / mc_.pt);
          } else if (fabs(mc_.eta) < 2.5) {
            h_L1Puppi_jet_ptratio_vs_pt_endcap_->Fill(mc_.pt, recopair.second.pt / mc_.pt);
          } else if (fabs(mc_.eta) < 3.) {
            h_L1Puppi_jet_ptratio_vs_pt_ecnotk_->Fill(mc_.pt, recopair.second.pt / mc_.pt);
          } else if (fabs(mc_.eta) < 5.) {
            h_L1Puppi_jet_ptratio_vs_pt_hf_->Fill(mc_.pt, recopair.second.pt / mc_.pt);
          }
          h_L1Puppi_jet_ptratio_vs_eta_->Fill(mc_.eta, recopair.second.pt / mc_.pt);
        }
      }
    }
  }
  for (auto& recopair : reco_) {
    recopair.first.clear();
    recopair.second.clear();
  }

  for (auto& pfToken : phase2PFToken_) {
    edm::Handle<std::vector<l1t::PFCandidate>> l1pfs;
    e.getByToken(pfToken, l1pfs);

    for (const auto& pfc : *l1pfs) {
      h_L1PF_pt_->Fill(pfc.pt());
      h_L1PF_eta_->Fill(pfc.eta());
      if (abs(pfc.pdgId()) == 13) {
        h_L1PF_pt_mu_->Fill(pfc.pt());
        h_L1PF_eta_mu_->Fill(pfc.eta());
      } else if (abs(pfc.pdgId()) == 11) {
        h_L1PF_pt_el_->Fill(pfc.pt());
        h_L1PF_eta_el_->Fill(pfc.eta());
      } else if (abs(pfc.pdgId()) == 22) {
        h_L1PF_pt_pho_->Fill(pfc.pt());
        h_L1PF_eta_pho_->Fill(pfc.eta());
      } else if (abs(pfc.pdgId()) == 211) {
        h_L1PF_pt_ch_->Fill(pfc.pt());
        h_L1PF_eta_ch_->Fill(pfc.eta());
      } else if (abs(pfc.pdgId()) == 130) {
        h_L1PF_pt_nh_->Fill(pfc.pt());
        h_L1PF_eta_nh_->Fill(pfc.eta());
      }
    }  // loop over L1 PF
  }
  for (auto& pupToken : phase2PuppiToken_) {
    edm::Handle<std::vector<l1t::PFCandidate>> l1pups;
    e.getByToken(pupToken, l1pups);
    for (const auto& pupc : *l1pups) {
      h_L1Puppi_pt_->Fill(pupc.pt());
      h_L1Puppi_eta_->Fill(pupc.eta());
      if (abs(pupc.pdgId()) == 13) {
        h_L1Puppi_pt_mu_->Fill(pupc.pt());
        h_L1Puppi_eta_mu_->Fill(pupc.eta());
      } else if (abs(pupc.pdgId()) == 11) {
        h_L1Puppi_pt_el_->Fill(pupc.pt());
        h_L1Puppi_eta_el_->Fill(pupc.eta());
      } else if (abs(pupc.pdgId()) == 22) {
        h_L1Puppi_pt_pho_->Fill(pupc.pt());
        h_L1Puppi_eta_pho_->Fill(pupc.eta());
      } else if (abs(pupc.pdgId()) == 211) {
        h_L1Puppi_pt_ch_->Fill(pupc.pt());
        h_L1Puppi_eta_ch_->Fill(pupc.eta());
      } else if (abs(pupc.pdgId()) == 130) {
        h_L1Puppi_pt_nh_->Fill(pupc.pt());
        h_L1Puppi_eta_nh_->Fill(pupc.eta());
      }
    }  // loop over L1 Puppi
  }
}

//
// -------------------------------------- endRun --------------------------------------------
//
//
// -------------------------------------- book histograms --------------------------------------------
//
void L1TPhase2CorrelatorOffline::bookPhase2CorrelatorHistos(DQMStore::IBooker& ibooker) {
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_);
  ibooker.setScope(MonitorElementData::Scope::RUN);

  dqmoffline::l1t::HistDefinition resVsPtDef = histDefinitions_[PlotConfig::resVsPt];
  dqmoffline::l1t::HistDefinition resVsEtaDef = histDefinitions_[PlotConfig::resVsEta];
  dqmoffline::l1t::HistDefinition ptDistDef = histDefinitions_[PlotConfig::ptDist];
  dqmoffline::l1t::HistDefinition etaDistDef = histDefinitions_[PlotConfig::etaDist];

  int ptratio_nbins = 300;
  float ptratio_lo = 0.;
  float ptratio_hi = 3.;

  h_L1PF_pt_ = ibooker.book1D("PF_pt", "L1 PF p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1PF_eta_ = ibooker.book1D("PF_eta", "L1 PF #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);
  h_L1Puppi_pt_ = ibooker.book1D("Puppi_pt", "L1 PUPPI p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1Puppi_eta_ = ibooker.book1D("Puppi_eta", "L1 PUPPI #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);

  h_L1PF_pt_mu_ = ibooker.book1D("PF_pt_mu", "L1 PF Muon p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1PF_eta_mu_ = ibooker.book1D("PF_eta_mu", "L1 PF Muon #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);
  h_L1Puppi_pt_mu_ =
      ibooker.book1D("Puppi_pt_mu", "L1 PUPPI Muon p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1Puppi_eta_mu_ =
      ibooker.book1D("Puppi_eta_mu", "L1 PUPPI Muon #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);

  h_L1PF_pt_el_ = ibooker.book1D("PF_pt_el", "L1 PF Electron p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1PF_eta_el_ =
      ibooker.book1D("PF_eta_el", "L1 PF Electron #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);
  h_L1Puppi_pt_el_ =
      ibooker.book1D("Puppi_pt_el", "L1 PUPPI Electron p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1Puppi_eta_el_ =
      ibooker.book1D("Puppi_eta_el", "L1 PUPPI Electron #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);

  h_L1PF_pt_pho_ = ibooker.book1D("PF_pt_pho", "L1 PF Photon p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1PF_eta_pho_ =
      ibooker.book1D("PF_eta_pho", "L1 PF Photon #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);
  h_L1Puppi_pt_pho_ =
      ibooker.book1D("Puppi_pt_pho", "L1 PUPPI Photon p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1Puppi_eta_pho_ =
      ibooker.book1D("Puppi_eta_pho", "L1 PUPPI Photon #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);

  h_L1PF_pt_ch_ =
      ibooker.book1D("PF_pt_ch", "L1 PF Charged Hadron p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1PF_eta_ch_ =
      ibooker.book1D("PF_eta_ch", "L1 PF Charged Hadron #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);
  h_L1Puppi_pt_ch_ =
      ibooker.book1D("Puppi_pt_ch", "L1 PUPPI Charged Hadron p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1Puppi_eta_ch_ = ibooker.book1D(
      "Puppi_eta_ch", "L1 PUPPI Charged Hadron #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);

  h_L1PF_pt_nh_ =
      ibooker.book1D("PF_pt_nh", "L1 PF Neutral Hadron p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1PF_eta_nh_ =
      ibooker.book1D("PF_eta_nh", "L1 PF Neutral Hadron #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);
  h_L1Puppi_pt_nh_ =
      ibooker.book1D("Puppi_pt_nh", "L1 PUPPI Neutral Hadron p_{T}", ptDistDef.nbinsX, ptDistDef.xmin, ptDistDef.xmax);
  h_L1Puppi_eta_nh_ = ibooker.book1D(
      "Puppi_eta_nh", "L1 PUPPI Neutral Hadron #eta", etaDistDef.nbinsX, etaDistDef.xmin, etaDistDef.xmax);

  ibooker.setCurrentFolder(respresolFolder_);

  h_L1PF_part_ptratio_0p2_vs_pt_barrel_ = ibooker.book2D("L1PFParticlePtRatio0p2VsPtBarrel",
                                                         "L1 PF Particle L1/Gen (#Delta R < 0.2) vs p_{T}, Barrel",
                                                         resVsPtDef.nbinsX,
                                                         resVsPtDef.xmin,
                                                         resVsPtDef.xmax,
                                                         ptratio_nbins,
                                                         ptratio_lo,
                                                         ptratio_hi);

  h_L1PF_part_ptratio_0p2_vs_pt_endcap_ = ibooker.book2D("L1PFParticlePtRatio0p2VsPtEndcap",
                                                         "L1 PF Particle L1/Gen (#Delta R < 0.2) vs p_{T}, Endcap",
                                                         resVsPtDef.nbinsX,
                                                         resVsPtDef.xmin,
                                                         resVsPtDef.xmax,
                                                         ptratio_nbins,
                                                         ptratio_lo,
                                                         ptratio_hi);

  h_L1PF_part_ptratio_0p2_vs_pt_ecnotk_ =
      ibooker.book2D("L1PFParticlePtRatio0p2VsPtEndcapNoTk",
                     "L1 PF Particle L1/Gen (#Delta R < 0.2) vs p_{T}, Endcap No Tk",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax,
                     ptratio_nbins,
                     ptratio_lo,
                     ptratio_hi);

  h_L1PF_part_ptratio_0p2_vs_pt_hf_ = ibooker.book2D("L1PFParticlePtRatio0p2VsPtHF",
                                                     "L1 PF Particle L1/Gen (#Delta R < 0.2) vs p_{T}, HF",
                                                     resVsPtDef.nbinsX,
                                                     resVsPtDef.xmin,
                                                     resVsPtDef.xmax,
                                                     ptratio_nbins,
                                                     ptratio_lo,
                                                     ptratio_hi);

  h_L1PF_part_ptratio_0p2_vs_eta_ = ibooker.book2D("L1PFParticlePtRatio0p2VsEta",
                                                   "L1 PF Particle L1/Gen (#Delta R < 0.2) vs #eta",
                                                   resVsEtaDef.nbinsX,
                                                   resVsEtaDef.xmin,
                                                   resVsEtaDef.xmax,
                                                   ptratio_nbins,
                                                   ptratio_lo,
                                                   ptratio_hi);

  h_L1Puppi_part_ptratio_0p2_vs_pt_barrel_ =
      ibooker.book2D("L1PUPPIParticlePtRatio0p2VsPtBarrel",
                     "L1 PUPPI Particle L1/Gen (#Delta R < 0.2) vs p_{T}, Barrel",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax,
                     ptratio_nbins,
                     ptratio_lo,
                     ptratio_hi);

  h_L1Puppi_part_ptratio_0p2_vs_pt_endcap_ =
      ibooker.book2D("L1PUPPIParticlePtRatio0p2VsPtEndcap",
                     "L1 PUPPI Particle L1/Gen (#Delta R < 0.2) vs p_{T}, Endcap",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax,
                     ptratio_nbins,
                     ptratio_lo,
                     ptratio_hi);

  h_L1Puppi_part_ptratio_0p2_vs_pt_ecnotk_ =
      ibooker.book2D("L1PUPPIParticlePtRatio0p2VsPtEndcapNoTk",
                     "L1 PUPPI Particle L1/Gen (#Delta R < 0.2) vs p_{T}, Endcap No Tk",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax,
                     ptratio_nbins,
                     ptratio_lo,
                     ptratio_hi);

  h_L1Puppi_part_ptratio_0p2_vs_pt_hf_ = ibooker.book2D("L1PUPPIParticlePtRatio0p2VsPtHF",
                                                        "L1 PUPPI Particle L1/Gen (#Delta R < 0.2) vs p_{T}, HF",
                                                        resVsPtDef.nbinsX,
                                                        resVsPtDef.xmin,
                                                        resVsPtDef.xmax,
                                                        ptratio_nbins,
                                                        ptratio_lo,
                                                        ptratio_hi);

  h_L1Puppi_part_ptratio_0p2_vs_eta_ = ibooker.book2D("L1PUPPIParticlePtRatio0p2VsEta",
                                                      "L1 PUPPI Particle L1/Gen (#Delta R < 0.2) vs #eta",
                                                      resVsEtaDef.nbinsX,
                                                      resVsEtaDef.xmin,
                                                      resVsEtaDef.xmax,
                                                      ptratio_nbins,
                                                      ptratio_lo,
                                                      ptratio_hi);

  h_L1PF_jet_ptratio_vs_pt_barrel_ = ibooker.book2D("L1PFJetPtRatioVsPtBarrel",
                                                    "L1 PF Jet L1/Gen vs p_{T}, Barrel",
                                                    resVsPtDef.nbinsX,
                                                    resVsPtDef.xmin,
                                                    resVsPtDef.xmax,
                                                    ptratio_nbins,
                                                    ptratio_lo,
                                                    ptratio_hi);

  h_L1PF_jet_ptratio_vs_pt_endcap_ = ibooker.book2D("L1PFJetPtRatioVsPtEndcap",
                                                    "L1 PF Jet L1/Gen vs p_{T}, Endcap",
                                                    resVsPtDef.nbinsX,
                                                    resVsPtDef.xmin,
                                                    resVsPtDef.xmax,
                                                    ptratio_nbins,
                                                    ptratio_lo,
                                                    ptratio_hi);

  h_L1PF_jet_ptratio_vs_pt_ecnotk_ = ibooker.book2D("L1PFJetPtRatioVsPtEndcapNoTk",
                                                    "L1 PF Jet L1/Gen vs p_{T}, Endcap No Tk",
                                                    resVsPtDef.nbinsX,
                                                    resVsPtDef.xmin,
                                                    resVsPtDef.xmax,
                                                    ptratio_nbins,
                                                    ptratio_lo,
                                                    ptratio_hi);

  h_L1PF_jet_ptratio_vs_pt_hf_ = ibooker.book2D("L1PFJetPtRatioVsPtHF",
                                                "L1 PF Jet L1/Gen vs p_{T}, HF",
                                                resVsPtDef.nbinsX,
                                                resVsPtDef.xmin,
                                                resVsPtDef.xmax,
                                                ptratio_nbins,
                                                ptratio_lo,
                                                ptratio_hi);

  h_L1PF_jet_ptratio_vs_eta_ = ibooker.book2D("L1PFJetPtRatioVsEta",
                                              "L1 PF Jet L1/Gen vs #eta",
                                              resVsEtaDef.nbinsX,
                                              resVsEtaDef.xmin,
                                              resVsEtaDef.xmax,
                                              ptratio_nbins,
                                              ptratio_lo,
                                              ptratio_hi);

  h_L1Puppi_jet_ptratio_vs_pt_barrel_ = ibooker.book2D("L1PUPPIJetPtRatioVsPtBarrel",
                                                       "L1 PUPPI Jet L1/Gen vs p_{T}, Barrel",
                                                       resVsPtDef.nbinsX,
                                                       resVsPtDef.xmin,
                                                       resVsPtDef.xmax,
                                                       ptratio_nbins,
                                                       ptratio_lo,
                                                       ptratio_hi);

  h_L1Puppi_jet_ptratio_vs_pt_endcap_ = ibooker.book2D("L1PUPPIJetPtRatioVsPtEndcap",
                                                       "L1 PUPPI Jet L1/Gen vs p_{T}, Endcap",
                                                       resVsPtDef.nbinsX,
                                                       resVsPtDef.xmin,
                                                       resVsPtDef.xmax,
                                                       ptratio_nbins,
                                                       ptratio_lo,
                                                       ptratio_hi);

  h_L1Puppi_jet_ptratio_vs_pt_ecnotk_ = ibooker.book2D("L1PUPPIJetPtRatioVsPtEndcapNoTk",
                                                       "L1 PUPPI Jet L1/Gen vs p_{T}, EndcapNoTk",
                                                       resVsPtDef.nbinsX,
                                                       resVsPtDef.xmin,
                                                       resVsPtDef.xmax,
                                                       ptratio_nbins,
                                                       ptratio_lo,
                                                       ptratio_hi);

  h_L1Puppi_jet_ptratio_vs_pt_hf_ = ibooker.book2D("L1PUPPIJetPtRatioVsPtHF",
                                                   "L1 PUPPI Jet L1/Gen vs p_{T}, HF",
                                                   resVsPtDef.nbinsX,
                                                   resVsPtDef.xmin,
                                                   resVsPtDef.xmax,
                                                   ptratio_nbins,
                                                   ptratio_lo,
                                                   ptratio_hi);

  h_L1Puppi_jet_ptratio_vs_eta_ = ibooker.book2D("L1PUPPIJetPtRatioVsEta",
                                                 "L1 PUPPI Jet L1/Gen vs #eta",
                                                 resVsEtaDef.nbinsX,
                                                 resVsEtaDef.xmin,
                                                 resVsEtaDef.xmax,
                                                 ptratio_nbins,
                                                 ptratio_lo,
                                                 ptratio_hi);

  ibooker.setCurrentFolder(histFolder_);
  //Response

  h_L1PF_part_response_0p2_pt_barrel_ = ibooker.book1D("L1PFParticleResponse0p2VsPtBarrel",
                                                       "L1 PF Particle Response (#Delta R < 0.2) vs p_{T}, Barrel",
                                                       resVsPtDef.nbinsX,
                                                       resVsPtDef.xmin,
                                                       resVsPtDef.xmax);

  h_L1PF_part_response_0p2_pt_endcap_ = ibooker.book1D("L1PFParticleResponse0p2VsPtEndcap",
                                                       "L1 PF Particle Response (#Delta R < 0.2) vs p_{T}, Endcap",
                                                       resVsPtDef.nbinsX,
                                                       resVsPtDef.xmin,
                                                       resVsPtDef.xmax);

  h_L1PF_part_response_0p2_pt_ecnotk_ =
      ibooker.book1D("L1PFParticleResponse0p2VsPtEndcapNoTk",
                     "L1 PF Particle Response (#Delta R < 0.2) vs p_{T}, Endcap No Tk",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax);

  h_L1PF_part_response_0p2_pt_hf_ = ibooker.book1D("L1PFParticleResponse0p2VsPtHF",
                                                   "L1 PF Particle Response (#Delta R < 0.2) vs p_{T}, HF",
                                                   resVsPtDef.nbinsX,
                                                   resVsPtDef.xmin,
                                                   resVsPtDef.xmax);

  h_L1PF_part_response_0p2_eta_ = ibooker.book1D("L1PFParticleResponse0p2VsEta",
                                                 "L1 PF Particle Response (#Delta R < 0.2) vs #eta",
                                                 resVsEtaDef.nbinsX,
                                                 resVsEtaDef.xmin,
                                                 resVsEtaDef.xmax);

  h_L1Puppi_part_response_0p2_pt_barrel_ =
      ibooker.book1D("L1PUPPIParticleResponse0p2VsPtBarrel",
                     "L1 PUPPI Particle Response (#Delta R < 0.2) vs p_{T}, Barrel",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax);

  h_L1Puppi_part_response_0p2_pt_endcap_ =
      ibooker.book1D("L1PUPPIParticleResponse0p2VsPtEndcap",
                     "L1 PUPPI Particle Response (#Delta R < 0.2) vs p_{T}, Endcap",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax);

  h_L1Puppi_part_response_0p2_pt_ecnotk_ =
      ibooker.book1D("L1PUPPIParticleResponse0p2VsPtEndcapNoTk",
                     "L1 PUPPI Particle Response (#Delta R < 0.2) vs p_{T}, Endcap No Tk",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax);

  h_L1Puppi_part_response_0p2_pt_hf_ = ibooker.book1D("L1PUPPIParticleResponse0p2VsPtHF",
                                                      "L1 PUPPI Particle Response (#Delta R < 0.2) vs p_{T}, HF",
                                                      resVsPtDef.nbinsX,
                                                      resVsPtDef.xmin,
                                                      resVsPtDef.xmax);

  h_L1Puppi_part_response_0p2_eta_ = ibooker.book1D("L1PUPPIParticleResponse0p2VsEta",
                                                    "L1 PUPPI Particle Response (#Delta R < 0.2) vs #eta",
                                                    resVsEtaDef.nbinsX,
                                                    resVsEtaDef.xmin,
                                                    resVsEtaDef.xmax);

  h_L1PF_jet_response_pt_barrel_ = ibooker.book1D("L1PFJetResponseVsPtBarrel",
                                                  "L1 PF Jet Response vs p_{T}, Barrel",
                                                  resVsPtDef.nbinsX,
                                                  resVsPtDef.xmin,
                                                  resVsPtDef.xmax);

  h_L1PF_jet_response_pt_endcap_ = ibooker.book1D("L1PFJetResponseVsPtEndcap",
                                                  "L1 PF Jet Response vs p_{T}, Endcap",
                                                  resVsPtDef.nbinsX,
                                                  resVsPtDef.xmin,
                                                  resVsPtDef.xmax);

  h_L1PF_jet_response_pt_ecnotk_ = ibooker.book1D("L1PFJetResponseVsPtEndcapNoTk",
                                                  "L1 PF Jet Response vs p_{T}, Endcap No Tk",
                                                  resVsPtDef.nbinsX,
                                                  resVsPtDef.xmin,
                                                  resVsPtDef.xmax);

  h_L1PF_jet_response_pt_hf_ = ibooker.book1D(
      "L1PFJetResponseVsPtHF", "L1 PF Jet Response vs p_{T}, HF", resVsPtDef.nbinsX, resVsPtDef.xmin, resVsPtDef.xmax);

  h_L1PF_jet_response_eta_ = ibooker.book1D(
      "L1PFJetResponseVsEta", "L1 PF Jet Response vs #eta", resVsEtaDef.nbinsX, resVsEtaDef.xmin, resVsEtaDef.xmax);

  h_L1Puppi_jet_response_pt_barrel_ = ibooker.book1D("L1PUPPIJetResponseVsPtBarrel",
                                                     "L1 PUPPI Jet Response vs p_{T}, Barrel",
                                                     resVsPtDef.nbinsX,
                                                     resVsPtDef.xmin,
                                                     resVsPtDef.xmax);

  h_L1Puppi_jet_response_pt_endcap_ = ibooker.book1D("L1PUPPIJetResponseVsPtEndcap",
                                                     "L1 PUPPI Jet Response vs p_{T}, Endcap",
                                                     resVsPtDef.nbinsX,
                                                     resVsPtDef.xmin,
                                                     resVsPtDef.xmax);

  h_L1Puppi_jet_response_pt_ecnotk_ = ibooker.book1D("L1PUPPIJetResponseVsPtEndcapNoTk",
                                                     "L1 PUPPI Jet Response vs p_{T}, EndcapNoTk",
                                                     resVsPtDef.nbinsX,
                                                     resVsPtDef.xmin,
                                                     resVsPtDef.xmax);

  h_L1Puppi_jet_response_pt_hf_ = ibooker.book1D("L1PUPPIJetResponseVsPtHF",
                                                 "L1 PUPPI Jet Response vs p_{T}, HF",
                                                 resVsPtDef.nbinsX,
                                                 resVsPtDef.xmin,
                                                 resVsPtDef.xmax);

  h_L1Puppi_jet_response_eta_ = ibooker.book1D("L1PUPPIJetResponseVsEta",
                                               "L1 PUPPI Jet Response vs #eta",
                                               resVsEtaDef.nbinsX,
                                               resVsEtaDef.xmin,
                                               resVsEtaDef.xmax);

  //Resolution

  h_L1PF_part_resolution_0p2_pt_barrel_ = ibooker.book1D("L1PFParticleResolution0p2VsPtBarrel",
                                                         "L1 PF Particle Resolution (#Delta R < 0.2) vs p_{T}, Barrel",
                                                         resVsPtDef.nbinsX,
                                                         resVsPtDef.xmin,
                                                         resVsPtDef.xmax);

  h_L1PF_part_resolution_0p2_pt_endcap_ = ibooker.book1D("L1PFParticleResolution0p2VsPtEndcap",
                                                         "L1 PF Particle Resolution (#Delta R < 0.2) vs p_{T}, Endcap",
                                                         resVsPtDef.nbinsX,
                                                         resVsPtDef.xmin,
                                                         resVsPtDef.xmax);

  h_L1PF_part_resolution_0p2_pt_ecnotk_ =
      ibooker.book1D("L1PFParticleResolution0p2VsPtEndcapNoTk",
                     "L1 PF Particle Resolution (#Delta R < 0.2) vs p_{T}, Endcap No Tk",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax);

  h_L1PF_part_resolution_0p2_pt_hf_ = ibooker.book1D("L1PFParticleResolution0p2VsPtHF",
                                                     "L1 PF Particle Resolution (#Delta R < 0.2) vs p_{T}, HF",
                                                     resVsPtDef.nbinsX,
                                                     resVsPtDef.xmin,
                                                     resVsPtDef.xmax);

  h_L1Puppi_part_resolution_0p2_pt_barrel_ =
      ibooker.book1D("L1PUPPIParticleResolution0p2VsPtBarrel",
                     "L1 PUPPI Particle Resolution (#Delta R < 0.2) vs p_{T}, Barrel",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax);

  h_L1Puppi_part_resolution_0p2_pt_endcap_ =
      ibooker.book1D("L1PUPPIParticleResolution0p2VsPtEndcap",
                     "L1 PUPPI Particle Resolution (#Delta R < 0.2) vs p_{T}, Endcap",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax);

  h_L1Puppi_part_resolution_0p2_pt_ecnotk_ =
      ibooker.book1D("L1PUPPIParticleResolution0p2VsPtEndcapNoTk",
                     "L1 PUPPI Particle Resolution (#Delta R < 0.2) vs p_{T}, Endcap No Tk",
                     resVsPtDef.nbinsX,
                     resVsPtDef.xmin,
                     resVsPtDef.xmax);

  h_L1Puppi_part_resolution_0p2_pt_hf_ = ibooker.book1D("L1PUPPIParticleResolution0p2VsPtHF",
                                                        "L1 PUPPI Particle Resolution (#Delta R < 0.2) vs p_{T}, HF",
                                                        resVsPtDef.nbinsX,
                                                        resVsPtDef.xmin,
                                                        resVsPtDef.xmax);

  h_L1PF_jet_resolution_pt_barrel_ = ibooker.book1D("L1PFJetResolutionVsPtBarrel",
                                                    "L1 PF Jet Resolution vs p_{T}, Barrel",
                                                    resVsPtDef.nbinsX,
                                                    resVsPtDef.xmin,
                                                    resVsPtDef.xmax);

  h_L1PF_jet_resolution_pt_endcap_ = ibooker.book1D("L1PFJetResolutionVsPtEndcap",
                                                    "L1 PF Jet Resolution vs p_{T}, Endcap",
                                                    resVsPtDef.nbinsX,
                                                    resVsPtDef.xmin,
                                                    resVsPtDef.xmax);

  h_L1PF_jet_resolution_pt_ecnotk_ = ibooker.book1D("L1PFJetResolutionVsPtEndcapNoTk",
                                                    "L1 PF Jet Resolution vs p_{T}, Endcap No Tk",
                                                    resVsPtDef.nbinsX,
                                                    resVsPtDef.xmin,
                                                    resVsPtDef.xmax);

  h_L1PF_jet_resolution_pt_hf_ = ibooker.book1D("L1PFJetResolutionVsPtHF",
                                                "L1 PF Jet Resolution vs p_{T}, HF",
                                                resVsPtDef.nbinsX,
                                                resVsPtDef.xmin,
                                                resVsPtDef.xmax);

  h_L1Puppi_jet_resolution_pt_barrel_ = ibooker.book1D("L1PUPPIJetResolutionVsPtBarrel",
                                                       "L1 PUPPI Jet Resolution vs p_{T}, Barrel",
                                                       resVsPtDef.nbinsX,
                                                       resVsPtDef.xmin,
                                                       resVsPtDef.xmax);

  h_L1Puppi_jet_resolution_pt_endcap_ = ibooker.book1D("L1PUPPIJetResolutionVsPtEndcap",
                                                       "L1 PUPPI Jet Resolution vs p_{T}, Endcap",
                                                       resVsPtDef.nbinsX,
                                                       resVsPtDef.xmin,
                                                       resVsPtDef.xmax);

  h_L1Puppi_jet_resolution_pt_ecnotk_ = ibooker.book1D("L1PUPPIJetResolutionVsPtEndcapNoTk",
                                                       "L1 PUPPI Jet Resolution vs p_{T}, EndcapNoTk",
                                                       resVsPtDef.nbinsX,
                                                       resVsPtDef.xmin,
                                                       resVsPtDef.xmax);

  h_L1Puppi_jet_resolution_pt_hf_ = ibooker.book1D("L1PUPPIJetResolutionVsPtHF",
                                                   "L1 PUPPI Jet Resolution vs p_{T}, HF",
                                                   resVsPtDef.nbinsX,
                                                   resVsPtDef.xmin,
                                                   resVsPtDef.xmax);

  ibooker.cd();

  return;
}

//
// -------------------------------------- endRun --------------------------------------------
//
void L1TPhase2CorrelatorOffline::dqmEndRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  computeResponseResolution();
}

void L1TPhase2CorrelatorOffline::computeResponseResolution() {
  std::vector<MonitorElement*> monElementstoComputeIn = {h_L1PF_part_ptratio_0p2_vs_pt_barrel_,
                                                         h_L1PF_part_ptratio_0p2_vs_pt_endcap_,
                                                         h_L1PF_part_ptratio_0p2_vs_pt_ecnotk_,
                                                         h_L1PF_part_ptratio_0p2_vs_pt_hf_,
                                                         h_L1PF_part_ptratio_0p2_vs_eta_,
                                                         h_L1Puppi_part_ptratio_0p2_vs_pt_barrel_,
                                                         h_L1Puppi_part_ptratio_0p2_vs_pt_endcap_,
                                                         h_L1Puppi_part_ptratio_0p2_vs_pt_ecnotk_,
                                                         h_L1Puppi_part_ptratio_0p2_vs_pt_hf_,
                                                         h_L1Puppi_part_ptratio_0p2_vs_eta_,
                                                         h_L1PF_jet_ptratio_vs_pt_barrel_,
                                                         h_L1PF_jet_ptratio_vs_pt_endcap_,
                                                         h_L1PF_jet_ptratio_vs_pt_ecnotk_,
                                                         h_L1PF_jet_ptratio_vs_pt_hf_,
                                                         h_L1PF_jet_ptratio_vs_eta_,
                                                         h_L1Puppi_jet_ptratio_vs_pt_barrel_,
                                                         h_L1Puppi_jet_ptratio_vs_pt_endcap_,
                                                         h_L1Puppi_jet_ptratio_vs_pt_ecnotk_,
                                                         h_L1Puppi_jet_ptratio_vs_pt_hf_,
                                                         h_L1Puppi_jet_ptratio_vs_eta_};
  std::vector<MonitorElement*> monElementstoComputeResp = {h_L1PF_part_response_0p2_pt_barrel_,
                                                           h_L1PF_part_response_0p2_pt_endcap_,
                                                           h_L1PF_part_response_0p2_pt_ecnotk_,
                                                           h_L1PF_part_response_0p2_pt_hf_,
                                                           h_L1PF_part_response_0p2_eta_,
                                                           h_L1Puppi_part_response_0p2_pt_barrel_,
                                                           h_L1Puppi_part_response_0p2_pt_endcap_,
                                                           h_L1Puppi_part_response_0p2_pt_ecnotk_,
                                                           h_L1Puppi_part_response_0p2_pt_hf_,
                                                           h_L1Puppi_part_response_0p2_eta_,
                                                           h_L1PF_jet_response_pt_barrel_,
                                                           h_L1PF_jet_response_pt_endcap_,
                                                           h_L1PF_jet_response_pt_ecnotk_,
                                                           h_L1PF_jet_response_pt_hf_,
                                                           h_L1PF_jet_response_eta_,
                                                           h_L1Puppi_jet_response_pt_barrel_,
                                                           h_L1Puppi_jet_response_pt_endcap_,
                                                           h_L1Puppi_jet_response_pt_ecnotk_,
                                                           h_L1Puppi_jet_response_pt_hf_,
                                                           h_L1Puppi_jet_response_eta_};
  std::vector<MonitorElement*> monElementstoComputeResol = {h_L1PF_part_resolution_0p2_pt_barrel_,
                                                            h_L1PF_part_resolution_0p2_pt_endcap_,
                                                            h_L1PF_part_resolution_0p2_pt_ecnotk_,
                                                            h_L1PF_part_resolution_0p2_pt_hf_,
                                                            nullptr,
                                                            h_L1Puppi_part_resolution_0p2_pt_barrel_,
                                                            h_L1Puppi_part_resolution_0p2_pt_endcap_,
                                                            h_L1Puppi_part_resolution_0p2_pt_ecnotk_,
                                                            h_L1Puppi_part_resolution_0p2_pt_hf_,
                                                            nullptr,
                                                            h_L1PF_jet_resolution_pt_barrel_,
                                                            h_L1PF_jet_resolution_pt_endcap_,
                                                            h_L1PF_jet_resolution_pt_ecnotk_,
                                                            h_L1PF_jet_resolution_pt_hf_,
                                                            nullptr,
                                                            h_L1Puppi_jet_resolution_pt_barrel_,
                                                            h_L1Puppi_jet_resolution_pt_endcap_,
                                                            h_L1Puppi_jet_resolution_pt_ecnotk_,
                                                            h_L1Puppi_jet_resolution_pt_hf_,
                                                            nullptr};

  for (unsigned int i = 0; i < monElementstoComputeIn.size(); i++) {
    if (monElementstoComputeIn[i] != nullptr && monElementstoComputeResp[i] != nullptr) {
      medianResponseCorrResolution(
          monElementstoComputeIn[i], monElementstoComputeResp[i], monElementstoComputeResol[i]);
    }
  }
}

std::vector<float> L1TPhase2CorrelatorOffline::getQuantile(float quant, TH2F* hist) {
  std::vector<float> quantiles(hist->GetNbinsX(), 1.);
  for (int ix = 1; ix < hist->GetNbinsX() + 1; ix++) {
    float thresh = quant * (hist->Integral(ix, ix, 0, -1));
    if (hist->Integral(ix, ix, 0, -1) == 0.) {
    } else if (quant <= 0. || thresh < hist->GetBinContent(ix, 0)) {
      quantiles[ix - 1] = hist->GetYaxis()->GetBinLowEdge(1);
    } else if (quant >= 1. || thresh >= hist->Integral(ix, ix, 0, hist->GetNbinsY())) {
      quantiles[ix - 1] = hist->GetYaxis()->GetBinUpEdge(hist->GetNbinsY());
    } else {
      float sum = hist->GetBinContent(ix, 0);
      for (int iy = 1; iy < hist->GetNbinsY() + 1; iy++) {
        float add = hist->GetBinContent(ix, iy);
        if (sum + add >= thresh) {
          quantiles[ix - 1] =
              hist->GetYaxis()->GetBinLowEdge(iy) + hist->GetYaxis()->GetBinWidth(iy) * ((thresh - sum) / add);
          break;
        }
        sum += add;
      }
    }
  }
  return quantiles;
}

void L1TPhase2CorrelatorOffline::medianResponseCorrResolution(MonitorElement* in2D,
                                                              MonitorElement* response,
                                                              MonitorElement* resolution) {
  auto hbase = in2D->getTH2F();
  auto hresp = response->getTH1F();
  if (hbase != nullptr && hresp != nullptr) {
    if (hbase->GetNbinsX() == hresp->GetNbinsX()) {
      auto med = getQuantile(0.5, hbase);
      TGraph* ptrecgen = new TGraph(hbase->GetNbinsX());
      for (int ib = 1; ib < hbase->GetNbinsX() + 1; ib++) {
        float corr = med[ib - 1];
        float xval = hbase->GetXaxis()->GetBinCenter(ib);
        ptrecgen->SetPoint(ib - 1, xval * corr, xval);
        hresp->SetBinContent(ib, corr);
      }
      if (resolution != nullptr) {
        auto hresol = resolution->getTH1F();
        if (hresol != nullptr) {
          if (hbase->GetNbinsX() == hresol->GetNbinsX()) {
            ptrecgen->Sort();
            TH2F* ch = new TH2F(*hbase);
            ch->Reset("ICE");
            for (int ibx = 1; ibx < ch->GetNbinsX() + 1; ibx++) {
              float xval = hbase->GetXaxis()->GetBinCenter(ibx);
              for (int iby = 1; iby < ch->GetNbinsY() + 1; iby++) {
                float yval = hbase->GetYaxis()->GetBinCenter(iby);
                float newyval = ptrecgen->Eval(yval * xval) / xval;
                int ycb = ch->FindBin(xval, newyval);
                ch->SetBinContent(ycb, ch->GetBinContent(ycb) + hbase->GetBinContent(ibx, iby));
              }
            }
            delete ptrecgen;
            auto qc = getQuantile(0.5, ch);
            auto qhi = getQuantile(0.84, ch);
            auto qlo = getQuantile(0.16, ch);
            delete ch;
            for (int ibx = 1; ibx < hbase->GetNbinsX() + 1; ibx++) {
              hresol->SetBinContent(ibx, qc[ibx - 1] > 0.2 ? (qhi[ibx - 1] - qlo[ibx - 1]) / 2. : 0.);
            }
          }
        }
      } else {
        delete ptrecgen;
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(L1TPhase2CorrelatorOffline);
