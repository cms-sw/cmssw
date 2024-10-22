/**
 * \file L1TPhase2MuonOffline.cc
 *
 * \author S. Folgueras 
 *
 */

#include "DQMOffline/L1Trigger/interface/L1TPhase2MuonOffline.h"

// To convert from HW to Physical coordinates
#include "DataFormats/L1TMuonPhase2/interface/Constants.h"

using namespace reco;
using namespace edm;
using namespace std;
using namespace l1t;

//__________RECO-GMT Muon Pair Helper Class____________________________
GenMuonGMTPair::GenMuonGMTPair(const reco::GenParticle* muon, const l1t::L1Candidate* gmtmu)
    : mu_(muon), gmtmu_(gmtmu) {
  if (gmtmu) {
    gmtEta_ = gmtmu_->eta();
    gmtPhi_ = gmtmu_->phi();
  } else {
    gmtEta_ = -5.;
    gmtPhi_ = -5.;
  }
  if (mu_) {
    muEta_ = mu_->eta();
    muPhi_ = mu_->phi();
  } else {
    muEta_ = 999.;
    muPhi_ = 999.;
  }
};

GenMuonGMTPair::GenMuonGMTPair(const GenMuonGMTPair& muonGmtPair) {
  mu_ = muonGmtPair.mu_;
  gmtmu_ = muonGmtPair.gmtmu_;

  gmtEta_ = muonGmtPair.gmtEta_;
  gmtPhi_ = muonGmtPair.gmtPhi_;

  muEta_ = muonGmtPair.muEta_;
  muPhi_ = muonGmtPair.muPhi_;
}

float GenMuonGMTPair::dR2() {
  if (!gmtmu_)
    return 999.;
  float dEta = gmtEta_ - muEta_;
  float dPhi = reco::deltaPhi(gmtPhi_, muPhi_);
  return dEta * dEta + dPhi * dPhi;
}

L1TPhase2MuonOffline::EtaRegion GenMuonGMTPair::etaRegion() const {
  if (std::abs(muEta_) < 0.83)
    return L1TPhase2MuonOffline::kEtaRegionBmtf;
  if (std::abs(muEta_) < 1.24)
    return L1TPhase2MuonOffline::kEtaRegionOmtf;
  if (std::abs(muEta_) < 2.4)
    return L1TPhase2MuonOffline::kEtaRegionEmtf;
  return L1TPhase2MuonOffline::kEtaRegionAll;
}

double GenMuonGMTPair::getDeltaVar(const L1TPhase2MuonOffline::ResType type) const {
  if (type == L1TPhase2MuonOffline::kResPt)
    return (gmtPt() - pt()) / pt();
  if (type == L1TPhase2MuonOffline::kRes1OverPt)
    return (pt() - gmtPt()) / gmtPt();  // (1/gmtPt - 1/pt) / (1/pt)
  if (type == L1TPhase2MuonOffline::kResQOverPt)
    return (gmtCharge() * charge() * pt() - gmtPt()) /
           gmtPt();  // (gmtCharge/gmtPt - charge/pt) / (charge/pt) with gmtCharge/charge = gmtCharge*charge
  if (type == L1TPhase2MuonOffline::kResPhi)
    return reco::deltaPhi(gmtPhi(), muPhi_);
  if (type == L1TPhase2MuonOffline::kResEta)
    return gmtEta() - muEta_;
  if (type == L1TPhase2MuonOffline::kResCh)
    return gmtCharge() - charge();
  return -999.;
}

double GenMuonGMTPair::getVar(const L1TPhase2MuonOffline::EffType type) const {
  if (type == L1TPhase2MuonOffline::kEffPt)
    return pt();
  if (type == L1TPhase2MuonOffline::kEffPhi)
    return muPhi_;
  if (type == L1TPhase2MuonOffline::kEffEta)
    return muEta_;
  return -999.;
}

//__________DQM_base_class_______________________________________________
L1TPhase2MuonOffline::L1TPhase2MuonOffline(const ParameterSet& ps)
    : gmtMuonToken_(consumes<l1t::SAMuonCollection>(ps.getParameter<edm::InputTag>("gmtMuonToken"))),
      gmtTkMuonToken_(consumes<l1t::TrackerMuonCollection>(ps.getParameter<edm::InputTag>("gmtTkMuonToken"))),
      genParticleToken_(
          consumes<std::vector<reco::GenParticle>>(ps.getUntrackedParameter<edm::InputTag>("genParticlesInputTag"))),
      muonTypes_({kSAMuon, kTkMuon}),
      effTypes_({kEffPt, kEffPhi, kEffEta}),
      resTypes_({kResPt, kResQOverPt, kResPhi, kResEta}),
      etaRegions_({kEtaRegionAll, kEtaRegionBmtf, kEtaRegionOmtf, kEtaRegionEmtf}),
      qualLevels_({kQualOpen, kQualDouble, kQualSingle}),
      resNames_({{kResPt, "pt"},
                 {kRes1OverPt, "1overpt"},
                 {kResQOverPt, "qoverpt"},
                 {kResPhi, "phi"},
                 {kResEta, "eta"},
                 {kResCh, "charge"}}),
      resLabels_({{kResPt, "(p_{T}^{L1} - p_{T}^{reco})/p_{T}^{reco}"},
                  {kRes1OverPt, "(p_{T}^{reco} - p_{T}^{L1})/p_{T}^{L1}"},
                  {kResQOverPt, "(q^{L1}*q^{reco}*p_{T}^{reco} - p_{T}^{L1})/p_{T}^{L1}"},
                  {kResPhi, "#phi_{L1} - #phi_{reco}"},
                  {kResEta, "#eta_{L1} - #eta_{reco}"},
                  {kResCh, "charge^{L1} - charge^{reco}"}}),
      etaNames_({{kEtaRegionAll, "etaMin0_etaMax2p4"},
                 {kEtaRegionBmtf, "etaMin0_etaMax0p83"},
                 {kEtaRegionOmtf, "etaMin0p83_etaMax1p24"},
                 {kEtaRegionEmtf, "etaMin1p24_etaMax2p4"}}),
      qualNames_({{kQualOpen, "qualOpen"}, {kQualDouble, "qualDouble"}, {kQualSingle, "qualSingle"}}),
      muonNames_({{kSAMuon, "SAMuon"}, {kTkMuon, "TkMuon"}}),
      histFolder_(ps.getUntrackedParameter<string>("histFolder")),
      cutsVPSet_(ps.getUntrackedParameter<std::vector<edm::ParameterSet>>("cuts")),
      effVsPtBins_(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsPtBins")),
      effVsPhiBins_(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsPhiBins")),
      effVsEtaBins_(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsEtaBins")),
      maxGmtMuonDR_(ps.getUntrackedParameter<double>("maxDR")) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::L1TPhase2MuonOffline()" << endl;

  for (const auto& c : cutsVPSet_) {
    const auto qCut = c.getUntrackedParameter<int>("qualCut");
    QualLevel qLevel = kQualOpen;
    if (qCut > 11) {
      qLevel = kQualSingle;
    } else if (qCut > 7) {
      qLevel = kQualDouble;
    } else if (qCut > 3) {
      qLevel = kQualOpen;
    }
    cuts_.emplace_back(std::make_pair(c.getUntrackedParameter<int>("ptCut"), qLevel));
  }
}

//----------------------------------------------------------------------
void L1TPhase2MuonOffline::dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  edm::LogInfo("L1TPhase2MuonOFfline") << "L1TPhase2MuonOffline::dqmBeginRun" << endl;
}

//_____________________________________________________________________
void L1TPhase2MuonOffline::bookHistograms(DQMStore::IBooker& ibooker,
                                          const edm::Run& run,
                                          const edm::EventSetup& iSetup) {
  edm::LogInfo("L1TPhase2MuonOFfline") << "L1TPhase2MuonOffline::bookHistograms" << endl;

  //book histos
  for (const auto mutype : muonTypes_) {
    bookControlHistos(ibooker, mutype);
    bookEfficiencyHistos(ibooker, mutype);
    bookResolutionHistos(ibooker, mutype);
  }
}

//_____________________________________________________________________
void L1TPhase2MuonOffline::analyze(const Event& iEvent, const EventSetup& eventSetup) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::analyze() " << endl;

  // COLLECT GEN MUONS
  iEvent.getByToken(genParticleToken_, genparticles_);

  std::vector<const reco::GenParticle*> genmus;
  for (const reco::GenParticle& gen : *genparticles_) {
    if (std::abs(gen.pdgId()) != 13)
      continue;
    genmus.push_back(&gen);
  }
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::analyze() N of genmus: " << genmus.size() << endl;

  // Collect both muon collection:
  iEvent.getByToken(gmtMuonToken_, gmtSAMuon_);
  iEvent.getByToken(gmtTkMuonToken_, gmtTkMuon_);

  // Fill Control histograms
  edm::LogInfo("L1TPhase2MuonOffline") << "Fill Control histograms for GMT Muons" << endl;
  fillControlHistos();

  // Match each muon to a gen muon, if possible.
  if (genmus.empty())
    return;
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::analyze() calling matchMuonsToGen() " << endl;
  matchMuonsToGen(genmus);

  // Fill efficiency and resolution once, matching has been done...
  fillEfficiencyHistos();
  fillResolutionHistos();
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::analyze() Computation finished" << endl;
}

//_____________________________________________________________________
void L1TPhase2MuonOffline::bookControlHistos(DQMStore::IBooker& ibooker, MuType mutype) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::bookControlHistos()" << endl;

  ibooker.setCurrentFolder(histFolder_ + "/" + muonNames_[mutype] + "/control_variables");

  controlHistos_[mutype][kPt] = ibooker.book1D(muonNames_[mutype] + "Pt", "MuonPt; p_{T}", 50, 0., 100.);
  controlHistos_[mutype][kPhi] = ibooker.book1D(muonNames_[mutype] + "Phi", "MuonPhi; #phi", 66, -3.3, 3.3);
  controlHistos_[mutype][kEta] = ibooker.book1D(muonNames_[mutype] + "Eta", "MuonEta; #eta", 50, -2.5, 2.5);
  controlHistos_[mutype][kIso] = ibooker.book1D(muonNames_[mutype] + "Iso", "MuonIso; RelIso", 50, 0, 1.0);
  controlHistos_[mutype][kQual] = ibooker.book1D(muonNames_[mutype] + "Qual", "MuonQual; Quality", 15, 0.5, 15.5);
  controlHistos_[mutype][kZ0] = ibooker.book1D(muonNames_[mutype] + "Z0", "MuonZ0; Z_{0}", 50, 0, 50.0);
  controlHistos_[mutype][kD0] = ibooker.book1D(muonNames_[mutype] + "D0", "MuonD0; D_{0}", 50, 0, 200.);
}

void L1TPhase2MuonOffline::bookEfficiencyHistos(DQMStore::IBooker& ibooker, MuType mutype) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::bookEfficiencyHistos()" << endl;

  ibooker.setCurrentFolder(histFolder_ + "/" + muonNames_[mutype] + "/nums_and_dens");

  std::string histoname = "";
  for (const auto eta : etaRegions_) {
    for (const auto q : qualLevels_) {
      histoname = "Eff_" + muonNames_[mutype] + "_" + etaNames_[eta] + "_" + qualNames_[q];

      auto histBins = getHistBinsEff(kEffPt);
      efficiencyNum_[mutype][eta][q][kEffPt] =
          ibooker.book1D(histoname + "_Pt_Num", "MuonPt; p_{T} ;", histBins.size() - 1, &histBins[0]);
      efficiencyDen_[mutype][eta][q][kEffPt] =
          ibooker.book1D(histoname + "_Pt_Den", "MuonPt; p_{T} ;", histBins.size() - 1, &histBins[0]);

      histBins = getHistBinsEff(kEffEta);
      efficiencyNum_[mutype][eta][q][kEffEta] =
          ibooker.book1D(histoname + "_Eta_Num", "MuonEta; #eta ;", histBins.size() - 1, &histBins[0]);
      efficiencyDen_[mutype][eta][q][kEffEta] =
          ibooker.book1D(histoname + "_Eta_Den", "MuonEta; #eta ;", histBins.size() - 1, &histBins[0]);

      histBins = getHistBinsEff(kEffPhi);
      efficiencyNum_[mutype][eta][q][kEffPhi] =
          ibooker.book1D(histoname + "_Phi_Num", "MuonPhi; #phi ;", histBins.size() - 1, &histBins[0]);
      efficiencyDen_[mutype][eta][q][kEffPhi] =
          ibooker.book1D(histoname + "_Phi_Den", "MuonPhi; #phi ;", histBins.size() - 1, &histBins[0]);
    }
  }
}

void L1TPhase2MuonOffline::bookResolutionHistos(DQMStore::IBooker& ibooker, MuType mutype) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::bookResolutionHistos()" << endl;

  ibooker.setCurrentFolder(histFolder_ + "/" + muonNames_[mutype] + "/resolution");
  std::string histoname = "";
  for (const auto eta : etaRegions_) {
    for (const auto q : qualLevels_) {
      for (const auto var : resTypes_) {
        histoname = "Res_" + muonNames_[mutype] + "_" + etaNames_[eta] + "_" + qualNames_[q] + "_" + resNames_[var];
        auto nbins = std::get<0>(getHistBinsRes(var));
        auto xmin = std::get<1>(getHistBinsRes(var));
        auto xmax = std::get<2>(getHistBinsRes(var));
        resolutionHistos_[mutype][eta][q][var] =
            ibooker.book1D(histoname, resNames_[var] + ";" + resLabels_[var], nbins, xmin, xmax);
      }
    }
  }
}

//____________________________________________________________________
void L1TPhase2MuonOffline::fillControlHistos() {
  for (auto& muIt : *gmtSAMuon_) {
    controlHistos_[kSAMuon][kPt]->Fill(lsb_pt * muIt.hwPt());
    controlHistos_[kSAMuon][kPhi]->Fill(lsb_phi * muIt.hwPhi());
    controlHistos_[kSAMuon][kEta]->Fill(lsb_eta * muIt.hwEta());
    controlHistos_[kSAMuon][kIso]->Fill(muIt.hwIso());
    controlHistos_[kSAMuon][kQual]->Fill(muIt.hwQual());
    controlHistos_[kSAMuon][kZ0]->Fill(lsb_z0 * muIt.hwZ0());
    controlHistos_[kSAMuon][kD0]->Fill(lsb_d0 * muIt.hwD0());
  }

  for (auto& muIt : *gmtTkMuon_) {
    controlHistos_[kTkMuon][kPt]->Fill(lsb_pt * muIt.hwPt());
    controlHistos_[kTkMuon][kPhi]->Fill(lsb_phi * muIt.hwPhi());
    controlHistos_[kTkMuon][kEta]->Fill(lsb_eta * muIt.hwEta());
    controlHistos_[kTkMuon][kIso]->Fill(muIt.hwIso());
    controlHistos_[kTkMuon][kQual]->Fill(muIt.hwQual());
    controlHistos_[kTkMuon][kZ0]->Fill(lsb_z0 * muIt.hwZ0());
    controlHistos_[kTkMuon][kD0]->Fill(lsb_d0 * muIt.hwD0());
  }
}

void L1TPhase2MuonOffline::fillEfficiencyHistos() {
  for (const auto& muIt : gmtSAMuonPairs_) {
    auto eta = muIt.etaRegion();
    for (const auto var : effTypes_) {
      auto varToFill = muIt.getVar(var);
      for (const auto& cut : cuts_) {
        const auto q = cut.second;
        efficiencyDen_[kSAMuon][eta][q][var]->Fill(varToFill);
        if (muIt.gmtPt() < 0)
          continue;  // there is not an assciated gmt muon
        if (muIt.gmtQual() < q * 4)
          continue;  //quality requirements
        const auto gmtPtCut = cut.first;
        if (var != kEffPt && muIt.gmtPt() < gmtPtCut)
          continue;  // pt requirement
        efficiencyNum_[kSAMuon][eta][q][var]->Fill(varToFill);
      }
    }
  }

  /// FOR TK MUONS
  for (const auto& muIt : gmtTkMuonPairs_) {
    auto eta = muIt.etaRegion();
    for (const auto var : effTypes_) {
      auto varToFill = muIt.getVar(var);
      for (const auto& cut : cuts_) {
        const auto q = cut.second;
        efficiencyDen_[kTkMuon][eta][q][var]->Fill(varToFill);
        if (muIt.gmtPt() < 0)
          continue;  // there is not an assciated gmt muon
        if (muIt.gmtQual() < q * 4)
          continue;  //quality requirements
        const auto gmtPtCut = cut.first;
        if (var != kEffPt && muIt.gmtPt() < gmtPtCut)
          continue;  // pt requirement
        efficiencyNum_[kTkMuon][eta][q][var]->Fill(varToFill);
      }
    }
  }
}

void L1TPhase2MuonOffline::fillResolutionHistos() {
  for (const auto& muIt : gmtSAMuonPairs_) {
    if (muIt.gmtPt() < 0)
      continue;

    auto eta = muIt.etaRegion();
    for (const auto q : qualLevels_) {
      if (muIt.gmtQual() < q * 4)
        continue;
      for (const auto var : resTypes_) {
        auto varToFill = muIt.getDeltaVar(var);

        resolutionHistos_[kSAMuon][eta][q][var]->Fill(varToFill);
      }
    }
  }

  for (const auto& muIt : gmtTkMuonPairs_) {
    if (muIt.gmtPt() < 0)
      continue;

    auto eta = muIt.etaRegion();
    for (const auto q : qualLevels_) {
      if (muIt.gmtQual() < q * 4)
        continue;
      for (const auto var : resTypes_) {
        auto varToFill = muIt.getDeltaVar(var);

        resolutionHistos_[kTkMuon][eta][q][var]->Fill(varToFill);
      }
    }
  }
}

//_____________________________________________________________________
void L1TPhase2MuonOffline::matchMuonsToGen(std::vector<const reco::GenParticle*> genmus) {
  gmtSAMuonPairs_.clear();
  gmtTkMuonPairs_.clear();

  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::matchMuonsToGen() " << endl;

  for (const reco::GenParticle* gen : genmus) {
    edm::LogInfo("L1TPhase2MuonOffline") << "Looping on genmus: " << gen << endl;
    GenMuonGMTPair pairBestCand(&(*gen), nullptr);
    float dr2Best = maxGmtMuonDR_ * maxGmtMuonDR_;
    for (auto& muIt : *gmtSAMuon_) {
      GenMuonGMTPair pairTmpCand(&(*gen), &(muIt));
      float dr2Tmp = pairTmpCand.dR2();
      if (dr2Tmp < dr2Best) {
        dr2Best = dr2Tmp;
        pairBestCand = pairTmpCand;
      }
    }
    gmtSAMuonPairs_.emplace_back(pairBestCand);

    GenMuonGMTPair pairBestCand2(&(*gen), nullptr);
    dr2Best = maxGmtMuonDR_ * maxGmtMuonDR_;
    for (auto& tkmuIt : *gmtTkMuon_) {
      GenMuonGMTPair pairTmpCand(&(*gen), &(tkmuIt));
      float dr2Tmp = pairTmpCand.dR2();
      if (dr2Tmp < dr2Best) {
        dr2Best = dr2Tmp;
        pairBestCand2 = pairTmpCand;
      }
    }
    gmtTkMuonPairs_.emplace_back(pairBestCand2);
  }
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::matchMuonsToGen() gmtSAMuons: "
                                       << gmtSAMuonPairs_.size() << endl;
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::matchMuonsToGen() gmtTkMuons: "
                                       << gmtTkMuonPairs_.size() << endl;
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::matchMuonsToGen() END " << endl;
}

std::vector<float> L1TPhase2MuonOffline::getHistBinsEff(EffType eff) {
  if (eff == kEffPt) {
    std::vector<float> effVsPtBins(effVsPtBins_.begin(), effVsPtBins_.end());
    return effVsPtBins;
  }
  if (eff == kEffPhi) {
    std::vector<float> effVsPhiBins(effVsPhiBins_.begin(), effVsPhiBins_.end());
    return effVsPhiBins;
  }
  if (eff == kEffEta) {
    std::vector<float> effVsEtaBins(effVsEtaBins_.begin(), effVsEtaBins_.end());
    return effVsEtaBins;
  }
  return {0., 1.};
}

std::tuple<int, double, double> L1TPhase2MuonOffline::getHistBinsRes(ResType res) {
  if (res == kResPt)
    return {50, -2., 2.};
  if (res == kRes1OverPt)
    return {50, -2., 2.};
  if (res == kResQOverPt)
    return {50, -2., 2.};
  if (res == kResPhi)
    return {96, -0.2, 0.2};
  if (res == kResEta)
    return {100, -0.1, 0.1};
  if (res == kResCh)
    return {5, -2, 3};
  return {1, 0, 1};
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TPhase2MuonOffline);
