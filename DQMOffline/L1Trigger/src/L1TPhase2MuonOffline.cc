/**
 * \file L1TPhase2MuonOffline.cc
 *
 * \author S. Folgueras 
 *
 */

#include "DQMOffline/L1Trigger/interface/L1TPhase2MuonOffline.h"

// To convert from HW to Physical coordinates
#include "L1Trigger/Phase2L1GMT/interface/Constants.h"

using namespace reco;
using namespace trigger;
using namespace edm;
using namespace std;
using namespace l1t;

//__________RECO-GMT Muon Pair Helper Class____________________________
GenMuonGMTPair::GenMuonGMTPair(const reco::GenParticle* muon, const l1t::L1Candidate* gmtmu,
			       const PropagateToMuon& propagator)
  : mu_(muon), gmtmu_(gmtmu) {

  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline GenMuonGMTPair::GenMuonGMTPair()" << endl;  
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline GenMuonGMTPair::GenMuonGMTPair() gmtmu " << 
				       gmtmu << endl;  
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline GenMuonGMTPair::GenMuonGMTPair() gen " << 
				       muon  << endl;  
  
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
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline GenMuonGMTPair::GenMuonGMTPair() END" << endl;  
};

GenMuonGMTPair::GenMuonGMTPair(const GenMuonGMTPair& muonGmtPair) {
  mu_    = muonGmtPair.mu_;
  gmtmu_ = muonGmtPair.gmtmu_;

  gmtEta_ = muonGmtPair.gmtEta_;
  gmtPhi_ = muonGmtPair.gmtPhi_;

  muEta_ = muonGmtPair.muEta_;
  muPhi_ = muonGmtPair.muPhi_;
}

float GenMuonGMTPair::dR() {
  float dEta = gmtmu_ ? (gmtEta_ - muEta_) : 999.;
  float dPhi = gmtmu_ ? reco::deltaPhi(gmtPhi_, muPhi_) : 999.;
  return sqrt(dEta * dEta + dPhi * dPhi);
}

L1TPhase2MuonOffline::EtaRegion GenMuonGMTPair::etaRegion() const {
  if (std::abs(muEta_) < 0.83)
    return L1TPhase2MuonOffline::kEtaRegionBmtf;
  if (std::abs(muEta_) < 1.24)
    return L1TPhase2MuonOffline::kEtaRegionOmtf;
  if (std::abs(muEta_) < 2.4)
    return L1TPhase2MuonOffline::kEtaRegionEmtf;
  return L1TPhase2MuonOffline::kEtaRegionOut;
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
L1TPhase2MuonOffline::L1TPhase2MuonOffline(const ParameterSet& ps) :
  gmtMuonToken_(consumes<l1t::SAMuonCollection>(ps.getParameter<edm::InputTag>("gmtMuonToken"))),
  gmtTkMuonToken_(consumes<l1t::TrackerMuonCollection>(ps.getParameter<edm::InputTag>("gmtTkMuonToken"))),
  genParticleToken_(consumes<std::vector<reco::GenParticle>>(ps.getUntrackedParameter<edm::InputTag>("genParticlesInputTag"))),
  muonpropagator_(ps.getParameter<edm::ParameterSet>("muProp"), consumesCollector()),
  muonTypes_({kSAMuon, kTkMuon}),
  effTypes_({kEffPt, kEffPhi, kEffEta}),
  resTypes_({kResPt, kResQOverPt, kResPhi, kResEta}),
  etaRegions_({kEtaRegionAll, kEtaRegionBmtf, kEtaRegionOmtf, kEtaRegionEmtf}),
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
  qualNames_({{kQualAll, "qualAll"}, 
	     {kQualOpen, "qualOpen"}, 
	     {kQualDouble, "qualDouble"}, 
	     {kQualSingle, "qualSingle"}}),
  muonNames_({{kSAMuon, "SAMuon"}, {kTkMuon, "TkMuon"}}),
  histFolder_(ps.getUntrackedParameter<string>("histFolder")),
//  cutsVPSet_(ps.getUntrackedParameter<std::vector<edm::ParameterSet>>("cuts")),
  effVsPtBins_(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsPtBins")),
  effVsPhiBins_(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsPhiBins")),
  effVsEtaBins_(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsEtaBins")),
  maxGmtMuonDR_(0.3){
  
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::L1TPhase2MuonOffline()" << endl;

  /*  for (const auto& cutsPSet : cutsVPSet_) {
    const auto qCut = cutsPSet.getUntrackedParameter<int>("qualCut");
    QualLevel qLevel = kQualAll;
    if (qCut > 11) {
      qLevel = kQualSingle;
    } else if (qCut > 7) {
      qLevel = kQualDouble;
    } else if (qCut > 3) {
      qLevel = kQualOpen;
    }
    cuts_.emplace_back(std::make_pair(cutsPSet.getUntrackedParameter<int>("ptCut"), qLevel));
  }
  */
  // Get Muon constants
  lsb_pt = Phase2L1GMT::LSBpt;
  lsb_phi = Phase2L1GMT::LSBphi;
  lsb_eta = Phase2L1GMT::LSBeta;
  lsb_z0 = Phase2L1GMT::LSBSAz0;
  lsb_d0 = Phase2L1GMT::LSBSAd0;
}

//_____________________________________________________________________
L1TPhase2MuonOffline::~L1TPhase2MuonOffline() {}
//----------------------------------------------------------------------
void L1TPhase2MuonOffline::dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  edm::LogInfo("L1TPhase2MuonOFfline") << "L1TPhase2MuonOffline::dqmBeginRun" << endl;
}

//_____________________________________________________________________
void L1TPhase2MuonOffline::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup& iSetup) {
  edm::LogInfo("L1TPhase2MuonOFfline") << "L1TPhase2MuonOffline::bookHistograms" << endl;

  //book histos
  for (const auto mutype : muonTypes_) {
    bookControlHistos(ibooker, mutype);
    //    bookEfficiencyHistos(ibooker, mutype);
    //    bookResolutionHistos(ibooker, mutype);
  }
}

//_____________________________________________________________________
void L1TPhase2MuonOffline::analyze(const Event& iEvent, const EventSetup& eventSetup) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::analyze() " << endl;
  // Initialise propagator (do we need this?) 
  muonpropagator_.init(eventSetup);
  
  // COLLECT GEN MUONS 
  iEvent.getByToken(genParticleToken_, genparticles_);
  
  std::vector<const reco::GenParticle*> genmus;
  for (const reco::GenParticle& gen : *genparticles_) {
    if (std::abs(gen.pdgId()) != 13) continue;
    genmus.push_back(&gen);
  }
  edm::LogInfo("L1TPhase2MuonOffline") << 
    "L1TPhase2MuonOffline::analyze() N of genmus: "<< genmus.size() << endl;

  // Collect both muon collection: 
  iEvent.getByToken(gmtMuonToken_, gmtSAMuon_);
  iEvent.getByToken(gmtTkMuonToken_, gmtTkMuon_);
  
   
  // Fill Control histograms
  edm::LogInfo("L1TPhase2MuonOffline") << "Fill Control histograms for GMT Muons" << endl;
  fillControlHistos(); 
  

  // Match each muon to a gen muon, if possible.
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::analyze() calling matchMuonsToGen() "<< endl;
  matchMuonsToGen(genmus);
  
  
  /*

  vector<MuonGmtPair>::const_iterator muonGmtPairsIt = m_MuonGmtPairs.begin();
  vector<MuonGmtPair>::const_iterator muonGmtPairsEnd = m_MuonGmtPairs.end();

  // To fill once for global eta and once for TF eta region of the L1T muon.
  // The second entry is a placeholder and will be replaced by the TF eta region of the L1T muon.
  std::array<EtaRegion, 2> regsToFill{{kEtaRegionAll, kEtaRegionAll}};

  for (; muonGmtPairsIt != muonGmtPairsEnd; ++muonGmtPairsIt) {
    // Fill the resolution histograms
    if ((muonGmtPairsIt->etaRegion() != kEtaRegionOut) && (muonGmtPairsIt->gmtPt() > 0)) {
      regsToFill[1] = muonGmtPairsIt->etaRegion();
      m_histoKeyResType histoKeyRes = {kResPt, kEtaRegionAll, kQualAll};
      for (const auto var : m_resTypes) {
        const auto varToFill = muonGmtPairsIt->getDeltaVar(var);
        std::get<0>(histoKeyRes) = var;
        // Fill for the global eta and for TF eta region that the probe muon is in
        for (const auto regToFill : regsToFill) {
          std::get<1>(histoKeyRes) = regToFill;
          for (const auto qualLevel : m_qualLevelsRes) {
            // This assumes that the qualLevel enum has increasing qualities
            // HW quality levels can be 0, 4, 8, or 12
            int qualCut = qualLevel * 4;
            if (muonGmtPairsIt->gmtQual() >= qualCut) {
              std::get<2>(histoKeyRes) = qualLevel;
              m_ResolutionHistos[histoKeyRes]->Fill(varToFill);
            }
          }
        }
      }
    }

    // Fill the efficiency numerator and denominator histograms
    if (muonGmtPairsIt->etaRegion() != kEtaRegionOut) {
      unsigned int cutsCounter = 0;
      for (const auto& cut : m_cuts) {
        const auto gmtPtCut = cut.first;
        const auto qualLevel = cut.second;
        const bool gmtAboveCut = (muonGmtPairsIt->gmtPt() > gmtPtCut);

        // default keys
        m_histoKeyEffDenVarType histoKeyEffDenVar = {kEffPt, gmtPtCut, kEtaRegionAll};
        m_histoKeyEffNumVarType histoKeyEffNumVar = {kEffPt, gmtPtCut, kEtaRegionAll, qualLevel};

        regsToFill[1] = muonGmtPairsIt->etaRegion();
        for (const auto var : m_effTypes) {
          if (var != kEffPt) {
            if (muonGmtPairsIt->pt() < m_recoToL1PtCutFactor * gmtPtCut)
              break;  // efficiency at plateau
          }
          double varToFill;
          if (var == kEffVtx) {
            varToFill = static_cast<double>(nVtx);
          } else {
            varToFill = muonGmtPairsIt->getVar(var);
          }
          // Fill denominators
          if (var == kEffEta) {
            m_EfficiencyDenEtaHistos[gmtPtCut]->Fill(varToFill);
          } else {
            std::get<0>(histoKeyEffDenVar) = var;
            // Fill for the global eta and for TF eta region that the probe muon is in
            for (const auto regToFill : regsToFill) {
              if (var == kEffPt) {
                if (cutsCounter == 0) {
                  m_EfficiencyDenPtHistos[regToFill]->Fill(varToFill);
                }
              } else {
                std::get<2>(histoKeyEffDenVar) = regToFill;
                m_EfficiencyDenVarHistos[histoKeyEffDenVar]->Fill(varToFill);
              }
            }
          }
          // Fill numerators
          std::get<0>(histoKeyEffNumVar) = var;
          // This assumes that the qualLevel enum has increasing qualities
          if (gmtAboveCut && muonGmtPairsIt->gmtQual() >= qualLevel * 4) {
            if (var == kEffEta) {
              m_histoKeyEffNumEtaType histoKeyEffNumEta = {gmtPtCut, qualLevel};
              m_EfficiencyNumEtaHistos[histoKeyEffNumEta]->Fill(varToFill);
            } else {
              std::get<3>(histoKeyEffNumVar) = qualLevel;
              // Fill for the global eta and for TF eta region that the probe muon is in
              for (const auto regToFill : regsToFill) {
                std::get<2>(histoKeyEffNumVar) = regToFill;
                m_EfficiencyNumVarHistos[histoKeyEffNumVar]->Fill(varToFill);
              }
            }
          }
        }
        ++cutsCounter;
      }
    }
  }
  */  
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::analyze() Computation finished" << endl;
}

//_____________________________________________________________________
void L1TPhase2MuonOffline::bookControlHistos(DQMStore::IBooker& ibooker, MuType mutype) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::bookControlHistos()" << endl;
  
  ibooker.setCurrentFolder(histFolder_ + "/" + muonNames_[mutype] + "/control_variables"); 

  controlHistos_[mutype][kPt]   = ibooker.book1D(muonNames_[mutype]+"Pt"  , "MuonPt; p_{T}"    , 50, 0., 100.);
  controlHistos_[mutype][kPhi]  = ibooker.book1D(muonNames_[mutype]+"Phi" , "MuonPhi; #phi"    , 66, -3.3, 3.3);
  controlHistos_[mutype][kEta]  = ibooker.book1D(muonNames_[mutype]+"Eta" , "MuonEta; #eta"    , 50, -2.5, 2.5);
  controlHistos_[mutype][kIso]  = ibooker.book1D(muonNames_[mutype]+"Iso" , "MuonIso; RelIso"  , 50, 0, 1.0);
  controlHistos_[mutype][kQual] = ibooker.book1D(muonNames_[mutype]+"Qual", "MuonQual; Quality", 15, 0.5, 15.5); 
  controlHistos_[mutype][kZ0]   = ibooker.book1D(muonNames_[mutype]+"Z0"  , "MuonZ0; Z_{0}"    , 50, 0, 50.0);
  controlHistos_[mutype][kD0]   = ibooker.book1D(muonNames_[mutype]+"D0"  , "MuonD0; D_{0}"    , 50, 0, 200.); 
}
void L1TPhase2MuonOffline::bookEfficiencyHistos(DQMStore::IBooker& ibooker, MuType mutype) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::bookEfficiencyHistos()" << endl;
}
void L1TPhase2MuonOffline::bookResolutionHistos(DQMStore::IBooker& ibooker, MuType mutype) {
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::bookResolutionHistos()" << endl;
}

/*
void L1TPhase2MuonOffline
//_____________________________________________________________________
void L1TPhase2MuonOffline::bookEfficiencyHistos(DQMStore::IBooker& ibooker, MuType mutype) {
  edm::LogInfo("L1tphase2muonoffline") << "L1tphase2muonoffline::bookEfficiencyHistos()" << endl;
  ibooker.setCurrentFolder(m_HistFolder + "/" + muonNames_[mutype] + "/efficiencies");
  
  for (const auto var : effTypes_) {
    auto histBins = getHistBinsEff(var);
    
    // histograms for eta variable get a special treatment
    for (const auto& cut : m_cuts) {
      const auto gmtPtCut = cut.first;
      const auto qualLevel = cut.second;
      
      std::string name = "effDen_" + effNames_[var] + "_" + std::to_string(gmtPtCut);
      
      efficiencyHistos_[gmtPtCut] =
	ibooker.book1D(name, name + ";" + m_effLabelStrings[var], histBins.size() - 1, &histBins[0]);
      name = "effNum_" + m_effStrings[var] + "_" + std::to_string(gmtPtCut) + "_" + m_qualStrings[qualLevel];
      m_histoKeyEffNumEtaType histoKeyEffNumEta = {gmtPtCut, qualLevel};
      m_EfficiencyNumEtaHistos[histoKeyEffNumEta] =
	ibooker.book1D(name, name + ";" + m_effLabelStrings[var], histBins.size() - 1, &histBins[0]);
    }
    
    else {
      for (const auto etaReg : m_etaRegions) {
        // denominator histograms for pt variable get a special treatment
        if (var == kEffPt) {
          std::string name = "effDen_" + m_effStrings[var] + "_" + m_etaStrings[etaReg];
          m_EfficiencyDenPtHistos[etaReg] =
              ibooker.book1D(name, name + ";" + m_effLabelStrings[var], histBins.size() - 1, &histBins[0]);
        } else {
          for (const auto& cut : m_cuts) {
            const int gmtPtCut = cut.first;
            std::string name =
	    "effDen_" + m_effStrings[var] + "_" + std::to_string(gmtPtCut) + "_" + m_etaStrings[etaReg];
            m_histoKeyEffDenVarType histoKeyEffDenVar = {var, gmtPtCut, etaReg};
            m_EfficiencyDenVarHistos[histoKeyEffDenVar] =
                ibooker.book1D(name, name + ";" + m_effLabelStrings[var], histBins.size() - 1, &histBins[0]);
          }
        }
        for (const auto& cut : m_cuts) {
          const auto gmtPtCut = cut.first;
          const auto qualLevel = cut.second;
          std::string name = "effNum_" + m_effStrings[var] + "_" + std::to_string(gmtPtCut) + "_" +
                             m_etaStrings[etaReg] + "_" + m_qualStrings[qualLevel];
          m_histoKeyEffNumVarType histoKeyEffNum = {var, gmtPtCut, etaReg, qualLevel};
          m_EfficiencyNumVarHistos[histoKeyEffNum] =
              ibooker.book1D(name, name + ";" + m_effLabelStrings[var], histBins.size() - 1, &histBins[0]);
        }
      }
    }
  }
}

void L1TPhase2MuonOffline::bookResolutionHistos(DQMStore::IBooker& ibooker, MuType mutype) {
  edmLogInfo("L1TPhase2MuonOffline")  << "L1TPhase2MuonOffline::bookResolutionHistos()" << endl;

  ibooker.setCurrentFolder(m_HistFolder + "/" + muonNames_[mutype] + "/resolution");

  for (const auto var : m_resTypes) {
    auto nbins = std::get<0>(getHistBinsRes(var));
    auto xmin = std::get<1>(getHistBinsRes(var));
    auto xmax = std::get<2>(getHistBinsRes(var));
    for (const auto etaReg : m_etaRegions) {
      for (const auto qualLevel : m_qualLevelsRes) {
        m_histoKeyResType histoKeyRes = {var, etaReg, qualLevel};
        std::string name = "resolution_" + m_resStrings[var] + "_" + m_etaStrings[etaReg] + "_" + m_qualStrings[qualLevel];
        m_ResolutionHistos[histoKeyRes] = ibooker.book1D(name, name + ";" + m_resLabelStrings[var], nbins, xmin, xmax);
      }
    }
  }
}
*/
//____________________________________________________________________
void L1TPhase2MuonOffline::fillControlHistos(){
    
  for (auto& muIt : *gmtSAMuon_) {   
    controlHistos_[kSAMuon][kPt]  ->Fill(lsb_pt  * muIt.hwPt());
    controlHistos_[kSAMuon][kPhi] ->Fill(lsb_phi * muIt.hwPhi());
    controlHistos_[kSAMuon][kEta] ->Fill(lsb_eta * muIt.hwEta());
    controlHistos_[kSAMuon][kIso] ->Fill(muIt.hwIso());
    controlHistos_[kSAMuon][kQual]->Fill(muIt.hwQual());
    controlHistos_[kSAMuon][kZ0]  ->Fill(lsb_z0 * muIt.hwZ0());
    controlHistos_[kSAMuon][kD0]  ->Fill(lsb_d0 * muIt.hwD0());
  }
  
  for (auto& muIt : *gmtTkMuon_) {
    controlHistos_[kTkMuon][kPt]  ->Fill(lsb_pt  * muIt.hwPt());
    controlHistos_[kTkMuon][kPhi] ->Fill(lsb_phi * muIt.hwPhi());
    controlHistos_[kTkMuon][kEta] ->Fill(lsb_eta * muIt.hwEta());
    controlHistos_[kTkMuon][kIso] ->Fill(muIt.hwIso());
    controlHistos_[kTkMuon][kQual]->Fill(muIt.hwQual());
    controlHistos_[kTkMuon][kZ0]  ->Fill(lsb_z0 * muIt.hwZ0());
    controlHistos_[kTkMuon][kD0]  ->Fill(lsb_d0 * muIt.hwD0());   
  }
  
}

//_____________________________________________________________________
void L1TPhase2MuonOffline::matchMuonsToGen(std::vector<const reco::GenParticle*> genmus) {
  gmtSAMuonPairs_.clear();
  gmtTkMuonPairs_.clear();

  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::matchMuonsToGen() " << endl;
  
  
  for (const reco::GenParticle * gen : genmus){
    edm::LogInfo("L1TPhase2MuonOffline") << "Looping on genmus: "<< gen << endl;
    GenMuonGMTPair pairBestCand(&(*gen), nullptr, muonpropagator_);
    for (auto& muIt : *gmtSAMuon_) {   
      GenMuonGMTPair pairTmpCand(&(*gen), &(muIt), muonpropagator_);
      if ((pairTmpCand.dR() < maxGmtMuonDR_) && (pairTmpCand.dR() < pairBestCand.dR())) {
        pairBestCand = pairTmpCand;
      }
    }
    gmtSAMuonPairs_.emplace_back(pairBestCand);
  
    GenMuonGMTPair pairBestCand2(&(*gen), nullptr, muonpropagator_);
    for (auto& tkmuIt : *gmtTkMuon_) {
      GenMuonGMTPair pairTmpCand(&(*gen), &(tkmuIt), muonpropagator_);
      if ((pairTmpCand.dR() < maxGmtMuonDR_) && (pairTmpCand.dR() < pairBestCand2.dR())) {
        pairBestCand2 = pairTmpCand;
      }
    }
    gmtTkMuonPairs_.emplace_back(pairBestCand2);    
    
  } 
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::matchMuonsToGen() gmtSAMuons: " << gmtSAMuonPairs_.size()<< endl;
  edm::LogInfo("L1TPhase2MuonOffline") << "L1TPhase2MuonOffline::matchMuonsToGen() gmtTkMuons: " << gmtTkMuonPairs_.size()<< endl;
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
