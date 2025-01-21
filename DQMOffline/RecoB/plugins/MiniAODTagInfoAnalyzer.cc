#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/BTauReco/interface/JetFeatures.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/LostTracksFeatures.h"
//For DeepJet
#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepFlavourFeatures.h"
//For ParticleNet
#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepBoostedJetFeatures.h"
//For UParT
#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4TagInfo.h"
#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4Features.h"

/** \class MiniAODTagInfoAnalyzer
 *
 *  TagInfo to run on MiniAOD
 *
 */
class MiniAODTagInfoAnalyzer : public DQMEDAnalyzer {
public:

  explicit MiniAODTagInfoAnalyzer(const edm::ParameterSet& pSet);
  ~MiniAODTagInfoAnalyzer() override = default;

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyzeTagInfoDeepJet(const reco::DeepFlavourTagInfo*);
  void analyzeTagInfoParticleNet(const reco::DeepBoostedJetTagInfo*);
  void analyzeTagInfoUnifiedParticleTransformerAK4(const reco::UnifiedParticleTransformerAK4TagInfo*);

  const edm::EDGetTokenT<edm::View<reco::Jet>> jetToken_;
  const std::vector<std::string> jetTagInfos_;
  const double ptMin_;
  const double absEtaMin_;
  const double absEtaMax_;
  std::vector<edm::InputTag> tagInfoTags_;
  std::vector<edm::EDGetTokenT<edm::View<reco::BaseTagInfo>>> tagInfoTokens_;

  bool doDeepJet_;
  bool doParticleNetCentral_;
  bool doParticleNetForward_;
  bool doUnifiedParticleTransformerAK4_;

  std::unordered_map<std::string,MonitorElement*> map_ME_DeepJet_;
  std::unordered_map<std::string,MonitorElement*> map_ME_ParticleNet_;
  std::unordered_map<std::string,MonitorElement*> map_ME_UParT_;

};

MiniAODTagInfoAnalyzer::MiniAODTagInfoAnalyzer(const edm::ParameterSet& pSet):
  jetToken_(consumes<edm::View<reco::Jet>>(pSet.getParameter<edm::InputTag>("jets"))),
  jetTagInfos_(pSet.getParameter<std::vector<std::string>>("jetTagInfos")),
  ptMin_(pSet.getParameter<double>("ptMin")),
  absEtaMin_(pSet.getParameter<double>("absEtaMin")),
  absEtaMax_(pSet.getParameter<double>("absEtaMax")){
    tagInfoTokens_ = edm::vector_transform(jetTagInfos_, [this](std::string const & jetTagInfoStr){
      return mayConsume<edm::View<reco::BaseTagInfo>>(edm::InputTag(jetTagInfoStr));}
    );

    doDeepJet_ = false;
    doParticleNetCentral_ = false;
    doParticleNetForward_ = false;
    doUnifiedParticleTransformerAK4_ = false;
    for (const auto& jetTagInfoLabel : jetTagInfos_) {
      if (jetTagInfoLabel.find("pfDeepFlavour") != std::string::npos){
        doDeepJet_ = true;
      }
      else if (jetTagInfoLabel.find("pfParticleNetFromMiniAODAK4PuppiCentral") != std::string::npos){
        doParticleNetCentral_ = true;
      }
      else if (jetTagInfoLabel.find("pfParticleNetFromMiniAODAK4PuppiForward") != std::string::npos){
        doParticleNetForward_ = true;
      }
      else if (jetTagInfoLabel.find("pfUnifiedParticleTransformerAK4") != std::string::npos){
        doUnifiedParticleTransformerAK4_ = true;
      }
    }
  }

void MiniAODTagInfoAnalyzer::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& es) {
  if (doDeepJet_){
    ibook.setCurrentFolder("Btag/TagInfo_pfDeepFlavour");
    map_ME_DeepJet_["c_pf_btagPf_trackEtaRel"]     = ibook.book1D("c_pf_btagPf_trackEtaRel",     "", 50,   0., 10.);
    map_ME_DeepJet_["c_pf_btagPf_trackPtRel"]      = ibook.book1D("c_pf_btagPf_trackPtRel",      "", 100,  0., 5.);
    map_ME_DeepJet_["c_pf_btagPf_trackPPar"]       = ibook.book1D("c_pf_btagPf_trackPPar",       "", 100,  0., 500.);
    map_ME_DeepJet_["c_pf_btagPf_trackDeltaR"]     = ibook.book1D("c_pf_btagPf_trackDeltaR",     "", 50,   0., 1);
    map_ME_DeepJet_["c_pf_btagPf_trackPParRatio"]  = ibook.book1D("c_pf_btagPf_trackPParRatio",  "", 150, 0.7, 1.);
    map_ME_DeepJet_["c_pf_btagPf_trackSip2dVal"]   = ibook.book1D("c_pf_btagPf_trackSip2dVal",   "", 60, -0.15, 0.15);
    map_ME_DeepJet_["c_pf_btagPf_trackSip2dSig"]   = ibook.book1D("c_pf_btagPf_trackSip2dSig",   "", 120, -5., 25.);
    map_ME_DeepJet_["c_pf_btagPf_trackSip3dVal"]   = ibook.book1D("c_pf_btagPf_trackSip3dVal",   "", 60, -0.15, 0.15);
    map_ME_DeepJet_["c_pf_btagPf_trackSip3dSig"]   = ibook.book1D("c_pf_btagPf_trackSip3dSig",   "", 120, -5., 25.);
    map_ME_DeepJet_["c_pf_btagPf_trackJetDistVal"] = ibook.book1D("c_pf_btagPf_trackJetDistVal", "", 110, -1.0, 0.1);
    map_ME_DeepJet_["c_pf_ptrel"]                  = ibook.book1D("c_pf_ptrel",                  "", 100, -1.2, 0.2);
    map_ME_DeepJet_["c_pf_drminsv"]                = ibook.book1D("c_pf_drminsv",                "", 120, -0.5, 0.1);
    map_ME_DeepJet_["c_pf_vtx_ass"]                = ibook.book1D("c_pf_vtx_ass",                "", 10, 0, 10);
    map_ME_DeepJet_["c_pf_puppiw"]                 = ibook.book1D("c_pf_puppiw",                 "", 20, 0., 1.);
    map_ME_DeepJet_["c_pf_chi2"]                   = ibook.book1D("c_pf_chi2",                   "", 15, 0., 15.);
    map_ME_DeepJet_["c_pf_quality"]                = ibook.book1D("c_pf_quality",                "", 10, 0., 10);
    map_ME_DeepJet_["n_pf_ptrel"]                  = ibook.book1D("n_pf_ptrel",                  "", 50, -1., 0.);
    map_ME_DeepJet_["n_pf_deltaR"]                 = ibook.book1D("n_pf_deltaR",                 "", 50, -1., 0.);
    map_ME_DeepJet_["n_pf_isGamma"]                = ibook.book1D("n_pf_isGamma",                "", 2, -0.5, 1.5);
    map_ME_DeepJet_["n_pf_hadFrac"]                = ibook.book1D("n_pf_hadFrac",                "", 20, 0., 1.);
    map_ME_DeepJet_["n_pf_drminsv"]                = ibook.book1D("n_pf_drminsv",                "", 10, 0., 0.5);
    map_ME_DeepJet_["n_pf_puppiw"]                 = ibook.book1D("n_pf_puppiw",                 "", 20, 0., 1.);
    map_ME_DeepJet_["sv_pt"]                       = ibook.book1D("sv_pt",                       "", 50, 0.,  200.);
    map_ME_DeepJet_["sv_deltaR"]                   = ibook.book1D("sv_deltaR",                   "", 100, -0.8, 0.2);
    map_ME_DeepJet_["sv_mass"]                     = ibook.book1D("sv_mass",                     "", 40, 0., 10.);
    map_ME_DeepJet_["sv_ntracks"]                  = ibook.book1D("sv_ntracks",                  "", 15, -0.5, 14.5);
    map_ME_DeepJet_["sv_chi2"]                     = ibook.book1D("sv_chi2",                     "", 40., 0., 20.);
    map_ME_DeepJet_["sv_normchi2"]                 = ibook.book1D("sv_normchi2",                 "", 40., 0., 20.);
    map_ME_DeepJet_["sv_dxy"]                      = ibook.book1D("sv_dxy",                      "", 100,  0., 0.5);
    map_ME_DeepJet_["sv_dxysig"]                   = ibook.book1D("sv_dxysig",                   "", 140, 0., 70.);
    map_ME_DeepJet_["sv_d3d"]                      = ibook.book1D("sv_d3d",                      "", 20,  0., 1.);
    map_ME_DeepJet_["sv_d3dsig"]                   = ibook.book1D("sv_d3dsig",                   "", 140, 0., 70.);
    map_ME_DeepJet_["sv_costhetasvpv"]             = ibook.book1D("sv_costhetasvpv",             "", 100, -1., 1.);
    map_ME_DeepJet_["sv_enratio"]                  = ibook.book1D("sv_enratio",                  "",  20,  0., 1.);
    map_ME_DeepJet_["max_cpf_n"]                   = ibook.book1D("max_cpf_n",                   "", 31,  -0.5, 30.5);
    map_ME_DeepJet_["max_npf_n"]                   = ibook.book1D("max_npf_n",                   "", 31,  -0.5, 30.5);
    map_ME_DeepJet_["max_sv_n"]                    = ibook.book1D("max_sv_n",                    "",  6,  -0.5, 5.5);
    map_ME_DeepJet_["pv_n"]                        = ibook.book1D("pv_n",                        "", 100, -0.5, 100.5);
  }
  if (doParticleNetCentral_ || doParticleNetForward_){
    if (doParticleNetCentral_){
      ibook.setCurrentFolder("Btag/TagInfo_pfParticleNetFromMiniAODAK4PuppiCentral");
    }
    else if(doParticleNetForward_){
      ibook.setCurrentFolder("Btag/TagInfo_pfParticleNetFromMiniAODAK4PuppiForward");
    }
    map_ME_ParticleNet_["pfcand_pt_log"]                = ibook.book1D("pfcand_pt_log","",               120,  -0.5, 5.5);
    map_ME_ParticleNet_["pfcand_energy_log"]            = ibook.book1D("pfcand_energy_log","",           120,  -0.5, 5.5);
    map_ME_ParticleNet_["pfcand_deta"]                  = ibook.book1D("pfcand_deta","",                 60,  -0.6, 0.6);
    map_ME_ParticleNet_["pfcand_dphi"]                  = ibook.book1D("pfcand_dphi","",                 60,  -0.6, 0.6);
    map_ME_ParticleNet_["pfcand_eta"]                   = ibook.book1D("pfcand_eta","",                  100, -5.0, 5.0);
    map_ME_ParticleNet_["pfcand_charge"]                = ibook.book1D("pfcand_charge","",               3, -1.5, 1.5);
    map_ME_ParticleNet_["pfcand_frompv"]                = ibook.book1D("pfcand_frompv","",               10, 0, 10);
    map_ME_ParticleNet_["pfcand_nlostinnerhits"]        = ibook.book1D("pfcand_nlostinnerhits","",       100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_track_chi2"]            = ibook.book1D("pfcand_track_chi2","",           15, 0., 15);
    map_ME_ParticleNet_["pfcand_track_qual"]            = ibook.book1D("pfcand_track_qual","",           10, 0., 10);
    map_ME_ParticleNet_["pfcand_dz"]                    = ibook.book1D("pfcand_dz","",                   100, -10., 10.);
    map_ME_ParticleNet_["pfcand_dzsig"]                 = ibook.book1D("pfcand_dzsig","",                100,   0., 20.);
    map_ME_ParticleNet_["pfcand_dxy"]                   = ibook.book1D("pfcand_dxy","",                  100, -10., 10.);
    map_ME_ParticleNet_["pfcand_dxysig"]                = ibook.book1D("pfcand_dxysig","",               100,   0., 20.);
    map_ME_ParticleNet_["pfcand_etarel"]                = ibook.book1D("pfcand_etarel","",                40, -10., 10.);
    map_ME_ParticleNet_["pfcand_pperp_ratio"]           = ibook.book1D("pfcand_pperp_ratio","",           20, 0., 1.);
    map_ME_ParticleNet_["pfcand_ppara_ratio"]           = ibook.book1D("pfcand_ppara_ratio","",           20, 0., 1.);
    map_ME_ParticleNet_["pfcand_trackjet_d3d"]          = ibook.book1D("pfcand_trackjet_d3d","",         100, -20., 20.);
    map_ME_ParticleNet_["pfcand_trackjet_d3dsig"]       = ibook.book1D("pfcand_trackjet_d3dsig","",      100, -20., 20.);
    map_ME_ParticleNet_["pfcand_trackjet_dist"]         = ibook.book1D("pfcand_trackjet_dist","",        100, -20., 20.);
    map_ME_ParticleNet_["pfcand_nhits"]                 = ibook.book1D("pfcand_nhits","",                21, -0.5, 20.5);
    map_ME_ParticleNet_["pfcand_npixhits"]              = ibook.book1D("pfcand_npixhits","",             21, -0.5, 20.5);
    map_ME_ParticleNet_["pfcand_nstriphits"]            = ibook.book1D("pfcand_nstriphits","",           21, -0.5, 20.5);
    map_ME_ParticleNet_["pfcand_trackjet_decayL"]       = ibook.book1D("pfcand_trackjet_decayL","",      100, -20., 20.);
    map_ME_ParticleNet_["pfcand_id"]                    = ibook.book1D("pfcand_id","",                   11, -0.5, 10.5);
    map_ME_ParticleNet_["pfcand_calofraction"]          = ibook.book1D("pfcand_calofraction","",         60, 0., 3.);
    map_ME_ParticleNet_["pfcand_hcalfraction"]          = ibook.book1D("pfcand_hcalfraction","",         20, 0., 1.);
    map_ME_ParticleNet_["pfcand_puppiw"]                = ibook.book1D("pfcand_puppiw","",               20, 0., 1.);
    map_ME_ParticleNet_["pfcand_muon_id"]               = ibook.book1D("pfcand_muon_id","",              100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_muon_isglobal"]         = ibook.book1D("pfcand_muon_isglobal","",          2, -0.5,1.5);
    map_ME_ParticleNet_["pfcand_muon_segcomp"]          = ibook.book1D("pfcand_muon_segcomp","",         40., 0., 20.);
    map_ME_ParticleNet_["pfcand_muon_chi2"]             = ibook.book1D("pfcand_muon_chi2","",            40., 0., 20.);
    map_ME_ParticleNet_["pfcand_muon_nvalidhit"]        = ibook.book1D("pfcand_muon_nvalidhit","",       11, -0.5, 10.5);
    map_ME_ParticleNet_["pfcand_muon_nstation"]         = ibook.book1D("pfcand_muon_nstation","",        11, -0.5, 10.5);
    map_ME_ParticleNet_["pfcand_electron_detaIn"]       = ibook.book1D("pfcand_electron_detaIn","",      100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_electron_dphiIn"]       = ibook.book1D("pfcand_electron_dphiIn","",      100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_electron_sigIetaIeta"]  = ibook.book1D("pfcand_electron_sigIetaIeta","", 100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_electron_sigIphiIphi"]  = ibook.book1D("pfcand_electron_sigIphiIphi","", 100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_electron_r9"]           = ibook.book1D("pfcand_electron_r9","",          100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_electron_convProb"]     = ibook.book1D("pfcand_electron_convProb","",    100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_photon_sigIetaIeta"]    = ibook.book1D("pfcand_photon_sigIetaIeta","",   100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_photon_r9"]             = ibook.book1D("pfcand_photon_r9","",            100, -1.2, 0.2);
    map_ME_ParticleNet_["pfcand_photon_eVeto"]          = ibook.book1D("pfcand_photon_eVeto","",         2, 0, 2);
    map_ME_ParticleNet_["pfcand_tau_signal"]            = ibook.book1D("pfcand_tau_signal","",           2, 0, 2);

    map_ME_ParticleNet_["sv_pt_log"]                    = ibook.book1D("sv_pt_log", "", 120,  -0.5, 5.5);
    map_ME_ParticleNet_["sv_mass"]                      = ibook.book1D("sv_mass",   "", 40, 0., 10.);
    map_ME_ParticleNet_["sv_deta"]                      = ibook.book1D("sv_deta",   "", 80, -0.4, 0.4);
    map_ME_ParticleNet_["sv_dphi"]                      = ibook.book1D("sv_dphi",   "", 80, -0.4, 0.4);
    map_ME_ParticleNet_["sv_eta"]                       = ibook.book1D("sv_eta",    "", 60, -3.0, 3.0);
    map_ME_ParticleNet_["sv_ntrack"]                    = ibook.book1D("sv_ntrack", "", 15, -0.5, 14.5);
    map_ME_ParticleNet_["sv_chi2"]                      = ibook.book1D("sv_chi2",   "", 40., 0., 20.);
    map_ME_ParticleNet_["sv_dxy"]                       = ibook.book1D("sv_dxy",    "", 100,  -5.,  5.);
    map_ME_ParticleNet_["sv_dxysig"]                    = ibook.book1D("sv_dxysig", "", 120,   0., 30.);
    map_ME_ParticleNet_["sv_d3d"]                       = ibook.book1D("sv_d3d",    "", 100,   0.,  5.);
    map_ME_ParticleNet_["sv_d3dsig"]                    = ibook.book1D("sv_d3dsig", "", 120,   0., 30.);

    map_ME_ParticleNet_["losttrack_pt_log"]             = ibook.book1D("losttrack_pt_log",          "", 120,  -0.5, 5.5);
    map_ME_ParticleNet_["losttrack_eta"]                = ibook.book1D("losttrack_eta",             "", 100, -5.0, 5.0);
    map_ME_ParticleNet_["losttrack_deta"]               = ibook.book1D("losttrack_deta",            "", 80, -0.4, 0.4);
    map_ME_ParticleNet_["losttrack_dphi"]               = ibook.book1D("losttrack_dphi",            "", 80, -0.4, 0.4);
    map_ME_ParticleNet_["losttrack_charge"]             = ibook.book1D("losttrack_charge",          "", 3, -1.5, 1.5);
    map_ME_ParticleNet_["losttrack_frompv"]             = ibook.book1D("losttrack_frompv",          "", 10, 0, 10);
    map_ME_ParticleNet_["losttrack_track_chi2"]         = ibook.book1D("losttrack_track_chi2",      "", 15, 0., 15.);
    map_ME_ParticleNet_["losttrack_track_qual"]         = ibook.book1D("losttrack_track_qual",      "", 100, -1.2, 0.2);
    map_ME_ParticleNet_["losttrack_dz"]                 = ibook.book1D("losttrack_dz",              "", 100,  0., 0.5);
    map_ME_ParticleNet_["losttrack_dxy"]                = ibook.book1D("losttrack_dxy",             "", 140, 0., 70.);
    map_ME_ParticleNet_["losttrack_dzsig"]              = ibook.book1D("losttrack_dzsig",           "", 50,  0., 1.0);
    map_ME_ParticleNet_["losttrack_dxysig"]             = ibook.book1D("losttrack_dxysig",          "", 140, 0., 70.);
    map_ME_ParticleNet_["losttrack_etarel"]             = ibook.book1D("losttrack_etarel",          "", 40, -10., 10.);
    map_ME_ParticleNet_["losttrack_trackjet_d3d"]       = ibook.book1D("losttrack_trackjet_d3d",    "", 60, -0.15, 0.15);
    map_ME_ParticleNet_["losttrack_trackjet_d3dsig"]    = ibook.book1D("losttrack_trackjet_d3dsig", "", 120, -5., 25.);
    map_ME_ParticleNet_["losttrack_trackjet_dist"]      = ibook.book1D("losttrack_trackjet_dist",   "", 20, -10., 10.);
    map_ME_ParticleNet_["losttrack_trackjet_decayL"]    = ibook.book1D("losttrack_trackjet_decayL", "", 20, -10., 10.);
    map_ME_ParticleNet_["losttrack_npixhits"]           = ibook.book1D("losttrack_npixhits",        "", 21, -0.5, 20.5);
    map_ME_ParticleNet_["losttrack_nstriphits"]         = ibook.book1D("losttrack_nstriphits",      "", 21, -0.5, 20.5);
  }
  if (doUnifiedParticleTransformerAK4_){
    ibook.setCurrentFolder("Btag/TagInfo_pfUnifiedParticleTransformerAK4");
    map_ME_UParT_["c_pf_btagPf_trackEtaRel"]      = ibook.book1D("c_pf_btagPf_trackEtaRel",     "", 50,   0., 10.);
    map_ME_UParT_["c_pf_btagPf_trackPtRel"]       = ibook.book1D("c_pf_btagPf_trackPtRel",      "", 100,  0., 5.);
    map_ME_UParT_["c_pf_btagPf_trackPPar"]        = ibook.book1D("c_pf_btagPf_trackPPar",       "", 100,  0., 500.);
    map_ME_UParT_["c_pf_btagPf_trackDeltaR"]      = ibook.book1D("c_pf_btagPf_trackDeltaR",     "", 50,   0., 1);
    map_ME_UParT_["c_pf_btagPf_trackPParRatio"]   = ibook.book1D("c_pf_btagPf_trackPParRatio",  "", 120, 0.4, 1.);
    map_ME_UParT_["c_pf_btagPf_trackSip2dVal"]    = ibook.book1D("c_pf_btagPf_trackSip2dVal",   "", 60, -0.15, 0.15);
    map_ME_UParT_["c_pf_btagPf_trackSip2dSig"]    = ibook.book1D("c_pf_btagPf_trackSip2dSig",   "", 100, -5., 15.);
    map_ME_UParT_["c_pf_btagPf_trackSip3dVal"]    = ibook.book1D("c_pf_btagPf_trackSip3dVal",   "", 60, -0.15, 0.15);
    map_ME_UParT_["c_pf_btagPf_trackSip3dSig"]    = ibook.book1D("c_pf_btagPf_trackSip3dSig",   "", 100, -5., 15.);
    map_ME_UParT_["c_pf_btagPf_trackJetDistVal"]  = ibook.book1D("c_pf_btagPf_trackJetDistVal", "", 110, -1.0, 0.1);
    map_ME_UParT_["c_pf_btagPf_trackDecayLen"]    = ibook.book1D("c_pf_btagPf_trackDecayLen",   "", 50, 0., 50.);
    map_ME_UParT_["c_pf_ptrel"]                   = ibook.book1D("c_pf_ptrel",                  "", 100, -1.2, 0.2);
    map_ME_UParT_["c_pf_drminsv"]                 = ibook.book1D("c_pf_drminsv",                "", 120, -0.5, 0.);
    map_ME_UParT_["c_pf_vtx_ass"]                 = ibook.book1D("c_pf_vtx_ass",                "", 10, 0, 10);
    map_ME_UParT_["c_pf_puppiw"]                  = ibook.book1D("c_pf_puppiw",                 "", 20, 0., 1.);
    map_ME_UParT_["c_pf_chi2"]                    = ibook.book1D("c_pf_chi2",                   "", 15, 0., 15);
    map_ME_UParT_["c_pf_quality"]                 = ibook.book1D("c_pf_quality",                "", 10, 0., 10);
    map_ME_UParT_["c_pf_charge"]                  = ibook.book1D("c_pf_charge",                 "", 3, -1.5, 1.5);
    map_ME_UParT_["c_pf_dz"]                      = ibook.book1D("c_pf_dz",                     "", 50, 0., 50.);
    map_ME_UParT_["c_pf_HadFrac"]                 = ibook.book1D("c_pf_HadFrac",                "", 20, 0., 1.);
    map_ME_UParT_["c_pf_CaloFrac"]                = ibook.book1D("c_pf_CaloFrac",               "", 60, 0., 3.);
    map_ME_UParT_["c_pf_pdgID"]                   = ibook.book1D("c_pf_pdgID",                  "", 11, -0.5, 10.5);
    map_ME_UParT_["c_pf_lostInnerHits"]           = ibook.book1D("c_pf_lostInnerHits",          "", 11, -0.5, 10.5);
    map_ME_UParT_["c_pf_numberOfPixelHits"]       = ibook.book1D("c_pf_numberOfPixelHits",      "", 21, -0.5, 20.5);
    map_ME_UParT_["c_pf_numberOfStripHits"]       = ibook.book1D("c_pf_numberOfStripHits",      "", 21, -0.5, 20.5);
    map_ME_UParT_["lt_btagPf_trackEtaRel"]        = ibook.book1D("lt_btagPf_trackEtaRel",       "", 50,0., 10.);
    map_ME_UParT_["lt_btagPf_trackPtRel"]         = ibook.book1D("lt_btagPf_trackPtRel",        "", 100,  0., 5.);
    map_ME_UParT_["lt_btagPf_trackPPar"]          = ibook.book1D("lt_btagPf_trackPPar",         "", 100,  0., 500.);
    map_ME_UParT_["lt_btagPf_trackDeltaR"]        = ibook.book1D("lt_btagPf_trackDeltaR",       "", 50,   0., 1);
    map_ME_UParT_["lt_btagPf_trackPParRatio"]     = ibook.book1D("lt_btagPf_trackPParRatio",    "", 150,  0.7, 1.);
    map_ME_UParT_["lt_btagPf_trackSip2dVal"]      = ibook.book1D("lt_btagPf_trackSip2dVal",     "", 60, -0.15, 0.15);
    map_ME_UParT_["lt_btagPf_trackSip2dSig"]      = ibook.book1D("lt_btagPf_trackSip2dSig",     "", 120, -5., 25.);
    map_ME_UParT_["lt_btagPf_trackSip3dVal"]      = ibook.book1D("lt_btagPf_trackSip3dVal",     "", 60, -0.15, 0.15);
    map_ME_UParT_["lt_btagPf_trackSip3dSig"]      = ibook.book1D("lt_btagPf_trackSip3dSig",     "", 120, -5., 25.);
    map_ME_UParT_["lt_btagPf_trackJetDistVal"]    = ibook.book1D("lt_btagPf_trackJetDistVal",   "", 110, -1.0, 0.1);
    map_ME_UParT_["lt_drminsv"]                   = ibook.book1D("lt_drminsv",                  "", 500, -0.5, 0.5);
    map_ME_UParT_["lt_charge"]                    = ibook.book1D("lt_charge",                   "", 3, -1.5, 1.5);
    map_ME_UParT_["lt_puppiw"]                    = ibook.book1D("lt_puppiw",                   "", 20, 0., 1.);
    map_ME_UParT_["lt_chi2"]                      = ibook.book1D("lt_chi2",                     "", 15, 0., 15.);
    map_ME_UParT_["lt_quality"]                   = ibook.book1D("lt_quality",                  "", 10, 0., 10);
    map_ME_UParT_["lt_lostInnerHits"]             = ibook.book1D("lt_lostInnerHits",            "", 11, -0.5, 10.5);
    map_ME_UParT_["lt_numberOfPixelHits"]         = ibook.book1D("lt_numberOfPixelHits",        "", 21, -0.5, 20.5);
    map_ME_UParT_["lt_numberOfStripHits"]         = ibook.book1D("lt_numberOfStripHits",        "", 21, -0.5, 20.5);
    map_ME_UParT_["n_pf_ptrel"]                   = ibook.book1D("n_pf_ptrel",                  "", 100, -1.2, 0.2);
    map_ME_UParT_["n_pf_etarel"]                  = ibook.book1D("n_pf_etarel",                 "", 40, -10., 10.);
    map_ME_UParT_["n_pf_phirel"]                  = ibook.book1D("n_pf_phirel",                 "", 40, -10., 10.);
    map_ME_UParT_["n_pf_deltaR"]                  = ibook.book1D("n_pf_deltaR",                 "", 50, -1., 0.);
    map_ME_UParT_["n_pf_isGamma"]                 = ibook.book1D("n_pf_isGamma",                "", 2, -0.5, 1.5);
    map_ME_UParT_["n_pf_hadFrac"]                 = ibook.book1D("n_pf_hadFrac",                "", 20, 0., 4.);
    map_ME_UParT_["n_pf_drminsv"]                 = ibook.book1D("n_pf_drminsv",                "", 500, -0.5, 0.5);
    map_ME_UParT_["n_pf_puppiw"]                  = ibook.book1D("n_pf_puppiw",                 "", 20, 0., 1.);
    map_ME_UParT_["sv_pt"]                        = ibook.book1D("sv_pt",                       "", 50, 0.,  200.);
    map_ME_UParT_["sv_deltaR"]                    = ibook.book1D("sv_deltaR",                   "", 100, -0.8, 0.2);
    map_ME_UParT_["sv_mass"]                      = ibook.book1D("sv_mass",                     "", 40, 0., 10.);
    map_ME_UParT_["sv_etarel"]                    = ibook.book1D("sv_etarel",                   "", 100, -1., 1.);
    map_ME_UParT_["sv_phirel"]                    = ibook.book1D("sv_phirel",                   "", 100, -1., 1.);
    map_ME_UParT_["sv_ntracks"]                   = ibook.book1D("sv_ntracks",                  "", 15, -0.5, 14.5);
    map_ME_UParT_["sv_chi2"]                      = ibook.book1D("sv_chi2",                     "", 40., 0., 20.);
    map_ME_UParT_["sv_normchi2"]                  = ibook.book1D("sv_normchi2",                 "", 40., 0., 20.);
    map_ME_UParT_["sv_dxy"]                       = ibook.book1D("sv_dxy",                      "", 100,  -5.,  5.);
    map_ME_UParT_["sv_dxysig"]                    = ibook.book1D("sv_dxysig",                   "", 120,   0., 30.);
    map_ME_UParT_["sv_d3d"]                       = ibook.book1D("sv_d3d",                      "", 100,   0.,  5.);
    map_ME_UParT_["sv_d3dsig"]                    = ibook.book1D("sv_d3dsig",                   "", 120,   0., 30.);
    map_ME_UParT_["sv_costhetasvpv"]              = ibook.book1D("sv_costhetasvpv",             "", 100, -1., 1.);
    map_ME_UParT_["sv_enratio"]                   = ibook.book1D("sv_enratio",                  "", 100,  0., 1.);
    map_ME_UParT_["max_cpf_n"]                    = ibook.book1D("max_cpf_n",                   "", 31, -0.5, 30.5);
    map_ME_UParT_["max_lt_n"]                     = ibook.book1D("max_lt_n",                    "", 11, -0.5, 10.5);
    map_ME_UParT_["max_npf_n"]                    = ibook.book1D("max_npf_n",                   "", 31, -0.5, 30.5);
    map_ME_UParT_["max_sv_n"]                     = ibook.book1D("max_sv_n",                    "", 11, -0.5, 10.5);
  }
}

void MiniAODTagInfoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Get the vector of jets
  edm::Handle<edm::View<reco::Jet>> jetCollection;
  iEvent.getByToken(jetToken_, jetCollection);

  //
  std::vector<edm::Handle<edm::View<reco::BaseTagInfo>>> jetTagInfos;
  jetTagInfos.resize(tagInfoTokens_.size());
  for (size_t i = 0; i < tagInfoTokens_.size(); ++i) {
    iEvent.getByToken(tagInfoTokens_[i], jetTagInfos[i]);
  }

  //
  // Loop over the pat::Jets
  //
  size_t idx = 0;
  edm::View<reco::Jet>::const_iterator jet;

  for (jet = jetCollection->begin(), idx = 0; jet != jetCollection->end(); ++jet, ++idx) {
    if (jet->pt() < ptMin_) continue;
    if (std::abs(jet->eta()) > absEtaMax_ || std::abs(jet->eta()) < absEtaMin_) continue;

    edm::RefToBase<reco::Jet> jetRef = jetCollection->refAt(idx);

    //
    // Loop over tag infos
    //
    for (size_t k = 0; k < jetTagInfos.size(); ++k) {

      const edm::View<reco::BaseTagInfo> &taginfos = *jetTagInfos[k];

      //
      // Same procedure in PATJetProducer
      //
      // This is not associative, so we have to search the jet
      edm::Ptr<reco::BaseTagInfo> match;
      // Try first by 'same index'
      if ((idx < taginfos.size()) && (taginfos[idx].jet() == jetRef)) {
        match = taginfos.ptrAt(idx);
      } else {
        // otherwise fail back to a simple search
        for (edm::View<reco::BaseTagInfo>::const_iterator itTI = taginfos.begin(), edTI = taginfos.end(); itTI != edTI; ++itTI) {
          if (itTI->jet() == jetRef) {
            match = taginfos.ptrAt(itTI - taginfos.begin());
            break;
          }
        }
      }
      //
      //
      //
      if (match.isNonnull()) {
        if (jetTagInfos_[k].find("pfDeepFlavour") != std::string::npos && doDeepJet_){
          const reco::DeepFlavourTagInfo* taginfo  = static_cast<const reco::DeepFlavourTagInfo*>(match.get());
          if (!taginfo) {
            throw cms::Exception("Configuration") << "MiniAODTagInfoAnalyzer: not of type DeepFlavourTagInfo. " << std::endl;
          }
          analyzeTagInfoDeepJet(taginfo);
        }
        else if (jetTagInfos_[k].find("pfParticleNetFromMiniAODAK4PuppiCentral") != std::string::npos && doParticleNetCentral_){
          const reco::DeepBoostedJetTagInfo* taginfo  = static_cast<const reco::DeepBoostedJetTagInfo*>(match.get());
          if (!taginfo) {
            throw cms::Exception("Configuration") << "MiniAODTagInfoAnalyzer: not of type DeepBoostedJetTagInfo. " << std::endl;
          }
          analyzeTagInfoParticleNet(taginfo);
        }
        else if (jetTagInfos_[k].find("pfParticleNetFromMiniAODAK4PuppiForward") != std::string::npos && doParticleNetForward_){
          const reco::DeepBoostedJetTagInfo* taginfo  = static_cast<const reco::DeepBoostedJetTagInfo*>(match.get());
          if (!taginfo) {
            throw cms::Exception("Configuration") << "MiniAODTagInfoAnalyzer: not of type DeepBoostedJetTagInfo. " << std::endl;
          }
          analyzeTagInfoParticleNet(taginfo);
        }
        else if (jetTagInfos_[k].find("pfUnifiedParticleTransformerAK4") != std::string::npos && doUnifiedParticleTransformerAK4_){
          const reco::UnifiedParticleTransformerAK4TagInfo* taginfo  = static_cast<const reco::UnifiedParticleTransformerAK4TagInfo*>(match.get());
          if (!taginfo) {
            throw cms::Exception("Configuration") << "MiniAODTagInfoAnalyzer: not of type UnifiedParticleTransformerAK4TagInfo. " << std::endl;
          }
          analyzeTagInfoUnifiedParticleTransformerAK4(taginfo);
        }
      }
    }
  }
}

void MiniAODTagInfoAnalyzer::analyzeTagInfoDeepJet(const reco::DeepFlavourTagInfo* taginfo){
  const auto& features = taginfo->features();

  size_t n_cpf_ = (unsigned int)25;
  size_t n_npf_ = (unsigned int)25;
  size_t n_sv_ = (unsigned int)4;

  // c_pf candidates
  auto max_c_pf_n = std::min(features.c_pf_features.size(), n_cpf_);
  for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
    const auto& c_pf_features = features.c_pf_features.at(c_pf_n);
    map_ME_DeepJet_["c_pf_btagPf_trackEtaRel"]->Fill(c_pf_features.btagPf_trackEtaRel);
    map_ME_DeepJet_["c_pf_btagPf_trackPtRel"]->Fill(c_pf_features.btagPf_trackPtRel);
    map_ME_DeepJet_["c_pf_btagPf_trackPPar"]->Fill(c_pf_features.btagPf_trackPPar);
    map_ME_DeepJet_["c_pf_btagPf_trackDeltaR"]->Fill(c_pf_features.btagPf_trackDeltaR);
    map_ME_DeepJet_["c_pf_btagPf_trackPParRatio"]->Fill(c_pf_features.btagPf_trackPParRatio);
    map_ME_DeepJet_["c_pf_btagPf_trackSip2dVal"]->Fill(c_pf_features.btagPf_trackSip2dVal);
    map_ME_DeepJet_["c_pf_btagPf_trackSip2dSig"]->Fill(c_pf_features.btagPf_trackSip2dSig);
    map_ME_DeepJet_["c_pf_btagPf_trackSip3dVal"]->Fill(c_pf_features.btagPf_trackSip3dVal);
    map_ME_DeepJet_["c_pf_btagPf_trackSip3dSig"]->Fill(c_pf_features.btagPf_trackSip3dSig);
    map_ME_DeepJet_["c_pf_btagPf_trackJetDistVal"]->Fill(c_pf_features.btagPf_trackJetDistVal);
    map_ME_DeepJet_["c_pf_ptrel"]->Fill(c_pf_features.ptrel);
    map_ME_DeepJet_["c_pf_drminsv"]->Fill(c_pf_features.drminsv);
    map_ME_DeepJet_["c_pf_vtx_ass"]->Fill(c_pf_features.vtx_ass);
    map_ME_DeepJet_["c_pf_puppiw"]->Fill(c_pf_features.puppiw);
    map_ME_DeepJet_["c_pf_chi2"]->Fill(c_pf_features.chi2);
    map_ME_DeepJet_["c_pf_quality"]->Fill(c_pf_features.quality);
  }
  // n_pf candidates
  auto max_n_pf_n = std::min(features.n_pf_features.size(), n_npf_);
  for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
    const auto& n_pf_features = features.n_pf_features.at(n_pf_n);
    map_ME_DeepJet_["n_pf_ptrel"]->Fill(n_pf_features.ptrel);
    map_ME_DeepJet_["n_pf_deltaR"]->Fill(n_pf_features.deltaR);
    map_ME_DeepJet_["n_pf_isGamma"]->Fill(n_pf_features.isGamma);
    map_ME_DeepJet_["n_pf_hadFrac"]->Fill(n_pf_features.hadFrac);
    map_ME_DeepJet_["n_pf_drminsv"]->Fill(n_pf_features.drminsv);
    map_ME_DeepJet_["n_pf_puppiw"]->Fill(n_pf_features.puppiw);
  }
  // sv candidates
  auto max_sv_n = std::min(features.sv_features.size(), n_sv_);
  for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
    const auto& sv_features = features.sv_features.at(sv_n);
    map_ME_DeepJet_["sv_pt"]->Fill(sv_features.pt);
    map_ME_DeepJet_["sv_deltaR"]->Fill(sv_features.deltaR);
    map_ME_DeepJet_["sv_mass"]->Fill(sv_features.mass);
    map_ME_DeepJet_["sv_ntracks"]->Fill(sv_features.ntracks);
    map_ME_DeepJet_["sv_chi2"]->Fill(sv_features.chi2);
    map_ME_DeepJet_["sv_normchi2"]->Fill(sv_features.normchi2);
    map_ME_DeepJet_["sv_dxy"]->Fill(sv_features.dxy);
    map_ME_DeepJet_["sv_dxysig"]->Fill(sv_features.dxysig);
    map_ME_DeepJet_["sv_d3d"]->Fill(sv_features.d3d);
    map_ME_DeepJet_["sv_d3dsig"]->Fill(sv_features.d3dsig);
    map_ME_DeepJet_["sv_costhetasvpv"]->Fill(sv_features.costhetasvpv);
    map_ME_DeepJet_["sv_enratio"]->Fill(sv_features.enratio);
  }
  map_ME_DeepJet_["max_cpf_n"]->Fill(max_c_pf_n);
  map_ME_DeepJet_["max_npf_n"]->Fill(max_n_pf_n);
  map_ME_DeepJet_["max_sv_n"]->Fill(max_sv_n);
  map_ME_DeepJet_["pv_n"]->Fill(features.npv);
}

void MiniAODTagInfoAnalyzer::analyzeTagInfoParticleNet(const reco::DeepBoostedJetTagInfo* taginfo){
  const auto& features = taginfo->features();

  int nfeatures_pfcand = features.get("pfcand_mask").size();
  int nfeatures_sv = features.get("sv_mask").size();
  int nfeatures_lt = features.get("lt_mask").size();

  for (int i=0; i < nfeatures_pfcand; i++){
    map_ME_ParticleNet_["pfcand_pt_log"]->Fill(features.get("jet_pfcand_pt_log")[i]);
    map_ME_ParticleNet_["pfcand_energy_log"]->Fill(features.get("jet_pfcand_energy_log")[i]);
    map_ME_ParticleNet_["pfcand_deta"]->Fill(features.get("jet_pfcand_deta")[i]);
    map_ME_ParticleNet_["pfcand_dphi"]->Fill(features.get("jet_pfcand_dphi")[i]);
    map_ME_ParticleNet_["pfcand_eta"]->Fill(features.get("jet_pfcand_eta")[i]);
    map_ME_ParticleNet_["pfcand_charge"]->Fill(features.get("jet_pfcand_charge")[i]);
    map_ME_ParticleNet_["pfcand_frompv"]->Fill(features.get("jet_pfcand_frompv")[i]);
    map_ME_ParticleNet_["pfcand_nlostinnerhits"]->Fill(features.get("jet_pfcand_nlostinnerhits")[i]);
    map_ME_ParticleNet_["pfcand_track_chi2"]->Fill(features.get("jet_pfcand_track_chi2")[i]);
    map_ME_ParticleNet_["pfcand_track_qual"]->Fill(features.get("jet_pfcand_track_qual")[i]);
    map_ME_ParticleNet_["pfcand_dz"]->Fill(features.get("jet_pfcand_dz")[i]);
    map_ME_ParticleNet_["pfcand_dzsig"]->Fill(features.get("jet_pfcand_dzsig")[i]);
    map_ME_ParticleNet_["pfcand_dxy"]->Fill(features.get("jet_pfcand_dxy")[i]);
    map_ME_ParticleNet_["pfcand_dxysig"]->Fill(features.get("jet_pfcand_dxysig")[i]);
    map_ME_ParticleNet_["pfcand_etarel"]->Fill(features.get("jet_pfcand_etarel")[i]);
    map_ME_ParticleNet_["pfcand_pperp_ratio"]->Fill(features.get("jet_pfcand_pperp_ratio")[i]);
    map_ME_ParticleNet_["pfcand_ppara_ratio"]->Fill(features.get("jet_pfcand_ppara_ratio")[i]);
    map_ME_ParticleNet_["pfcand_trackjet_d3d"]->Fill(features.get("jet_pfcand_trackjet_d3d")[i]);
    map_ME_ParticleNet_["pfcand_trackjet_d3dsig"]->Fill(features.get("jet_pfcand_trackjet_d3dsig")[i]);
    map_ME_ParticleNet_["pfcand_trackjet_dist"]->Fill(features.get("jet_pfcand_trackjet_dist")[i]);
    map_ME_ParticleNet_["pfcand_nhits"]->Fill(features.get("jet_pfcand_nhits")[i]);
    map_ME_ParticleNet_["pfcand_npixhits"]->Fill(features.get("jet_pfcand_npixhits")[i]);
    map_ME_ParticleNet_["pfcand_nstriphits"]->Fill(features.get("jet_pfcand_nstriphits")[i]);
    map_ME_ParticleNet_["pfcand_trackjet_decayL"]->Fill(features.get("jet_pfcand_trackjet_decayL")[i]);
    map_ME_ParticleNet_["pfcand_id"]->Fill(features.get("jet_pfcand_id")[i]);
    map_ME_ParticleNet_["pfcand_calofraction"]->Fill(features.get("jet_pfcand_calofraction")[i]);
    map_ME_ParticleNet_["pfcand_hcalfraction"]->Fill(features.get("jet_pfcand_hcalfraction")[i]);
    map_ME_ParticleNet_["pfcand_puppiw"]->Fill(features.get("jet_pfcand_puppiw")[i]);
    map_ME_ParticleNet_["pfcand_muon_id"]->Fill(features.get("jet_pfcand_muon_id")[i]);
    map_ME_ParticleNet_["pfcand_muon_isglobal"]->Fill(features.get("jet_pfcand_muon_isglobal")[i]);
    map_ME_ParticleNet_["pfcand_muon_segcomp"]->Fill(features.get("jet_pfcand_muon_segcomp")[i]);
    map_ME_ParticleNet_["pfcand_muon_chi2"]->Fill(features.get("jet_pfcand_muon_chi2")[i]);
    map_ME_ParticleNet_["pfcand_muon_nvalidhit"]->Fill(features.get("jet_pfcand_muon_nvalidhit")[i]);
    map_ME_ParticleNet_["pfcand_muon_nstation"]->Fill(features.get("jet_pfcand_muon_nstation")[i]);
    map_ME_ParticleNet_["pfcand_electron_detaIn"]->Fill(features.get("jet_pfcand_electron_detaIn")[i]);
    map_ME_ParticleNet_["pfcand_electron_dphiIn"]->Fill(features.get("jet_pfcand_electron_dphiIn")[i]);
    map_ME_ParticleNet_["pfcand_electron_sigIetaIeta"]->Fill(features.get("jet_pfcand_electron_sigIetaIeta")[i]);
    map_ME_ParticleNet_["pfcand_electron_sigIphiIphi"]->Fill(features.get("jet_pfcand_electron_sigIphiIphi")[i]);
    map_ME_ParticleNet_["pfcand_electron_r9"]->Fill(features.get("jet_pfcand_electron_r9")[i]);
    map_ME_ParticleNet_["pfcand_electron_convProb"]->Fill(features.get("jet_pfcand_electron_convProb")[i]);
    map_ME_ParticleNet_["pfcand_photon_sigIetaIeta"]->Fill(features.get("jet_pfcand_photon_sigIetaIeta")[i]);
    map_ME_ParticleNet_["pfcand_photon_r9"]->Fill(features.get("jet_pfcand_photon_r9")[i]);
    map_ME_ParticleNet_["pfcand_photon_eVeto"]->Fill(features.get("jet_pfcand_photon_eVeto")[i]);
    map_ME_ParticleNet_["pfcand_tau_signal"]->Fill(features.get("jet_pfcand_tau_signal")[i]);
  }
  for (int i=0; i < nfeatures_sv; i++){
    map_ME_ParticleNet_["sv_pt_log"]->Fill(features.get("jet_sv_pt_log")[i]);
    map_ME_ParticleNet_["sv_mass"]->Fill(features.get("jet_sv_mass")[i]);
    map_ME_ParticleNet_["sv_deta"]->Fill(features.get("jet_sv_deta")[i]);
    map_ME_ParticleNet_["sv_dphi"]->Fill(features.get("jet_sv_dphi")[i]);
    map_ME_ParticleNet_["sv_eta"]->Fill(features.get("jet_sv_eta")[i]);
    map_ME_ParticleNet_["sv_ntrack"]->Fill(features.get("jet_sv_ntrack")[i]);
    map_ME_ParticleNet_["sv_chi2"]->Fill(features.get("jet_sv_chi2")[i]);
    map_ME_ParticleNet_["sv_dxy"]->Fill(features.get("jet_sv_dxy")[i]);
    map_ME_ParticleNet_["sv_dxysig"]->Fill(features.get("jet_sv_dxysig")[i]);
    map_ME_ParticleNet_["sv_d3d"]->Fill(features.get("jet_sv_d3d")[i]);
    map_ME_ParticleNet_["sv_d3dsig"]->Fill(features.get("jet_sv_d3dsig")[i]);
  }
  for (int i=0; i < nfeatures_lt; i++){
    map_ME_ParticleNet_["losttrack_pt_log"]->Fill(features.get("jet_losttrack_pt_log")[i]);
    map_ME_ParticleNet_["losttrack_eta"]->Fill(features.get("jet_losttrack_eta")[i]);
    map_ME_ParticleNet_["losttrack_deta"]->Fill(features.get("jet_losttrack_deta")[i]);
    map_ME_ParticleNet_["losttrack_dphi"]->Fill(features.get("jet_losttrack_dphi")[i]);
    map_ME_ParticleNet_["losttrack_charge"]->Fill(features.get("jet_losttrack_charge")[i]);
    map_ME_ParticleNet_["losttrack_frompv"]->Fill(features.get("jet_losttrack_frompv")[i]);
    map_ME_ParticleNet_["losttrack_track_chi2"]->Fill(features.get("jet_losttrack_track_chi2")[i]);
    map_ME_ParticleNet_["losttrack_track_qual"]->Fill(features.get("jet_losttrack_track_qual")[i]);
    map_ME_ParticleNet_["losttrack_dz"]->Fill(features.get("jet_losttrack_dz")[i]);
    map_ME_ParticleNet_["losttrack_dxy"]->Fill(features.get("jet_losttrack_dxy")[i]);
    map_ME_ParticleNet_["losttrack_dzsig"]->Fill(features.get("jet_losttrack_dzsig")[i]);
    map_ME_ParticleNet_["losttrack_dxysig"]->Fill(features.get("jet_losttrack_dxysig")[i]);
    map_ME_ParticleNet_["losttrack_etarel"]->Fill(features.get("jet_losttrack_etarel")[i]);
    map_ME_ParticleNet_["losttrack_trackjet_d3d"]->Fill(features.get("jet_losttrack_trackjet_d3d")[i]);
    map_ME_ParticleNet_["losttrack_trackjet_d3dsig"]->Fill(features.get("jet_losttrack_trackjet_d3dsig")[i]);
    map_ME_ParticleNet_["losttrack_trackjet_dist"]->Fill(features.get("jet_losttrack_trackjet_dist")[i]);
    map_ME_ParticleNet_["losttrack_trackjet_decayL"]->Fill(features.get("jet_losttrack_trackjet_decayL")[i]);
    map_ME_ParticleNet_["losttrack_npixhits"]->Fill(features.get("jet_losttrack_npixhits")[i]);
    map_ME_ParticleNet_["losttrack_nstriphits"]->Fill(features.get("jet_losttrack_nstriphits")[i]);
  }
}

void MiniAODTagInfoAnalyzer::analyzeTagInfoUnifiedParticleTransformerAK4(const reco::UnifiedParticleTransformerAK4TagInfo* taginfo){
  const auto& features = taginfo->features();

  size_t n_cpf_ = (unsigned int)29;
  size_t n_lt_ = (unsigned int)5;
  size_t n_npf_ = (unsigned int)25;
  size_t n_sv_ = (unsigned int)5;

  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
    const auto& c_pf_features = features.c_pf_features.at(c_pf_n);
    map_ME_UParT_["c_pf_btagPf_trackEtaRel"]->Fill(c_pf_features.btagPf_trackEtaRel);
    map_ME_UParT_["c_pf_btagPf_trackPtRel"]->Fill(c_pf_features.btagPf_trackPtRel);
    map_ME_UParT_["c_pf_btagPf_trackPPar"]->Fill(c_pf_features.btagPf_trackPPar);
    map_ME_UParT_["c_pf_btagPf_trackDeltaR"]->Fill(c_pf_features.btagPf_trackDeltaR);
    map_ME_UParT_["c_pf_btagPf_trackPParRatio"]->Fill(c_pf_features.btagPf_trackPParRatio);
    map_ME_UParT_["c_pf_btagPf_trackSip2dVal"]->Fill(c_pf_features.btagPf_trackSip2dVal);
    map_ME_UParT_["c_pf_btagPf_trackSip2dSig"]->Fill(c_pf_features.btagPf_trackSip2dSig);
    map_ME_UParT_["c_pf_btagPf_trackSip3dVal"]->Fill(c_pf_features.btagPf_trackSip3dVal);
    map_ME_UParT_["c_pf_btagPf_trackSip3dSig"]->Fill(c_pf_features.btagPf_trackSip3dSig);
    map_ME_UParT_["c_pf_btagPf_trackJetDistVal"]->Fill(c_pf_features.btagPf_trackJetDistVal);
    map_ME_UParT_["c_pf_btagPf_trackDecayLen"]->Fill(c_pf_features.btagPf_trackDecayLen);
    map_ME_UParT_["c_pf_ptrel"]->Fill(c_pf_features.ptrel);
    map_ME_UParT_["c_pf_drminsv"]->Fill(c_pf_features.drminsv);
    map_ME_UParT_["c_pf_vtx_ass"]->Fill(c_pf_features.vtx_ass);
    map_ME_UParT_["c_pf_puppiw"]->Fill(c_pf_features.puppiw);
    map_ME_UParT_["c_pf_chi2"]->Fill(c_pf_features.chi2);
    map_ME_UParT_["c_pf_quality"]->Fill(c_pf_features.quality);
    map_ME_UParT_["c_pf_charge"]->Fill(c_pf_features.charge);
    map_ME_UParT_["c_pf_dz"]->Fill(c_pf_features.dz);
    map_ME_UParT_["c_pf_HadFrac"]->Fill(c_pf_features.HadFrac);
    map_ME_UParT_["c_pf_CaloFrac"]->Fill(c_pf_features.CaloFrac);
    map_ME_UParT_["c_pf_pdgID"]->Fill(c_pf_features.pdgID);
    map_ME_UParT_["c_pf_lostInnerHits"]->Fill(c_pf_features.lostInnerHits);
    map_ME_UParT_["c_pf_numberOfPixelHits"]->Fill(c_pf_features.numberOfPixelHits);
    map_ME_UParT_["c_pf_numberOfStripHits"]->Fill(c_pf_features.numberOfStripHits);
  }

  auto max_lt_n = std::min(features.lt_features.size(), (std::size_t)n_lt_);
  for (std::size_t lt_n = 0; lt_n < max_lt_n; lt_n++) {

    const auto& lt_features = features.lt_features.at(lt_n);
    map_ME_UParT_["lt_btagPf_trackEtaRel"]->Fill(lt_features.btagPf_trackEtaRel);
    map_ME_UParT_["lt_btagPf_trackPtRel"]->Fill(lt_features.btagPf_trackPtRel);
    map_ME_UParT_["lt_btagPf_trackPPar"]->Fill(lt_features.btagPf_trackPPar);
    map_ME_UParT_["lt_btagPf_trackDeltaR"]->Fill(lt_features.btagPf_trackDeltaR);
    map_ME_UParT_["lt_btagPf_trackPParRatio"]->Fill(lt_features.btagPf_trackPParRatio);
    map_ME_UParT_["lt_btagPf_trackSip2dVal"]->Fill(lt_features.btagPf_trackSip2dVal);
    map_ME_UParT_["lt_btagPf_trackSip2dSig"]->Fill(lt_features.btagPf_trackSip2dSig);
    map_ME_UParT_["lt_btagPf_trackSip3dVal"]->Fill(lt_features.btagPf_trackSip3dVal);
    map_ME_UParT_["lt_btagPf_trackSip3dSig"]->Fill(lt_features.btagPf_trackSip3dSig);
    map_ME_UParT_["lt_btagPf_trackJetDistVal"]->Fill(lt_features.btagPf_trackJetDistVal);
    map_ME_UParT_["lt_drminsv"]->Fill(lt_features.drminsv);
    map_ME_UParT_["lt_charge"]->Fill(lt_features.charge);
    map_ME_UParT_["lt_puppiw"]->Fill(lt_features.puppiw);
    map_ME_UParT_["lt_chi2"]->Fill(lt_features.chi2);
    map_ME_UParT_["lt_quality"]->Fill(lt_features.quality);
    map_ME_UParT_["lt_lostInnerHits"]->Fill(lt_features.lostInnerHits);
    map_ME_UParT_["lt_numberOfPixelHits"]->Fill(lt_features.numberOfPixelHits);
    map_ME_UParT_["lt_numberOfStripHits"]->Fill(lt_features.numberOfStripHits);
  }

  auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
  for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
    const auto& n_pf_features = features.n_pf_features.at(n_pf_n);
    map_ME_UParT_["n_pf_ptrel"]->Fill(n_pf_features.ptrel);
    map_ME_UParT_["n_pf_etarel"]->Fill(n_pf_features.etarel);
    map_ME_UParT_["n_pf_phirel"]->Fill(n_pf_features.phirel);
    map_ME_UParT_["n_pf_deltaR"]->Fill(n_pf_features.deltaR);
    map_ME_UParT_["n_pf_isGamma"]->Fill(n_pf_features.isGamma);
    map_ME_UParT_["n_pf_hadFrac"]->Fill(n_pf_features.hadFrac);
    map_ME_UParT_["n_pf_drminsv"]->Fill(n_pf_features.drminsv);
    map_ME_UParT_["n_pf_puppiw"]->Fill(n_pf_features.puppiw);
  }

  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
    const auto& sv_features = features.sv_features.at(sv_n);
    map_ME_UParT_["sv_pt"]->Fill(sv_features.pt);
    map_ME_UParT_["sv_deltaR"]->Fill(sv_features.deltaR);
    map_ME_UParT_["sv_mass"]->Fill(sv_features.mass);
    map_ME_UParT_["sv_etarel"]->Fill(sv_features.etarel);
    map_ME_UParT_["sv_phirel"]->Fill(sv_features.phirel);
    map_ME_UParT_["sv_ntracks"]->Fill(sv_features.ntracks);
    map_ME_UParT_["sv_chi2"]->Fill(sv_features.chi2);
    map_ME_UParT_["sv_normchi2"]->Fill(sv_features.normchi2);
    map_ME_UParT_["sv_dxy"]->Fill(sv_features.dxy);
    map_ME_UParT_["sv_dxysig"]->Fill(sv_features.dxysig);
    map_ME_UParT_["sv_d3d"]->Fill(sv_features.d3d);
    map_ME_UParT_["sv_d3dsig"]->Fill(sv_features.d3dsig);
    map_ME_UParT_["sv_costhetasvpv"]->Fill(sv_features.costhetasvpv);
    map_ME_UParT_["sv_enratio"]->Fill(sv_features.enratio);
  }

  map_ME_UParT_["max_cpf_n"]->Fill(max_c_pf_n);
  map_ME_UParT_["max_lt_n"]->Fill(max_lt_n);
  map_ME_UParT_["max_npf_n"]->Fill(max_n_pf_n);
  map_ME_UParT_["max_sv_n"]->Fill(max_sv_n);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MiniAODTagInfoAnalyzer);
