#include "DQMOffline/Trigger/plugins/LepHTMonitor.h"

#include <limits>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoEgamma/EgammaTools/src/ConversionTools.cc" //Would prefer to include header, but fails to find hasMatchedConversion definition without this..
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

namespace{

  //Offline electron definition
  bool isGood(const reco::GsfElectron &el, const reco::Vertex::Point &pv_position,
              const reco::BeamSpot::Point &bs_position,
              const edm::Handle<reco::ConversionCollection> &convs, bool pass_id,
              const double lep_counting_threshold, const double lep_iso_cut, const double lep_eta_cut,
              const double d0_cut_b, const double dz_cut_b,  const double d0_cut_e,  const double dz_cut_e) {

    //Electron ID
    if (!pass_id) return false;

    //pT
    if (el.pt()<lep_counting_threshold || std::abs(el.superCluster()->eta()) > lep_eta_cut) return false;

    //Isolation
    const auto &iso = el.pfIsolationVariables();
    const float absiso = iso.sumChargedHadronPt
      + std::max(0.0, iso.sumNeutralHadronEt + iso.sumPhotonEt -0.5 * iso.sumPUPt);
    const float relisowithdb = absiso/el.pt();
    if (relisowithdb>lep_iso_cut) return false;

    //Conversion matching
    bool pass_conversion = false;
    if (convs.isValid()) {
      pass_conversion = !ConversionTools::hasMatchedConversion(el, convs, bs_position);
    }
    else{
      edm::LogError("LepHTMonitor") << "Electron conversion matching failed.\n";
    }
    if (!pass_conversion) return false;


    //Impact parameter
    float d0 = 999., dz=999.;
    if (el.gsfTrack().isNonnull()) {
      d0=-(el.gsfTrack()->dxy(pv_position));
      dz = el.gsfTrack()->dz(pv_position);
    }
    else{
      edm::LogError("LepHTMonitor") << "Could not read electron.gsfTrack().\n";
      return false;
    }
    float etasc = el.superCluster()->eta();
    if (std::abs(etasc)>1.479) {//Endcap
      if (std::abs(d0)>d0_cut_e || std::abs(dz)>dz_cut_e) return false;

    }else{//Barrel
      if (std::abs(d0)>d0_cut_b || std::abs(dz)>dz_cut_b) return false;
    }

    return true;
  }

  //Offline muon definition
  bool isGood(const reco::Muon &mu, const reco::Vertex &pv,
              const double lep_counting_threshold, const double lep_iso_cut, const double lep_eta_cut,
              const double d0_cut, const double dz_cut, int muonIDlevel) {

    const reco::Vertex::Point &pv_position = pv.position();

    // Muon pt and eta acceptance
    if (mu.pt()<lep_counting_threshold || std::abs(mu.eta())>lep_eta_cut) return false;

    // Muon isolation
    const auto &iso = mu.pfIsolationR04();
    const float absiso = iso.sumChargedHadronPt
      + std::max(0.0, iso.sumNeutralHadronEt + iso.sumPhotonEt -0.5 * iso.sumPUPt);
    const float relisowithdb = absiso/mu.pt();
    if (relisowithdb>lep_iso_cut) return false;

    // Muon ID
    bool pass_id = false;
    if (muonIDlevel == 1) pass_id = muon::isLooseMuon(mu);
    else if (muonIDlevel == 3) pass_id = muon::isTightMuon(mu,pv);
    else pass_id = muon::isMediumMuon(mu);

    if (!pass_id) return false;

    // Muon impact parameter
    float d0 = std::abs(mu.muonBestTrack()->dxy(pv_position));
    float dz = std::abs(mu.muonBestTrack()->dz(pv_position));
    if (d0>d0_cut || dz>dz_cut) return false;

    return true;
  }
}

LepHTMonitor::LepHTMonitor(const edm::ParameterSet &ps):
  theElectronTag_(ps.getParameter<edm::InputTag>("electronCollection")),
  theElectronCollection_(consumes<edm::View<reco::GsfElectron> >(theElectronTag_)),
  theElectronVIDTag_(ps.getParameter<edm::InputTag>("electronVID")),
  theElectronVIDMap_(consumes<edm::ValueMap<bool> >(theElectronVIDTag_)),
  theMuonTag_(ps.getParameter<edm::InputTag>("muonCollection")),
  theMuonCollection_(consumes<reco::MuonCollection>(theMuonTag_)),
  thePfMETTag_(ps.getParameter<edm::InputTag>("pfMetCollection")),
  thePfMETCollection_(consumes<reco::PFMETCollection>(thePfMETTag_)),
  thePfJetTag_(ps.getParameter<edm::InputTag>("pfJetCollection")),
  thePfJetCollection_(consumes<reco::PFJetCollection>(thePfJetTag_)),
  theJetTagTag_(ps.getParameter<edm::InputTag>("jetTagCollection")),
  theJetTagCollection_(consumes<reco::JetTagCollection>(theJetTagTag_)),
  theVertexCollectionTag_(ps.getParameter<edm::InputTag>("vertexCollection")),
  theVertexCollection_(consumes<reco::VertexCollection>(theVertexCollectionTag_)),
  theConversionCollectionTag_(ps.getParameter<edm::InputTag>("conversionCollection")),
  theConversionCollection_(consumes<reco::ConversionCollection>(theConversionCollectionTag_)),
  theBeamSpotTag_(ps.getParameter<edm::InputTag>("beamSpot")),
  theBeamSpot_(consumes<reco::BeamSpot>(theBeamSpotTag_)),

  num_genTriggerEventFlag_(new GenericTriggerEventFlag(ps.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this)),
  den_lep_genTriggerEventFlag_(new GenericTriggerEventFlag(ps.getParameter<edm::ParameterSet>("den_lep_GenericTriggerEventPSet"),consumesCollector(), *this)),
  den_HT_genTriggerEventFlag_(new GenericTriggerEventFlag(ps.getParameter<edm::ParameterSet>("den_HT_GenericTriggerEventPSet"),consumesCollector(), *this)),

  folderName_(ps.getParameter<std::string>("folderName")),

  muonIDlevel_(ps.getUntrackedParameter<int>("muonIDlevel")),

  jetPtCut_(ps.getUntrackedParameter<double>("jetPtCut")),
  jetEtaCut_(ps.getUntrackedParameter<double>("jetEtaCut")),
  metCut_(ps.getUntrackedParameter<double>("metCut")),
  htCut_(ps.getUntrackedParameter<double>("htCut")),

  nmusCut_(ps.getUntrackedParameter<double>("nmus")),
  nelsCut_(ps.getUntrackedParameter<double>("nels")),
  lep_pt_plateau_(ps.getUntrackedParameter<double>("leptonPtPlateau")),
  lep_counting_threshold_(ps.getUntrackedParameter<double>("leptonCountingThreshold")),
  lep_iso_cut_(ps.getUntrackedParameter<double>("lepIsoCut")),
  lep_eta_cut_(ps.getUntrackedParameter<double>("lepEtaCut")),
  lep_d0_cut_b_(ps.getUntrackedParameter<double>("lep_d0_cut_b")),
  lep_dz_cut_b_(ps.getUntrackedParameter<double>("lep_dz_cut_b")),
  lep_d0_cut_e_(ps.getUntrackedParameter<double>("lep_d0_cut_e")),
  lep_dz_cut_e_(ps.getUntrackedParameter<double>("lep_dz_cut_e")),
  ptbins_(ps.getParameter< std::vector<double> >("ptbins")),
  htbins_(ps.getParameter< std::vector<double> >("htbins")),

  nbins_eta_(ps.getUntrackedParameter<int>("nbins_eta")),
  nbins_phi_(ps.getUntrackedParameter<int>("nbins_phi")),
  nbins_npv_(ps.getUntrackedParameter<int>("nbins_npv")),
  etabins_min_(ps.getUntrackedParameter<double>("etabins_min")),
  etabins_max_(ps.getUntrackedParameter<double>("etabins_max")),
  phibins_min_(ps.getUntrackedParameter<double>("phibins_min")),
  phibins_max_(ps.getUntrackedParameter<double>("phibins_max")),
  npvbins_min_(ps.getUntrackedParameter<double>("npvbins_min")),
  npvbins_max_(ps.getUntrackedParameter<double>("npvbins_max")),

  h_pfHTTurnOn_num_(nullptr),
  h_pfHTTurnOn_den_(nullptr),
  h_lepPtTurnOn_num_(nullptr),
  h_lepPtTurnOn_den_(nullptr),
  h_lepEtaTurnOn_num_(nullptr),
  h_lepEtaTurnOn_den_(nullptr),
  h_lepPhiTurnOn_num_(nullptr),
  h_lepPhiTurnOn_den_(nullptr),
  h_NPVTurnOn_num_(nullptr),
  h_NPVTurnOn_den_(nullptr)
  {
  edm::LogInfo("LepHTMonitor")
    << "Constructor LepHTMonitor::LepHTMonitor\n";
  }

LepHTMonitor::~LepHTMonitor() {
  edm::LogInfo("LepHTMonitor")
    << "Destructor LepHTMonitor::~LepHTMonitor\n";
}

void LepHTMonitor::dqmBeginRun(const edm::Run &run, const edm::EventSetup &e) {
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::beginRun\n";
}

void LepHTMonitor::bookHistograms(DQMStore::IBooker &ibooker,
                                           const edm::Run &iRun, const edm::EventSetup &iSetup) {
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::bookHistograms\n";
  //book at beginRun
  ibooker.cd();
  ibooker.setCurrentFolder("HLT/LepHT/" + folderName_);


  bool is_mu = false;
  bool is_ele = false;
  if (theElectronTag_.label().empty() and not theMuonTag_.label().empty()) {
    is_mu=true;
  } else if (not theElectronTag_.label().empty() and theMuonTag_.label().empty()) {
    is_ele=true;
  }
  //Cosmetic axis names
  std::string lepton="lepton", Lepton="Lepton";
  if (is_mu && !is_ele) {
    lepton="muon";
    Lepton="Muon";
  }else if (is_ele && !is_mu) {
    lepton="electron";
    Lepton="Electron";
  }
  //Initialize trigger flags
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_lep_genTriggerEventFlag_ && den_lep_genTriggerEventFlag_->on() ) den_lep_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_HT_genTriggerEventFlag_ && den_HT_genTriggerEventFlag_->on() ) den_HT_genTriggerEventFlag_->initRun( iRun, iSetup );


  //Convert to vfloat for picky TH1F constructor
  vector<float> f_ptbins;
  for(uint ib=0;ib<ptbins_.size();ib++) f_ptbins.push_back(static_cast<float>(ptbins_[ib]));
  vector<float> f_htbins;
  for(uint ib=0;ib<htbins_.size();ib++) f_htbins.push_back(static_cast<float>(htbins_[ib]));

  //num and den hists to be divided in harvesting step to make turn on curves
  h_pfHTTurnOn_num_ = ibooker.book1D("pfHTTurnOn_num", "Numerator;Offline H_{T} [GeV];",  f_htbins.size()-1, f_htbins.data());
  h_pfHTTurnOn_den_ = ibooker.book1D("pfHTTurnOn_den","Denominator;Offline H_{T} [GeV];",  f_htbins.size()-1, f_htbins.data());

  h_lepPtTurnOn_num_  = ibooker.book1D("lepPtTurnOn_num", ("Numerator;Offline "+lepton+" p_{T} [GeV];").c_str(),   f_ptbins.size()-1, f_ptbins.data());
  h_lepPtTurnOn_den_  = ibooker.book1D("lepPtTurnOn_den", ("Denominator;Offline "+lepton+" p_{T} [GeV];").c_str(), f_ptbins.size()-1, f_ptbins.data());
  h_lepEtaTurnOn_num_ = ibooker.book1D("lepEtaTurnOn_num", "Numerator;Offline lepton #eta;", nbins_eta_,etabins_min_,etabins_max_);
  h_lepEtaTurnOn_den_ = ibooker.book1D("lepEtaTurnOn_den", "Denominator;Offline lepton #eta;", nbins_eta_,etabins_min_,etabins_max_);
  h_lepPhiTurnOn_num_ = ibooker.book1D("lepPhiTurnOn_num", "Numerator;Offline lepton #phi;", nbins_phi_,phibins_min_,phibins_max_);
  h_lepPhiTurnOn_den_ = ibooker.book1D("lepPhiTurnOn_den", "Denominator;Offline lepton #phi;", nbins_phi_,phibins_min_,phibins_max_);

  h_lepEtaPhiTurnOn_num_ = ibooker.book2D("lepEtaPhiTurnOn_num", "Numerator;Offline lepton #eta;Offline lepton #phi;", nbins_eta_/2,etabins_min_,etabins_max_,nbins_phi_/2,phibins_min_,phibins_max_);
  h_lepEtaPhiTurnOn_den_ = ibooker.book2D("lepEtaPhiTurnOn_den", "Denominator;Offline lepton #eta;Offline lepton #phi;", nbins_eta_/2,etabins_min_,etabins_max_,nbins_phi_/2,phibins_min_,phibins_max_);

  h_NPVTurnOn_num_  = ibooker.book1D("NPVTurnOn_num", "Numerator;N_{PV};", nbins_npv_,npvbins_min_,npvbins_max_);
  h_NPVTurnOn_den_  = ibooker.book1D("NPVTurnOn_den", "Denominator;N_{PV};", nbins_npv_,npvbins_min_,npvbins_max_);

  ibooker.cd();
}

void LepHTMonitor::beginLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
                                                 const edm::EventSetup &context) {
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::beginLuminosityBlock\n";
}

void LepHTMonitor::analyze(const edm::Event &e, const edm::EventSetup &eSetup) {
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::analyze\n";

  //Find whether main and auxilliary triggers fired
  bool hasFired = false;
  bool hasFiredAuxiliary = false;
  bool hasFiredLeptonAuxiliary = false;
  if (den_lep_genTriggerEventFlag_->on() && den_lep_genTriggerEventFlag_->accept( e, eSetup) ) hasFiredLeptonAuxiliary=true;
  if (den_HT_genTriggerEventFlag_->on() && den_HT_genTriggerEventFlag_->accept( e, eSetup) ) hasFiredAuxiliary=true;
  if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept( e, eSetup) ) hasFired=true;

  if (!(hasFiredAuxiliary || hasFiredLeptonAuxiliary)) return;
  int npv=0;
  //Vertex
  edm::Handle<reco::VertexCollection> VertexCollection;
  if (not theVertexCollectionTag_.label().empty()) {
    e.getByToken(theVertexCollection_, VertexCollection);
    if ( !VertexCollection.isValid() ) {
      edm::LogWarning("LepHTMonitor")
        << "Invalid VertexCollection: " << theVertexCollectionTag_.label() << '\n';
    }
    else npv = VertexCollection->size();
  }


  //Get electron ID map
  edm::Handle<edm::ValueMap<bool> > ele_id_decisions;
  if (not theElectronVIDTag_.label().empty()) {
    e.getByToken(theElectronVIDMap_ ,ele_id_decisions);
    if (!ele_id_decisions.isValid()) {
      edm::LogWarning("LepHTMonitor")
        << "Invalid Electron VID map: " << theElectronVIDTag_.label() << '\n';
    }
  }

  //Conversions
  edm::Handle<reco::ConversionCollection> ConversionCollection;
  if (not theConversionCollectionTag_.label().empty()) {
    e.getByToken(theConversionCollection_, ConversionCollection);
    if ( !ConversionCollection.isValid() ) {
      edm::LogWarning("LepHTMonitor")
        << "Invalid ConversionCollection: " << theConversionCollectionTag_.label() << '\n';
    }
  }

  //Beam Spot
  edm::Handle<reco::BeamSpot> BeamSpot;
  if (not theBeamSpotTag_.label().empty()) {
    e.getByToken(theBeamSpot_, BeamSpot);
    if ( !BeamSpot.isValid() ) {
      edm::LogWarning("LepHTMonitor")
        << "Invalid BeamSpot: " << theBeamSpotTag_.label() << '\n';
    }
  }

  //MET
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  if (not thePfMETTag_.label().empty()) {
    e.getByToken(thePfMETCollection_, pfMETCollection);
    if ( !pfMETCollection.isValid() ) {
      edm::LogWarning("LepHTMonitor")
        << "Invalid PFMETCollection: " << thePfMETTag_.label() << '\n';
    }
  }

  //Jets
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  if (not thePfJetTag_.label().empty()) {
    e.getByToken (thePfJetCollection_,pfJetCollection);
    if ( !pfJetCollection.isValid() ) {
      edm::LogWarning("LepHTMonitor")
        << "Invalid PFJetCollection: " << thePfJetTag_.label() << '\n';
    }
  }

  //Electron
  edm::Handle<edm::View<reco::GsfElectron> > ElectronCollection;
  if (not theElectronTag_.label().empty()) {
    e.getByToken (theElectronCollection_, ElectronCollection);
    if ( !ElectronCollection.isValid() ) {
      edm::LogWarning("LepHTMonitor")
        << "Invalid GsfElectronCollection: " << theElectronTag_.label() << '\n';
    }
  }

  //Muon
  edm::Handle<reco::MuonCollection> MuonCollection;
  if (not theMuonTag_.label().empty()) {
    e.getByToken (theMuonCollection_, MuonCollection);
    if ( !MuonCollection.isValid() ) {
      edm::LogWarning("LepHTMonitor")
        << "Invalid MuonCollection: " << theMuonTag_.label() << '\n';
    }
  }

  //Get offline HT
  double pfHT = -1.0;
  if (pfJetCollection.isValid()) {
    pfHT=0.0;
    for(const auto &pfjet: *pfJetCollection) {
      if (pfjet.pt() < jetPtCut_) continue;
      if (std::abs(pfjet.eta()) > jetEtaCut_) continue;
      pfHT += pfjet.pt();
    }
  }

  //Get offline MET
  double pfMET = -1.0;
  if (pfMETCollection.isValid() && pfMETCollection->size()) {
    pfMET = pfMETCollection->front().et();
  }


  //Find offline leptons and keep track of pt,eta of leading and trailing leptons
  double lep_max_pt = -1.0;
  double lep_eta=0;
  double lep_phi=0;
  double trailing_ele_eta=0;
  double trailing_ele_phi=0;
  double trailing_mu_eta=0;
  double trailing_mu_phi=0;
  double min_ele_pt= -1.0;
  double min_mu_pt=-1.0;
  int nels=0;
  int nmus=0;
  if (VertexCollection.isValid() && VertexCollection->size()) {//for quality checks
    //Try to find a reco electron
    if (ElectronCollection.isValid()
       && ConversionCollection.isValid()
       && BeamSpot.isValid()
       && ele_id_decisions.isValid()) {
      size_t index=0;
      for(const auto &electron: *ElectronCollection) {
        const auto el = ElectronCollection->ptrAt(index);
        bool pass_id = (*ele_id_decisions)[el];
        if (isGood(electron, VertexCollection->front().position(),
              BeamSpot->position(), ConversionCollection, pass_id,
              lep_counting_threshold_,lep_iso_cut_,lep_eta_cut_,
              lep_d0_cut_b_, lep_dz_cut_b_, lep_d0_cut_e_, lep_dz_cut_e_))
        {
          if (electron.pt()>lep_max_pt) {lep_max_pt=electron.pt(); lep_eta=electron.eta();lep_phi=electron.phi();}
          if (electron.pt()<min_ele_pt || min_ele_pt<0) { min_ele_pt=electron.pt(); trailing_ele_eta=electron.eta(); trailing_ele_phi=electron.phi();}
          nels++;
        }
        index++;
      }
    }

    //Try to find a reco muon
    if (MuonCollection.isValid()) {
      for(const auto &muon: *MuonCollection) {
        if (isGood(muon, VertexCollection->front(),lep_counting_threshold_,lep_iso_cut_,lep_eta_cut_, lep_d0_cut_b_, lep_dz_cut_b_, muonIDlevel_)) {
          if (muon.pt()>lep_max_pt) {lep_max_pt=muon.pt(); lep_eta=muon.eta();lep_phi=muon.phi();}
          if (muon.pt()<min_mu_pt || min_mu_pt<0) {min_mu_pt=muon.pt(); trailing_mu_eta=muon.eta(); trailing_mu_phi=muon.phi();}
          nmus++;
        }
      }
    }
  }


  //Fill single lepton triggers with leading lepton pT
  float lep_pt = lep_max_pt;

  //For dilepton triggers, use trailing rather than leading lepton
  if (nmusCut_>=2) {lep_pt = min_mu_pt; lep_eta = trailing_mu_eta; lep_phi = trailing_mu_phi;}
  if (nelsCut_>=2) {lep_pt = min_ele_pt; lep_eta = trailing_ele_eta; lep_phi = trailing_ele_phi;}
  if (nelsCut_>=1 && nmusCut_>=1) {
    if (min_ele_pt<min_mu_pt) {
      lep_pt = min_ele_pt;
      lep_eta = trailing_ele_eta;
      lep_phi = trailing_ele_phi;
    } else {
      lep_pt = min_mu_pt;
      lep_eta = trailing_mu_eta;
      lep_phi = trailing_mu_phi;
    }
  }

  const bool nleps_cut = nels>=nelsCut_ && nmus>=nmusCut_;
  bool lep_plateau = lep_pt>lep_pt_plateau_ || lep_pt_plateau_<0.0;

  //Fill lepton pT and eta histograms
  if (hasFiredLeptonAuxiliary || !e.isRealData()) {

    if (nleps_cut  && (pfMET>metCut_ || metCut_<0.0) && (pfHT>htCut_ || htCut_<0.0)) {
      if (h_lepPtTurnOn_den_) {
        if (lep_pt > ptbins_.back()) lep_pt = ptbins_.back()-1; //Overflow protection
        h_lepPtTurnOn_den_->Fill(lep_pt);
      }
      if (h_lepPtTurnOn_num_ && hasFired) h_lepPtTurnOn_num_->Fill(lep_pt);

      if (lep_plateau) {
        //Fill Eta and Phi histograms for leptons above pT threshold
        if (h_lepEtaTurnOn_den_) h_lepEtaTurnOn_den_->Fill(lep_eta);
        if (h_lepEtaTurnOn_num_ && hasFired) h_lepEtaTurnOn_num_->Fill(lep_eta);
        if (h_lepPhiTurnOn_den_) h_lepPhiTurnOn_den_->Fill(lep_phi);
        if (h_lepPhiTurnOn_num_ && hasFired) h_lepPhiTurnOn_num_->Fill(lep_phi);
        if (h_lepEtaPhiTurnOn_den_) h_lepEtaPhiTurnOn_den_->Fill(lep_eta,lep_phi);
        if (h_lepEtaPhiTurnOn_num_ && hasFired) h_lepEtaPhiTurnOn_num_->Fill(lep_eta,lep_phi);

        //Fill NPV histograms
        if (h_NPVTurnOn_den_) h_NPVTurnOn_den_->Fill(npv);
        if (h_NPVTurnOn_num_ && hasFired) h_NPVTurnOn_num_->Fill(npv);
      }
    }
  }

  //Fill HT turn-on histograms
  if (hasFiredAuxiliary || !e.isRealData()) {
    if (nleps_cut && lep_plateau ) {
      if (h_pfHTTurnOn_den_) {
        if (pfHT > htbins_.back()) pfHT = htbins_.back()-1; //Overflow protection
        h_pfHTTurnOn_den_->Fill(pfHT);
      }
      if (h_pfHTTurnOn_num_ && hasFired) h_pfHTTurnOn_num_->Fill(pfHT);
    }
  }
}

void LepHTMonitor::endLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
                                               const edm::EventSetup &eSetup) {
  edm::LogInfo("LepHTMonitor")
    << "LepHTMonitor::endLuminosityBlock\n";
}

void LepHTMonitor::endRun(const edm::Run &run, const edm::EventSetup &eSetup) {
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::endRun\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(LepHTMonitor);
