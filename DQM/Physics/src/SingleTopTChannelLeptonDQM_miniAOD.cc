#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DQM/Physics/src/SingleTopTChannelLeptonDQM_miniAOD.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <iostream>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
using namespace std;
namespace SingleTopTChannelLepton_miniAOD {

// maximal number of leading jets
// to be used for top mass estimate
static const unsigned int MAXJETS = 4;
// nominal mass of the W boson to
// be used for the top mass estimate
static const double WMASS = 80.4;

MonitorEnsemble::MonitorEnsemble(const char* label,
                                 const edm::ParameterSet& cfg,
                                 const edm::VParameterSet& vcfg,
                                 edm::ConsumesCollector&& iC)
    : label_(label),
      pvSelect_(nullptr),
      jetIDSelect_(nullptr),
      includeBTag_(false),
      lowerEdge_(-1.),
      upperEdge_(-1.),
      logged_(0) {
  // sources have to be given; this PSet is not optional
  edm::ParameterSet sources = cfg.getParameter<edm::ParameterSet>("sources");
  muons_ = iC.consumes<edm::View<pat::Muon>>(
      sources.getParameter<edm::InputTag>("muons"));
  elecs_gsf_ = iC.consumes<edm::View<pat::Electron>>(
      sources.getParameter<edm::InputTag>("elecs_gsf"));

  jets_ = iC.consumes<edm::View<pat::Jet>>(
      sources.getParameter<edm::InputTag>("jets"));
  for (edm::InputTag const& tag :
       sources.getParameter<std::vector<edm::InputTag>>("mets"))
    mets_.push_back(iC.consumes<edm::View<pat::MET>>(tag));
  pvs_ = iC.consumes<edm::View<reco::Vertex>>(
      sources.getParameter<edm::InputTag>("pvs"));
  // electronExtras are optional; they may be omitted or
  // empty
  if (cfg.existsAs<edm::ParameterSet>("elecExtras")) {
    edm::ParameterSet elecExtras =
        cfg.getParameter<edm::ParameterSet>("elecExtras");
    // select is optional; in case it's not found no
    // selection will be applied
    if (elecExtras.existsAs<std::string>("select")) {
      elecSelect_ = vcfg[1].getParameter<std::string>("select");
    }
    // isolation is optional; in case it's not found no
    // isolation will be applied
    if (elecExtras.existsAs<std::string>("isolation")) {
      elecIso_ = elecExtras.getParameter<std::string>("isolation");
    }


    // electronId is optional; in case it's not found the
    // InputTag will remain empty
    if (elecExtras.existsAs<edm::ParameterSet>("electronId")) {
      edm::ParameterSet elecId =
          elecExtras.getParameter<edm::ParameterSet>("electronId");
      electronId_ = iC.consumes<edm::ValueMap<float>>(
          elecId.getParameter<edm::InputTag>("src"));
      eidCutValue_ = elecId.getParameter<double>("cutValue");
    }
  }
  // pvExtras are opetional; they may be omitted or empty
  if (cfg.existsAs<edm::ParameterSet>("pvExtras")) {
    edm::ParameterSet pvExtras =
        cfg.getParameter<edm::ParameterSet>("pvExtras");
    // select is optional; in case it's not found no
    // selection will be applied
    if (pvExtras.existsAs<std::string>("select")) {
      pvSelect_.reset(new StringCutObjectSelector<reco::Vertex>(
          pvExtras.getParameter<std::string>("select")));
    }
  }
  // muonExtras are optional; they may be omitted or empty
  if (cfg.existsAs<edm::ParameterSet>(
          "muonExtras")) { 
    edm::ParameterSet muonExtras =
        cfg.getParameter<edm::ParameterSet>("muonExtras");

    // select is optional; in case it's not found no
    // selection will be applied
    if (muonExtras.existsAs<std::string>("select")) {
      muonSelect_ = vcfg[1].getParameter<std::string>("select");
    }
    // isolation is optional; in case it's not found no
    // isolation will be applied
    if (muonExtras.existsAs<std::string>("isolation")) {
      muonIso_ = muonExtras.getParameter<std::string>("isolation");
    }
  }


  // jetExtras are optional; they may be omitted or
  // empty
  if (cfg.existsAs<edm::ParameterSet>("jetExtras")) {
    edm::ParameterSet jetExtras =
        cfg.getParameter<edm::ParameterSet>("jetExtras");
    // jetCorrector is optional; in case it's not found
    // the InputTag will remain empty
    if (jetExtras.existsAs<std::string>("jetCorrector")) {
      jetCorrector_ = jetExtras.getParameter<std::string>("jetCorrector");
    }
    // read jetID information if it exists
    if (jetExtras.existsAs<edm::ParameterSet>("jetID")) {
      edm::ParameterSet jetID =
          jetExtras.getParameter<edm::ParameterSet>("jetID");
      jetIDLabel_ = iC.consumes<reco::JetIDValueMap>(
          jetID.getParameter<edm::InputTag>("label"));
      jetIDSelect_.reset(new StringCutObjectSelector<reco::JetID>(
          jetID.getParameter<std::string>("select")));
    }
    // select is optional; in case it's not found no
    // selection will be applied (only implemented for
    // CaloJets at the moment)
    if (jetExtras.existsAs<std::string>("select")) {

      jetSelect_ = jetExtras.getParameter<std::string>("select");
      jetSelect_ = vcfg[2].getParameter<std::string>("select");
    }

  }

  // triggerExtras are optional; they may be omitted or empty
  if (cfg.existsAs<edm::ParameterSet>("triggerExtras")) {
    edm::ParameterSet triggerExtras =
        cfg.getParameter<edm::ParameterSet>("triggerExtras");
    triggerTable_ = iC.consumes<edm::TriggerResults>(
        triggerExtras.getParameter<edm::InputTag>("src"));
    triggerPaths_ =
        triggerExtras.getParameter<std::vector<std::string>>("paths");
  }

  // massExtras is optional; in case it's not found no mass
  // window cuts are applied for the same flavor monitor
  // histograms
  if (cfg.existsAs<edm::ParameterSet>("massExtras")) {
    edm::ParameterSet massExtras =
        cfg.getParameter<edm::ParameterSet>("massExtras");
    lowerEdge_ = massExtras.getParameter<double>("lowerEdge");
    upperEdge_ = massExtras.getParameter<double>("upperEdge");
  }

  // setup the verbosity level for booking histograms;
  // per default the verbosity level will be set to
  // STANDARD. This will also be the chosen level in
  // the case when the monitoring PSet is not found
  verbosity_ = STANDARD;
  if (cfg.existsAs<edm::ParameterSet>("monitoring")) {
    edm::ParameterSet monitoring =
        cfg.getParameter<edm::ParameterSet>("monitoring");
    if (monitoring.getParameter<std::string>("verbosity") == "DEBUG")
      verbosity_ = DEBUG;
    if (monitoring.getParameter<std::string>("verbosity") == "VERBOSE")
      verbosity_ = VERBOSE;
    if (monitoring.getParameter<std::string>("verbosity") == "STANDARD")
      verbosity_ = STANDARD;
  }
  // and don't forget to do the histogram booking
  directory_ = cfg.getParameter<std::string>("directory");


  muonSelect.reset(new StringCutObjectSelector<pat::Muon, true>(muonSelect_)); 
  muonIso.reset(new StringCutObjectSelector<pat::Muon, true>(muonIso_)); 
  
  elecSelect.reset(new StringCutObjectSelector<pat::Electron, true>(elecSelect_)); 
  elecIso.reset(new StringCutObjectSelector<pat::Electron, true>(elecIso_)); 

}

void MonitorEnsemble::book(DQMStore::IBooker & ibooker) {
  // set up the current directory path
  std::string current(directory_);
  current += label_;
  ibooker.setCurrentFolder(current);

  // determine number of bins for trigger monitoring
  unsigned int nPaths = triggerPaths_.size();

  // --- [STANDARD] --- //
  // number of selected primary vertices
  hists_["pvMult_"] = ibooker.book1D("PvMult", "N_{pvs}", 100, 0., 100.);
  // pt of the leading muon
  hists_["muonPt_"] = ibooker.book1D("MuonPt", "pt(#mu)", 50, 0., 250.);
  // muon multiplicity before std isolation
  hists_["muonMult_"] = ibooker.book1D("MuonMult", "N_{20}(#mu)", 10, 0., 10.);
  // muon multiplicity after  std isolation
  hists_["muonMultIso_"] = ibooker.book1D("MuonMultIso", "N_{Iso}(#mu)", 10, 0., 10.);
  // pt of the leading electron
  hists_["elecPt_"] = ibooker.book1D("ElecPt", "pt(e)", 50, 0., 250.);
  // electron multiplicity before std isolation
  hists_["elecMult_"] = ibooker.book1D("ElecMult", "N_{30}(e)", 10, 0., 10.);
  // electron multiplicity after  std isolation
  hists_["elecMultIso_"] = ibooker.book1D("ElecMultIso", "N_{Iso}(e)", 10, 0., 10.);
  // multiplicity of jets with pt>20 (corrected to L2+L3)
  hists_["jetMult_"] = ibooker.book1D("JetMult", "N_{30}(jet)", 10, 0., 10.);
  // trigger efficiency estimates for single lepton triggers
  hists_["triggerEff_"] = ibooker.book1D("TriggerEff",
      "Eff(trigger)", nPaths, 0., nPaths);
  // monitored trigger occupancy for single lepton triggers
  hists_["triggerMon_"] = ibooker.book1D("TriggerMon",
      "Mon(trigger)", nPaths, 0., nPaths);

  hists_["slimmedMETs_"] = ibooker.book1D("slimmedMETs", "MET_{slimmed}", 50, 0., 200.);

  // MET (calo)
  hists_["slimmedMETsNoHF_"] = ibooker.book1D("slimmedMETsNoHF", "MET_{slimmedNoHF}", 50, 0., 200.);
  // MET (pflow)
  hists_["slimmedMETsPuppi_"] = ibooker.book1D("slimmedMETsPuppi", "MET_{slimmedPuppi}", 50, 0., 200.);
  // W mass estimate
  hists_["massW_"] = ibooker.book1D("MassW", "M(W)", 60, 0., 300.);
  // Top mass estimate
  hists_["massTop_"] = ibooker.book1D("MassTop", "M(Top)", 50, 0., 500.);
  // W mass transverse estimate mu
  hists_["MTWm_"] = ibooker.book1D("MTWm", "M_{T}^{W}(#mu)", 60, 0., 300.);
  // Top mass transverse estimate mu
  hists_["mMTT_"] = ibooker.book1D("mMTT", "M_{T}^{t}(#mu)", 50, 0., 500.);

  // W mass transverse estimate e
  hists_["MTWe_"] = ibooker.book1D("MTWe", "M_{T}^{W}(e)", 60, 0., 300.);
  // Top mass transverse estimate e
  hists_["eMTT_"] = ibooker.book1D("eMTT", "M_{T}^{t}(e)", 50, 0., 500.);

  // set bin labels for trigger monitoring
  triggerBinLabels(std::string("trigger"), triggerPaths_);

  if (verbosity_ == STANDARD) return;

  // --- [VERBOSE] --- //

  // eta of the leading muon
  hists_["muonEta_"] = ibooker.book1D("MuonEta", "#eta(#mu)", 30, -3., 3.);
  // std isolation variable of the leading muon
  hists_["muonPFRelIso_"] = ibooker.book1D("MuonPFRelIso",
      "PFIso_{Rel}(#mu)", 50, 0., 1.);
  hists_["muonRelIso_"] = ibooker.book1D("MuonRelIso", "Iso_{Rel}(#mu)", 50, 0., 1.);

  // eta of the leading electron
  hists_["elecEta_"] = ibooker.book1D("ElecEta", "#eta(e)", 30, -3., 3.);
  // std isolation variable of the leading electron
  hists_["elecRelIso_"] = ibooker.book1D("ElecRelIso", "Iso_{Rel}(e)", 50, 0., 1.);
  hists_["elecPFRelIso_"] = ibooker.book1D("ElecPFRelIso",
      "PFIso_{Rel}(e)", 50, 0., 1.);

  // multiplicity of btagged jets (for track counting high efficiency) with
  // pt(L2L3)>30
  hists_["jetMultBEff_"] = ibooker.book1D("JetMultBEff",
      "N_{30}(b/eff)", 10, 0., 10.);
  // btag discriminator for track counting high efficiency for jets with
  // pt(L2L3)>30
  hists_["jetBDiscEff_"] = ibooker.book1D("JetBDiscEff",
      "Disc_{b/eff}(jet)", 100, 0., 10.);

  // eta of the 1. leading jet
  hists_["jet1Eta_"] = ibooker.book1D("Jet1Eta", "#eta (jet1)", 50, -5., 5.);
  // eta of the 2. leading jet
  hists_["jet2Eta_"] = ibooker.book1D("Jet2Eta", "#eta (jet2)", 50, -5., 5.);

  // pt of the 1. leading jet (corrected to L2+L3)
  hists_["jet1Pt_"] = ibooker.book1D("Jet1Pt", "pt_{L2L3}(jet1)", 60, 0., 300.);
  // pt of the 2. leading jet (corrected to L2+L3)
  hists_["jet2Pt_"] = ibooker.book1D("Jet2Pt", "pt_{L2L3}(jet2)", 60, 0., 300.);

  // eta and pt of the b-tagged jet (filled only when nJets==2)
  hists_["TaggedJetEta_"] = ibooker.book1D("TaggedJetEta",
      "#eta (Tagged jet)", 50, -5., 5.);
  hists_["TaggedJetPt_"] = ibooker.book1D("TaggedJetPt",
      "pt_{L2L3}(Tagged jet)", 60, 0., 300.);

  // eta and pt of the jet not passing b-tag (filled only when nJets==2)
  hists_["UnTaggedJetEta_"] = ibooker.book1D("UnTaggedJetEta",
      "#eta (UnTagged jet)", 50, -5., 5.);
  hists_["UnTaggedJetPt_"] = ibooker.book1D("UnTaggedJetPt",
      "pt_{L2L3}(UnTagged jet)", 60, 0., 300.);

  // eta and pt of the most forward jet in the event with nJets==2
  hists_["FwdJetEta_"] = ibooker.book1D("FwdJetEta", "#eta (Fwd jet)", 50, -5., 5.);
  hists_["FwdJetPt_"] = ibooker.book1D("FwdJetPt",
      "pt_{L2L3}(Fwd jet)", 60, 0., 300.);

  // 2D histogram (pt,eta) of the b-tagged jet (filled only when nJets==2)
  hists_["TaggedJetPtEta_"] = ibooker.book2D("TaggedJetPt_Eta",
      "(pt vs #eta)_{L2L3}(Tagged jet)", 60, 0., 300., 50, -5., 5.);

  // 2D histogram (pt,eta) of the not-b tagged jet (filled only when nJets==2)
  hists_["UnTaggedJetPtEta_"] = ibooker.book2D("UnTaggedJetPt_Eta",
    "(pt vs #eta)_{L2L3}(UnTagged jet)", 60, 0., 300., 50, -5., 5.);



  // dz for muons (to suppress cosmis)
  hists_["muonDelZ_"] = ibooker.book1D("MuonDelZ", "d_{z}(#mu)", 50, -25., 25.);
  // dxy for muons (to suppress cosmics)
  hists_["muonDelXY_"] = ibooker.book2D("MuonDelXY",
      "d_{xy}(#mu)", 50, -0.1, 0.1, 50, -0.1, 0.1);

  // set axes titles for dxy for muons
  hists_["muonDelXY_"]->setAxisTitle("x [cm]", 1);
  hists_["muonDelXY_"]->setAxisTitle("y [cm]", 2);

  if (verbosity_ == VERBOSE) return;

  // --- [DEBUG] --- //

  // relative muon isolation from charged hadrons  for the leading muon
  hists_["muonChHadIso_"] = ibooker.book1D("MuonChHadIso",
      "Iso_{ChHad}(#mu)", 100, 0., 1.);
  // relative muon isolation from neutral hadrons for the leading muon
  hists_["muonNeuHadIso_"] = ibooker.book1D("MuonNeuHadIso",
      "Iso_{NeuHad}(#mu)", 100, 0., 1.);
  // relative muon isolation from photons for the leading muon
  hists_["muonPhIso_"] = ibooker.book1D("MuonPhIso", "Iso_{Ph}(#mu)", 100, 0., 1.);

  // relative electron isolation from charged hadrons for the leading electron
  hists_["elecChHadIso_"] = ibooker.book1D("ElecChHadIso",
      "Iso_{ChHad}(e)", 100, 0., 1.);
  // relative electron isolation from neutral hadrons for the leading electron
  hists_["elecNeuHadIso_"] = ibooker.book1D("ElecNeuHadIso",
      "Iso_{NeuHad}(e)", 100, 0., 1.);
  // relative electron isolation from photons for the leading electron
  hists_["elecPhIso_"] = ibooker.book1D("ElecPhIso", "Iso_{Ph}(e)", 100, 0., 1.);

  // multiplicity of btagged jets (for track counting high purity) with
  // pt(L2L3)>30
  hists_["jetMultBPur_"] = ibooker.book1D("JetMultBPur",
      "N_{30}(b/pur)", 10, 0., 10.);
  // btag discriminator for track counting high purity
  hists_["jetBDiscPur_"] = ibooker.book1D("JetBDiscPur",
      "Disc_{b/pur}(Jet)", 200, -10., 10.);
  // btag discriminator for track counting high purity for 1. leading jet
  hists_["jet1BDiscPur_"] = ibooker.book1D("Jet1BDiscPur",
      "Disc_{b/pur}(Jet1)", 200, -10., 10.);
  // btag discriminator for track counting high purity for 2. leading jet
  hists_["jet2BDiscPur_"] = ibooker.book1D("Jet2BDiscPur",
      "Disc_{b/pur}(Jet2)", 200, -10., 10.);

  // multiplicity of btagged jets (for simple secondary vertex) with pt(L2L3)>30
  hists_["jetMultBVtx_"] = ibooker.book1D("JetMultBVtx",
      "N_{30}(b/vtx)", 10, 0., 10.);
  // btag discriminator for simple secondary vertex
  hists_["jetBDiscVtx_"] = ibooker.book1D("JetBDiscVtx",
      "Disc_{b/vtx}(Jet)", 35, -1., 6.);

  // multiplicity of btagged jets (for combined secondary vertex) with
  // pt(L2L3)>30
  hists_["jetMultBCombVtx_"] = ibooker.book1D("JetMultBCombVtx",
      "N_{30}(b/CSV)", 10, 0., 10.);
  // btag discriminator for combined secondary vertex
  hists_["jetBDiscCombVtx_"] = ibooker.book1D("JetBDiscCombVtx",
      "Disc_{b/CSV}(Jet)", 60, -1., 2.);
  // btag discriminator for combined secondary vertex for 1. leading jet
  hists_["jet1BDiscCombVtx_"] = ibooker.book1D("Jet1BDiscCombVtx",
      "Disc_{b/CSV}(Jet1)", 60, -1., 2.);
  // btag discriminator for combined secondary vertex for 2. leading jet
  hists_["jet2BDiscCombVtx_"] = ibooker.book1D("Jet2BDiscCombVtx",
      "Disc_{b/CSV}(Jet2)", 60, -1., 2.);

  // pt of the 1. leading jet (uncorrected)
  hists_["jet1PtRaw_"] = ibooker.book1D("Jet1PtRaw", "pt_{Raw}(jet1)", 60, 0., 300.);
  // pt of the 2. leading jet (uncorrected)
  hists_["jet2PtRaw_"] = ibooker.book1D("Jet2PtRaw", "pt_{Raw}(jet2)", 60, 0., 300.);

  // selected events
  hists_["eventLogger_"] = ibooker.book2D("EventLogger",
      "Logged Events", 9, 0., 9., 10, 0., 10.);

  // set axes titles for selected events
  hists_["eventLogger_"]->getTH1()->SetOption("TEXT");
  hists_["eventLogger_"]->setBinLabel(1, "Run", 1);
  hists_["eventLogger_"]->setBinLabel(2, "Block", 1);
  hists_["eventLogger_"]->setBinLabel(3, "Event", 1);
  hists_["eventLogger_"]->setBinLabel(4, "pt_{L2L3}(jet1)", 1);
  hists_["eventLogger_"]->setBinLabel(5, "pt_{L2L3}(jet2)", 1);
  hists_["eventLogger_"]->setBinLabel(6, "pt_{L2L3}(jet3)", 1);
  hists_["eventLogger_"]->setBinLabel(7, "pt_{L2L3}(jet4)", 1);
  hists_["eventLogger_"]->setBinLabel(8, "M_{W}", 1);
  hists_["eventLogger_"]->setBinLabel(9, "M_{Top}", 1);
  hists_["eventLogger_"]->setAxisTitle("logged evts", 2);
  return;
}

void MonitorEnsemble::fill(const edm::Event& event,
                           const edm::EventSetup& setup) {
  // fetch trigger event if configured such
  edm::Handle<edm::TriggerResults> triggerTable;
  if (!triggerTable_.isUninitialized()) {
    if (!event.getByToken(triggerTable_, triggerTable)) return;
  }
  /*
    ------------------------------------------------------------

    Primary Vertex Monitoring

    ------------------------------------------------------------
  */

  // fill monitoring plots for primary vertices
  edm::Handle<edm::View<reco::Vertex>> pvs;
  if (!event.getByToken(pvs_, pvs)) return;
  unsigned int pvMult = 0;
  for (edm::View<reco::Vertex>::const_iterator pv = pvs->begin();
       pv != pvs->end(); ++pv) {
    if (!pvSelect_ || (*pvSelect_)(*pv)) pvMult++;
  }
  fill("pvMult_", pvMult);

  /*
     ------------------------------------------------------------

     Electron Monitoring

     ------------------------------------------------------------
  */

  // fill monitoring plots for electrons
  edm::Handle<edm::View<pat::Electron>> elecs_gsf;






  if (!event.getByToken(elecs_gsf_, elecs_gsf)) return;
 
  // loop electron collection
  unsigned int eMult = 0, eMultIso = 0;
  std::vector<const pat::Electron*> isoElecs;
  pat::Electron e;

  unsigned int idx_gsf = 0;
  for ( edm::View<pat::Electron>::const_iterator elec = elecs_gsf->begin(); elec != elecs_gsf->end(); ++elec) {

   if (true){
      if ((*elecSelect)(*elec)) {
        double isolationRel =
            (elec->dr03TkSumPt() + elec->dr03EcalRecHitSumEt() +
             elec->dr03HcalTowerSumEt()) /
            elec->pt();

        double isolationChHad =
            elec->pt() /
            (elec->pt() + elec->pfIsolationVariables().sumChargedHadronPt);
        double isolationNeuHad =
            elec->pt() /
            (elec->pt() + elec->pfIsolationVariables().sumNeutralHadronEt);
        double isolationPhoton =
            elec->pt() /
            (elec->pt() + elec->pfIsolationVariables().sumPhotonEt);
        double el_ChHadIso = elec->pfIsolationVariables().sumChargedHadronPt;
        double el_NeHadIso = elec->pfIsolationVariables().sumNeutralHadronEt;
        double el_PhIso = elec->pfIsolationVariables().sumPhotonEt;
        double PFisolationRel =
            (el_ChHadIso +
             max(0., el_NeHadIso + el_PhIso -
                         0.5 * elec->pfIsolationVariables().sumPUPt)) /
            elec->pt();

        if (eMult == 0) {
          fill("elecPt_", elec->pt());
          fill("elecEta_", elec->eta());
          fill("elecRelIso_", isolationRel);
          fill("elecPFRelIso_", PFisolationRel);
          fill("elecChHadIso_", isolationChHad);
          fill("elecNeuHadIso_", isolationNeuHad);
          fill("elecPhIso_", isolationPhoton);
        }

        ++eMult;
        if ((*elecIso)(*elec)) {
          if (eMultIso == 0) e = *elec;
          isoElecs.push_back(&(*elec));
          ++eMultIso;
        }
      }
    }
    idx_gsf++;
  }

  fill("elecMult_", eMult);
  fill("elecMultIso_", eMultIso);

  /*
     ------------------------------------------------------------

     Muon Monitoring

     ------------------------------------------------------------
  */

  // fill monitoring plots for muons
  unsigned int mMult = 0, mMultIso = 0;

  edm::Handle<edm::View<pat::Muon>> muons;

  pat::Muon mu;

  if (!event.getByToken(muons_, muons)) return;
  for (edm::View<pat::Muon>::const_iterator muon = muons->begin(); muon != muons->end();
       ++muon) {  

    // restrict to globalMuons
    if (muon->isGlobalMuon()) {
      fill("muonDelZ_", muon->globalTrack()->vz());
      fill("muonDelXY_", muon->globalTrack()->vx(), muon->globalTrack()->vy());

      // apply selection
      if ((*muonSelect)(*muon)) {

        double isolationRel =
            (muon->isolationR03().sumPt + muon->isolationR03().emEt +
             muon->isolationR03().hadEt) /
            muon->pt();
        double isolationChHad =
            muon->pt() /
            (muon->pt() + muon->pfIsolationR04().sumChargedHadronPt);
        double isolationNeuHad =
            muon->pt() /
            (muon->pt() + muon->pfIsolationR04().sumNeutralHadronEt);
        double isolationPhoton =
            muon->pt() / (muon->pt() + muon->pfIsolationR04().sumPhotonEt);
        double PFisolationRel = (muon->pfIsolationR04().sumChargedHadronPt +
                                 muon->pfIsolationR04().sumNeutralHadronEt +
                                 muon->pfIsolationR04().sumPhotonEt) /
                                muon->pt();

        if (mMult == 0) {
          // restrict to leading muon
          fill("muonPt_", muon->pt());
          fill("muonEta_", muon->eta());
          fill("muonRelIso_", isolationRel);
          fill("muonChHadIso_", isolationChHad);
          fill("muonNeuHadIso_", isolationNeuHad);
          fill("muonPhIso_", isolationPhoton);
          fill("muonPFRelIso_", PFisolationRel);
        }
        ++mMult;

        if ((*muonIso)(*muon)) {
          if (mMultIso == 0) mu = *muon;
          ++mMultIso;
        }
      }
    }
  }
  fill("muonMult_", mMult);
  fill("muonMultIso_", mMultIso);

  /*
     ------------------------------------------------------------

     Jet Monitoring

     ------------------------------------------------------------
  */

  // loop jet collection
  std::vector<pat::Jet> correctedJets;
  unsigned int mult = 0, multBEff = 0, multBPur = 0, multNoBPur = 0,
               multBVtx = 0, multBCombVtx = 0;

  edm::Handle<edm::View<pat::Jet>> jets;
  if (!event.getByToken(jets_, jets)) return;

  vector<double> bJetDiscVal;
  vector<double> NobJetDiscVal;
  pat::Jet TaggedJetCand;
  pat::Jet UnTaggedJetCand;
  pat::Jet FwdJetCand;
  for (edm::View<pat::Jet>::const_iterator jet = jets->begin();
       jet != jets->end(); ++jet) {

      pat::Jet sel = *jet;

      if ( jetSelectJet==0)
	jetSelectJet.reset(new StringCutObjectSelector<pat::Jet>(jetSelect_));

      if (!((*jetSelectJet)(sel))) continue;

    // prepare jet to fill monitor histograms
    pat::Jet monitorJet = *jet;
    correctedJets.push_back(monitorJet);

    ++mult;  // determine jet multiplicity


      fill("jetBDiscEff_", monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags")); //hard coded discriminator and value right now.
      if (monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") > 0.89) ++multBEff;



      if (monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") > 0.89) {
        if (multBPur == 0) {
          TaggedJetCand = monitorJet;
          bJetDiscVal.push_back(monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));

        } else if (multBPur == 1) {
          bJetDiscVal.push_back(monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));
          if (bJetDiscVal[1] > bJetDiscVal[0]) TaggedJetCand = monitorJet;
        }
        ++multBPur;
      } else {
        if (multNoBPur == 0) {
          UnTaggedJetCand = monitorJet;
          NobJetDiscVal.push_back(monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));

        } else if (multNoBPur == 1) {
          NobJetDiscVal.push_back(monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));
          if (NobJetDiscVal[1] < NobJetDiscVal[0]) UnTaggedJetCand = monitorJet;
        }

        ++multNoBPur;
      }


      if (monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") > 0.89) ++multBEff;
      if (mult == 1) {
        fill("jet1BDiscPur_", monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));
      } else if (mult == 2) {
        fill("jet2BDiscPur_", monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));
      }

      fill("jetBDiscPur_", monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));


    // fill pt (raw or L2L3) for the leading jets
    if (mult == 1) {
      fill("jet1Pt_", monitorJet.pt());
      fill("jet1Eta_", monitorJet.eta());
      fill("jet1PtRaw_", jet->pt());
      FwdJetCand = monitorJet;
    }

    if (mult == 2) {
      fill("jet2Pt_", monitorJet.pt());
      fill("jet2Eta_", monitorJet.eta());
      fill("jet2PtRaw_", jet->pt());

      if (abs(monitorJet.eta()) > abs(FwdJetCand.eta())) {
        FwdJetCand = monitorJet;
      }

      fill("FwdJetPt_", FwdJetCand.pt());
      fill("FwdJetEta_", FwdJetCand.eta());
    }
  }

  if (multNoBPur == 1 && multBPur == 1) {

    fill("TaggedJetPtEta_", TaggedJetCand.pt(), TaggedJetCand.eta());
    fill("UnTaggedJetPtEta_", UnTaggedJetCand.pt(), UnTaggedJetCand.eta());

    fill("TaggedJetPt_", TaggedJetCand.pt());
    fill("TaggedJetEta_", TaggedJetCand.eta());
    fill("UnTaggedJetPt_", UnTaggedJetCand.pt());
    fill("UnTaggedJetEta_", UnTaggedJetCand.eta());
  }

  fill("jetMult_", mult);
  fill("jetMultBEff_", multBEff);
  fill("jetMultBPur_", multBPur);
  fill("jetMultBVtx_", multBVtx);
  fill("jetMultBCombVtx_", multBCombVtx);

  /*
  ------------------------------------------------------------

  MET Monitoring

  ------------------------------------------------------------
  */

  // fill monitoring histograms for met
  pat::MET mET;

  for (std::vector<edm::EDGetTokenT<edm::View<pat::MET>>>::const_iterator
           met_ = mets_.begin();
       met_ != mets_.end(); ++met_) {
    edm::Handle<edm::View<pat::MET>> met;
    if (!event.getByToken(*met_, met)) continue;
    mET = *(met->begin());
    if (met->begin() != met->end()) {
      unsigned int idx = met_ - mets_.begin();
      if (idx == 0) fill("slimmedMETs_", met->begin()->et());
      if (idx == 1) fill("slimmedMETsNoHF_", met->begin()->et());
      if (idx == 2) fill("slimmedMETsPuppi_", met->begin()->et());

    }
  }

  /*
     ------------------------------------------------------------

     Event Monitoring

     ------------------------------------------------------------
  */

  // fill W boson and top mass estimates
  Calculate_miniAOD eventKinematics(MAXJETS, WMASS);
  double wMass = eventKinematics.massWBoson(correctedJets);
  double topMass = eventKinematics.massTopQuark(correctedJets);
  if (wMass >= 0 && topMass >= 0) {
    fill("massW_", wMass);
    fill("massTop_", topMass);
  }
  // fill plots for trigger monitoring
  if ((lowerEdge_ == -1. && upperEdge_ == -1.) ||
      (lowerEdge_ < wMass && wMass < upperEdge_)) {
    if (!triggerTable_.isUninitialized())
      fill(event, *triggerTable, "trigger", triggerPaths_);
    if (logged_ <= hists_.find("eventLogger_")->second->getNbinsY()) {
      // log runnumber, lumi block, event number & some
      // more pysics infomation for interesting events
      fill("eventLogger_", 0.5, logged_ + 0.5, event.eventAuxiliary().run());
      fill("eventLogger_", 1.5, logged_ + 0.5,
           event.eventAuxiliary().luminosityBlock());
      fill("eventLogger_", 2.5, logged_ + 0.5, event.eventAuxiliary().event());
      if (correctedJets.size() > 0)
        fill("eventLogger_", 3.5, logged_ + 0.5, correctedJets[0].pt());
      if (correctedJets.size() > 1)
        fill("eventLogger_", 4.5, logged_ + 0.5, correctedJets[1].pt());
      if (correctedJets.size() > 2)
        fill("eventLogger_", 5.5, logged_ + 0.5, correctedJets[2].pt());
      if (correctedJets.size() > 3)
        fill("eventLogger_", 6.5, logged_ + 0.5, correctedJets[3].pt());
      fill("eventLogger_", 7.5, logged_ + 0.5, wMass);
      fill("eventLogger_", 8.5, logged_ + 0.5, topMass);
      ++logged_;
    }
  }
  if (multBPur != 0 && mMultIso == 1) {

    double mtW = eventKinematics.tmassWBoson(&mu, mET, TaggedJetCand);
    fill("MTWm_", mtW);
    double MTT = eventKinematics.tmassTopQuark(&mu, mET, TaggedJetCand);
    fill("mMTT_", MTT);
  }

  if (multBPur != 0 && eMultIso == 1) {
    double mtW = eventKinematics.tmassWBoson(&e, mET, TaggedJetCand);
    fill("MTWe_", mtW);
    double MTT = eventKinematics.tmassTopQuark(&e, mET, TaggedJetCand);
    fill("eMTT_", MTT);
  }
}
}

SingleTopTChannelLeptonDQM_miniAOD::SingleTopTChannelLeptonDQM_miniAOD(
    const edm::ParameterSet& cfg)
    : vertexSelect_(nullptr),
      beamspot_(""),
      beamspotSelect_(nullptr),
      muonStep_(nullptr),
      electronStep_(nullptr),
      pvStep_(nullptr),
      metStep_(nullptr) {
  jetSteps_.clear();


  // configure preselection
  edm::ParameterSet presel =
      cfg.getParameter<edm::ParameterSet>("preselection");
  if (presel.existsAs<edm::ParameterSet>("trigger")) {
    edm::ParameterSet trigger =
        presel.getParameter<edm::ParameterSet>("trigger");
    triggerTable__ = consumes<edm::TriggerResults>(
        trigger.getParameter<edm::InputTag>("src"));
    triggerPaths_ = trigger.getParameter<std::vector<std::string>>("select");
  }
  if (presel.existsAs<edm::ParameterSet>("vertex")) {
    edm::ParameterSet vertex = presel.getParameter<edm::ParameterSet>("vertex");
    vertex_ = vertex.getParameter<edm::InputTag>("src");
    vertex__ =
        consumes<reco::Vertex>(vertex.getParameter<edm::InputTag>("src"));
    vertexSelect_.reset(new StringCutObjectSelector<reco::Vertex>(
        vertex.getParameter<std::string>("select")));
  }
  if (presel.existsAs<edm::ParameterSet>("beamspot")) {
    edm::ParameterSet beamspot =
        presel.getParameter<edm::ParameterSet>("beamspot");
    beamspot_ = beamspot.getParameter<edm::InputTag>("src");
    beamspot__ =
        consumes<reco::BeamSpot>(beamspot.getParameter<edm::InputTag>("src"));
    beamspotSelect_.reset(new StringCutObjectSelector<reco::BeamSpot>(
        beamspot.getParameter<std::string>("select")));
  }
  // conifgure the selection
   std::vector<edm::ParameterSet> sel = 
      cfg.getParameter<std::vector<edm::ParameterSet>>("selection");

  for (unsigned int i = 0; i < sel.size(); ++i) {
    selectionOrder_.push_back(sel.at(i).getParameter<std::string>("label"));
    selection_[selectionStep(selectionOrder_.back())] = std::make_pair(
        sel.at(i),
        std::unique_ptr<SingleTopTChannelLepton_miniAOD::MonitorEnsemble>(
        new SingleTopTChannelLepton_miniAOD::MonitorEnsemble(
            selectionStep(selectionOrder_.back()).c_str(),
            cfg.getParameter<edm::ParameterSet>("setup"),
            cfg.getParameter<std::vector<edm::ParameterSet>>("selection"),
            consumesCollector())));
  }
  for (std::vector<std::string>::const_iterator selIt = selectionOrder_.begin();
       selIt != selectionOrder_.end(); ++selIt) {
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if (selection_.find(key) != selection_.end()) {
      using std::unique_ptr;

      if (type == "muons") {
        muonStep_.reset(new SelectionStep<pat::Muon>(selection_[key].first,
                                                     consumesCollector()));
      }
      if (type == "elecs") {
        electronStep_.reset(new SelectionStep<pat::Electron>(
            selection_[key].first, consumesCollector()));
      }

      if (type == "pvs") {
        pvStep_.reset(new SelectionStep<reco::Vertex>(selection_[key].first,
                                                     consumesCollector()));
      }
      if (type == "jets") {
        jetSteps_.push_back(std::unique_ptr<SelectionStep<pat::Jet>>(
            new SelectionStep<pat::Jet>(selection_[key].first,
                                         consumesCollector())));
      }
      if (type == "met") {
        metStep_.reset(new SelectionStep<pat::MET>(selection_[key].first,
                                                   consumesCollector()));
      }
    }
  }
}
void SingleTopTChannelLeptonDQM_miniAOD::bookHistograms(DQMStore::IBooker & ibooker,
  edm::Run const &, edm::EventSetup const & ){

  for (auto selIt = selection_.begin(); selIt != selection_.end(); ++selIt) {
    selIt->second.second->book(ibooker);
  }
}
void SingleTopTChannelLeptonDQM_miniAOD::analyze(const edm::Event& event,
                                         const edm::EventSetup& setup) {
  if (!triggerTable__.isUninitialized()) {
    edm::Handle<edm::TriggerResults> triggerTable;
    if (!event.getByToken(triggerTable__, triggerTable)) return;
    if (!accept(event, *triggerTable, triggerPaths_)) return;
  }
  if (!beamspot__.isUninitialized()) {
    edm::Handle<reco::BeamSpot> beamspot;
    if (!event.getByToken(beamspot__, beamspot)) return;
    if (!(*beamspotSelect_)(*beamspot)) return;
  }

  if (!vertex__.isUninitialized()) {
    edm::Handle<edm::View<reco::Vertex>> vertex;
    if (!event.getByToken(vertex__, vertex)) return;
    edm::View<reco::Vertex>::const_iterator pv = vertex->begin();
    if (!(*vertexSelect_)(*pv)) return;
  }

  // apply selection steps
  unsigned int passed = 0;
  unsigned int nJetSteps = -1;

  for (std::vector<std::string>::const_iterator selIt = selectionOrder_.begin();
       selIt != selectionOrder_.end(); ++selIt) {
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if (selection_.find(key) != selection_.end()) {
      if (type == "empty") {
        selection_[key].second->fill(event, setup);
      }
      if (type == "presel") {
        selection_[key].second->fill(event, setup);
      }
      if (type == "elecs" && electronStep_ != 0) {
        if (electronStep_->select(event)) {
          ++passed;
          selection_[key].second->fill(event, setup);
        } else
          break;
      }
      if (type == "muons" && muonStep_ != 0) {
        if (muonStep_->select(event)) {
          ++passed;
          selection_[key].second->fill(event, setup);
        } else
          break;
      }
      if (type == "jets") {
        nJetSteps++;
        if (jetSteps_[nJetSteps]) {
          if (jetSteps_[nJetSteps]->select(event, setup)) {
            ++passed;
            selection_[key].second->fill(event, setup);
          } else
            break;
        }
      }
      if (type == "met" && metStep_ != 0) {
        if (metStep_->select(event)) {
          ++passed;
          selection_[key].second->fill(event, setup);
        } else
          break;
      }
    }
  }
}
