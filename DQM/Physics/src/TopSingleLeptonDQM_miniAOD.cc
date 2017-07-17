#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DQM/Physics/src/TopSingleLeptonDQM_miniAOD.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <iostream>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
using namespace std;
namespace TopSingleLepton_miniAOD {

// maximal number of leading jets
// to be used for top mass estimate
static const unsigned int MAXJETS = 4;
// nominal mass of the W boson to
// be used for the top mass estimate
static const double WMASS = 80.4;

MonitorEnsemble::MonitorEnsemble(const char* label,
                                 const edm::ParameterSet& cfg,
                                 edm::ConsumesCollector&& iC)
    : label_(label),
      elecIso_(nullptr),
      elecSelect_(nullptr),
      pvSelect_(nullptr),
      muonIso_(nullptr),
      muonSelect_(nullptr),
      jetIDSelect_(nullptr),
     jetSelect(nullptr),
      includeBTag_(false),
      lowerEdge_(-1.),
      upperEdge_(-1.),
      logged_(0) {

  // sources have to be given; this PSet is not optional
  edm::ParameterSet sources = cfg.getParameter<edm::ParameterSet>("sources");
 // muons_ = iC.consumes<edm::View<reco::PFCandidate> >(
 //     sources.getParameter<edm::InputTag>("muons"));


  muons_ = iC.consumes<edm::View<pat::Muon> >(
    sources.getParameter<edm::InputTag>("muons"));


  elecs_ = iC.consumes<edm::View<pat::Electron> >(
      sources.getParameter<edm::InputTag>("elecs"));
  pvs_ = iC.consumes<edm::View<reco::Vertex> >(
      sources.getParameter<edm::InputTag>("pvs"));
  jets_ = iC.consumes<edm::View<pat::Jet> >(
      sources.getParameter<edm::InputTag>("jets"));
  for (edm::InputTag const& tag :
       sources.getParameter<std::vector<edm::InputTag> >("mets"))
    mets_.push_back(iC.consumes<edm::View<pat::MET> >(tag));
  // electronExtras are optional; they may be omitted or
  // empty
  if (cfg.existsAs<edm::ParameterSet>("elecExtras")) {


    edm::ParameterSet elecExtras =
        cfg.getParameter<edm::ParameterSet>("elecExtras");
    // select is optional; in case it's not found no
    // selection will be applied
    if (elecExtras.existsAs<std::string>("select")) {
      elecSelect_.reset( new StringCutObjectSelector<pat::Electron>(
          elecExtras.getParameter<std::string>("select")));
    }
    // isolation is optional; in case it's not found no
    // isolation will be applied
    if (elecExtras.existsAs<std::string>("isolation")) {
      elecIso_.reset( new StringCutObjectSelector<pat::Electron>(
          elecExtras.getParameter<std::string>("isolation")));
    }
    // electronId is optional; in case it's not found the
    // InputTag will remain empty
    if (elecExtras.existsAs<edm::ParameterSet>("electronId")) {
      edm::ParameterSet elecId =
          elecExtras.getParameter<edm::ParameterSet>("electronId");
      electronId_ = iC.consumes<edm::ValueMap<float> >(
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
  if (cfg.existsAs<edm::ParameterSet>("muonExtras")) {
    edm::ParameterSet muonExtras =
        cfg.getParameter<edm::ParameterSet>("muonExtras");
    // select is optional; in case it's not found no
    // selection will be applied
    if (muonExtras.existsAs<std::string>("select")) {
      muonSelect_.reset(new StringCutObjectSelector<pat::Muon>(
          muonExtras.getParameter<std::string>("select")));
    }
    // isolation is optional; in case it's not found no
    // isolation will be applied
    if (muonExtras.existsAs<std::string>("isolation")) {
      muonIso_.reset(new StringCutObjectSelector<pat::Muon>(
          muonExtras.getParameter<std::string>("isolation")));
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
      jetSelect.reset(new StringCutObjectSelector<pat::Jet> (jetSelect_));
    }
  }

  // triggerExtras are optional; they may be omitted or empty
  if (cfg.existsAs<edm::ParameterSet>("triggerExtras")) {
    edm::ParameterSet triggerExtras =
        cfg.getParameter<edm::ParameterSet>("triggerExtras");
    triggerTable_ = iC.consumes<edm::TriggerResults>(
        triggerExtras.getParameter<edm::InputTag>("src"));
    triggerPaths_ =
        triggerExtras.getParameter<std::vector<std::string> >("paths");
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
  // book(ibooker);
}

void MonitorEnsemble::book(DQMStore::IBooker & ibooker) {
  // set up the current directory path
  std::string current(directory_);
  current += label_;
  ibooker.setCurrentFolder(current);

  // determine number of bins for trigger monitoring
  unsigned int nPaths = triggerPaths_.size();

  // --- [STANDARD] --- //
  // Run Number
  hists_["RunNumb_"] = ibooker.book1D("RunNumber", "Run Nr.", 1.e4, 1.5e5, 3.e5);
  // instantaneous luminosity
  hists_["InstLumi_"] = ibooker.book1D("InstLumi", "Inst. Lumi.", 100, 0., 1.e3);
  // number of selected primary vertices
  hists_["pvMult_"] = ibooker.book1D("PvMult", "N_{pvs}", 100, 0., 100.);
  // pt of the leading muon
  hists_["muonPt_"] = ibooker.book1D("MuonPt", "pt(#mu)", 50, 0., 250.);
  // muon multiplicity before std isolation
  hists_["muonMult_"] = ibooker.book1D("MuonMult", "N_{All}(#mu)", 10, 0., 10.);
  // muon multiplicity after  std isolation
  hists_["muonMultIso_"] = ibooker.book1D("MuonMultIso",
      "N_{Iso}(#mu)", 10, 0., 10.);
  // pt of the leading electron
  hists_["elecPt_"] = ibooker.book1D("ElecPt", "pt(e)", 50, 0., 250.);
  // electron multiplicity before std isolation
  hists_["elecMult_"] = ibooker.book1D("ElecMult", "N_{All}(e)", 10, 0., 10.);
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
  // MET (calo)
  hists_["slimmedMETs_"] = ibooker.book1D("slimmedMETs", "MET_{slimmed}", 50, 0., 200.);
  // W mass estimate
  hists_["massW_"] = ibooker.book1D("MassW", "M(W)", 60, 0., 300.);
  // Top mass estimate
  hists_["massTop_"] = ibooker.book1D("MassTop", "M(Top)", 50, 0., 500.);
  // b-tagged Top mass
  hists_["massBTop_"] = ibooker.book1D("MassBTop", "M(Top, 1 b-tag)", 50, 0., 500.);
  // set bin labels for trigger monitoring
  triggerBinLabels(std::string("trigger"), triggerPaths_);

  if (verbosity_ == STANDARD) return;

  // --- [VERBOSE] --- //
  // eta of the leading muon
  hists_["muonEta_"] = ibooker.book1D("MuonEta", "#eta(#mu)", 30, -3., 3.);
  // relative isolation of the candidate muon (depending on the decay channel)
  hists_["muonRelIso_"] = ibooker.book1D(
      "MuonRelIso", "Iso_{Rel}(#mu) (#Delta#beta Corrected)", 50, 0., 1.);
  // eta of the leading electron
  hists_["elecEta_"] = ibooker.book1D("ElecEta", "#eta(e)", 30, -3., 3.);
  // std isolation variable of the leading electron
  hists_["elecRelIso_"] = ibooker.book1D("ElecRelIso", "Iso_{Rel}(e)", 50, 0., 1.);
  // multiplicity of btagged jets (for track counting high efficiency) with
  // pt(L2L3)>30
  hists_["jetMultBEff_"] = ibooker.book1D("JetMultBEff",
      "N_{30}(TCHE)", 10, 0., 10.);
  // btag discriminator for track counting high efficiency for jets with
  // pt(L2L3)>30
  hists_["jetBDiscEff_"] = ibooker.book1D("JetBDiscEff",
      "Disc_{TCHE}(jet)", 100, 0., 10.);
  // eta of the 1. leading jet (corrected to L2+L3)
  hists_["jet1Eta_"] = ibooker.book1D("Jet1Eta", "#eta_{L2L3}(jet1)", 60, -3., 3.);
  // pt of the 1. leading jet (corrected to L2+L3)
  hists_["jet1Pt_"] = ibooker.book1D("Jet1Pt", "pt_{L2L3}(jet1)", 60, 0., 300.);
  // eta of the 2. leading jet (corrected to L2+L3)
  hists_["jet2Eta_"] = ibooker.book1D("Jet2Eta", "#eta_{L2L3}(jet2)", 60, -3., 3.);
  // pt of the 2. leading jet (corrected to L2+L3)
  hists_["jet2Pt_"] = ibooker.book1D("Jet2Pt", "pt_{L2L3}(jet2)", 60, 0., 300.);
  // eta of the 3. leading jet (corrected to L2+L3)
  hists_["jet3Eta_"] = ibooker.book1D("Jet3Eta", "#eta_{L2L3}(jet3)", 60, -3., 3.);
  // pt of the 3. leading jet (corrected to L2+L3)
  hists_["jet3Pt_"] = ibooker.book1D("Jet3Pt", "pt_{L2L3}(jet3)", 60, 0., 300.);
  // eta of the 4. leading jet (corrected to L2+L3)
  hists_["jet4Eta_"] = ibooker.book1D("Jet4Eta", "#eta_{L2L3}(jet4)", 60, -3., 3.);
  // pt of the 4. leading jet (corrected to L2+L3)
  hists_["jet4Pt_"] = ibooker.book1D("Jet4Pt", "pt_{L2L3}(jet4)", 60, 0., 300.);
  // MET (tc)
  hists_["slimmedMETsNoHF_"] = ibooker.book1D("slimmedMETsNoHF", "MET_{slimmedNoHF}", 50, 0., 200.);
  // MET (pflow)
  hists_["slimmedMETsPuppi_"] = ibooker.book1D("slimmedMETsPuppi", "MET_{slimmedPuppi}", 50, 0., 200.);
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
  // charged hadron isolation component of the candidate muon (depending on the
  // decay channel)
  hists_["muonChHadIso_"] = ibooker.book1D("MuonChHadIsoComp",
      "ChHad_{IsoComponent}(#mu)", 50, 0., 5.);
  // neutral hadron isolation component of the candidate muon (depending on the
  // decay channel)
  hists_["muonNeHadIso_"] = ibooker.book1D("MuonNeHadIsoComp",
      "NeHad_{IsoComponent}(#mu)", 50, 0., 5.);
  // photon isolation component of the candidate muon (depending on the decay
  // channel)
  hists_["muonPhIso_"] = ibooker.book1D("MuonPhIsoComp",
      "Photon_{IsoComponent}(#mu)", 50, 0., 5.);
  // charged hadron isolation component of the candidate electron (depending on
  // the decay channel)
  hists_["elecChHadIso_"] = ibooker.book1D("ElectronChHadIsoComp",
      "ChHad_{IsoComponent}(e)", 50, 0., 5.);
  // neutral hadron isolation component of the candidate electron (depending on
  // the decay channel)
  hists_["elecNeHadIso_"] = ibooker.book1D("ElectronNeHadIsoComp",
      "NeHad_{IsoComponent}(e)", 50, 0., 5.);
  // photon isolation component of the candidate electron (depending on the
  // decay channel)
  hists_["elecPhIso_"] = ibooker.book1D("ElectronPhIsoComp",
      "Photon_{IsoComponent}(e)", 50, 0., 5.);
  // multiplicity of btagged jets (for track counting high purity) with
  // pt(L2L3)>30
  hists_["jetMultBPur_"] = ibooker.book1D("JetMultBPur",
      "N_{30}(TCHP)", 10, 0., 10.);
  // btag discriminator for track counting high purity
  hists_["jetBDiscPur_"] = ibooker.book1D("JetBDiscPur",
      "Disc_{TCHP}(Jet)", 100, 0., 10.);
  // multiplicity of btagged jets (for simple secondary vertex) with pt(L2L3)>30
  hists_["jetMultBVtx_"] = ibooker.book1D("JetMultBVtx",
      "N_{30}(SSVHE)", 10, 0., 10.);
  // btag discriminator for simple secondary vertex
  hists_["jetBDiscVtx_"] = ibooker.book1D("JetBDiscVtx",
      "Disc_{SSVHE}(Jet)", 35, -1., 6.);
  // multiplicity for combined secondary vertex
  hists_["jetMultCSVtx_"] = ibooker.book1D("JetMultCSV", "N_{30}(CSV)", 10, 0., 10.);
  // btag discriminator for combined secondary vertex
  hists_["jetBCVtx_"] = ibooker.book1D("JetDiscCSV",
      "Disc_{CSV}(JET)", 100, -1., 2.);
  // pt of the 1. leading jet (uncorrected)
  hists_["jet1PtRaw_"] = ibooker.book1D("Jet1PtRaw", "pt_{Raw}(jet1)", 60, 0., 300.);
  // pt of the 2. leading jet (uncorrected)
  hists_["jet2PtRaw_"] = ibooker.book1D("Jet2PtRaw", "pt_{Raw}(jet2)", 60, 0., 300.);
  // pt of the 3. leading jet (uncorrected)
  hists_["jet3PtRaw_"] = ibooker.book1D("Jet3PtRaw", "pt_{Raw}(jet3)", 60, 0., 300.);
  // pt of the 4. leading jet (uncorrected)
  hists_["jet4PtRaw_"] = ibooker.book1D("Jet4PtRaw", "pt_{Raw}(jet4)", 60, 0., 300.);
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
  // fill monitoring plots for primary verices
  edm::Handle<edm::View<reco::Vertex> > pvs;
  if (!event.getByToken(pvs_, pvs)) return;
  unsigned int pvMult = 0;
  for (edm::View<reco::Vertex>::const_iterator pv = pvs->begin();
       pv != pvs->end(); ++pv) {
    if (!pvSelect_ || (*pvSelect_)(*pv)) pvMult++;
  }
  fill("pvMult_", pvMult);

  /*
  ------------------------------------------------------------

  Run and Inst. Luminosity information (Inst. Lumi. filled now with a dummy
  value=5.0)

  ------------------------------------------------------------
  */
  if (!event.eventAuxiliary().run()) return;
  fill("RunNumb_", event.eventAuxiliary().run());

  double dummy = 5.;
  fill("InstLumi_", dummy);

  /*
  ------------------------------------------------------------

  Electron Monitoring

  ------------------------------------------------------------
  */

  // fill monitoring plots for electrons
  edm::Handle<edm::View<pat::Electron> > elecs;
  if (!event.getByToken(elecs_, elecs)) return;

  // check availability of electron id
  edm::Handle<edm::ValueMap<float> > electronId;
  if (!electronId_.isUninitialized()) {
    if (!event.getByToken(electronId_, electronId)) return;
  }

  // loop electron collection
  unsigned int eMult = 0, eMultIso = 0;
  std::vector<const pat::Electron*> isoElecs;


 for (edm::View<pat::Electron>::const_iterator elec = elecs->begin();
       elec != elecs->end(); ++elec) {

      if(true){//no electron id applied yet!
      if (!elecSelect_ || (*elecSelect_)(*elec)) {

        double el_ChHadIso = elec->pfIsolationVariables().sumChargedHadronPt;
        double el_NeHadIso = elec->pfIsolationVariables().sumNeutralHadronEt;
        double el_PhIso = elec->pfIsolationVariables().sumPhotonEt;
        double el_pfRelIso =
            (el_ChHadIso +
             max(0., el_NeHadIso + el_PhIso -
                         0.5 * elec->pfIsolationVariables().sumPUPt)) /
            elec->pt();
        if (eMult == 0) {
          // restrict to the leading electron
          fill("elecPt_", elec->pt());
          fill("elecEta_", elec->eta());
          fill("elecRelIso_", el_pfRelIso);
          fill("elecChHadIso_", el_ChHadIso);
          fill("elecNeHadIso_", el_NeHadIso);
          fill("elecPhIso_", el_PhIso);
        }

        ++eMult;
        if (!elecIso_ || (*elecIso_)(*elec)) {
          isoElecs.push_back(&(*elec));
          ++eMultIso;
        }
      }
    }
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

  edm::Handle<edm::View<pat::Muon> > muons;
  edm::View<pat::Muon>::const_iterator muonit;

  if (!event.getByToken(muons_, muons)) return;



  for (edm::View<pat::Muon>::const_iterator muon = muons->begin();
       muon != muons->end(); ++muon) {


    // restrict to globalMuons
    if (muon->isGlobalMuon()) {
      fill("muonDelZ_", muon->innerTrack()->vz());  // CB using inner track!
      fill("muonDelXY_", muon->innerTrack()->vx(), muon->innerTrack()->vy());



      // apply preselection
      if (!muonSelect_ || (*muonSelect_)(*muon)) {
 

        double chHadPt = muon->pfIsolationR04().sumChargedHadronPt;
        double neHadEt = muon->pfIsolationR04().sumNeutralHadronEt;
        double phoEt = muon->pfIsolationR04().sumPhotonEt;

        double pfRelIso =
            (chHadPt +
             max(0., neHadEt + phoEt - 0.5 * muon->pfIsolationR04().sumPUPt)) /
            muon->pt();  // CB dBeta corrected iso!

        if (mMult == 0) {
          // restrict to leading muon
          fill("muonPt_", muon->pt());
          fill("muonEta_", muon->eta());

          fill("muonRelIso_", pfRelIso);

          fill("muonChHadIso_", chHadPt);
          fill("muonNeHadIso_", neHadEt);
          fill("muonPhIso_", phoEt);
        }
        ++mMult;
        if (!muonIso_ || (*muonIso_)(*muon)) ++mMultIso;
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
  std::vector<double> JetTagValues;
  unsigned int mult = 0, multBEff = 0, multBPur = 0, multBVtx = 0, multCSV = 0;

  edm::Handle<edm::View<pat::Jet> > jets;
  if (!event.getByToken(jets_, jets)) {
    return;
  }

  for (edm::View<pat::Jet>::const_iterator jet = jets->begin();
       jet != jets->end(); ++jet) {
    // check jetID for calo jets
    unsigned int idx = jet - jets->begin();

      pat::Jet sel = *jet;

      if(!(*jetSelect)(sel))continue;
//      if (!jetSelect(sel)) continue;

    // prepare jet to fill monitor histograms
    pat::Jet monitorJet = *jet;

    correctedJets.push_back(monitorJet);
    ++mult;  // determine jet multiplicity

      fill("jetBDiscEff_", monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags")); //hard coded discriminator and value right now.
      if (monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") > 0.89) ++multBEff;


      // Fill a vector with Jet b-tag WP for later M3+1tag calculation: CSV
      // tagger
      JetTagValues.push_back(monitorJet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags"));
//    }
    // fill pt (raw or L2L3) for the leading four jets
    if (idx == 0) {
      fill("jet1Pt_", monitorJet.pt());
      fill("jet1PtRaw_", jet->pt());
      fill("jet1Eta_", monitorJet.eta());
    };
    if (idx == 1) {
      fill("jet2Pt_", monitorJet.pt());
      fill("jet2PtRaw_", jet->pt());
      fill("jet2Eta_", monitorJet.eta());
    }
    if (idx == 2) {
      fill("jet3Pt_", monitorJet.pt());
      fill("jet3PtRaw_", jet->pt());
      fill("jet3Eta_", monitorJet.eta());
    }
    if (idx == 3) {
      fill("jet4Pt_", monitorJet.pt());
      fill("jet4PtRaw_", jet->pt());
      fill("jet4Eta_", monitorJet.eta());
    }
  }
  fill("jetMult_", mult);
  fill("jetMultBEff_", multBEff);
  fill("jetMultBPur_", multBPur);
  fill("jetMultBVtx_", multBVtx);
  fill("jetMultCSVtx_", multCSV);

  /*
  ------------------------------------------------------------

  MET Monitoring

  ------------------------------------------------------------
  */

  // fill monitoring histograms for met
  for (std::vector<edm::EDGetTokenT<edm::View<pat::MET> > >::const_iterator
           met_ = mets_.begin();
       met_ != mets_.end(); ++met_) {
    edm::Handle<edm::View<pat::MET> > met;
    if (!event.getByToken(*met_, met)) continue;
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

  // Fill M3 with Btag (CSV Tight) requirement

 // if (!includeBTag_) return;
  if (correctedJets.size() != JetTagValues.size()) return;
  double btopMass =
      eventKinematics.massBTopQuark(correctedJets, JetTagValues, 0.89); //hard coded CSVv2 value
  
  if (btopMass >= 0) fill("massBTop_", btopMass);

}
}

TopSingleLeptonDQM_miniAOD::TopSingleLeptonDQM_miniAOD(const edm::ParameterSet& cfg)
    : vertexSelect_(nullptr),
      beamspot_(""),
      beamspotSelect_(nullptr),
      MuonStep(nullptr),
      ElectronStep(nullptr),
      PvStep(nullptr),
      METStep(nullptr) {
  JetSteps.clear();

  // configure preselection
  edm::ParameterSet presel =
      cfg.getParameter<edm::ParameterSet>("preselection");
  if (presel.existsAs<edm::ParameterSet>("trigger")) {
    edm::ParameterSet trigger =
        presel.getParameter<edm::ParameterSet>("trigger");
    triggerTable__ = consumes<edm::TriggerResults>(
        trigger.getParameter<edm::InputTag>("src"));
    triggerPaths_ = trigger.getParameter<std::vector<std::string> >("select");
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
  sel_ = cfg.getParameter<std::vector<edm::ParameterSet> >("selection");
  setup_ = cfg.getParameter<edm::ParameterSet>("setup");
  for (unsigned int i = 0; i < sel_.size(); ++i) {
    selectionOrder_.push_back(sel_.at(i).getParameter<std::string>("label"));
    selection_[selectionStep(selectionOrder_.back())] = std::make_pair(
        sel_.at(i),
        std::unique_ptr<TopSingleLepton_miniAOD::MonitorEnsemble>(
          new TopSingleLepton_miniAOD::MonitorEnsemble(
            selectionStep(selectionOrder_.back()).c_str(),
            setup_, consumesCollector())));
  }
  for (std::vector<std::string>::const_iterator selIt = selectionOrder_.begin();
       selIt != selectionOrder_.end(); ++selIt) {
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if (selection_.find(key) != selection_.end()) {
      if (type == "muons") {
        MuonStep.reset(new SelectionStep<pat::Muon>(selection_[key].first,
                                                        consumesCollector()));
      }
      if (type == "elecs") {
        ElectronStep.reset(new SelectionStep<pat::Electron>(
            selection_[key].first, consumesCollector()));
      }
      if (type == "pvs") {
        PvStep.reset(new SelectionStep<reco::Vertex>(selection_[key].first,
                                                 consumesCollector()));
      }
      if (type == "jets") {
        JetSteps.push_back(std::unique_ptr<SelectionStep<pat::Jet>>(
            new SelectionStep<pat::Jet>(selection_[key].first, consumesCollector())));
      }

      if (type == "met") {
        METStep.reset(new SelectionStep<pat::MET>(selection_[key].first,
                                               consumesCollector()));
      }
    }
  }
}
void TopSingleLeptonDQM_miniAOD::bookHistograms(DQMStore::IBooker & ibooker,
  edm::Run const &, edm::EventSetup const & ){

  for (auto selIt = selection_.begin(); selIt != selection_.end(); ++selIt) {
    selIt->second.second->book(ibooker);
  }
}
void TopSingleLeptonDQM_miniAOD::analyze(const edm::Event& event,
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

  unsigned int passed = 0;
  unsigned int nJetSteps = -1;

  for (std::vector<std::string>::const_iterator selIt = selectionOrder_.begin();
       selIt != selectionOrder_.end(); ++selIt) {
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if (selection_.find(key) != selection_.end()) {
      if (type == "empty") {
        selection_[key].second->fill(event, setup);
      }
      if (type == "muons" && MuonStep != 0) {
        if (MuonStep->select(event)) {
          ++passed;

          selection_[key].second->fill(event, setup);
        } else
          break;
      }

      if (type == "elecs" && ElectronStep != 0) {

        if (ElectronStep->select(event)) {
          ++passed;
          selection_[key].second->fill(event, setup);
        } else
          break;
      }

      if (type == "pvs" && PvStep != 0) {
        if (PvStep->selectVertex(event)) {
          ++passed;
          selection_[key].second->fill(event, setup);
        } else
          break;
      }

      if (type == "jets") {
        nJetSteps++;
        if (JetSteps[nJetSteps] != NULL) {
          if (JetSteps[nJetSteps]->select(event, setup)) {
            ++passed;
            selection_[key].second->fill(event, setup);
          } else
            break;
        }
      }

      if (type == "met" && METStep != 0) {
        if (METStep->select(event)) {
          ++passed;
          selection_[key].second->fill(event, setup);
        } else
          break;
      }
    }
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
