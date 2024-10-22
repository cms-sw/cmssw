#include "DQM/Physics/src/TopSingleLeptonDQM.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include <iostream>
#include <memory>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

using namespace std;
namespace TopSingleLepton {

  // maximal number of leading jets
  // to be used for top mass estimate
  static const unsigned int MAXJETS = 4;
  // nominal mass of the W boson to
  // be used for the top mass estimate
  static const double WMASS = 80.4;

  MonitorEnsemble::MonitorEnsemble(const char* label, const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : label_(label),
        elecSelect_(nullptr),
        pvSelect_(nullptr),
        muonIso_(nullptr),
        muonSelect_(nullptr),
        jetIDSelect_(nullptr),
        jetlooseSelection_(nullptr),
        jetSelection_(nullptr),
        includeBTag_(false),
        lowerEdge_(-1.),
        upperEdge_(-1.),
        logged_(0) {
    // sources have to be given; this PSet is not optional
    edm::ParameterSet sources = cfg.getParameter<edm::ParameterSet>("sources");
    muons_ = iC.consumes<edm::View<reco::PFCandidate>>(sources.getParameter<edm::InputTag>("muons"));
    elecs_ = iC.consumes<edm::View<reco::PFCandidate>>(sources.getParameter<edm::InputTag>("elecs"));
    pvs_ = iC.consumes<edm::View<reco::Vertex>>(sources.getParameter<edm::InputTag>("pvs"));
    jets_ = iC.consumes<edm::View<reco::Jet>>(sources.getParameter<edm::InputTag>("jets"));
    for (edm::InputTag const& tag : sources.getParameter<std::vector<edm::InputTag>>("mets"))
      mets_.push_back(iC.consumes<edm::View<reco::MET>>(tag));
    // electronExtras are optional; they may be omitted or
    // empty
    if (cfg.existsAs<edm::ParameterSet>("elecExtras")) {
      // rho for PF isolation with EA corrections
      // eventrhoToken_ =
      // iC.consumes<double>(edm::InputTag("fixedGridRhoFastjetAll"));

      edm::ParameterSet elecExtras = cfg.getParameter<edm::ParameterSet>("elecExtras");
      // select is optional; in case it's not found no
      // selection will be applied
      if (elecExtras.existsAs<std::string>("select")) {
        elecSelect_ = std::make_unique<StringCutObjectSelector<reco::PFCandidate>>(
            elecExtras.getParameter<std::string>("select"));
      }

      if (elecExtras.existsAs<std::string>("rho")) {
        rhoTag = elecExtras.getParameter<edm::InputTag>("rho");
      }
      // electronId is optional; in case it's not found the
      // InputTag will remain empty
      if (elecExtras.existsAs<edm::ParameterSet>("electronId")) {
        edm::ParameterSet elecId = elecExtras.getParameter<edm::ParameterSet>("electronId");
        electronId_ = iC.consumes<edm::ValueMap<float>>(elecId.getParameter<edm::InputTag>("src"));
        eidCutValue_ = elecId.getParameter<double>("cutValue");
      }
    }
    // pvExtras are optional; they may be omitted or empty
    if (cfg.existsAs<edm::ParameterSet>("pvExtras")) {
      edm::ParameterSet pvExtras = cfg.getParameter<edm::ParameterSet>("pvExtras");
      // select is optional; in case it's not found no
      // selection will be applied
      if (pvExtras.existsAs<std::string>("select")) {
        pvSelect_ =
            std::make_unique<StringCutObjectSelector<reco::Vertex>>(pvExtras.getParameter<std::string>("select"));
      }
    }
    // muonExtras are optional; they may be omitted or empty
    if (cfg.existsAs<edm::ParameterSet>("muonExtras")) {
      edm::ParameterSet muonExtras = cfg.getParameter<edm::ParameterSet>("muonExtras");
      // select is optional; in case it's not found no
      // selection will be applied
      if (muonExtras.existsAs<std::string>("select")) {
        muonSelect_ = std::make_unique<StringCutObjectSelector<reco::PFCandidate>>(
            muonExtras.getParameter<std::string>("select"));
      }
      // isolation is optional; in case it's not found no
      // isolation will be applied
      if (muonExtras.existsAs<std::string>("isolation")) {
        muonIso_ = std::make_unique<StringCutObjectSelector<reco::PFCandidate>>(
            muonExtras.getParameter<std::string>("isolation"));
      }
    }

    // jetExtras are optional; they may be omitted or
    // empty
    if (cfg.existsAs<edm::ParameterSet>("jetExtras")) {
      edm::ParameterSet jetExtras = cfg.getParameter<edm::ParameterSet>("jetExtras");
      // jetCorrector is optional; in case it's not found
      // the InputTag will remain empty
      if (jetExtras.existsAs<std::string>("jetCorrector")) {
        jetCorrector_ =
            iC.consumes<reco::JetCorrector>(edm::InputTag(jetExtras.getParameter<std::string>("jetCorrector")));
      }
      // read jetID information if it exists
      if (jetExtras.existsAs<edm::ParameterSet>("jetID")) {
        edm::ParameterSet jetID = jetExtras.getParameter<edm::ParameterSet>("jetID");
        jetIDLabel_ = iC.consumes<reco::JetIDValueMap>(jetID.getParameter<edm::InputTag>("label"));
        jetIDSelect_ =
            std::make_unique<StringCutObjectSelector<reco::JetID>>(jetID.getParameter<std::string>("select"));
      }
      // select is optional; in case it's not found no
      // selection will be applied (only implemented for
      // CaloJets at the moment)
      if (jetExtras.existsAs<std::string>("select")) {
        jetSelect_ = jetExtras.getParameter<std::string>("select");
        jetSelection_ = std::make_unique<StringCutObjectSelector<reco::PFJet>>(jetSelect_);
        jetlooseSelection_ = std::make_unique<StringCutObjectSelector<reco::PFJet>>(
            "chargedHadronEnergyFraction()>0 && chargedMultiplicity()>0 && chargedEmEnergyFraction()<0.99 && "
            "neutralHadronEnergyFraction()<0.99 && neutralEmEnergyFraction()<0.99 && "
            "(chargedMultiplicity()+neutralMultiplicity())>1");
      }
      // jetBDiscriminators are optional; in case they are
      // not found the InputTag will remain empty; they
      // consist of pairs of edm::JetFlavorAssociation's &
      // corresponding working points
      includeBTag_ = jetExtras.existsAs<edm::ParameterSet>("jetBTaggers");
      if (includeBTag_) {
        edm::ParameterSet btagCSV =
            jetExtras.getParameter<edm::ParameterSet>("jetBTaggers").getParameter<edm::ParameterSet>("cvsVertex");
        btagCSV_ = iC.consumes<reco::JetTagCollection>(btagCSV.getParameter<edm::InputTag>("label"));
        btagCSVWP_ = btagCSV.getParameter<double>("workingPoint");
      }
    }

    // triggerExtras are optional; they may be omitted or empty
    if (cfg.existsAs<edm::ParameterSet>("triggerExtras")) {
      edm::ParameterSet triggerExtras = cfg.getParameter<edm::ParameterSet>("triggerExtras");
      triggerTable_ = iC.consumes<edm::TriggerResults>(triggerExtras.getParameter<edm::InputTag>("src"));
      triggerPaths_ = triggerExtras.getParameter<std::vector<std::string>>("paths");
    }

    // massExtras is optional; in case it's not found no mass
    // window cuts are applied for the same flavor monitor
    // histograms
    if (cfg.existsAs<edm::ParameterSet>("massExtras")) {
      edm::ParameterSet massExtras = cfg.getParameter<edm::ParameterSet>("massExtras");
      lowerEdge_ = massExtras.getParameter<double>("lowerEdge");
      upperEdge_ = massExtras.getParameter<double>("upperEdge");
    }

    // setup the verbosity level for booking histograms;
    // per default the verbosity level will be set to
    // STANDARD. This will also be the chosen level in
    // the case when the monitoring PSet is not found
    verbosity_ = STANDARD;
    if (cfg.existsAs<edm::ParameterSet>("monitoring")) {
      edm::ParameterSet monitoring = cfg.getParameter<edm::ParameterSet>("monitoring");
      if (monitoring.getParameter<std::string>("verbosity") == "DEBUG")
        verbosity_ = DEBUG;
      if (monitoring.getParameter<std::string>("verbosity") == "VERBOSE")
        verbosity_ = VERBOSE;
      if (monitoring.getParameter<std::string>("verbosity") == "STANDARD")
        verbosity_ = STANDARD;
    }
    // and don't forget to do the histogram booking
    directory_ = cfg.getParameter<std::string>("directory");
  }

  void MonitorEnsemble::book(DQMStore::IBooker& ibooker) {
    // set up the current directory path
    std::string current(directory_);
    current += label_;
    ibooker.setCurrentFolder(current);

    // determine number of bins for trigger monitoring
    // unsigned int nPaths = triggerPaths_.size();

    // --- [STANDARD] --- //
    // Run Number
    // hists_["RunNumb_"] = ibooker.book1D("RunNumber", "Run Nr.", 10, 0, 10);
    // instantaneous luminosity
    // hists_["InstLumi_"] = ibooker.book1D("InstLumi", "Inst. Lumi.", 100, 0., 1.e3);
    // number of selected primary vertices
    hists_["pvMult_"] = ibooker.book1D("PvMult", "N_{good pvs}", 50, 0., 50.);
    // pt of the leading muon
    hists_["muonPt_"] = ibooker.book1D("MuonPt", "pt(#mu TightId, TightIso)", 40, 0., 200.);
    // muon multiplicity after  tight Id
    hists_["muonMult_"] = ibooker.book1D("MuonMult", "N_{loose}(#mu)", 10, 0., 10.);
    // muon multiplicity after  isolation
    //hists_["muonMultIso_"] = ibooker.book1D("MuonMultIso",
    //    "N_{TightIso}(#mu)", 10, 0., 10.);
    // muon multiplicity after  tight Id and isolation
    hists_["muonMultTight_"] = ibooker.book1D("MuonMultTight", "N_{TightIso,TightId}(#mu)", 10, 0., 10.);
    // pt of the leading electron
    hists_["elecPt_"] = ibooker.book1D("ElecPt", "pt(e TightId, TightIso)", 40, 0., 200.);
    // electron multiplicity before std isolation
    //hists_["elecMult_"] = ibooker.book1D("ElecMult", "N_{All}(e, tightId)", 10, 0., 10.);
    // electron multiplicity after  std isolation
    //hists_["elecMultIso_"] = ibooker.book1D("ElecMultIso", "N_{Iso}(e, tightId, tightIso)", 10, 0., 10.);
    // multiplicity of jets with pt>30
    hists_["jetMult_"] = ibooker.book1D("JetMult", "N_{30}(jet)", 10, 0., 10.);
    // multiplicity of loose jets with pt>30
    hists_["jetMultLoose_"] = ibooker.book1D("JetMultLoose", "N_{30, loose}(jet)", 10, 0., 10.);

    // trigger efficiency estimates for single lepton triggers
    // hists_["triggerEff_"] = ibooker.book1D("TriggerEff",
    //    "Eff(trigger)", nPaths, 0., nPaths);
    // monitored trigger occupancy for single lepton triggers
    // hists_["triggerMon_"] = ibooker.book1D("TriggerMon",
    //    "Mon(trigger)", nPaths, 0., nPaths);
    // MET (pflow)
    hists_["metPflow_"] = ibooker.book1D("METPflow", "MET_{Pflow}", 50, 0., 200.);
    // W mass estimate
    hists_["massW_"] = ibooker.book1D("MassW", "M(W)", 60, 0., 300.);
    // Top mass estimate
    hists_["massTop_"] = ibooker.book1D("MassTop", "M(Top)", 50, 0., 500.);
    // b-tagged Top mass
    hists_["massBTop_"] = ibooker.book1D("MassBTop", "M(Top, 1 b-tag)", 50, 0., 500.);
    // set bin labels for trigger monitoring
    triggerBinLabels(std::string("trigger"), triggerPaths_);

    if (verbosity_ == STANDARD)
      return;

    // --- [VERBOSE] --- //
    // eta of the leading muon
    hists_["muonEta_"] = ibooker.book1D("MuonEta", "#eta(#mu TightId, TightIso)", 30, -3., 3.);
    // relative isolation of the candidate muon (depending on the decay channel)
    hists_["muonRelIso_"] = ibooker.book1D("MuonRelIso", "Iso_{Rel}(#mu TightId) (#Delta#beta Corrected)", 50, 0., 1.);
    // phi of the leading muon
    hists_["muonPhi_"] = ibooker.book1D("MuonPhi", "#phi(#mu TightId, TightIso)", 40, -4., 4.);
    // eta of the leading electron
    hists_["elecEta_"] = ibooker.book1D("ElecEta", "#eta(e tightId)", 30, -3., 3.);
    // std isolation variable of the leading electron
    hists_["elecRelIso_"] = ibooker.book1D("ElecRelIso", "Iso_{Rel}(e TightId, TightIso)", 50, 0., 1.);
    // phi of the leading electron
    hists_["elecPhi_"] = ibooker.book1D("ElecPhi", "#phi(e tightId)", 40, -4., 4.);
    // multiplicity of tight Id, tight Iso electorns
    hists_["elecMultTight_"] = ibooker.book1D("ElecMultTight", "N_{TightIso,TightId}(e)", 10, 0., 10.);
    // multiplicity of btagged jets (for track counting high efficiency) with
    // pt(L2L3)>30
    // hists_["jetMultBEff_"] = ibooker.book1D("JetMultBEff",
    //    "N_{30}(TCHE)", 10, 0., 10.);
    // btag discriminator for track counting high efficiency for jets with
    // pt(L2L3)>30
    //hists_["jetBDiscEff_"] = ibooker.book1D("JetBDiscEff",
    //    "Disc_{TCHE}(jet)", 100, 0., 10.);
    // eta of the 1. leading jet (corrected to L2+L3)
    hists_["jet1Eta_"] = ibooker.book1D("Jet1Eta", "#eta_{30,loose}(jet1)", 60, -3., 3.);
    // pt of the 1. leading jet (corrected to L2+L3)
    hists_["jet1Pt_"] = ibooker.book1D("Jet1Pt", "pt_{30,loose}(jet1)", 60, 0., 300.);
    // eta of the 2. leading jet (corrected to L2+L3)
    hists_["jet2Eta_"] = ibooker.book1D("Jet2Eta", "#eta_{30,loose}(jet2)", 60, -3., 3.);
    // pt of the 2. leading jet (corrected to L2+L3)
    hists_["jet2Pt_"] = ibooker.book1D("Jet2Pt", "pt_{30,loose}(jet2)", 60, 0., 300.);
    // eta of the 3. leading jet (corrected to L2+L3)
    hists_["jet3Eta_"] = ibooker.book1D("Jet3Eta", "#eta_{30,loose}(jet3)", 60, -3., 3.);
    // pt of the 3. leading jet (corrected to L2+L3)
    hists_["jet3Pt_"] = ibooker.book1D("Jet3Pt", "pt_{30,loose}(jet3)", 60, 0., 300.);
    // eta of the 4. leading jet (corrected to L2+L3)
    hists_["jet4Eta_"] = ibooker.book1D("Jet4Eta", "#eta_{30,loose}(jet4)", 60, -3., 3.);
    // pt of the 4. leading jet (corrected to L2+L3)
    hists_["jet4Pt_"] = ibooker.book1D("Jet4Pt", "pt_{30,loose}(jet4)", 60, 0., 300.);
    // dz for muons (to suppress cosmis)
    hists_["muonDelZ_"] = ibooker.book1D("MuonDelZ", "d_{z}(#mu)", 50, -25., 25.);
    // dxy for muons (to suppress cosmics)
    hists_["muonDelXY_"] = ibooker.book2D("MuonDelXY", "d_{xy}(#mu)", 50, -0.1, 0.1, 50, -0.1, 0.1);

    // set axes titles for dxy for muons
    hists_["muonDelXY_"]->setAxisTitle("x [cm]", 1);
    hists_["muonDelXY_"]->setAxisTitle("y [cm]", 2);

    if (verbosity_ == VERBOSE)
      return;

    // --- [DEBUG] --- //
    // charged hadron isolation component of the candidate muon (depending on the
    // decay channel)
    hists_["muonChHadIso_"] = ibooker.book1D("MuonChHadIsoComp", "ChHad_{IsoComponent}(#mu TightId)", 50, 0., 5.);
    // neutral hadron isolation component of the candidate muon (depending on the
    // decay channel)
    hists_["muonNeHadIso_"] = ibooker.book1D("MuonNeHadIsoComp", "NeHad_{IsoComponent}(#mu TightId)", 50, 0., 5.);
    // photon isolation component of the candidate muon (depending on the decay
    // channel)
    hists_["muonPhIso_"] = ibooker.book1D("MuonPhIsoComp", "Photon_{IsoComponent}(#mu )", 50, 0., 5.);
    // charged hadron isolation component of the candidate electron (depending on
    // the decay channel)
    hists_["elecChHadIso_"] = ibooker.book1D("ElectronChHadIsoComp", "ChHad_{IsoComponent}(e tightId)", 50, 0., 5.);
    // neutral hadron isolation component of the candidate electron (depending on
    // the decay channel)
    hists_["elecNeHadIso_"] = ibooker.book1D("ElectronNeHadIsoComp", "NeHad_{IsoComponent}(e tightId)", 50, 0., 5.);
    // photon isolation component of the candidate electron (depending on the
    // decay channel)
    hists_["elecPhIso_"] = ibooker.book1D("ElectronPhIsoComp", "Photon_{IsoComponent}(e tightId)", 50, 0., 5.);

    // multiplicity for combined secondary vertex
    hists_["jetMultBCSVM_"] = ibooker.book1D("JetMultBCSVM", "N_{30}(CSVM)", 10, 0., 10.);
    // btag discriminator for combined secondary vertex
    hists_["jetBCSV_"] = ibooker.book1D("JetDiscCSV", "BJet Disc_{CSV}(JET)", 100, -1., 2.);
    // pt of the 1. leading jet (uncorrected)
    // hists_["jet1PtRaw_"] = ibooker.book1D("Jet1PtRaw", "pt_{Raw}(jet1)", 60, 0., 300.);
    // pt of the 2. leading jet (uncorrected)
    // hists_["jet2PtRaw_"] = ibooker.book1D("Jet2PtRaw", "pt_{Raw}(jet2)", 60, 0., 300.);
    // pt of the 3. leading jet (uncorrected)
    // hists_["jet3PtRaw_"] = ibooker.book1D("Jet3PtRaw", "pt_{Raw}(jet3)", 60, 0., 300.);
    // pt of the 4. leading jet (uncorrected)
    // hists_["jet4PtRaw_"] = ibooker.book1D("Jet4PtRaw", "pt_{Raw}(jet4)", 60, 0., 300.);
    // selected events
    hists_["eventLogger_"] = ibooker.book2D("EventLogger", "Logged Events", 9, 0., 9., 10, 0., 10.);

    // set axes titles for selected events
    hists_["eventLogger_"]->getTH1()->SetOption("TEXT");
    hists_["eventLogger_"]->setBinLabel(1, "Run", 1);
    hists_["eventLogger_"]->setBinLabel(2, "Block", 1);
    hists_["eventLogger_"]->setBinLabel(3, "Event", 1);
    hists_["eventLogger_"]->setBinLabel(4, "pt_{30,loose}(jet1)", 1);
    hists_["eventLogger_"]->setBinLabel(5, "pt_{30,loose}(jet2)", 1);
    hists_["eventLogger_"]->setBinLabel(6, "pt_{30,loose}(jet3)", 1);
    hists_["eventLogger_"]->setBinLabel(7, "pt_{30,loose}(jet4)", 1);
    hists_["eventLogger_"]->setBinLabel(8, "M_{W}", 1);
    hists_["eventLogger_"]->setBinLabel(9, "M_{Top}", 1);
    hists_["eventLogger_"]->setAxisTitle("logged evts", 2);
    return;
  }

  void MonitorEnsemble::fill(const edm::Event& event, const edm::EventSetup& setup) {
    // fetch trigger event if configured such
    edm::Handle<edm::TriggerResults> triggerTable;

    if (!triggerTable_.isUninitialized()) {
      if (!event.getByToken(triggerTable_, triggerTable))
        return;
    }

    /*
  ------------------------------------------------------------

  Primary Vertex Monitoring

  ------------------------------------------------------------
  */
    // fill monitoring plots for primary verices
    edm::Handle<edm::View<reco::Vertex>> pvs;

    if (!event.getByToken(pvs_, pvs))
      return;
    const reco::Vertex& Pvertex = pvs->front();
    unsigned int pvMult = 0;
    for (edm::View<reco::Vertex>::const_iterator pv = pvs->begin(); pv != pvs->end(); ++pv) {
      if (!pvSelect_ || (*pvSelect_)(*pv))
        pvMult++;
    }
    fill("pvMult_", pvMult);

    /*
  ------------------------------------------------------------

  Electron Monitoring

  ------------------------------------------------------------
  */

    // fill monitoring plots for electrons
    edm::Handle<edm::View<reco::PFCandidate>> elecs;
    edm::Handle<double> _rhoHandle;
    event.getByLabel(rhoTag, _rhoHandle);
    if (!event.getByToken(elecs_, elecs))
      return;

    // check availability of electron id
    edm::Handle<edm::ValueMap<float>> electronId;
    if (!electronId_.isUninitialized()) {
      if (!event.getByToken(electronId_, electronId))
        return;
    }

    // loop electron collection
    unsigned int eMult = 0, eMultIso = 0;
    std::vector<const reco::PFCandidate*> isoElecs;
    for (edm::View<reco::PFCandidate>::const_iterator elec = elecs->begin(); elec != elecs->end(); ++elec) {
      if (elec->gsfElectronRef().isNull()) {
        continue;
      }
      reco::GsfElectronRef gsf_el = elec->gsfElectronRef();
      // restrict to electrons with good electronId
      if (electronId_.isUninitialized()
              ? true
              : ((double)(*electronId)[gsf_el] >=
                 eidCutValue_)) {  //This Electron Id is not currently used, but we can keep this for future needs
        if (!elecSelect_ || (*elecSelect_)(*elec)) {
          double el_ChHadIso = gsf_el->pfIsolationVariables().sumChargedHadronPt;
          double el_NeHadIso = gsf_el->pfIsolationVariables().sumNeutralHadronEt;
          double el_PhIso = gsf_el->pfIsolationVariables().sumPhotonEt;
          double absEta = std::fabs(gsf_el->superCluster()->eta());

          //Effective Area computation
          double eA = 0;
          if (absEta < 1.000)
            eA = 0.1703;
          else if (absEta < 1.479)
            eA = 0.1715;
          else if (absEta < 2.000)
            eA = 0.1213;
          else if (absEta < 2.200)
            eA = 0.1230;
          else if (absEta < 2.300)
            eA = 0.1635;
          else if (absEta < 2.400)
            eA = 0.1937;
          else if (absEta < 5.000)
            eA = 0.2393;

          double rho = _rhoHandle.isValid() ? (float)(*_rhoHandle) : 0;
          double el_pfRelIso = (el_ChHadIso + max(0., el_NeHadIso + el_PhIso - rho * eA)) / gsf_el->pt();

          //Only TightId
          if (eMult == 0) {  // Restricted to the leading tight electron
            fill("elecRelIso_", el_pfRelIso);
            fill("elecChHadIso_", el_ChHadIso);
            fill("elecNeHadIso_", el_NeHadIso);
            fill("elecPhIso_", el_PhIso);
          }
          ++eMult;

          if (!((el_pfRelIso < 0.0588 && absEta < 1.479) || (el_pfRelIso < 0.0571 && absEta > 1.479)))
            continue;  // PF Isolation with Effective Area Corrections according to https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2

          // TightId and TightIso
          if (eMultIso == 0) {  //Only leading
            fill("elecPt_", elec->pt());
            fill("elecEta_", elec->eta());
            fill("elecPhi_", elec->phi());
          }
          ++eMultIso;
        }
      }
    }
    //fill("elecMult_", eMult);
    fill("elecMultTight_", eMultIso);

    /*
  ------------------------------------------------------------

  Muon Monitoring

  ------------------------------------------------------------
  */

    // fill monitoring plots for muons
    unsigned int mMult = 0, mTight = 0, mTightId = 0;

    edm::Handle<edm::View<reco::PFCandidate>> muons;
    edm::View<reco::PFCandidate>::const_iterator muonit;

    if (!event.getByToken(muons_, muons))
      return;

    for (edm::View<reco::PFCandidate>::const_iterator muonit = muons->begin(); muonit != muons->end(); ++muonit) {
      if (muonit->muonRef().isNull())
        continue;
      reco::MuonRef muon = muonit->muonRef();

      // restrict to globalMuons
      if (muon->isGlobalMuon()) {
        fill("muonDelZ_", muon->innerTrack()->vz());  // CB using inner track!
        fill("muonDelXY_", muon->innerTrack()->vx(), muon->innerTrack()->vy());

        // apply preselection
        if ((!muonSelect_ || (*muonSelect_)(*muonit))) {
          mMult++;
          double chHadPt = muon->pfIsolationR04().sumChargedHadronPt;
          double neHadEt = muon->pfIsolationR04().sumNeutralHadronEt;
          double phoEt = muon->pfIsolationR04().sumPhotonEt;
          double pfRelIso = (chHadPt + max(0., neHadEt + phoEt - 0.5 * muon->pfIsolationR04().sumPUPt)) /
                            muon->pt();  // CB dBeta corrected iso!

          if (!(muon->isGlobalMuon() && muon->isPFMuon() && muon->globalTrack()->normalizedChi2() < 10. &&
                muon->globalTrack()->hitPattern().numberOfValidMuonHits() > 0 && muon->numberOfMatchedStations() > 1 &&
                muon->innerTrack()->hitPattern().numberOfValidPixelHits() > 0 &&
                muon->innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
                fabs(muon->muonBestTrack()->dxy(Pvertex.position())) < 0.2 &&
                fabs(muon->muonBestTrack()->dz(Pvertex.position())) < 0.5))
            continue;  //Only tight muons

          if (mTightId == 0) {
            fill("muonRelIso_", pfRelIso);
            fill("muonChHadIso_", chHadPt);
            fill("muonNeHadIso_", neHadEt);
            fill("muonPhIso_", phoEt);
          }
          mTightId++;

          if (!(pfRelIso < 0.15))
            continue;  //Tight Iso

          if (mTight == 0) {  //Leading tightId tightIso muon
            fill("muonPt_", muon->pt());
            fill("muonEta_", muon->eta());
            fill("muonPhi_", muon->phi());
          }
          mTight++;
        }
      }
    }
    fill("muonMult_", mMult);
    fill("muonMultTight_", mTight);

    /*
  ------------------------------------------------------------

  Jet Monitoring

  ------------------------------------------------------------
  */

    const reco::JetCorrector* corrector = nullptr;
    if (!jetCorrector_.isUninitialized()) {
      // check whether a jet corrector is in the event or not
      edm::Handle<reco::JetCorrector> correctorHandle = event.getHandle(jetCorrector_);
      if (correctorHandle.isValid()) {
        corrector = correctorHandle.product();
      } else {
        edm::LogVerbatim("TopDiLeptonOfflineDQM") << "\n"
                                                  << "-----------------------------------------------------------------"
                                                     "-------------------- \n"
                                                  << " No JetCorrector available from Event:              "
                                                     "                     \n"
                                                  << "  - Jets will not be corrected.                                  "
                                                     "                     \n"
                                                  << "-----------------------------------------------------------------"
                                                     "-------------------- \n";
      }
    }

    // check availability of the btaggers
    edm::Handle<reco::JetTagCollection> btagEff, btagPur, btagVtx, btagCSV;
    if (includeBTag_) {
      if (!event.getByToken(btagCSV_, btagCSV))
        return;
    }

    // loop jet collection
    std::vector<reco::Jet> correctedJets;
    std::vector<double> JetTagValues;
    unsigned int mult = 0, multLoose = 0, multCSV = 0;

    edm::Handle<edm::View<reco::Jet>> jets;
    if (!event.getByToken(jets_, jets)) {
      return;
    }

    for (edm::View<reco::Jet>::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
      bool isLoose = false;
      // check additional jet selection for pf jets
      if (dynamic_cast<const reco::PFJet*>(&*jet)) {
        reco::PFJet sel = dynamic_cast<const reco::PFJet&>(*jet);
        if ((*jetlooseSelection_)(sel))
          isLoose = true;
        sel.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);
        if (!(*jetSelection_)(sel))
          continue;
      }

      // prepare jet to fill monitor histograms
      reco::Jet monitorJet = *jet;

      ++mult;  // determine jet (no Id) multiplicity
      monitorJet.scaleEnergy(corrector ? corrector->correction(*jet) : 1.);

      if (isLoose) {  //Loose Id
        unsigned int idx = jet - jets->begin();
        correctedJets.push_back(monitorJet);
        if (includeBTag_) {
          // fill b-discriminators
          edm::RefToBase<reco::Jet> jetRef = jets->refAt(idx);
          fill("jetBCSV_", (*btagCSV)[jetRef]);
          if ((*btagCSV)[jetRef] > btagCSVWP_)
            ++multCSV;
          // Fill a vector with Jet b-tag WP for later M3+1tag calculation: CSV
          // tagger
          JetTagValues.push_back((*btagCSV)[jetRef]);
        }

        // fill pt/eta for the leading four jets
        if (multLoose == 0) {
          fill("jet1Pt_", monitorJet.pt());
          fill("jet1Eta_", monitorJet.eta());
        };
        if (multLoose == 1) {
          fill("jet2Pt_", monitorJet.pt());
          fill("jet2Eta_", monitorJet.eta());
        }
        if (multLoose == 2) {
          fill("jet3Pt_", monitorJet.pt());
          fill("jet3Eta_", monitorJet.eta());
        }
        if (multLoose == 3) {
          fill("jet4Pt_", monitorJet.pt());
          fill("jet4Eta_", monitorJet.eta());
        }
        multLoose++;
      }
    }
    fill("jetMult_", mult);
    fill("jetMultLoose_", multLoose);
    fill("jetMultBCSVM_", multCSV);

    /*
  ------------------------------------------------------------

  MET Monitoring

  ------------------------------------------------------------
  */

    // fill monitoring histograms for met
    for (std::vector<edm::EDGetTokenT<edm::View<reco::MET>>>::const_iterator met_ = mets_.begin(); met_ != mets_.end();
         ++met_) {
      edm::Handle<edm::View<reco::MET>> met;
      if (!event.getByToken(*met_, met))
        continue;
      if (met->begin() != met->end()) {  //If we ever have to use more than one type of met again
        unsigned int idx = met_ - mets_.begin();
        if (idx == 0)
          fill("metPflow_", met->begin()->et());
      }
    }

    /*
  ------------------------------------------------------------

  Event Monitoring

  ------------------------------------------------------------
  */

    // fill W boson and top mass estimates

    Calculate eventKinematics(MAXJETS, WMASS);
    double wMass = eventKinematics.massWBoson(correctedJets);
    double topMass = eventKinematics.massTopQuark(correctedJets);
    if (wMass >= 0 && topMass >= 0) {
      fill("massW_", wMass);
      fill("massTop_", topMass);
    }

    // Fill M3 with Btag (CSV Tight) requirement

    if (!includeBTag_)
      return;
    if (correctedJets.size() != JetTagValues.size())
      return;
    double btopMass = eventKinematics.massBTopQuark(correctedJets, JetTagValues, btagCSVWP_);
    if (btopMass >= 0)
      fill("massBTop_", btopMass);

    // fill plots for trigger monitoring
    if ((lowerEdge_ == -1. && upperEdge_ == -1.) || (lowerEdge_ < wMass && wMass < upperEdge_)) {
      if (!triggerTable_.isUninitialized())
        fill(event, *triggerTable, "trigger", triggerPaths_);
      if (logged_ <= hists_.find("eventLogger_")->second->getNbinsY()) {
        // log runnumber, lumi block, event number & some
        // more pysics infomation for interesting events
        fill("eventLogger_", 0.5, logged_ + 0.5, event.eventAuxiliary().run());
        fill("eventLogger_", 1.5, logged_ + 0.5, event.eventAuxiliary().luminosityBlock());
        fill("eventLogger_", 2.5, logged_ + 0.5, event.eventAuxiliary().event());
        if (!correctedJets.empty())
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
  }
}  // namespace TopSingleLepton

TopSingleLeptonDQM::TopSingleLeptonDQM(const edm::ParameterSet& cfg)
    : vertexSelect_(nullptr),
      beamspot_(""),
      beamspotSelect_(nullptr),
      MuonStep(nullptr),
      ElectronStep(nullptr),
      PvStep(nullptr),
      METStep(nullptr) {
  JetSteps.clear();
  CaloJetSteps.clear();
  PFJetSteps.clear();
  // configure preselection
  edm::ParameterSet presel = cfg.getParameter<edm::ParameterSet>("preselection");
  if (presel.existsAs<edm::ParameterSet>("trigger")) {
    edm::ParameterSet trigger = presel.getParameter<edm::ParameterSet>("trigger");
    triggerTable__ = consumes<edm::TriggerResults>(trigger.getParameter<edm::InputTag>("src"));
    triggerPaths_ = trigger.getParameter<std::vector<std::string>>("select");
  }
  if (presel.existsAs<edm::ParameterSet>("beamspot")) {
    edm::ParameterSet beamspot = presel.getParameter<edm::ParameterSet>("beamspot");
    beamspot_ = beamspot.getParameter<edm::InputTag>("src");
    beamspot__ = consumes<reco::BeamSpot>(beamspot.getParameter<edm::InputTag>("src"));
    beamspotSelect_ =
        std::make_unique<StringCutObjectSelector<reco::BeamSpot>>(beamspot.getParameter<std::string>("select"));
  }

  // configure the selection
  sel_ = cfg.getParameter<std::vector<edm::ParameterSet>>("selection");
  setup_ = cfg.getParameter<edm::ParameterSet>("setup");
  for (unsigned int i = 0; i < sel_.size(); ++i) {
    selectionOrder_.push_back(sel_.at(i).getParameter<std::string>("label"));
    selection_[selectionStep(selectionOrder_.back())] =
        std::make_pair(sel_.at(i),
                       std::make_unique<TopSingleLepton::MonitorEnsemble>(
                           selectionStep(selectionOrder_.back()).c_str(), setup_, consumesCollector()));
  }
  for (std::vector<std::string>::const_iterator selIt = selectionOrder_.begin(); selIt != selectionOrder_.end();
       ++selIt) {
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if (selection_.find(key) != selection_.end()) {
      if (type == "muons") {
        MuonStep = std::make_unique<SelectionStep<reco::PFCandidate>>(selection_[key].first, consumesCollector());
      }
      if (type == "elecs") {
        ElectronStep = std::make_unique<SelectionStep<reco::PFCandidate>>(selection_[key].first, consumesCollector());
      }
      if (type == "pvs") {
        PvStep = std::make_unique<SelectionStep<reco::Vertex>>(selection_[key].first, consumesCollector());
      }
      if (type == "jets") {
        JetSteps.push_back(std::make_unique<SelectionStep<reco::Jet>>(selection_[key].first, consumesCollector()));
      }
      if (type == "jets/pf") {
        PFJetSteps.push_back(std::make_unique<SelectionStep<reco::PFJet>>(selection_[key].first, consumesCollector()));
      }
      if (type == "jets/calo") {
        CaloJetSteps.push_back(
            std::make_unique<SelectionStep<reco::CaloJet>>(selection_[key].first, consumesCollector()));
      }
      if (type == "met") {
        METStep = std::make_unique<SelectionStep<reco::MET>>(selection_[key].first, consumesCollector());
      }
    }
  }
}
void TopSingleLeptonDQM::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  for (auto selIt = selection_.begin(); selIt != selection_.end(); ++selIt) {
    selIt->second.second->book(ibooker);
  }
}
void TopSingleLeptonDQM::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  if (!triggerTable__.isUninitialized()) {
    edm::Handle<edm::TriggerResults> triggerTable;
    if (!event.getByToken(triggerTable__, triggerTable))
      return;
    if (!accept(event, *triggerTable, triggerPaths_))
      return;
  }
  if (!beamspot__.isUninitialized()) {
    edm::Handle<reco::BeamSpot> beamspot;
    if (!event.getByToken(beamspot__, beamspot))
      return;
    if (!(*beamspotSelect_)(*beamspot))
      return;
  }
  //  cout<<" apply selection steps"<<endl;
  unsigned int nJetSteps = -1;
  unsigned int nPFJetSteps = -1;
  unsigned int nCaloJetSteps = -1;
  for (std::vector<std::string>::const_iterator selIt = selectionOrder_.begin(); selIt != selectionOrder_.end();
       ++selIt) {
    std::string key = selectionStep(*selIt), type = objectType(*selIt);
    if (selection_.find(key) != selection_.end()) {
      if (type == "empty") {
        selection_[key].second->fill(event, setup);
      }
      if (type == "muons" && MuonStep != nullptr) {
        if (MuonStep->select(event)) {
          //      cout<<"selected event! "<<selection_[key].second<<endl;
          selection_[key].second->fill(event, setup);
        } else
          break;
      }
      // cout<<" apply selection steps 2"<<endl;
      if (type == "elecs" && ElectronStep != nullptr) {
        // cout<<"In electrons ..."<<endl;
        if (ElectronStep->select(event, "electron")) {
          selection_[key].second->fill(event, setup);
        } else
          break;
      }
      // cout<<" apply selection steps 3"<<endl;
      if (type == "pvs" && PvStep != nullptr) {
        if (PvStep->selectVertex(event)) {
          selection_[key].second->fill(event, setup);
        } else
          break;
      }
      // cout<<" apply selection steps 4"<<endl;
      if (type == "jets") {
        nJetSteps++;
        if (JetSteps[nJetSteps] != nullptr) {
          if (JetSteps[nJetSteps]->select(event, setup)) {
            selection_[key].second->fill(event, setup);
          } else
            break;
        }
      }
      if (type == "jets/pf") {
        nPFJetSteps++;
        if (PFJetSteps[nPFJetSteps] != nullptr) {
          if (PFJetSteps[nPFJetSteps]->select(event, setup)) {
            selection_[key].second->fill(event, setup);
          } else
            break;
        }
      }
      if (type == "jets/calo") {
        nCaloJetSteps++;
        if (CaloJetSteps[nCaloJetSteps] != nullptr) {
          if (CaloJetSteps[nCaloJetSteps]->select(event, setup)) {
            selection_[key].second->fill(event, setup);
          } else
            break;
        }
      }
      if (type == "met" && METStep != nullptr) {
        if (METStep->select(event)) {
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
