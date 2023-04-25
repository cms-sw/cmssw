// -*- C++ -*- //
// Package:    EopTreeWriter
// Class:      EopTreeWriter
//
/**\class EopTreeWriter EopTreeWriter.cc Alignment/OfflineValidation/plugins/EopTreeWriter.cc
 //
 Description: <one line class summary>
 
 Implementation:
 <Notes on implementation>
*/
//
// Original Author:  Holger Enderle
//         Created:  Thu Dec  4 11:22:48 CET 2008
// $Id: EopElecTreeWriter.cc,v 1.8 2012/08/30 08:40:51 cgoetzma Exp $
//
//

// framework include files
#include "Alignment/OfflineValidation/interface/EopElecVariables.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoEcal/EgammaCoreTools/interface/SuperClusterShapeAlgo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "TCanvas.h"
#include "TH1.h"
#include "TH2D.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TTree.h"

struct EopTriggerType {
  bool fired;
  double prescale;
  int index;

  EopTriggerType() {
    fired = false;
    prescale = 0;
    index = 0;
  }
};

//
// class decleration
//
using namespace std;

class EopElecTreeWriter : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit EopElecTreeWriter(const edm::ParameterSet&);
  ~EopElecTreeWriter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  // methods
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override{};
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  // ES tokens
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;

  // EDM tokens
  const edm::EDGetTokenT<reco::VertexCollection> theVertexCollectionToken_;
  const edm::EDGetTokenT<HBHERecHitCollection> theHBHERecHitCollectionToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> theEcalRecHitCollectionToken_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> theBarrelSupClusCollectionToken_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> theEndCapSupClusCollectionToken_;
  const edm::EDGetTokenT<edm::TriggerResults> theTriggerResultsToken_;
  const edm::EDGetTokenT<trigger::TriggerEvent> theTriggerEventToken_;
  const edm::EDGetTokenT<reco::GsfTrackCollection> theGsfTrackCollectionToken_;
  const edm::EDGetTokenT<reco::GsfElectronCoreCollection> theGsfElectronCoreCollectionToken_;

  // some static constants
  static constexpr float k_etaBarrel = 1.55;
  static constexpr float k_etaEndcap = 1.44;

  // other member data
  std::string theTrigger_;
  std::string theFilter_;
  bool debugTriggerSelection_;
  edm::Service<TFileService> fs_;
  TTree* tree_;
  EopElecVariables* treeMemPtr_;
  HLTConfigProvider hltConfig_;
  std::vector<std::string> triggerNames_;
  MultiTrajectoryStateTransform* mtsTransform_;

  // histograms

  // Cut flow (events number)
  TH1D* h_nEvents;
  TH1D* h_nEventsWithVertex;
  TH1D* h_nEventsTriggered;
  TH1D* h_nEventsHLTFilter;
  TH1D* h_nEventsHLTelectron;
  TH1D* h_nEventsHLTrejected;
  TH1D* h_nEvents2Elec;
  TH1D* h_nHLTelectrons;
  TH1D* h_nTrkRejectedPerEvt;
  TH1D* h_nTrkSelectedPerEvt;

  // Cut flow (tracks number)
  TH1D* h_nTracks;
  TH1D* h_nTracksFiltered;
  TH1D* h_cut_Ptmin;
  TH1D* h_cut_OneSCmatch;

  TH1D* h_counter1;
  TH1D* h_counter2;

  TH1D* h_distToClosestSCgsf;
  TH1D* h_distToClosestSC;
  TH1D* h_EcalEnergy;
  TH1D* h_Momentum;
  TH1D* HcalEnergy;
  TH1D* h_fBREM;
  TH1D* h_Eop_InnerNegative;
  TH1D* h_Eop_InnerPositive;

  TH2D* HcalVSEcal;
};

namespace eopUtils {

  static constexpr float R_ECAL = 136.5;
  static constexpr float Z_Endcap = 328.0;
  static constexpr float etaBarrelEndcap = 1.479;

  // Function to convert the eta of the track to a detector eta (for matching with SC)
  float ecalEta(float EtaParticle, float Zvertex, float RhoVertex) {
    if (EtaParticle != 0.) {
      float Theta = 0.0;
      float ZEcal = (R_ECAL - RhoVertex) * sinh(EtaParticle) + Zvertex;

      if (ZEcal != 0.0)
        Theta = atan(R_ECAL / ZEcal);
      if (Theta < 0.0)
        Theta = Theta + M_PI;

      float ETA = -log(tan(0.5 * Theta));

      if (fabs(ETA) > etaBarrelEndcap) {
        float Zend = Z_Endcap;
        if (EtaParticle < 0.0)
          Zend = -Zend;
        float Zlen = Zend - Zvertex;
        float RR = Zlen / sinh(EtaParticle);
        Theta = atan((RR + RhoVertex) / Zend);
        if (Theta < 0.0)
          Theta = Theta + M_PI;
        ETA = -log(tan(0.5 * Theta));
      }
      return ETA;
    } else {
      edm::LogWarning("") << "[EcalPositionFromTrack::etaTransformation] Warning: Eta equals to zero, not correcting";
      return EtaParticle;
    }
  }
}  // namespace eopUtils

// constructors and destructor

EopElecTreeWriter::EopElecTreeWriter(const edm::ParameterSet& iConfig)
    : magFieldToken_(esConsumes()),
      tkGeomToken_(esConsumes()),
      caloGeomToken_(esConsumes()),
      theVertexCollectionToken_(consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"))),
      theHBHERecHitCollectionToken_(consumes<HBHERecHitCollection>(edm::InputTag("hbhereco", ""))),
      theEcalRecHitCollectionToken_(consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"))),
      theBarrelSupClusCollectionToken_(
          consumes<reco::SuperClusterCollection>(edm::InputTag("hybridSuperClusters", ""))),
      theEndCapSupClusCollectionToken_(consumes<reco::SuperClusterCollection>(
          edm::InputTag("multi5x5SuperClusters", "multi5x5EndcapSuperClusters"))),
      theTriggerResultsToken_(consumes<edm::TriggerResults>(edm::InputTag("TriggerResults", "", "HLT"))),
      theTriggerEventToken_(consumes<trigger::TriggerEvent>(edm::InputTag("hltTriggerSummaryAOD"))),
      theGsfTrackCollectionToken_(consumes<reco::GsfTrackCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      theGsfElectronCoreCollectionToken_(
          consumes<reco::GsfElectronCoreCollection>(edm::InputTag("gedGsfElectronCores"))),
      theTrigger_(iConfig.getParameter<std::string>("triggerPath")),
      theFilter_(iConfig.getParameter<std::string>("hltFilter")),
      debugTriggerSelection_(iConfig.getParameter<bool>("debugTriggerSelection")) {
  usesResource(TFileService::kSharedResource);

  // TTree creation
  tree_ = fs_->make<TTree>("EopTree", "EopTree");
  treeMemPtr_ = new EopElecVariables;
  tree_->Branch("EopElecVariables", &treeMemPtr_);  // address of pointer!

  // Control histograms declaration
  h_distToClosestSC = fs_->make<TH1D>("distToClosestSC", "distToClosestSC", 100, 0, 0.1);
  h_distToClosestSCgsf = fs_->make<TH1D>("distToClosestSCgsf", "distToClosestSCgsf", 100, 0, 0.1);
  h_EcalEnergy = fs_->make<TH1D>("EcalEnergy", "EcalEnergy", 100, 0, 200);
  h_Momentum = fs_->make<TH1D>("Momentum", "Momentum", 100, 0, 200);
  HcalEnergy = fs_->make<TH1D>("HcalEnergy", "HcalEnergy", 100, 0, 40);
  h_fBREM = fs_->make<TH1D>("fBREM", "fBREM", 100, -0.2, 1);
  h_Eop_InnerNegative = fs_->make<TH1D>("Eop_InnerNegative", "Eop_InnerNegative", 100, 0, 3);
  h_Eop_InnerPositive = fs_->make<TH1D>("Eop_InnerPositive", "Eop_InnerPositive", 100, 0, 3);
  HcalVSEcal = fs_->make<TH2D>("HcalVSEcal", "HcalVSEcal", 100, 0, 160, 100, 0, 10);

  h_nEvents = fs_->make<TH1D>("nEvents", "nEvents", 1, 0, 1);
  h_nEventsWithVertex = fs_->make<TH1D>("nEventsWithVertex", "nEventsWithVertex", 1, 0, 1);
  h_nEventsTriggered = fs_->make<TH1D>("nEventsTriggered", "nEventsTriggered", 1, 0, 1);
  h_nEventsHLTFilter = fs_->make<TH1D>("nEventsHLTFilter", "nEventsHLTFilter", 1, 0, 1);
  h_nEventsHLTelectron = fs_->make<TH1D>("nEventsHLTelectron", "nEventsHLTelectron", 1, 0, 1);
  h_nEventsHLTrejected = fs_->make<TH1D>("nEventsHLTrejected", "nEventsHLTrejected", 1, 0, 1);
  h_nEvents2Elec = fs_->make<TH1D>("nEvents2Elec", "nEvents2Elec", 1, 0, 1);

  h_nHLTelectrons = fs_->make<TH1D>("nHLTelectrons", "nHLTelectrons", 20, 0, 20);
  h_nTrkRejectedPerEvt = fs_->make<TH1D>("nTrkRejectedPerEvt", "nTrkRejectedPerEvt", 20, 0, 20);
  h_nTrkSelectedPerEvt = fs_->make<TH1D>("nTrkSelectedPerEvt", "nTrkSelectedPerEvt", 20, 0, 20);

  h_nTracks = fs_->make<TH1D>("nTracks", "nTracks", 1, 0, 1);
  h_nTracksFiltered = fs_->make<TH1D>("nTracksFiltered", "nTracksFiltered", 1, 0, 1);
  h_cut_Ptmin = fs_->make<TH1D>("cut_Ptmin", "cut_Ptmin", 1, 0, 1);
  h_cut_OneSCmatch = fs_->make<TH1D>("cut_OneSCmatch", "cut_OneSCmatch", 1, 0, 1);

  h_counter1 = fs_->make<TH1D>("counter1", "counter1", 1, 0, 1);
  h_counter2 = fs_->make<TH1D>("counter2", "counter2", 1, 0, 1);
}

EopElecTreeWriter::~EopElecTreeWriter() {
  // control histograms
  h_distToClosestSC->SetXTitle("distance from track to closest SuperCluster in eta-phi plan (weighted matching)");
  h_distToClosestSC->SetYTitle("# Tracks");

  h_distToClosestSCgsf->SetXTitle("distance from track to closest SuperCluster in eta-phi plan (gsfElectronCore)");
  h_distToClosestSCgsf->SetYTitle("# Tracks");

  h_EcalEnergy->SetXTitle("Ecal energy deposit (GeV)");
  h_EcalEnergy->SetYTitle("# tracks");

  HcalEnergy->SetXTitle("Hcal energy deposit (GeV)");
  HcalEnergy->SetYTitle("# tracks");

  h_Momentum->SetXTitle("Momentum magnitude (GeV)");
  h_Momentum->SetYTitle("# tracks");

  h_Eop_InnerNegative->SetXTitle("E/p");
  h_Eop_InnerNegative->SetYTitle("# tracks");

  h_Eop_InnerPositive->SetXTitle("E/p");
  h_Eop_InnerPositive->SetYTitle("# tracks");

  HcalVSEcal->SetXTitle("Ecal energy (GeV)");
  HcalVSEcal->SetYTitle("Hcal energy (GeV)");
}

//###########################################
//#     method called to for each event     #
//###########################################

void EopElecTreeWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  h_nEvents->Fill(0.5);

  Double_t EnergyHcalIn01;
  Double_t EnergyHcalIn02;
  Double_t EnergyHcalIn03;
  Double_t EnergyHcalIn04;
  Double_t EnergyHcalIn05;
  Double_t etaWidth;
  Double_t phiWidth;
  Int_t algo_ID;
  Double_t EnergyEcal;
  Double_t fbrem;
  Double_t pin;
  Double_t etaIn;
  Double_t phiIn;
  Double_t pout;
  Double_t etaOut;
  Double_t phiOut;
  Double_t MaxPtIn01;
  Double_t SumPtIn01;
  Bool_t NoTrackIn0015;
  Double_t MaxPtIn02;
  Double_t SumPtIn02;
  Bool_t NoTrackIn0020;
  Double_t MaxPtIn03;
  Double_t SumPtIn03;
  Bool_t NoTrackIn0025;
  Double_t MaxPtIn04;
  Double_t SumPtIn04;
  Bool_t NoTrackIn0030;
  Double_t MaxPtIn05;
  Double_t SumPtIn05;
  Bool_t NoTrackIn0035;
  Bool_t NoTrackIn0040;
  Double_t dRSC_first;
  Double_t dRSC_second;
  Double_t etaSC;
  Double_t phiSC;
  Int_t nbSC;        //to count the nb of SuperCluster matching with a given track
  Int_t nBasicClus;  //to count the nb of basic cluster in a given superCluster
  Bool_t isEcalDriven;
  Bool_t isTrackerDriven;
  Bool_t isBarrel;
  Bool_t isEndcap;
  Bool_t TrigTag;

  //---------------   for GsfTrack propagation through tracker  ---------------

  const MagneticField* magField_ = &iSetup.getData(magFieldToken_);
  const TrackerGeometry* trackerGeom_ = &iSetup.getData(tkGeomToken_);
  MultiTrajectoryStateTransform mtsTransform(trackerGeom_, magField_);

  //---------------    Super Cluster    -----------------

  // getting primary vertex (necessary to convert eta track to eta detector
  const reco::VertexCollection& vertex = iEvent.get(theVertexCollectionToken_);

  if (vertex.empty()) {
    edm::LogError("EopElecTreeWriter") << "Error: no primary vertex found!";
    return;
  }
  const reco::Vertex& vert = vertex.front();
  h_nEventsWithVertex->Fill(0.5);

  // getting calorimeter geometry
  const CaloGeometry* geo = &iSetup.getData(caloGeomToken_);
  const CaloSubdetectorGeometry* subGeo = geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  if (subGeo == nullptr)
    edm::LogError("EopElecTreeWriter") << "ERROR: unable to find SubDetector geometry!!!";

  // getting Hcal rechits
  const HBHERecHitCollection* HcalHits = &iEvent.get(theHBHERecHitCollectionToken_);

  // getting Ecal rechits
  const EcalRecHitCollection* rhc = &iEvent.get(theEcalRecHitCollectionToken_);
  if (rhc == nullptr)
    edm::LogError("EopElecTreeWriter") << "ERROR: unable to find the EcalRecHit collection !!!";

  // getting SuperCluster
  const reco::SuperClusterCollection* BarrelSupClusCollection = &iEvent.get(theBarrelSupClusCollectionToken_);
  const reco::SuperClusterCollection* EndcapSupClusCollection = &iEvent.get(theEndCapSupClusCollectionToken_);

  // necessary to re-calculate phi and eta width of SuperClusters
  SuperClusterShapeAlgo SCShape(rhc, subGeo);

  //---------------    Trigger   -----------------
  TrigTag = false;
  const edm::TriggerResults* trigRes = &iEvent.get(theTriggerResultsToken_);

  // trigger event
  const trigger::TriggerEvent* triggerEvent = &iEvent.get(theTriggerEventToken_);

  // our trigger table
  std::map<std::string, EopTriggerType> HLTpaths;
  for (const auto& triggerName : triggerNames_) {
    if (triggerName.find(theTrigger_) != 0)
      continue;
    EopTriggerType myTrigger;

    const unsigned int prescaleSize = hltConfig_.prescaleSize();
    for (unsigned int ps = 0; ps < prescaleSize; ps++) {
      auto const prescaleValue = hltConfig_.prescaleValue<double>(ps, triggerName);
      if (prescaleValue != 1) {
        myTrigger.prescale = prescaleValue;
      }
    }

    myTrigger.index = hltConfig_.triggerIndex(triggerName);
    if (myTrigger.index == -1)
      continue;
    myTrigger.fired =
        trigRes->wasrun(myTrigger.index) && trigRes->accept(myTrigger.index) && !trigRes->error(myTrigger.index);
    HLTpaths[triggerName] = myTrigger;
  }

  // First cut : trigger cut
  std::string firstFiredPath = "";
  for (const auto& it : HLTpaths) {
    if (it.second.fired) {
      TrigTag = true;
      firstFiredPath = it.first;
      break;
    }
  }
  if (!TrigTag)
    return;
  h_nEventsTriggered->Fill(0.5);

  // Displaying filters label from the first fired trigger
  // Useful for finding the good filter label
  std::vector<std::string> filters = hltConfig_.moduleLabels(firstFiredPath);

  if (debugTriggerSelection_) {
    edm::LogInfo("EopElecTreeWriter") << "filters : ";
    for (unsigned int i = 0; i < filters.size(); i++) {
      edm::LogInfo("EopElecTreeWriter") << filters[i] << " ";
    }
  }

  // Getting HLT electrons
  edm::InputTag testTag(theFilter_, "", "HLT");
  int testindex = triggerEvent->filterIndex(testTag);

  if (testindex >= triggerEvent->sizeFilters())
    return;
  h_nEventsHLTFilter->Fill(0.5);

  const trigger::Keys& KEYS_el(triggerEvent->filterKeys(testindex));

  std::vector<const trigger::TriggerObject*> HLTelectrons;
  for (unsigned int i = 0; i != KEYS_el.size(); ++i) {
    const trigger::TriggerObject* triggerObject_el = &(triggerEvent->getObjects().at(KEYS_el[i]));
    HLTelectrons.push_back(triggerObject_el);
  }

  h_nHLTelectrons->Fill(HLTelectrons.size());

  if (HLTelectrons.empty())
    return;
  h_nEventsHLTelectron->Fill(0.5);

  // finding the HLT electron with highest pt and saving the corresponding index
  unsigned int HighPtIndex = 0;
  double maxPtHLT = -5.;
  for (unsigned int j = 0; j < HLTelectrons.size(); j++) {
    if (HLTelectrons[j]->pt() > maxPtHLT) {
      maxPtHLT = HLTelectrons[j]->pt();
      HighPtIndex = j;
    }
  }

  //-----------------   Tracks   -------------------

  // getting GsfTrack
  const reco::GsfTrackCollection& tracks = iEvent.get(theGsfTrackCollectionToken_);

  // filtering track
  int nRejected = 0;
  int nSelected = 0;
  std::vector<const reco::GsfTrack*> filterTracks;
  for (const auto& track : tracks) {
    h_nTracks->Fill(0.5);
    double deltar =
        reco::deltaR(track.eta(), track.phi(), HLTelectrons[HighPtIndex]->eta(), HLTelectrons[HighPtIndex]->phi());
    // remove the triggered electron with highest pt
    if (deltar < 0.025) {
      treeMemPtr_->px_rejected_track = track.px();
      treeMemPtr_->py_rejected_track = track.py();
      treeMemPtr_->pz_rejected_track = track.pz();
      treeMemPtr_->p_rejected_track = track.p();
      nRejected++;
      continue;
    }
    filterTracks.push_back(&track);  // we use all the others
    nSelected++;
    h_nTracksFiltered->Fill(0.5);
  }
  h_nTrkRejectedPerEvt->Fill(nRejected);
  h_nTrkSelectedPerEvt->Fill(nSelected);

  if (nRejected == 0)
    return;
  h_nEventsHLTrejected->Fill(0.5);

  if (filterTracks.empty())
    return;
  h_nEvents2Elec->Fill(0.5);

  //-------- test:Matching SC/track using gsfElectonCore collection --------

  const reco::GsfElectronCoreCollection* electrons = &iEvent.get(theGsfElectronCoreCollectionToken_);
  for (const auto& elec : *electrons) {
    double etaGSF = eopUtils::ecalEta((elec.gsfTrack())->eta(), vert.z(), (vert.position()).rho());
    if ((elec.gsfTrack())->pt() < 10.)
      continue;

    double DELTAR = 0;
    DELTAR = reco::deltaR((elec.superCluster())->eta(), (elec.superCluster())->phi(), etaGSF, (elec.gsfTrack())->phi());

    if (DELTAR < 0.1)
      h_distToClosestSCgsf->Fill(DELTAR);
  }

  //------------------------------------------------------------
  //--------------    Loop on tracks   -------------------------

  for (const auto& track : filterTracks) {
    // initializing variables
    isEcalDriven = false;
    isTrackerDriven = false;
    isBarrel = false;
    isEndcap = false;
    etaWidth = 0;
    phiWidth = 0;
    etaSC = 0;
    phiSC = 0;
    fbrem = 0;
    pin = 0;
    etaIn = 0;
    phiIn = 0;
    pout = 0;
    etaOut = 0;
    phiOut = 0;
    algo_ID = 0;
    EnergyEcal = 0;
    dRSC_first = 999;
    dRSC_second = 9999;
    nbSC = 0;
    nBasicClus = 0;

    // First cut on momentum magnitude
    h_Momentum->Fill(track->p());
    if (track->pt() < 10.)
      continue;
    h_cut_Ptmin->Fill(0.5);

    // calculating track parameters at innermost and outermost for Gsf tracks
    TrajectoryStateOnSurface inTSOS = mtsTransform.innerStateOnSurface(*track);
    TrajectoryStateOnSurface outTSOS = mtsTransform.outerStateOnSurface(*track);
    if (inTSOS.isValid() && outTSOS.isValid()) {
      GlobalVector inMom;
      //multiTrajectoryStateMode::momentumFromModeCartesian(inTSOS, inMom_);
      multiTrajectoryStateMode::momentumFromModePPhiEta(inTSOS, inMom);
      pin = inMom.mag();
      etaIn = inMom.eta();
      phiIn = inMom.phi();
      GlobalVector outMom;
      //multiTrajectoryStateMode::momentumFromModeCartesian(outTSOS, outMom_);
      multiTrajectoryStateMode::momentumFromModePPhiEta(outTSOS, outMom);
      pout = outMom.mag();
      etaOut = outMom.eta();
      phiOut = outMom.phi();
      fbrem = (pin - pout) / pin;
      h_fBREM->Fill(fbrem);
    }

    // Matching track with Hcal rec hits
    EnergyHcalIn01 = 0;
    EnergyHcalIn02 = 0;
    EnergyHcalIn03 = 0;
    EnergyHcalIn04 = 0;
    EnergyHcalIn05 = 0;

    //for (std::vector<HBHERecHit>::const_iterator hcal = (*HcalHits).begin(); hcal != (*HcalHits).end(); hcal++) {
    for (const auto& hcal : *HcalHits) {
      GlobalPoint posH = geo->getPosition(hcal.detid());
      double phihit = posH.phi();
      double etahit = posH.eta();
      double dR = reco::deltaR(etahit, phihit, etaOut, phiOut);

      // saving Hcal energy deposit measured for different eta-phi radius
      if (dR < 0.1)
        EnergyHcalIn01 += hcal.energy();
      if (dR < 0.2)
        EnergyHcalIn02 += hcal.energy();
      if (dR < 0.3)
        EnergyHcalIn03 += hcal.energy();
      if (dR < 0.4)
        EnergyHcalIn04 += hcal.energy();
      if (dR < 0.5)
        EnergyHcalIn05 += hcal.energy();
    }

    HcalEnergy->Fill(EnergyHcalIn02);

    //Isolation against charged particles
    MaxPtIn01 = 0.;
    SumPtIn01 = 0.;
    NoTrackIn0015 = true;
    MaxPtIn02 = 0.;
    SumPtIn02 = 0.;
    NoTrackIn0020 = true;
    MaxPtIn03 = 0.;
    SumPtIn03 = 0.;
    NoTrackIn0025 = true;
    MaxPtIn04 = 0.;
    SumPtIn04 = 0.;
    NoTrackIn0030 = true;
    MaxPtIn05 = 0.;
    SumPtIn05 = 0.;
    NoTrackIn0035 = true;
    NoTrackIn0040 = true;

    for (const auto& track1 : filterTracks) {
      if (track == track1)
        continue;

      double etaIn1 = 0.;
      double phiIn1 = 0.;

      TrajectoryStateOnSurface inTSOS1 = mtsTransform.innerStateOnSurface(*track1);
      if (inTSOS1.isValid()) {
        GlobalVector inMom1;
        multiTrajectoryStateMode::momentumFromModePPhiEta(inTSOS1, inMom1);
        etaIn1 = inMom1.eta();
        phiIn1 = inMom1.phi();
      }

      if (etaIn1 == 0 && phiIn1 == 0)
        continue;

      double dR = reco::deltaR(etaIn1, phiIn1, etaIn, phiIn);

      // different radius of inner isolation cone
      if (dR < 0.015)
        NoTrackIn0015 = false;
      if (dR < 0.020)
        NoTrackIn0020 = false;
      if (dR < 0.025)
        NoTrackIn0025 = false;
      if (dR < 0.030)
        NoTrackIn0030 = false;
      if (dR < 0.035)
        NoTrackIn0035 = false;
      if (dR < 0.040)
        NoTrackIn0040 = false;

      //calculate maximum Pt and sum Pt inside cones of different radius
      if (dR < 0.1) {
        SumPtIn01 += track1->pt();
        if (track1->pt() > MaxPtIn01) {
          MaxPtIn01 = track1->pt();
        }
      }

      if (dR < 0.2) {
        SumPtIn02 += track1->pt();
        if (track1->pt() > MaxPtIn02) {
          MaxPtIn02 = track1->pt();
        }
      }
      if (dR < 0.3) {
        SumPtIn03 += track1->pt();
        if (track1->pt() > MaxPtIn03) {
          MaxPtIn03 = track1->pt();
        }
      }
      if (dR < 0.4) {
        SumPtIn04 += track1->pt();
        if (track1->pt() > MaxPtIn04) {
          MaxPtIn04 = track1->pt();
        }
      }
      if (dR < 0.5) {
        SumPtIn05 += track1->pt();
        if (track1->pt() > MaxPtIn05) {
          MaxPtIn05 = track1->pt();
        }
      }
    }

    // Track-SuperCluster matching

    double dRSC;
    double etaAtEcal = 0;  // to convert eta from track to detector frame
    etaAtEcal = eopUtils::ecalEta(etaIn, vert.z(), (vert.position()).rho());

    //Barrel
    if (std::abs(track->eta()) < k_etaBarrel) {
      for (const auto& SC : *BarrelSupClusCollection) {
        dRSC = reco::deltaR(SC.eta(), SC.phi(), etaAtEcal, phiIn);

        if (dRSC < dRSC_first) {
          dRSC_first = dRSC;
        }  // distance in eta-phi plan to closest SC
        else if (dRSC < dRSC_second) {
          dRSC_second = dRSC;
        }  // to next closest SC

        if (dRSC < 0.09) {
          //Calculate phiWidth & etaWidth for associated SuperClusters
          SCShape.Calculate_Covariances(SC);
          phiWidth = SCShape.phiWidth();
          etaWidth = SCShape.etaWidth();

          etaSC = SC.eta();
          phiSC = SC.phi();

          algo_ID = SC.algoID();
          EnergyEcal = SC.energy();
          nBasicClus = SC.clustersSize();
          nbSC++;
          isBarrel = true;
        }
      }
    }

    // Endcap
    if (std::abs(track->eta()) > k_etaEndcap) {
      for (const auto& SC : *EndcapSupClusCollection) {
        dRSC = reco::deltaR(SC.eta(), SC.phi(), etaAtEcal, phiIn);

        if (dRSC < dRSC_first) {
          dRSC_first = dRSC;
        }  // distance in eta-phi plan to closest SC
        else if (dRSC < dRSC_second) {
          dRSC_second = dRSC;
        }  // to next closest SC

        if (dRSC < 0.09) {
          //Calculate phiWidth & etaWidth for associated SuperClusters
          SCShape.Calculate_Covariances(SC);
          phiWidth = SCShape.phiWidth();
          etaWidth = SCShape.etaWidth();

          etaSC = SC.eta();
          phiSC = SC.phi();

          algo_ID = SC.algoID();
          EnergyEcal = SC.energy();
          nBasicClus = SC.clustersSize();
          nbSC++;
          isEndcap = true;
        }
      }
    }

    if (dRSC_first < 0.1)
      h_distToClosestSC->Fill(dRSC_first);
    if (nbSC == 1)
      h_counter1->Fill(0.5);
    if (nbSC == 0)
      h_counter2->Fill(0.5);

    if (nbSC > 1 || nbSC == 0)
      continue;
    h_cut_OneSCmatch->Fill(0.5);

    if (isBarrel && isEndcap) {
      edm::LogError("EopElecTreeWriter") << "Error: Super Cluster double matching!";
      return;
    }

    h_EcalEnergy->Fill(EnergyEcal);
    HcalVSEcal->Fill(EnergyEcal, EnergyHcalIn02);

    // E over p plots
    if (track->charge() < 0)
      h_Eop_InnerNegative->Fill(EnergyEcal / pin);
    if (track->charge() > 0)
      h_Eop_InnerPositive->Fill(EnergyEcal / pin);

    //Check if track-SuperCluster matching is Ecal or Tracker driven
    edm::RefToBase<TrajectorySeed> seed = track->extra()->seedRef();
    if (seed.isNull()) {
      edm::LogError("GsfElectronCore") << "The GsfTrack has no seed ?!";
    } else {
      reco::ElectronSeedRef elseed = seed.castTo<reco::ElectronSeedRef>();
      if (elseed.isNull()) {
        edm::LogError("GsfElectronCore") << "The GsfTrack seed is not an ElectronSeed ?!";
      } else {
        isEcalDriven = elseed->isEcalDriven();
        isTrackerDriven = elseed->isTrackerDriven();
      }
    }

    treeMemPtr_->charge = track->charge();
    treeMemPtr_->nHits = track->numberOfValidHits();
    treeMemPtr_->nLostHits = track->numberOfLostHits();
    treeMemPtr_->innerOk = track->innerOk();
    treeMemPtr_->outerRadius = track->outerRadius();
    treeMemPtr_->chi2 = track->chi2();
    treeMemPtr_->normalizedChi2 = track->normalizedChi2();
    treeMemPtr_->px = track->px();
    treeMemPtr_->py = track->py();
    treeMemPtr_->pz = track->pz();
    treeMemPtr_->p = track->p();
    treeMemPtr_->pIn = pin;
    treeMemPtr_->etaIn = etaIn;
    treeMemPtr_->phiIn = phiIn;
    treeMemPtr_->pOut = pout;
    treeMemPtr_->etaOut = etaOut;
    treeMemPtr_->phiOut = phiOut;
    treeMemPtr_->pt = track->pt();
    treeMemPtr_->ptError = track->ptError();
    treeMemPtr_->theta = track->theta();
    treeMemPtr_->eta = track->eta();
    treeMemPtr_->phi = track->phi();
    treeMemPtr_->fbrem = fbrem;
    treeMemPtr_->MaxPtIn01 = MaxPtIn01;
    treeMemPtr_->SumPtIn01 = SumPtIn01;
    treeMemPtr_->NoTrackIn0015 = NoTrackIn0015;
    treeMemPtr_->MaxPtIn02 = MaxPtIn02;
    treeMemPtr_->SumPtIn02 = SumPtIn02;
    treeMemPtr_->NoTrackIn0020 = NoTrackIn0020;
    treeMemPtr_->MaxPtIn03 = MaxPtIn03;
    treeMemPtr_->SumPtIn03 = SumPtIn03;
    treeMemPtr_->NoTrackIn0025 = NoTrackIn0025;
    treeMemPtr_->MaxPtIn04 = MaxPtIn04;
    treeMemPtr_->SumPtIn04 = SumPtIn04;
    treeMemPtr_->NoTrackIn0030 = NoTrackIn0030;
    treeMemPtr_->MaxPtIn05 = MaxPtIn05;
    treeMemPtr_->SumPtIn05 = SumPtIn05;
    treeMemPtr_->NoTrackIn0035 = NoTrackIn0035;
    treeMemPtr_->NoTrackIn0040 = NoTrackIn0040;
    treeMemPtr_->SC_algoID = algo_ID;
    treeMemPtr_->SC_energy = EnergyEcal;
    treeMemPtr_->SC_nBasicClus = nBasicClus;
    treeMemPtr_->SC_etaWidth = etaWidth;
    treeMemPtr_->SC_phiWidth = phiWidth;
    treeMemPtr_->SC_eta = etaSC;
    treeMemPtr_->SC_phi = phiSC;
    treeMemPtr_->SC_isBarrel = isBarrel;
    treeMemPtr_->SC_isEndcap = isEndcap;
    treeMemPtr_->dRto1stSC = dRSC_first;
    treeMemPtr_->dRto2ndSC = dRSC_second;
    treeMemPtr_->HcalEnergyIn01 = EnergyHcalIn01;
    treeMemPtr_->HcalEnergyIn02 = EnergyHcalIn02;
    treeMemPtr_->HcalEnergyIn03 = EnergyHcalIn03;
    treeMemPtr_->HcalEnergyIn04 = EnergyHcalIn04;
    treeMemPtr_->HcalEnergyIn05 = EnergyHcalIn05;
    treeMemPtr_->isEcalDriven = isEcalDriven;
    treeMemPtr_->isTrackerDriven = isTrackerDriven;
    treeMemPtr_->RunNumber = iEvent.id().run();
    treeMemPtr_->EvtNumber = iEvent.id().event();

    tree_->Fill();

  }  // loop on tracks
}  // analyze

// ------------ method called once each job just after ending the event loop  ------------
void EopElecTreeWriter::endJob() {
  delete treeMemPtr_;
  treeMemPtr_ = nullptr;
}

// ------------ method called once each job just before starting event loop  ------------
void EopElecTreeWriter::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  bool changed = true;

  // Load trigger configuration at each begin of run
  // Fill the trigger names
  if (hltConfig_.init(iRun, iSetup, "HLT", changed)) {
    if (changed) {
      triggerNames_ = hltConfig_.triggerNames();
    }
  }

  // Displaying the trigger names
  if (debugTriggerSelection_) {
    unsigned int i = 0;
    for (const auto& it : triggerNames_) {
      edm::LogInfo("EopElecTreeWriter") << "HLTpath: " << (i++) << " = " << it;
    }
  }
}

//*************************************************************
void EopElecTreeWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
//*************************************************************
{
  edm::ParameterSetDescription desc;
  desc.setComment("Generate tree for Tracker Alignment E/p validation");
  desc.add<edm::InputTag>("src", edm::InputTag("electronGsfTracks"));
  desc.add<std::string>("triggerPath", "HLT_Ele");
  desc.add<std::string>("hltFilter", "hltDiEle27L1DoubleEGWPTightHcalIsoFilter");
  desc.add<bool>("debugTriggerSelection", false);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EopElecTreeWriter);
