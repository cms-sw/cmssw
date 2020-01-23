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
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

// user include files
#include <DataFormats/TrackReco/interface/Track.h>
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <TMath.h>
#include <TH1.h>
#include <TH2D.h>
#include "TTree.h"
#include <TCanvas.h>
#include <TLorentzVector.h>
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

//#include "RecoParticleFlow/PFRootEvent/interface/JetRecoTypes.h"
//#include "RecoParticleFlow/PFRootEvent/interface/JetMaker.h"
//#include "RecoParticleFlow/PFRootEvent/interface/ProtoJet.h"

#include "RecoEcal/EgammaCoreTools/interface/SuperClusterShapeAlgo.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "Alignment/OfflineValidation/interface/EopElecVariables.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

//Trigger
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Math/interface/deltaR.h"

//Super cluster eta and phi width
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

struct EopTriggerType {
  bool fired;
  Int_t prescale;
  Int_t index;

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

class EopElecTreeWriter : public edm::EDAnalyzer {
public:
  explicit EopElecTreeWriter(const edm::ParameterSet&);
  ~EopElecTreeWriter() override;

private:
  // Cut flow (events number)
  TH1D* nEvents;
  TH1D* nEventsWithVertex;
  TH1D* nEventsTriggered;
  TH1D* nEventsHLTFilter;
  TH1D* nEventsHLTelectron;
  TH1D* nEventsHLTrejected;
  TH1D* nEvents2Elec;

  TH1D* nHLTelectrons;
  TH1D* nTrkRejectedPerEvt;
  TH1D* nTrkSelectedPerEvt;

  // Cut flow (tracks number)
  TH1D* nTracks;
  TH1D* nTracksFiltered;
  TH1D* cut_Ptmin;
  TH1D* cut_OneSCmatch;

  TH1D* counter1;
  TH1D* counter2;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::InputTag src_;

  edm::Service<TFileService> fs_;
  TTree* tree_;
  EopElecVariables* treeMemPtr_;

  HLTConfigProvider hltConfig_;
  std::vector<std::string> triggerNames_;
  TH1D* distToClosestSCgsf;
  TH1D* distToClosestSC;
  TH1D* EcalEnergy;
  TH1D* Momentum;
  TH1D* HcalEnergy;
  TH1D* fBREM;
  TH1D* Eop_InnerNegative;
  TH1D* Eop_InnerPositive;

  TH2D* HcalVSEcal;

  MultiTrajectoryStateTransform* mtsTransform_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexCollectionToken;
  edm::EDGetTokenT<HBHERecHitCollection> theHBHERecHitCollectionToken;
  edm::EDGetTokenT<EcalRecHitCollection> theEcalRecHitCollectionToken;
  edm::EDGetTokenT<reco::SuperClusterCollection> theBarrelSupClusCollectionToken;
  edm::EDGetTokenT<reco::SuperClusterCollection> theEndCapSupClusCollectionToken;
  edm::EDGetTokenT<edm::TriggerResults> theTriggerResultsToken;
  edm::EDGetTokenT<trigger::TriggerEvent> theTriggerEventToken;
  edm::EDGetTokenT<reco::GsfTrackCollection> theGsfTrackCollectionToken;
  edm::EDGetTokenT<reco::GsfElectronCoreCollection> theGsfElectronCoreCollectionToken;
};

// Function to convert the eta of the track to a detector eta (for matching with SC)
float ecalEta(float EtaParticle, float Zvertex, float RhoVertex) {
  const float R_ECAL = 136.5;
  const float Z_Endcap = 328.0;
  const float etaBarrelEndcap = 1.479;

  if (EtaParticle != 0.) {
    float Theta = 0.0;
    float ZEcal = (R_ECAL - RhoVertex) * sinh(EtaParticle) + Zvertex;

    if (ZEcal != 0.0)
      Theta = atan(R_ECAL / ZEcal);
    if (Theta < 0.0)
      Theta = Theta + Geom::pi();

    float ETA = -log(tan(0.5 * Theta));

    if (fabs(ETA) > etaBarrelEndcap) {
      float Zend = Z_Endcap;
      if (EtaParticle < 0.0)
        Zend = -Zend;
      float Zlen = Zend - Zvertex;
      float RR = Zlen / sinh(EtaParticle);
      Theta = atan((RR + RhoVertex) / Zend);
      if (Theta < 0.0)
        Theta = Theta + Geom::pi();
      ETA = -log(tan(0.5 * Theta));
    }
    return ETA;
  } else {
    edm::LogWarning("") << "[EcalPositionFromTrack::etaTransformation] Warning: Eta equals to zero, not correcting";
    return EtaParticle;
  }
}

// constructors and destructor

EopElecTreeWriter::EopElecTreeWriter(const edm::ParameterSet& iConfig) {
  theVertexCollectionToken = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  theHBHERecHitCollectionToken = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco", ""));
  theEcalRecHitCollectionToken = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  theBarrelSupClusCollectionToken = consumes<reco::SuperClusterCollection>(edm::InputTag("hybridSuperClusters", ""));
  theEndCapSupClusCollectionToken =
      consumes<reco::SuperClusterCollection>(edm::InputTag("multi5x5SuperClusters", "multi5x5EndcapSuperClusters"));
  theTriggerResultsToken = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults", "", "HLT"));
  theTriggerEventToken = consumes<trigger::TriggerEvent>(edm::InputTag("hltTriggerSummaryAOD"));

  src_ = iConfig.getParameter<edm::InputTag>("src");
  theGsfTrackCollectionToken = consumes<reco::GsfTrackCollection>(src_);

  theGsfElectronCoreCollectionToken = consumes<reco::GsfElectronCoreCollection>(edm::InputTag("gedGsfElectronCores"));

  // TTree creation
  tree_ = fs_->make<TTree>("EopTree", "EopTree");
  treeMemPtr_ = new EopElecVariables;
  tree_->Branch("EopElecVariables", &treeMemPtr_);  // address of pointer!

  // Control histograms declaration
  distToClosestSC = fs_->make<TH1D>("distToClosestSC", "distToClosestSC", 100, 0, 0.1);
  distToClosestSCgsf = fs_->make<TH1D>("distToClosestSCgsf", "distToClosestSCgsf", 100, 0, 0.1);
  EcalEnergy = fs_->make<TH1D>("EcalEnergy", "EcalEnergy", 100, 0, 200);
  Momentum = fs_->make<TH1D>("Momentum", "Momentum", 100, 0, 200);
  HcalEnergy = fs_->make<TH1D>("HcalEnergy", "HcalEnergy", 100, 0, 40);
  fBREM = fs_->make<TH1D>("fBREM", "fBREM", 100, -0.2, 1);
  Eop_InnerNegative = fs_->make<TH1D>("Eop_InnerNegative", "Eop_InnerNegative", 100, 0, 3);
  Eop_InnerPositive = fs_->make<TH1D>("Eop_InnerPositive", "Eop_InnerPositive", 100, 0, 3);
  HcalVSEcal = fs_->make<TH2D>("HcalVSEcal", "HcalVSEcal", 100, 0, 160, 100, 0, 10);

  nEvents = fs_->make<TH1D>("nEvents", "nEvents", 1, 0, 1);
  nEventsWithVertex = fs_->make<TH1D>("nEventsWithVertex", "nEventsWithVertex", 1, 0, 1);
  nEventsTriggered = fs_->make<TH1D>("nEventsTriggered", "nEventsTriggered", 1, 0, 1);
  nEventsHLTFilter = fs_->make<TH1D>("nEventsHLTFilter", "nEventsHLTFilter", 1, 0, 1);
  nEventsHLTelectron = fs_->make<TH1D>("nEventsHLTelectron", "nEventsHLTelectron", 1, 0, 1);
  nEventsHLTrejected = fs_->make<TH1D>("nEventsHLTrejected", "nEventsHLTrejected", 1, 0, 1);
  nEvents2Elec = fs_->make<TH1D>("nEvents2Elec", "nEvents2Elec", 1, 0, 1);

  nHLTelectrons = fs_->make<TH1D>("nHLTelectrons", "nHLTelectrons", 20, 0, 20);
  nTrkRejectedPerEvt = fs_->make<TH1D>("nTrkRejectedPerEvt", "nTrkRejectedPerEvt", 20, 0, 20);
  nTrkSelectedPerEvt = fs_->make<TH1D>("nTrkSelectedPerEvt", "nTrkSelectedPerEvt", 20, 0, 20);

  nTracks = fs_->make<TH1D>("nTracks", "nTracks", 1, 0, 1);
  nTracksFiltered = fs_->make<TH1D>("nTracksFiltered", "nTracksFiltered", 1, 0, 1);
  cut_Ptmin = fs_->make<TH1D>("cut_Ptmin", "cut_Ptmin", 1, 0, 1);
  cut_OneSCmatch = fs_->make<TH1D>("cut_OneSCmatch", "cut_OneSCmatch", 1, 0, 1);

  counter1 = fs_->make<TH1D>("counter1", "counter1", 1, 0, 1);
  counter2 = fs_->make<TH1D>("counter2", "counter2", 1, 0, 1);
}

EopElecTreeWriter::~EopElecTreeWriter() {
  cout << "Destructor..." << endl;

  // control histograms

  distToClosestSC->SetXTitle("distance from track to closest SuperCluster in eta-phi plan (weighted matching)");
  distToClosestSC->SetYTitle("# Tracks");

  distToClosestSCgsf->SetXTitle("distance from track to closest SuperCluster in eta-phi plan (gsfElectronCore)");
  distToClosestSCgsf->SetYTitle("# Tracks");

  EcalEnergy->SetXTitle("Ecal energy deposit (GeV)");
  EcalEnergy->SetYTitle("# tracks");

  HcalEnergy->SetXTitle("Hcal energy deposit (GeV)");
  HcalEnergy->SetYTitle("# tracks");

  Momentum->SetXTitle("Momentum magnitude (GeV)");
  Momentum->SetYTitle("# tracks");

  Eop_InnerNegative->SetXTitle("E/p");
  Eop_InnerNegative->SetYTitle("# tracks");

  Eop_InnerPositive->SetXTitle("E/p");
  Eop_InnerPositive->SetYTitle("# tracks");

  HcalVSEcal->SetXTitle("Ecal energy (GeV)");
  HcalVSEcal->SetYTitle("Hcal energy (GeV)");
  cout << "Total number of events : " << nEvents->GetEntries() << endl;
}

//###########################################
//#     method called to for each event     #
//###########################################

void EopElecTreeWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  nEvents->Fill(0.5);

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

  edm::ESHandle<MagneticField> magField_;
  edm::ESHandle<TrackerGeometry> trackerGeom_;

  iSetup.get<IdealMagneticFieldRecord>().get(magField_);
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeom_);
  MultiTrajectoryStateTransform mtsTransform(trackerGeom_.product(), magField_.product());

  //---------------    Super Cluster    -----------------

  // getting primary vertex (necessary to convert eta track to eta detector
  edm::Handle<reco::VertexCollection> vertex;
  //iEvent.getByLabel("offlinePrimaryVertices","", vertex);
  iEvent.getByToken(theVertexCollectionToken, vertex);

  if (vertex->empty()) {
    cout << "Error: no primary vertex found!" << endl;
    return;
  }
  reco::Vertex vert;
  vert = vertex->front();
  nEventsWithVertex->Fill(0.5);

  // getting calorimeter geometry
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<CaloGeometryRecord>().get(geometry);
  const CaloGeometry* geo = geometry.product();
  const CaloSubdetectorGeometry* subGeo = geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  if (subGeo == nullptr)
    cout << "ERROR: unable to find SubDetector geometry!!!" << std::endl;

  // getting Hcal rechits
  edm::Handle<HBHERecHitCollection> HcalHits;
  //iEvent.getByLabel("hbhereco","", HcalHits);
  iEvent.getByToken(theHBHERecHitCollectionToken, HcalHits);

  // getting Ecal rechits
  edm::Handle<EcalRecHitCollection> ecalrechitcollection;
  //iEvent.getByLabel("ecalRecHit","EcalRecHitsEB", ecalrechitcollection);
  iEvent.getByToken(theEcalRecHitCollectionToken, ecalrechitcollection);
  const EcalRecHitCollection* rhc = ecalrechitcollection.product();
  if (rhc == nullptr)
    cout << "ERROR !!!" << std::endl;

  // getting SuperCluster
  edm::Handle<reco::SuperClusterCollection> BarrelSupClusCollection;
  edm::Handle<reco::SuperClusterCollection> EndcapSupClusCollection;
  //iEvent.getByLabel("hybridSuperClusters","", BarrelSupClusCollection);
  //iEvent.getByLabel("multi5x5SuperClusters","multi5x5EndcapSuperClusters", EndcapSupClusCollection);
  iEvent.getByToken(theBarrelSupClusCollectionToken, BarrelSupClusCollection);
  iEvent.getByToken(theEndCapSupClusCollectionToken, EndcapSupClusCollection);

  //iEvent.getByLabel("hfEMClusters","", tmpSupClusCollection);
  //iEvent.getByLabel("correctedHybridSuperClusters","", tmpSupClusCollection);

  // necessary to re-calculate phi and eta width of SuperClusters
  SuperClusterShapeAlgo SCShape(rhc, subGeo);

  //---------------    Trigger   -----------------

  TrigTag = false;
  const edm::InputTag triggerTag("TriggerResults", "", "HLT");

  edm::Handle<edm::TriggerResults> trigRes;
  //iEvent.getByLabel(triggerTag, trigRes);
  iEvent.getByToken(theTriggerResultsToken, trigRes);

  // trigger event
  edm::Handle<trigger::TriggerEvent> triggerEvent;
  //iEvent.getByLabel("hltTriggerSummaryAOD", triggerEvent );
  iEvent.getByToken(theTriggerEventToken, triggerEvent);

  //std::string pattern = "HLT_Ele32_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_v5";
  std::string pattern = "HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v4";

  // our trigger table
  std::map<std::string, EopTriggerType> HLTpaths;
  for (unsigned int i = 0; i < triggerNames_.size(); i++) {
    if (triggerNames_[i].find(pattern) != 0)
      continue;
    EopTriggerType myTrigger;

    const unsigned int prescaleSize = hltConfig_.prescaleSize();
    for (unsigned int ps = 0; ps < prescaleSize; ps++) {
      const unsigned int prescaleValue = hltConfig_.prescaleValue(ps, triggerNames_[i]);
      if (prescaleValue != 1) {
        myTrigger.prescale = prescaleValue;
      }
    }

    myTrigger.index = hltConfig_.triggerIndex(triggerNames_[i]);
    if (myTrigger.index == -1)
      continue;
    myTrigger.fired =
        trigRes->wasrun(myTrigger.index) && trigRes->accept(myTrigger.index) && !trigRes->error(myTrigger.index);
    HLTpaths[triggerNames_[i]] = myTrigger;
  }

  // First cut : trigger cut
  std::string firstFiredPath = "";
  for (std::map<std::string, EopTriggerType>::const_iterator it = HLTpaths.begin(); it != HLTpaths.end(); it++) {
    if (it->second.fired) {
      TrigTag = true;
      firstFiredPath = it->first;
      break;
    }
  }
  if (!TrigTag)
    return;
  nEventsTriggered->Fill(0.5);

  // Displaying filters label from the first fired trigger
  // Useful for finding the good filter label
  std::vector<std::string> filters = hltConfig_.moduleLabels(firstFiredPath);

  /*
   std::cout << "filters : ";
   for (unsigned int i=0;i<filters.size();i++){ 
   std::cout << filters[i]<<" ";
   }
   std::cout<<std::endl;
 */

  // Getting HLT electrons
  //edm::InputTag testTag("hltEle32CaloIdVLCaloIsoVLTrkIdVLTrkIsoVLTrackIsoFilter","","HLT");
  edm::InputTag testTag("hltDiEle27L1DoubleEGWPTightHcalIsoFilter", "", "HLT");
  int testindex = triggerEvent->filterIndex(testTag);

  if (testindex >= triggerEvent->sizeFilters())
    return;
  nEventsHLTFilter->Fill(0.5);

  const trigger::Keys& KEYS_el(triggerEvent->filterKeys(testindex));

  std::vector<const trigger::TriggerObject*> HLTelectrons;
  for (unsigned int i = 0; i != KEYS_el.size(); ++i) {
    const trigger::TriggerObject* triggerObject_el = &(triggerEvent->getObjects().at(KEYS_el[i]));
    HLTelectrons.push_back(triggerObject_el);
  }

  nHLTelectrons->Fill(HLTelectrons.size());

  if (HLTelectrons.empty())
    return;
  nEventsHLTelectron->Fill(0.5);

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
  edm::Handle<reco::GsfTrackCollection> tracks;
  //iEvent.getByLabel(src_, tracks);
  iEvent.getByToken(theGsfTrackCollectionToken, tracks);

  // filtering track
  int nRejected = 0;
  int nSelected = 0;
  std::vector<const reco::GsfTrack*> filterTracks;
  for (unsigned int i = 0; i < tracks->size(); i++) {
    nTracks->Fill(0.5);
    double deltar = reco::deltaR(
        (*tracks)[i].eta(), (*tracks)[i].phi(), HLTelectrons[HighPtIndex]->eta(), HLTelectrons[HighPtIndex]->phi());
    // remove the triggered electron with highest pt
    if (deltar < 0.025) {
      treeMemPtr_->px_rejected_track = (*tracks)[i].px();
      treeMemPtr_->py_rejected_track = (*tracks)[i].py();
      treeMemPtr_->pz_rejected_track = (*tracks)[i].pz();
      treeMemPtr_->p_rejected_track = (*tracks)[i].p();
      nRejected++;
      continue;
    }
    filterTracks.push_back(&(*tracks)[i]);  // we use all the others
    nSelected++;
    nTracksFiltered->Fill(0.5);
  }
  nTrkRejectedPerEvt->Fill(nRejected);
  nTrkSelectedPerEvt->Fill(nSelected);

  if (nRejected == 0)
    return;
  nEventsHLTrejected->Fill(0.5);

  if (filterTracks.empty())
    return;
  nEvents2Elec->Fill(0.5);

  //-------- test:Matching SC/track using gsfElectonCore collection --------

  edm::Handle<reco::GsfElectronCoreCollection> electrons;
  //iEvent.getByLabel("gsfElectronCores", electrons);
  iEvent.getByToken(theGsfElectronCoreCollectionToken, electrons);

  for (std::vector<reco::GsfElectronCore>::const_iterator elec = electrons->begin(); elec != electrons->end(); elec++) {
    double etaGSF = ecalEta((elec->gsfTrack())->eta(), vert.z(), (vert.position()).rho());
    if ((elec->gsfTrack())->pt() < 10.)
      continue;

    double DELTAR = 0;
    DELTAR =
        reco::deltaR((elec->superCluster())->eta(), (elec->superCluster())->phi(), etaGSF, (elec->gsfTrack())->phi());

    if (DELTAR < 0.1)
      distToClosestSCgsf->Fill(DELTAR);
  }

  //------------------------------------------------------------

  //--------------    Loop on tracks   -----------------

  for (std::vector<const reco::GsfTrack*>::const_iterator itrack = filterTracks.begin(); itrack != filterTracks.end();
       ++itrack) {
    const reco::GsfTrack* track = (*itrack);

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
    Momentum->Fill(track->p());
    if (track->pt() < 10.)
      continue;
    cut_Ptmin->Fill(0.5);

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
      fBREM->Fill(fbrem);
    }

    // Matching track with Hcal rec hits
    EnergyHcalIn01 = 0;
    EnergyHcalIn02 = 0;
    EnergyHcalIn03 = 0;
    EnergyHcalIn04 = 0;
    EnergyHcalIn05 = 0;

    for (std::vector<HBHERecHit>::const_iterator hcal = (*HcalHits).begin(); hcal != (*HcalHits).end(); hcal++) {
      GlobalPoint posH = geo->getPosition((*hcal).detid());
      double phihit = posH.phi();
      double etahit = posH.eta();
      double dR = reco::deltaR(etahit, phihit, etaOut, phiOut);

      // saving Hcal energy deposit measured for different eta-phi radius
      if (dR < 0.1)
        EnergyHcalIn01 += hcal->energy();
      if (dR < 0.2)
        EnergyHcalIn02 += hcal->energy();
      if (dR < 0.3)
        EnergyHcalIn03 += hcal->energy();
      if (dR < 0.4)
        EnergyHcalIn04 += hcal->energy();
      if (dR < 0.5)
        EnergyHcalIn05 += hcal->energy();
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

    for (std::vector<const reco::GsfTrack*>::const_iterator itrack1 = filterTracks.begin();
         itrack1 != filterTracks.end();
         ++itrack1) {
      const reco::GsfTrack* track1 = (*itrack1);

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
    etaAtEcal = ecalEta(etaIn, vert.z(), (vert.position()).rho());

    //Barrel
    if (track->eta() < 1.55 || track->eta() > -1.55) {
      for (std::vector<reco::SuperCluster>::const_iterator SC = BarrelSupClusCollection->begin();
           SC != BarrelSupClusCollection->end();
           SC++) {
        dRSC = 0;
        dRSC = reco::deltaR(SC->eta(), SC->phi(), etaAtEcal, phiIn);

        if (dRSC < dRSC_first) {
          dRSC_first = dRSC;
        }  // distance in eta-phi plan to closest SC
        else if (dRSC < dRSC_second) {
          dRSC_second = dRSC;
        }  // to next closest SC

        if (dRSC < 0.09) {
          //Calculate phiWidth & etaWidth for associated SuperClusters
          SCShape.Calculate_Covariances(*SC);
          phiWidth = SCShape.phiWidth();
          etaWidth = SCShape.etaWidth();

          etaSC = SC->eta();
          phiSC = SC->phi();

          algo_ID = SC->algoID();
          EnergyEcal = SC->energy();
          nBasicClus = SC->clustersSize();
          nbSC++;
          isBarrel = true;
        }
      }
    }

    // Endcap
    if (track->eta() < -1.44 || track->eta() > 1.44) {
      for (std::vector<reco::SuperCluster>::const_iterator SC = EndcapSupClusCollection->begin();
           SC != EndcapSupClusCollection->end();
           SC++) {
        dRSC = 0;
        dRSC = reco::deltaR(SC->eta(), SC->phi(), etaAtEcal, phiIn);

        if (dRSC < dRSC_first) {
          dRSC_first = dRSC;
        }  // distance in eta-phi plan to closest SC
        else if (dRSC < dRSC_second) {
          dRSC_second = dRSC;
        }  // to next closest SC

        if (dRSC < 0.09) {
          //Calculate phiWidth & etaWidth for associated SuperClusters
          SCShape.Calculate_Covariances(*SC);
          phiWidth = SCShape.phiWidth();
          etaWidth = SCShape.etaWidth();

          etaSC = SC->eta();
          phiSC = SC->phi();

          algo_ID = SC->algoID();
          EnergyEcal = SC->energy();
          nBasicClus = SC->clustersSize();
          nbSC++;
          isEndcap = true;
        }
      }
    }

    if (dRSC_first < 0.1)
      distToClosestSC->Fill(dRSC_first);
    if (nbSC == 1)
      counter1->Fill(0.5);
    if (nbSC == 0)
      counter2->Fill(0.5);

    if (nbSC > 1 || nbSC == 0)
      continue;
    cut_OneSCmatch->Fill(0.5);

    if (isBarrel && isEndcap) {
      cout << "Error: Super Cluster double matching!" << endl;
      return;
    }

    EcalEnergy->Fill(EnergyEcal);
    HcalVSEcal->Fill(EnergyEcal, EnergyHcalIn02);

    // E over p plots
    if (track->charge() < 0)
      Eop_InnerNegative->Fill(EnergyEcal / pin);
    if (track->charge() > 0)
      Eop_InnerPositive->Fill(EnergyEcal / pin);

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

// ------------ method called once each job just before starting event loop  ------------
void EopElecTreeWriter::beginJob() {}

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
  unsigned int i = 0;
  for (std::vector<std::string>::const_iterator it = triggerNames_.begin(); it < triggerNames_.end(); ++it) {
    std::cout << "HLTpath: " << (i++) << " = " << (*it) << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(EopElecTreeWriter);
