// -*- C++ -*-
//
// Package:    DQM/HLTObjectMonitor
// Class:      HLTObjectMonitor
//
/**\class HLTObjectMonitor HLTObjectMonitor.cc DQM/HLTEvF/plugins/HLTObjectMonitor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Charles Nicholas Mueller
//         Created:  Sun, 22 Mar 2015 22:29:00 GMT
//
//


// system include files
#include <memory>
#include <sys/time.h>
#include <cstdlib>

// user include files
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//for collections
#include "HLTrigger/JetMET/interface/AlphaT.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
//#include "HLTrigger/JetMET/interface/HLTRHemisphere.h"


// #include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
// #include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
// #include "CommonTools/RecoAlgos/interface/TrackSelector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TMath.h"
#include "TStyle.h"
#include "TLorentzVector.h"

#include <unordered_map>
//
// class declaration
//

//using namespace edm;
using namespace trigger;
using std::vector;
using std::string;
using std::unordered_map;

class HLTObjectMonitor : public DQMEDAnalyzer {
   public:
      explicit HLTObjectMonitor(const edm::ParameterSet&);
      ~HLTObjectMonitor();

  //      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;
      void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

      void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void fillPlots(int, string, edm::Handle<trigger::TriggerEvent>);
  double get_wall_time(void);
      // ----------member data ---------------------------

  bool debugPrint;

  HLTConfigProvider hltConfig_;

  string topDirectoryName;
  string mainShifterFolder;
  string backupFolder;

  unordered_map<string, unsigned int> lookupIndex;
  unordered_map<string, string> lookupFilter;
  vector<string> quickCollectionPaths;

  //set Token(-s)
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> aodTriggerToken_;
  edm::EDGetTokenT<LumiScalersCollection> lumiScalersToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;  
  edm::EDGetTokenT<vector<reco::MET>> metToken_;  
  edm::EDGetTokenT<vector<reco::RecoChargedCandidate>> chargedCandToken_;  
  edm::EDGetTokenT<reco::JetTagCollection> csvCaloTagsToken_;
  edm::EDGetTokenT<reco::JetTagCollection> csvPfTagsToken_;
  edm::EDGetTokenT<vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>>> rHemisphereToken_;
  

  // edm::EDGetTokenT<edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>> csvCaloToken_;  
  // edm::EDGetTokenT<edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>> csvPfToken_;  
  
  // edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> siPixelClusterToken_;
  // edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClusterToken_;
  // edm::EDGetTokenT<TrackingRecHitCollection> trackingRecHitsToken_;  
  // edm::EDGetTokenT<reco::TrackExtraCollection> trackExtraToken_;  
  // edm::EDGetTokenT<reco::TrackCollection> trackToken_;  

  //declare params
  edm::ParameterSet alphaT_TH1;
  edm::ParameterSet photonPt_TH1;
  edm::ParameterSet photonEta_TH1;
  edm::ParameterSet photonPhi_TH1;
  edm::ParameterSet muonPt_TH1;
  edm::ParameterSet muonEta_TH1;
  edm::ParameterSet muonPhi_TH1;
  edm::ParameterSet electronPt_TH1;
  edm::ParameterSet electronEta_TH1;
  edm::ParameterSet electronPhi_TH1;
  edm::ParameterSet jetPt_TH1;
  edm::ParameterSet jetAK8Pt_TH1;
  edm::ParameterSet jetAK8Mass_TH1;
  edm::ParameterSet tauPt_TH1;
  edm::ParameterSet diMuonLowMass_TH1;
  edm::ParameterSet caloMetPt_TH1;
  edm::ParameterSet caloMetPhi_TH1;
  edm::ParameterSet pfMetPt_TH1;
  edm::ParameterSet pfMetPhi_TH1;
  edm::ParameterSet caloHtPt_TH1;
  edm::ParameterSet pfHtPt_TH1;
  edm::ParameterSet bJetPhi_TH1;
  edm::ParameterSet bJetEta_TH1;

  //setup path names
  string alphaT_pathName;
  string photonPlots_pathName;
  string muonPlots_pathName;
  string electronPlots_pathName;
  string jetPt_pathName;
  string tauPt_pathName;
  string diMuonLowMass_pathName;
  string caloMetPlots_pathName;
  string pfMetPlots_pathName;
  string caloHtPt_pathName;
  string pfHtPt_pathName;
  string bJetPlots_pathName;
  string bJetPlots_pathNameOR;
  string jetAK8Plots_pathName;


  //declare all MEs
  MonitorElement * alphaT_;
  MonitorElement * photonPt_;
  MonitorElement * photonEta_;
  MonitorElement * photonPhi_;
  MonitorElement * muonPt_;
  MonitorElement * muonEta_;
  MonitorElement * muonPhi_;
  MonitorElement * electronPt_;
  MonitorElement * electronEta_;
  MonitorElement * electronPhi_;
  MonitorElement * jetPt_;
  MonitorElement * jetAK8Pt_;
  MonitorElement * jetAK8Mass_;
  MonitorElement * tauPt_;
  MonitorElement * diMuonLowMass_;
  MonitorElement * caloMetPt_;
  MonitorElement * caloMetPhi_;
  MonitorElement * pfMetPt_;
  MonitorElement * pfMetPhi_;
  MonitorElement * caloHtPt_;
  MonitorElement * pfHtPt_;
  MonitorElement * bJetPhi_;
  MonitorElement * bJetEta_;

  MonitorElement * wallTimePerEvent_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HLTObjectMonitor::HLTObjectMonitor(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  debugPrint = false;

  topDirectoryName = "HLT/ObjectMonitor";
  mainShifterFolder = topDirectoryName+"/MainShifter";
  backupFolder = topDirectoryName+"/Backup";


  //parse params
  alphaT_TH1 = iConfig.getParameter<edm::ParameterSet> ("alphaT");
  photonPt_TH1 = iConfig.getParameter<edm::ParameterSet>("photonPt");
  photonEta_TH1 = iConfig.getParameter<edm::ParameterSet>("photonEta");
  photonPhi_TH1 = iConfig.getParameter<edm::ParameterSet>("photonPhi");
  muonPt_TH1 = iConfig.getParameter<edm::ParameterSet>("muonPt");
  muonEta_TH1 = iConfig.getParameter<edm::ParameterSet>("muonEta");
  muonPhi_TH1 = iConfig.getParameter<edm::ParameterSet>("muonPhi");
  electronPt_TH1 = iConfig.getParameter<edm::ParameterSet>("electronPt");
  electronEta_TH1 = iConfig.getParameter<edm::ParameterSet>("electronEta");
  electronPhi_TH1 = iConfig.getParameter<edm::ParameterSet>("electronPhi");
  jetPt_TH1 = iConfig.getParameter<edm::ParameterSet>("jetPt");
  jetAK8Pt_TH1 = iConfig.getParameter<edm::ParameterSet>("jetAK8Pt");
  jetAK8Mass_TH1 = iConfig.getParameter<edm::ParameterSet>("jetAK8Mass");
  tauPt_TH1 = iConfig.getParameter<edm::ParameterSet>("tauPt");
  diMuonLowMass_TH1 = iConfig.getParameter<edm::ParameterSet>("diMuonLowMass");
  caloMetPt_TH1 = iConfig.getParameter<edm::ParameterSet>("caloMetPt");
  caloMetPhi_TH1 = iConfig.getParameter<edm::ParameterSet>("caloMetPhi");
  pfMetPt_TH1 = iConfig.getParameter<edm::ParameterSet>("pfMetPt");
  pfMetPhi_TH1 = iConfig.getParameter<edm::ParameterSet>("pfMetPhi");
  caloHtPt_TH1 = iConfig.getParameter<edm::ParameterSet>("caloHtPt");
  pfHtPt_TH1 = iConfig.getParameter<edm::ParameterSet>("pfHtPt");
  bJetPhi_TH1 = iConfig.getParameter<edm::ParameterSet>("bJetPhi");
  bJetEta_TH1 = iConfig.getParameter<edm::ParameterSet>("bJetEta");

  //set Token(s) 
  //will need to change 'TEST' to 'HLT' or something else before implementation
  triggerResultsToken_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults","", "TEST"));
  aodTriggerToken_ = consumes<trigger::TriggerEvent>(edm::InputTag("hltTriggerSummaryAOD", "", "TEST"));
  lumiScalersToken_ = consumes<LumiScalersCollection>(edm::InputTag("hltScalersRawToDigi","",""));
  beamSpotToken_ = consumes<reco::BeamSpot>(edm::InputTag("hltOnlineBeamSpot","","TEST")); 
  metToken_ = consumes<vector<reco::MET>>(edm::InputTag("hltPFMETProducer","","TEST"));  
  chargedCandToken_ = consumes<vector<reco::RecoChargedCandidate>>(edm::InputTag("hltL3NoFiltersNoVtxMuonCandidates","","TEST"));  
  csvCaloTagsToken_ = consumes<reco::JetTagCollection>(edm::InputTag("hltCombinedSecondaryVertexBJetTagsCalo","","TEST"));
  csvPfTagsToken_ = consumes<reco::JetTagCollection>(edm::InputTag("hltCombinedSecondaryVertexBJetTagsPF","","TEST"));
  rHemisphereToken_ = consumes<vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>>>(edm::InputTag("hltRHemisphere","","TEST"));
  // csvCaloTagsToken_ = consumes<edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>>(edm::InputTag("hltCombinedSecondaryVertexBJetTagsCalo","","TEST"));  
  // csvPfTagsToken_ = consumes<edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>>(edm::InputTag("hltCombinedSecondaryVertexBJetTagsPF","","TEST"));  
  //  siPixelClusterToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(edm::InputTag("hltSiPixelClusters","","TEST"));
  // siStripClusterToken_ = consumes<edmNew::DetSetVector<SiStripCluster>>(edm::InputTag("hltSiStripRawToClustersFacility","","TEST"));
  // trackingRecHitsToken_ = consumes<TrackingRecHitCollection>(edm::InputTag("hltIter2Merged","","TEST"));
  // trackExtraToken_ = consumes<reco::TrackExtraCollection>(edm::InputTag("hltIter2Merged","","TEST"));
  // trackToken_ = consumes<reco::TrackCollection>(edm::InputTag("hltIter2Merged","","TEST"));

  // use this csvTagToken_ = consumes<reco::JetTagCollection>(InputTag("hltCombinedSecondaryVertexBJetTagsPF","","TEST")); 
  // prob not this csvTagToken_ = consumes<edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>>(InputTag("hltCombinedSecondaryVertexBJetTagsPF","","TEST")); 

}


HLTObjectMonitor::~HLTObjectMonitor()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTObjectMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  double start = get_wall_time();
  
  using namespace edm;
  
   if (debugPrint) std::cout << "Inside analyze(). " << std::endl;
   int eventNumber = iEvent.id().event();

   // access trigger results
   edm::Handle<edm::TriggerResults> triggerResults;
   iEvent.getByToken(triggerResultsToken_, triggerResults);
   if (!triggerResults.isValid()) return;

   edm::Handle<trigger::TriggerEvent> aodTriggerEvent;
   iEvent.getByToken(aodTriggerToken_, aodTriggerEvent);
   if (!aodTriggerEvent.isValid()) return;

   edm::Handle<reco::BeamSpot> recoBeamSpot;
   iEvent.getByToken(beamSpotToken_, recoBeamSpot);

   edm::Handle<vector<reco::MET>> recoMet;
   iEvent.getByToken(metToken_, recoMet);

   edm::Handle<vector<reco::RecoChargedCandidate>> recoChargedCands;
   iEvent.getByToken(chargedCandToken_, recoChargedCands);

   edm::Handle<reco::JetTagCollection> csvCaloTags;
   iEvent.getByToken(csvCaloTagsToken_, csvCaloTags);

   edm::Handle<reco::JetTagCollection> csvPfTags;
   iEvent.getByToken(csvPfTagsToken_, csvPfTags);

   edm::Handle<vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>>> rHemisphere;
   iEvent.getByToken(rHemisphereToken_, rHemisphere);

   // edm::Handle<edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>> csvCaloTags;
   // iEvent.getByToken(csvCaloToken_, csvCaloTags);

   // edm::Handle<edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>> csvPfTags;
   // iEvent.getByToken(csvPfToken_, csvPfTags);

   // edm::Handle<edmNew::DetSetVector<SiPixelCluster>> siPixelCluster;
   // iEvent.getByToken(siPixelClusterToken_, siPixelCluster);

   // edm::Handle<edmNew::DetSetVector<SiStripCluster>> siStripCluster;
   // iEvent.getByToken(siStripClusterToken_, siStripCluster);

   // edm::Handle<TrackingRecHitCollection> trackingRecHits;
   // iEvent.getByToken(trackingRecHitsToken_, trackingRecHits);

   // edm::Handle<reco::TrackExtraCollection> trackExtras;
   // iEvent.getByToken(trackExtraToken_, trackExtras);

   // edm::Handle<reco::TrackCollection> tracks;
   // iEvent.getByToken(trackToken_, tracks);

   for (string & pathName : quickCollectionPaths) //loop over paths
     {
       if (triggerResults->accept(lookupIndex[pathName]) && hltConfig_.saveTags(lookupFilter[pathName]))
	 {
	   fillPlots(eventNumber, pathName, aodTriggerEvent);
	 }
     }

   //   sleep(1); //sleep for 1s, used to calibrate timing 
   double end = get_wall_time();
   double wallTime = end - start;
   wallTimePerEvent_->Fill(wallTime);
}

// ------------ method called when starting to processes a run  ------------
void
HLTObjectMonitor::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  if (debugPrint) std::cout << "Calling beginRun. " << std::endl;
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, "TEST", changed))
    {
      if (debugPrint) std::cout << "Extracting HLTconfig. " << std::endl;
    }
  
  string pathName_noVersion;
  vector<string> datasetPaths;
  
  //ATTN!!! this will need to be changed to proper DQM stream before deploying
  vector<string> datasetNames = hltConfig_.streamContent("A");
  for (unsigned int i=0;i<datasetNames.size();i++) {
    datasetPaths = hltConfig_.datasetContent(datasetNames[i]);
    for (const auto & pathName : datasetPaths){
      pathName_noVersion = hltConfig_.removeVersion(pathName);
      //only add unique pathNames (keys) to the lookup table by only adding it if it's no t already in the table
      if (lookupIndex.find(pathName_noVersion) == lookupIndex.end()){
	lookupIndex[pathName_noVersion] = hltConfig_.triggerIndex(pathName);
      }
    }
  }

  // setup string names
  alphaT_pathName = alphaT_TH1.getParameter<string>("pathName");
  photonPlots_pathName = photonPt_TH1.getParameter<string>("pathName");
  muonPlots_pathName = muonPt_TH1.getParameter<string>("pathName");
  electronPlots_pathName = electronPt_TH1.getParameter<string>("pathName");
  jetPt_pathName = jetPt_TH1.getParameter<string>("pathName");
  jetAK8Plots_pathName = jetAK8Pt_TH1.getParameter<string>("pathName");
  tauPt_pathName = tauPt_TH1.getParameter<string>("pathName");
  diMuonLowMass_pathName = diMuonLowMass_TH1.getParameter<string>("pathName");
  caloMetPlots_pathName = caloMetPt_TH1.getParameter<string>("pathName");
  pfMetPlots_pathName = pfMetPt_TH1.getParameter<string>("pathName");
  caloHtPt_pathName = caloHtPt_TH1.getParameter<string>("pathName");
  pfHtPt_pathName = pfHtPt_TH1.getParameter<string>("pathName");
  bJetPlots_pathName = bJetPhi_TH1.getParameter<string>("pathName");
  bJetPlots_pathNameOR = bJetPhi_TH1.getParameter<string>("pathName_OR");

  //link all paths and filters needed
  
  if (lookupIndex.count(photonPlots_pathName) > 0)
    {
      quickCollectionPaths.push_back(photonPlots_pathName);
      lookupFilter[photonPlots_pathName] = photonPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(muonPlots_pathName) > 0)
    {
      quickCollectionPaths.push_back(muonPlots_pathName);
      lookupFilter[muonPlots_pathName] = muonPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(electronPlots_pathName) > 0)
    {
      quickCollectionPaths.push_back(electronPlots_pathName);
      lookupFilter[electronPlots_pathName] = electronPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(diMuonLowMass_pathName) > 0)
    {
      quickCollectionPaths.push_back(diMuonLowMass_pathName);
      lookupFilter[diMuonLowMass_pathName] = diMuonLowMass_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(alphaT_pathName) > 0)
    {
      quickCollectionPaths.push_back(alphaT_TH1.getParameter<string> ("pathName"));
      lookupFilter[alphaT_pathName] = alphaT_TH1.getParameter<string> ("moduleName");
    }
  if (lookupIndex.count(jetPt_pathName) > 0)
    {
      quickCollectionPaths.push_back(jetPt_pathName);
      lookupFilter[jetPt_pathName] = jetPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(jetAK8Plots_pathName) > 0)
    {
      quickCollectionPaths.push_back(jetAK8Plots_pathName);
      lookupFilter[jetAK8Plots_pathName] = jetAK8Pt_TH1.getParameter<string>("moduleName");      
    }
  if (lookupIndex.count(caloMetPlots_pathName) >0)
    {
      quickCollectionPaths.push_back(caloMetPlots_pathName);
      lookupFilter[caloMetPlots_pathName] = caloMetPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(pfHtPt_pathName) > 0)
    {
      quickCollectionPaths.push_back(pfHtPt_pathName);
      lookupFilter[pfHtPt_pathName] = pfHtPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(tauPt_pathName) > 0)
    {
      quickCollectionPaths.push_back(tauPt_pathName);
      lookupFilter[tauPt_pathName] = tauPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(pfMetPlots_pathName) >0)
    {
      quickCollectionPaths.push_back(pfMetPlots_pathName);
      lookupFilter[pfMetPlots_pathName] = pfMetPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(caloHtPt_pathName) >0)
    {
      quickCollectionPaths.push_back(caloHtPt_pathName);
      lookupFilter[caloHtPt_pathName] = caloHtPt_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(bJetPlots_pathName) >0)
    {
      quickCollectionPaths.push_back(bJetPlots_pathName);
      lookupFilter[bJetPlots_pathName] = bJetPhi_TH1.getParameter<string>("moduleName");
    }
  if (lookupIndex.count(bJetPlots_pathNameOR) >0)
    {
      quickCollectionPaths.push_back(bJetPlots_pathNameOR);
      lookupFilter[bJetPlots_pathNameOR] = bJetEta_TH1.getParameter<string>("moduleName_OR");
    }


}

// ------------ method called when ending the processing of a run  ------------

void
HLTObjectMonitor::endRun(edm::Run const&, edm::EventSetup const&)
{
  if (debugPrint) std::cout << "Calling endRun. " << std::endl;
}

void HLTObjectMonitor::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  
  ////////////////////////////////
  ///
  /// Main shifter workspace plots
  ///
  ////////////////////////////////
  ibooker.setCurrentFolder(mainShifterFolder);

  //wall time
  TH1F * hist_wallTime = new TH1F("wallTime","wall time per event",1000,0,0.01);
  hist_wallTime->SetMinimum(0);
  wallTimePerEvent_ = ibooker.book1D("wallTime",hist_wallTime);

  //photon pt
  if (lookupIndex.count(photonPlots_pathName) > 0)
    {
      TH1F * hist_photonPt = new TH1F("Photon_pT","Photon pT",photonPt_TH1.getParameter<int>("NbinsX"),photonPt_TH1.getParameter<int>("Xmin"),photonPt_TH1.getParameter<int>("Xmax"));
      hist_photonPt->SetMinimum(0);
      photonPt_ = ibooker.book1D("Photon_pT",hist_photonPt);
    }
  //muon pt
  if (lookupIndex.count(muonPlots_pathName) > 0)
    {
      TH1F * hist_muonPt = new TH1F("Muon_pT","Muon pT",muonPt_TH1.getParameter<int>("NbinsX"),muonPt_TH1.getParameter<int>("Xmin"),muonPt_TH1.getParameter<int>("Xmax"));
      hist_muonPt->SetMinimum(0);
      muonPt_ = ibooker.book1D("Muon_pT",hist_muonPt);
    }
  //electron pt
  if (lookupIndex.count(electronPlots_pathName) > 0)
    {
      TH1F * hist_electronPt = new TH1F("Electron_pT","Electron pT",electronPt_TH1.getParameter<int>("NbinsX"),electronPt_TH1.getParameter<int>("Xmin"),electronPt_TH1.getParameter<int>("Xmax"));
      hist_electronPt->SetMinimum(0);
      electronPt_ = ibooker.book1D("Electron_pT",hist_electronPt);
    }
  //jet pt
  if (lookupIndex.count(jetPt_pathName) > 0)
    {
      TH1F * hist_jetPt = new TH1F("Jet_pT","Jet pT",jetPt_TH1.getParameter<int>("NbinsX"),jetPt_TH1.getParameter<int>("Xmin"),jetPt_TH1.getParameter<int>("Xmax"));
      hist_jetPt->SetMinimum(0);
      jetPt_ = ibooker.book1D("Jet_pT",hist_jetPt);
    }
  //jetAK8 pt + mass
  if (lookupIndex.count(jetAK8Plots_pathName) > 0)
    {
      //ak8 jet pt
      TH1F * hist_jetAK8Pt = new TH1F("JetAK8_pT","JetAK8 pT",jetAK8Pt_TH1.getParameter<int>("NbinsX"),jetAK8Pt_TH1.getParameter<int>("Xmin"),jetAK8Pt_TH1.getParameter<int>("Xmax"));
      hist_jetAK8Pt->SetMinimum(0);
      jetAK8Pt_ = ibooker.book1D("JetAK8_pT",hist_jetAK8Pt);
      //ak8 jet mass
      TH1F * hist_jetAK8Mass = new TH1F("JetAK8_mass","JetAK8 mass",jetAK8Mass_TH1.getParameter<int>("NbinsX"),jetAK8Mass_TH1.getParameter<int>("Xmin"),jetAK8Mass_TH1.getParameter<int>("Xmax"));
      hist_jetAK8Mass->SetMinimum(0);
      jetAK8Mass_ = ibooker.book1D("JetAK8_mass",hist_jetAK8Mass);
    }
  //tau pt
  if (lookupIndex.count(tauPt_pathName) > 0)
    {
      TH1F * hist_tauPt = new TH1F("Tau_pT","Tau pT",tauPt_TH1.getParameter<int>("NbinsX"),tauPt_TH1.getParameter<int>("Xmin"),tauPt_TH1.getParameter<int>("Xmax"));
      hist_tauPt->SetMinimum(0);
      tauPt_ = ibooker.book1D("Tau_pT",hist_tauPt);
    }
  //dimuon low mass
  if (lookupIndex.count(diMuonLowMass_pathName) > 0)
    {
      TH1F * hist_dimuonLowMass = new TH1F("Dimuon_LowMass","Dimuon Low Mass",diMuonLowMass_TH1.getParameter<int>("NbinsX"),diMuonLowMass_TH1.getParameter<double>("Xmin"),diMuonLowMass_TH1.getParameter<double>("Xmax"));
      hist_dimuonLowMass->SetMinimum(0);
      diMuonLowMass_ = ibooker.book1D("Dimuon_LowMass",hist_dimuonLowMass);
    }
  //alphaT
  if (lookupIndex.count(alphaT_pathName) > 0)
    {
      TH1F * hist_alphaT = new TH1F("alphaT","alphaT",alphaT_TH1.getParameter<int>("NbinsX"),alphaT_TH1.getParameter<int>("Xmin"),alphaT_TH1.getParameter<int>("Xmax"));
      hist_alphaT->SetMinimum(0);
      alphaT_ = ibooker.book1D("AlphaT",hist_alphaT);
    }
  //caloMET pt
  if (lookupIndex.count(caloMetPlots_pathName) >0)
    {
      TH1F * hist_caloMetPt = new TH1F("CaloMET_pT","CaloMET pT",caloMetPt_TH1.getParameter<int>("NbinsX"),caloMetPt_TH1.getParameter<int>("Xmin"),caloMetPt_TH1.getParameter<int>("Xmax"));
      hist_caloMetPt->SetMinimum(0);
      caloMetPt_ = ibooker.book1D("CaloMET_pT",hist_caloMetPt);
    }
  //caloHT pt
  if (lookupIndex.count(caloHtPt_pathName) >0)
    {
      TH1F * hist_caloHtPt = new TH1F("CaloHT_pT","CaloHT pT",caloHtPt_TH1.getParameter<int>("NbinsX"),caloHtPt_TH1.getParameter<int>("Xmin"),caloHtPt_TH1.getParameter<int>("Xmax"));
      hist_caloHtPt->SetMinimum(0);
      caloHtPt_ = ibooker.book1D("CaloHT_pT",hist_caloHtPt);
    }
  //PFHT pt
  if (lookupIndex.count(pfHtPt_pathName) > 0)
    {
      TH1F * hist_pfHtPt = new TH1F("PFHT_pT","PFHT pT",pfHtPt_TH1.getParameter<int>("NbinsX"),pfHtPt_TH1.getParameter<int>("Xmin"),pfHtPt_TH1.getParameter<int>("Xmax"));
      hist_pfHtPt->SetMinimum(0);
      pfHtPt_ = ibooker.book1D("PFHT_pT",hist_pfHtPt);
    }
  //PFMET pt
  if (lookupIndex.count(pfMetPlots_pathName) >0)
    {
      TH1F * hist_PFMetPt = new TH1F("PFMET_pT","PFMET pT",pfMetPt_TH1.getParameter<int>("NbinsX"),pfMetPt_TH1.getParameter<int>("Xmin"),pfMetPt_TH1.getParameter<int>("Xmax"));
      hist_PFMetPt->SetMinimum(0);
      pfMetPt_ = ibooker.book1D("PFMET_pT",hist_PFMetPt);
    }

  ////////////////////////////////
  ///
  /// Backup workspace plots
  ///
  ////////////////////////////////
  ibooker.setCurrentFolder(backupFolder);

  if (lookupIndex.count(photonPlots_pathName) > 0)
    {
      //photon eta
      TH1F * hist_photonEta = new TH1F("Photon_eta","Photon eta",photonEta_TH1.getParameter<int>("NbinsX"),photonEta_TH1.getParameter<int>("Xmin"),photonEta_TH1.getParameter<int>("Xmax"));
      hist_photonEta->SetMinimum(0);
      photonEta_ = ibooker.book1D("Photon_eta",hist_photonEta);
      //photon phi
      TH1F * hist_photonPhi = new TH1F("Photon_phi","Photon phi",photonPhi_TH1.getParameter<int>("NbinsX"),photonPhi_TH1.getParameter<double>("Xmin"),photonPhi_TH1.getParameter<double>("Xmax"));
      hist_photonPhi->SetMinimum(0);
      photonPhi_ = ibooker.book1D("Photon_phi",hist_photonPhi);
    }
  if (lookupIndex.count(muonPlots_pathName) > 0)
    {
      //muon eta
      TH1F * hist_muonEta = new TH1F("Muon_eta","Muon eta",muonEta_TH1.getParameter<int>("NbinsX"),muonEta_TH1.getParameter<int>("Xmin"),muonEta_TH1.getParameter<int>("Xmax"));
      hist_muonEta->SetMinimum(0);
      muonEta_ = ibooker.book1D("Muon_eta",hist_muonEta);
      //muon phi
      TH1F * hist_muonPhi = new TH1F("Muon_phi","Muon phi",muonPhi_TH1.getParameter<int>("NbinsX"),muonPhi_TH1.getParameter<double>("Xmin"),muonPhi_TH1.getParameter<double>("Xmax"));
      hist_muonPhi->SetMinimum(0);
      muonPhi_ = ibooker.book1D("Muon_phi",hist_muonPhi);
    }
  if (lookupIndex.count(electronPlots_pathName) > 0)
    {
      //electron eta
      TH1F * hist_electronEta = new TH1F("Electron_eta","Electron eta",electronEta_TH1.getParameter<int>("NbinsX"),electronEta_TH1.getParameter<int>("Xmin"),electronEta_TH1.getParameter<int>("Xmax"));
      hist_electronEta->SetMinimum(0);
      electronEta_ = ibooker.book1D("Electron_eta",hist_electronEta);
      //electron phi
      TH1F * hist_electronPhi = new TH1F("Electron_phi","Electron phi",electronPhi_TH1.getParameter<int>("NbinsX"),electronPhi_TH1.getParameter<double>("Xmin"),electronPhi_TH1.getParameter<double>("Xmax"));
      hist_electronPhi->SetMinimum(0);
      electronPhi_ = ibooker.book1D("Electron_phi",hist_electronPhi);
    }
  if (lookupIndex.count(caloMetPlots_pathName) >0)
    {
      //caloMET phi
      TH1F * hist_caloMetPhi = new TH1F("CaloMET_phi","CaloMET phi",caloMetPhi_TH1.getParameter<int>("NbinsX"),caloMetPhi_TH1.getParameter<double>("Xmin"),caloMetPhi_TH1.getParameter<double>("Xmax"));
      hist_caloMetPhi->SetMinimum(0);
      caloMetPhi_ = ibooker.book1D("CaloMET_phi",hist_caloMetPhi);
    }
  if (lookupIndex.count(pfMetPlots_pathName) >0)
    {
      //PFMET phi
      TH1F * hist_PFMetPhi = new TH1F("PFMET_phi","PFMET phi",pfMetPhi_TH1.getParameter<int>("NbinsX"),pfMetPhi_TH1.getParameter<double>("Xmin"),pfMetPhi_TH1.getParameter<double>("Xmax"));
      hist_PFMetPhi->SetMinimum(0);
      pfMetPhi_ = ibooker.book1D("PFMET_phi",hist_PFMetPhi);
    }
  if (lookupIndex.count(bJetPlots_pathName) >0 || lookupIndex.count(bJetPlots_pathNameOR) >0) 
    {
      //bJet phi
      TH1F * hist_bJetPhi = new TH1F("bJet_phi","b-Jet phi",bJetPhi_TH1.getParameter<int>("NbinsX"),bJetPhi_TH1.getParameter<double>("Xmin"),bJetPhi_TH1.getParameter<double>("Xmax"));
      hist_bJetPhi->SetMinimum(0);
      bJetPhi_ = ibooker.book1D("bJet_phi",hist_bJetPhi);
      //bJet eta
      TH1F * hist_bJetEta = new TH1F("bJet_eta","b-Jet eta",bJetEta_TH1.getParameter<int>("NbinsX"),bJetEta_TH1.getParameter<int>("Xmin"),bJetEta_TH1.getParameter<int>("Xmax"));
      hist_bJetEta->SetMinimum(0);
      bJetEta_ = ibooker.book1D("bJet_eta",hist_bJetEta);
    }
}

void HLTObjectMonitor::fillPlots(int evtNum, string pathName, edm::Handle<trigger::TriggerEvent> aodTriggerEvent)
{
  if (debugPrint) std::cout << "Inside fillPlots( " << evtNum << " , " << pathName << " ) " << std::endl;

  const TriggerObjectCollection objects = aodTriggerEvent->getObjects();
  
  edm::InputTag moduleFilter(lookupFilter[pathName],"","TEST");
  unsigned int moduleFilterIndex = aodTriggerEvent->filterIndex(moduleFilter);
  const Keys &keys = aodTriggerEvent->filterKeys( moduleFilterIndex );

  ////////////////////////////////
  ///
  /// single-object plots
  ///
  ////////////////////////////////  
 
  //PFHT pt
  if (pathName == pfHtPt_pathName)
    {
      for (const auto & key : keys) pfHtPt_->Fill(objects[key].pt());
    }
  
  //jet pt
  else if (pathName == jetPt_pathName)
    {
      for (const auto & key : keys) jetPt_->Fill(objects[key].pt());
    }

  //photon pt + eta + phi (all use same path)
  else if (pathName == photonPlots_pathName)
    {
      for (const auto & key : keys) 
	{
	  photonPt_->Fill(objects[key].pt());
	  photonEta_->Fill(objects[key].eta());
	  photonPhi_->Fill(objects[key].phi());
	}
    }

  //electron pt + eta + phi (all use same path)
  else if (pathName == electronPlots_pathName)
    {
      for (const auto & key : keys) 
  	{
  	  electronPt_->Fill(objects[key].pt());
  	  electronEta_->Fill(objects[key].eta());
  	  electronPhi_->Fill(objects[key].phi());
  	}
    }

  //muon pt + eta + phi (all use same path)
  else if (pathName == muonPlots_pathName)
    {
      for (const auto & key : keys) 
  	{
  	  muonPt_->Fill(objects[key].pt());
  	  muonEta_->Fill(objects[key].eta());
  	  muonPhi_->Fill(objects[key].phi());
      	}
    }

  //alphaT
  else if (pathName == alphaT_pathName)
    {
      std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>> alphaT_jets;
      for (const auto & key : keys)
  	{
  	  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> JetLVec(objects[key].pt(),objects[key].eta(),objects[key].phi(),objects[key].mass());
  	  alphaT_jets.push_back(JetLVec);
  	}

      float alphaT = AlphaT(alphaT_jets,false).value(); 
      alphaT_->Fill(alphaT);
    }

  //tau pt
  else if (pathName == tauPt_pathName)
    {
      for (const auto & key : keys) tauPt_->Fill(objects[key].pt());
    }

  //caloMET pt+phi (same path)
  else if (pathName == caloMetPlots_pathName)
    {
      for (const auto & key : keys)
  	{
  	  caloMetPt_->Fill(objects[key].pt());
  	  caloMetPhi_->Fill(objects[key].phi());
  	}
    }

  //caloHT pt
  else if (pathName == caloHtPt_pathName)
    {
      for (const auto & key : keys) caloHtPt_->Fill(objects[key].pt());
    }

  //jetAK8 pt + mass
  else if (pathName == jetAK8Plots_pathName)
    {
      for (const auto & key : keys)
	{
	  jetAK8Pt_->Fill(objects[key].pt());
	  jetAK8Mass_->Fill(objects[key].mass());
	}
    }
  
  //PFMET pt + phi
  else if (pathName == pfMetPlots_pathName)
    {
      for (const auto & key : keys)
  	{
  	  pfMetPt_->Fill(objects[key].pt());
  	  pfMetPhi_->Fill(objects[key].phi());
  	}
    }

  // bjet eta + phi
  else if (pathName == bJetPlots_pathName || pathName == bJetPlots_pathNameOR)
    {
       for (const auto & key : keys)
  	{
   	  bJetEta_->Fill(objects[key].eta());
   	  bJetPhi_->Fill(objects[key].phi());
  	}
    }
  
  // ////////////////////////////////
  // ///
  // /// double-object plots
  // ///
  // ////////////////////////////////
  
  //double muon low mass 
  else if (pathName == diMuonLowMass_pathName)
    {
      const double mu_mass(.105658);
      unsigned int kCnt0 = 0;  
      for (const auto & key0: keys)
  	{
  	  unsigned int kCnt1 = 0;
  	  for (const auto & key1: keys)
  	    {
  	      if (key0 != key1 && kCnt1 > kCnt0) // avoid filling hists with same objs && avoid double counting separate objs
  		{
  		  if (abs(objects[key0].id()) == 13 && abs(objects[key1].id()) == 13 && (objects[key0].id()+objects[key1].id()==0))  // check muon id and dimuon charge
  		    {
  		      TLorentzVector mu1, mu2, dimu;
  		      mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
  		      mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
  		      dimu = mu1+mu2;
  		      diMuonLowMass_->Fill(dimu.M());
  		    }
  		}
  	      kCnt1 +=1;
  	    }
  	  kCnt0 +=1;
  	}
    } //end double object plot
  
}

double HLTObjectMonitor::get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)) return 0;
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
HLTObjectMonitor::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
HLTObjectMonitor::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
// void
// HLTObjectMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   //The following says we do not know what parameters are allowed so do no validation
//   // Please change this to state exactly what you do use, even if it is no parameters
//   edm::ParameterSetDescription desc;
//   desc.setUnknown();
//   descriptions.addDefault(desc);
// }

//define this as a plug-in
DEFINE_FWK_MODULE(HLTObjectMonitor);
