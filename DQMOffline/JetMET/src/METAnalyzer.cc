
/** \class JetMETAnalyzer
 *
 *  DQM jetMET analysis monitoring
 *
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 *
 *          Jan. '14: modified by
 *
 *          M. Artur Weber
 *          R. Schoefbeck
 *          V. Sordini
 */

#include "DQMOffline/JetMET/interface/METAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include <string>

using namespace edm;
using namespace reco;
using namespace math;

// ***********************************************************
METAnalyzer::METAnalyzer(const edm::ParameterSet& pSet) {
  parameters = pSet;

  outputMEsInRootFile   = parameters.getParameter<bool>("OutputMEsInRootFile");
  mOutputFile_   = parameters.getParameter<std::string>("OutputFile");

  LSBegin_     = pSet.getParameter<int>("LSBegin");
  LSEnd_       = pSet.getParameter<int>("LSEnd");

  MetType_ = parameters.getUntrackedParameter<std::string>("METType");

  triggerResultsLabel_        = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_= consumes<edm::TriggerResults>(edm::InputTag(triggerResultsLabel_));
  jetCorrectionService_ = pSet.getParameter<std::string> ("JetCorrections");

  isCaloMet_ = (std::string("calo")==MetType_);
  //isTCMet_ = (std::string("tc") ==MetType_);
  isPFMet_ = (std::string("pf") ==MetType_);

  // MET information
  metCollectionLabel_       = parameters.getParameter<edm::InputTag>("METCollectionLabel");

  if(/*isTCMet_ || */isCaloMet_){
    inputJetIDValueMap      = pSet.getParameter<edm::InputTag>("InputJetIDValueMap");
    jetID_ValueMapToken_= consumes< edm::ValueMap<reco::JetID> >(inputJetIDValueMap);
    jetIDFunctorLoose=JetIDSelectionFunctor(JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::LOOSE);
  }
  if(isPFMet_){
    pfjetIDFunctorLoose=PFJetIDSelectionFunctor(PFJetIDSelectionFunctor::FIRSTDATA, PFJetIDSelectionFunctor::LOOSE);
  }
  ptThreshold_ = parameters.getParameter<double>("ptThreshold");


  if(isPFMet_){
    pfMetToken_= consumes<reco::PFMETCollection>(edm::InputTag(metCollectionLabel_));
  }
  if(isCaloMet_){
    caloMetToken_= consumes<reco::CaloMETCollection>(edm::InputTag(metCollectionLabel_));
  }
  //if(isTCMet_){
  // tcMetToken_= consumes<reco::METCollection>(edm::InputTag(metCollectionLabel_));
  //}
  
  fill_met_high_level_histo = parameters.getParameter<bool>("fillMetHighLevel");
  
  hTriggerLabelsIsSet_ = false;
  //jet cleanup parameters
  cleaningParameters_ = pSet.getParameter<ParameterSet>("CleaningParameters");

  edm::ConsumesCollector iC  = consumesCollector();
  //DCS
  DCSFilter_ = new JetMETDQMDCSFilter(parameters.getParameter<ParameterSet>("DCSFilter"), iC );

  //Vertex requirements
  bypassAllPVChecks_    = cleaningParameters_.getParameter<bool>("bypassAllPVChecks");
  bypassAllDCSChecks_    = cleaningParameters_.getParameter<bool>("bypassAllDCSChecks");
  runcosmics_ = parameters.getUntrackedParameter<bool>("runcosmics");
  vertexTag_    = cleaningParameters_.getParameter<edm::InputTag>("vertexCollection");
  vertexToken_  = consumes<std::vector<reco::Vertex> >(edm::InputTag(vertexTag_));

  //Trigger parameters
  gtTag_          = cleaningParameters_.getParameter<edm::InputTag>("gtLabel");
  gtToken_= consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(gtTag_));

  //inputTrackLabel_         = parameters.getParameter<edm::InputTag>("InputTrackLabel");
  //inputMuonLabel_          = parameters.getParameter<edm::InputTag>("InputMuonLabel");
  //inputElectronLabel_      = parameters.getParameter<edm::InputTag>("InputElectronLabel");
  //inputBeamSpotLabel_      = parameters.getParameter<edm::InputTag>("InputBeamSpotLabel");
  //inputTCMETValueMap_      = parameters.getParameter<edm::InputTag>("InputTCMETValueMap");
  //TrackToken_= consumes<edm::View <reco::Track> >(inputTrackLabel_);
  //MuonToken_= consumes<reco::MuonCollection>(inputMuonLabel_);
  //ElectronToken_= consumes<edm::View<reco::GsfElectron> >(inputElectronLabel_);
  //BeamspotToken_= consumes<reco::BeamSpot>(inputBeamSpotLabel_);
  //tcMETValueMapToken_= consumes< edm::ValueMap<reco::MuonMETCorrectionData> >(inputTCMETValueMap_);

  // Other data collections
  jetCollectionLabel_       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  if (isCaloMet_) caloJetsToken_ = consumes<reco::CaloJetCollection>(jetCollectionLabel_);
  //if (isTCMet_)   jptJetsToken_ = consumes<reco::JPTJetCollection>(jetCollectionLabel_);
  if (isPFMet_)   pfJetsToken_ = consumes<reco::PFJetCollection>(jetCollectionLabel_);

  hcalNoiseRBXCollectionTag_   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  HcalNoiseRBXToken_ = consumes<reco::HcalNoiseRBXCollection>(hcalNoiseRBXCollectionTag_);

  beamHaloSummaryTag_          = parameters.getParameter<edm::InputTag>("BeamHaloSummaryLabel");
  beamHaloSummaryToken_       = consumes<BeamHaloSummary>(beamHaloSummaryTag_); 
  hbheNoiseFilterResultTag_    = parameters.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel");
  hbheNoiseFilterResultToken_=consumes<bool>(hbheNoiseFilterResultTag_);

  // 
  nbinsPV_ = parameters.getParameter<int>("pVBin");
  nPVMin_   = parameters.getParameter<double>("pVMin");
  nPVMax_  = parameters.getParameter<double>("pVMax");

  triggerSelectedSubFolders_ = parameters.getParameter<edm::VParameterSet>("triggerSelectedSubFolders");
  for (edm::VParameterSet::const_iterator it = triggerSelectedSubFolders_.begin(); it!= triggerSelectedSubFolders_.end(); it++) {
    triggerFolderEventFlag_.push_back(new GenericTriggerEventFlag( *it, consumesCollector() ));
    triggerFolderExpr_.push_back(it->getParameter<std::vector<std::string> >("hltPaths"));
    triggerFolderLabels_.push_back(it->getParameter<std::string>("label"));
  }

 cleaningParameters_ = parameters.getParameter<ParameterSet>("CleaningParameters");

 verbose_      = parameters.getParameter<int>("verbose");

 FolderName_              = parameters.getUntrackedParameter<std::string>("FolderName");

 dbe_ = edm::Service<DQMStore>().operator->();
}

// ***********************************************************
METAnalyzer::~METAnalyzer() {
  for (std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
    delete *it;
  }
  delete DCSFilter_;

  if(outputMEsInRootFile){
      //dbe->save(mOutputFile_);
    dbe_->save(mOutputFile_);
  }


  //delete DCSFilter_;

//  delete highPtJetEventFlag_;
//  delete lowPtJetEventFlag_;
//  delete minBiasEventFlag_;
//  delete highMETEventFlag_;
//  delete eleEventFlag_;
//  delete muonEventFlag_;
}


void METAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				 edm::EventSetup const &) {
  // DQStore stuff
  //dbe_ = edm::Service<DQMStore>().operator->();
  //LogTrace(metname)<<"[METAnalyzer] Parameters initialization";
  std::string DirName = FolderName_+metCollectionLabel_.label();
  ibooker.setCurrentFolder(DirName);

  //dbe_ = dbe;

  folderNames_.push_back("Uncleaned");
  folderNames_.push_back("Cleaned");
  folderNames_.push_back("DiJet");

  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    bookMESet(DirName+"/"+*ic, ibooker,map_dijet_MEs);
    }
}

// ***********************************************************
void METAnalyzer::endJob() {

  //delete DCSFilter_;

  //if(outputMEsInRootFile){
      //dbe->save(mOutputFile_);
  //dbe_->save(mOutputFile_);
  //}

}

// ***********************************************************
void METAnalyzer::bookMESet(std::string DirName, DQMStore::IBooker & ibooker, std::map<std::string,MonitorElement*>& map_of_MEs)
{
  bool bLumiSecPlot=fill_met_high_level_histo;
  //if (DirName.find("Uncleaned")!=std::string::npos)bLumiSecPlot=true;//now defined on configlevel
  bookMonitorElement(DirName,ibooker,map_of_MEs,bLumiSecPlot);

  if (DirName.find("Cleaned")!=std::string::npos) {
    for (unsigned int i = 0; i<triggerFolderEventFlag_.size(); i++) {
      if (triggerFolderEventFlag_[i]->on()) {
        bookMonitorElement(DirName+"/"+triggerFolderLabels_[i],ibooker,map_of_MEs,false);
      }
    }
  }
}

// ***********************************************************
void METAnalyzer::bookMonitorElement(std::string DirName,DQMStore::IBooker & ibooker, std::map<std::string,MonitorElement*>& map_of_MEs, bool bLumiSecPlot=false)
{
  if (verbose_) std::cout << "bookMonitorElement " << DirName << std::endl;

  ibooker.setCurrentFolder(DirName);

  hTrigger    = ibooker.book1D("triggerResults", "triggerResults", 500, 0, 500); 
  hMEx        = ibooker.book1D("MEx",        "MEx",        200, -500,  500);
  hMEy        = ibooker.book1D("MEy",        "MEy",        200, -500,  500);
  hMET        = ibooker.book1D("MET",        "MET",        200,    0, 1000);
  hSumET      = ibooker.book1D("SumET",      "SumET",      400,    0, 4000);
  hMETSig     = ibooker.book1D("METSig",     "METSig",      51,    0,   51);
  hMETPhi     = ibooker.book1D("METPhi",     "METPhi",      60, -3.2,  3.2);
  hMET_logx   = ibooker.book1D("MET_logx",   "MET_logx",    40,   -1,    7);
  hSumET_logx = ibooker.book1D("SumET_logx", "SumET_logx",  40,   -1,    7);

  hMEx       ->setAxisTitle("MEx [GeV]",        1);
  hMEy       ->setAxisTitle("MEy [GeV]",        1);
  hMET       ->setAxisTitle("MET [GeV]",        1);
  hSumET     ->setAxisTitle("SumET [GeV]",      1);
  hMETSig    ->setAxisTitle("METSig",       1);
  hMETPhi    ->setAxisTitle("METPhi [rad]",     1);
  hMET_logx  ->setAxisTitle("log(MET) [GeV]",   1);
  hSumET_logx->setAxisTitle("log(SumET) [GeV]", 1);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"triggerResults",hTrigger));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MEx",hMEx));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MEy",hMEy));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MET",hMET));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"SumET",hSumET));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METSig",hMETSig));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METPhi",hMETPhi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MET_logx",hMET_logx));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"SumET_logx",hSumET_logx));

  // Book NPV profiles --> would some of these profiles be interesting for other MET types too
  //----------------------------------------------------------------------------
  meMEx_profile   = ibooker.bookProfile("MEx_profile",   "met.px()",    nbinsPV_, nPVMin_, nPVMax_, 200, -500,  500);
  meMEy_profile   = ibooker.bookProfile("MEy_profile",   "met.py()",    nbinsPV_, nPVMin_, nPVMax_, 200, -500,  500);
  meMET_profile   = ibooker.bookProfile("MET_profile",   "met.pt()",    nbinsPV_, nPVMin_, nPVMax_, 200,    0, 1000);
  meSumET_profile = ibooker.bookProfile("SumET_profile", "met.sumEt()", nbinsPV_, nPVMin_, nPVMax_, 400,    0, 4000);
  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  meMEx_profile  ->setAxisTitle("nvtx", 1);
  meMEy_profile  ->setAxisTitle("nvtx", 1);
  meMET_profile  ->setAxisTitle("nvtx", 1);
  meSumET_profile->setAxisTitle("nvtx", 1);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MEx_profile",meMEx_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MEy_profile",meMEy_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MET_profile",meMET_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"SumET_profile",meSumET_profile));

    
  if(isCaloMet_){
    hCaloMaxEtInEmTowers    = ibooker.book1D("CaloMaxEtInEmTowers",   "CaloMaxEtInEmTowers"   ,100,0,2000);
    hCaloMaxEtInEmTowers->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    hCaloMaxEtInHadTowers   = ibooker.book1D("CaloMaxEtInHadTowers",  "CaloMaxEtInHadTowers"  ,100,0,2000);
    hCaloMaxEtInHadTowers->setAxisTitle("Et(Max) in Had Tower [GeV]",1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloMaxEtInEmTowers",hCaloMaxEtInEmTowers));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloMaxEtInHadTowers",hCaloMaxEtInHadTowers));

    hCaloHadEtInHB          = ibooker.book1D("CaloHadEtInHB","CaloHadEtInHB",100,0,2000);
    hCaloHadEtInHB->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHO          = ibooker.book1D("CaloHadEtInHO","CaloHadEtInHO",25,0,500);
    hCaloHadEtInHO->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHE          = ibooker.book1D("CaloHadEtInHE","CaloHadEtInHE",100,0,2000);
    hCaloHadEtInHE->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHF          = ibooker.book1D("CaloHadEtInHF","CaloHadEtInHF",50,0,1000);
    hCaloHadEtInHF->setAxisTitle("Had Et [GeV]",1);
    hCaloEmEtInHF           = ibooker.book1D("CaloEmEtInHF" ,"CaloEmEtInHF" ,25,0,500);
    hCaloEmEtInHF->setAxisTitle("EM Et [GeV]",1);
    hCaloEmEtInEE           = ibooker.book1D("CaloEmEtInEE" ,"CaloEmEtInEE" ,50,0,1000);
    hCaloEmEtInEE->setAxisTitle("EM Et [GeV]",1);
    hCaloEmEtInEB           = ibooker.book1D("CaloEmEtInEB" ,"CaloEmEtInEB" ,100,0,2000);
    hCaloEmEtInEB->setAxisTitle("EM Et [GeV]",1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloHadEtInHO",hCaloHadEtInHO));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloHadEtInHF",hCaloHadEtInHF));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloHadEtInHE",hCaloHadEtInHE));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloHadEtInHB",hCaloHadEtInHB));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloEmEtInHF",hCaloEmEtInHF));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloEmEtInEE",hCaloEmEtInEE));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloEmEtInEB",hCaloEmEtInEB));

    hCaloMETPhi020  = ibooker.book1D("CaloMETPhi020",  "CaloMETPhi020",   60, -3.2,  3.2);
    hCaloMETPhi020 ->setAxisTitle("METPhi [rad] (MET>20 GeV)", 1);

    //hCaloMaxEtInEmTowers    = ibooker.book1D("CaloMaxEtInEmTowers",   "CaloMaxEtInEmTowers"   ,100,0,2000);
    //hCaloMaxEtInEmTowers->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    //hCaloMaxEtInHadTowers   = ibooker.book1D("CaloMaxEtInHadTowers",  "CaloMaxEtInHadTowers"  ,100,0,2000);
    //hCaloMaxEtInHadTowers->setAxisTitle("Et(Max) in Had Tower [GeV]",1);
    hCaloEtFractionHadronic = ibooker.book1D("CaloEtFractionHadronic","CaloEtFractionHadronic",100,0,1);
    hCaloEtFractionHadronic->setAxisTitle("Hadronic Et Fraction",1);
    hCaloEmEtFraction       = ibooker.book1D("CaloEmEtFraction",      "CaloEmEtFraction"      ,100,0,1);
    hCaloEmEtFraction->setAxisTitle("EM Et Fraction",1);
    
    //hCaloEmEtFraction002    = ibooker.book1D("CaloEmEtFraction002",   "CaloEmEtFraction002"      ,100,0,1);
    //hCaloEmEtFraction002->setAxisTitle("EM Et Fraction (MET>2 GeV)",1);
    //hCaloEmEtFraction010    = ibooker.book1D("CaloEmEtFraction010",   "CaloEmEtFraction010"      ,100,0,1);
    //hCaloEmEtFraction010->setAxisTitle("EM Et Fraction (MET>10 GeV)",1);
    hCaloEmEtFraction020    = ibooker.book1D("CaloEmEtFraction020",   "CaloEmEtFraction020"      ,100,0,1);
    hCaloEmEtFraction020->setAxisTitle("EM Et Fraction (MET>20 GeV)",1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloMETPhi020",hCaloMETPhi020));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloEtFractionHadronic",hCaloEtFractionHadronic));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloEmEtFraction", hCaloEmEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloEmEtFraction020",hCaloEmEtFraction020));



    //if (metCollectionLabel_.label() == "corMetGlobalMuons" ) {
    //hCalomuPt    = ibooker.book1D("CalomuonPt", "CalomuonPt", 50, 0, 500);
    //hCalomuEta   = ibooker.book1D("CalomuonEta", "CalomuonEta", 60, -3.0, 3.0);
    //hCalomuNhits = ibooker.book1D("CalomuonNhits", "CalomuonNhits", 50, 0, 50);
    //hCalomuChi2  = ibooker.book1D("CalomuonNormalizedChi2", "CalomuonNormalizedChi2", 20, 0, 20);
    //hCalomuD0    = ibooker.book1D("CalomuonD0", "CalomuonD0", 50, -1, 1);
    //hCaloMExCorrection       = ibooker.book1D("CaloMExCorrection", "CaloMExCorrection", 100, -500.0,500.0);
    //hCaloMEyCorrection       = ibooker.book1D("CaloMEyCorrection", "CaloMEyCorrection", 100, -500.0,500.0);
    //hCaloMuonCorrectionFlag  = ibooker.book1D("CaloCorrectionFlag","CaloCorrectionFlag", 5, -0.5, 4.5);

    //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CalomuonPt",hCalomuPt));
    //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CalomuonEta",hCalomuEta));
    //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CalomuonNhit",hCalomuNhits));
    //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CalomuonNormalizedChi2",hCalomuChi2));
    //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloMExCorrection",hCaloMExCorrection));
    //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloMEyCorrection",hCaloMEyCorrection));
    //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CaloCorrectionFlag",hCaloMuonCorrectionFlag));
    //}

  }

  if(isPFMet_){
    mePhotonEtFraction        = ibooker.book1D("PfPhotonEtFraction",        "pfmet.photonEtFraction()",         50, 0,    1);
    mePhotonEt                = ibooker.book1D("PfPhotonEt",                "pfmet.photonEt()",                100, 0, 1000);
    meNeutralHadronEtFraction = ibooker.book1D("PfNeutralHadronEtFraction", "pfmet.neutralHadronEtFraction()",  50, 0,    1);
    meNeutralHadronEt         = ibooker.book1D("PfNeutralHadronEt",         "pfmet.neutralHadronEt()",         100, 0, 1000);
    meElectronEtFraction      = ibooker.book1D("PfElectronEtFraction",      "pfmet.electronEtFraction()",       50, 0,    1);
    meElectronEt              = ibooker.book1D("PfElectronEt",              "pfmet.electronEt()",              100, 0, 1000);
    meChargedHadronEtFraction = ibooker.book1D("PfChargedHadronEtFraction", "pfmet.chargedHadronEtFraction()",  50, 0,    1);
    meChargedHadronEt         = ibooker.book1D("PfChargedHadronEt",         "pfmet.chargedHadronEt()",         100, 0, 1000);
    meMuonEtFraction          = ibooker.book1D("PfMuonEtFraction",          "pfmet.muonEtFraction()",           50, 0,    1);
    meMuonEt                  = ibooker.book1D("PfMuonEt",                  "pfmet.muonEt()",                  100, 0, 1000);
    meHFHadronEtFraction      = ibooker.book1D("PfHFHadronEtFraction",      "pfmet.HFHadronEtFraction()",       50, 0,    1);
    meHFHadronEt              = ibooker.book1D("PfHFHadronEt",              "pfmet.HFHadronEt()",              100, 0, 1000);
    meHFEMEtFraction          = ibooker.book1D("PfHFEMEtFraction",          "pfmet.HFEMEtFraction()",           50, 0,    1);
    meHFEMEt                  = ibooker.book1D("PfHFEMEt",                  "pfmet.HFEMEt()",                  100, 0, 1000);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction"       ,mePhotonEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEt"               ,mePhotonEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFraction",meNeutralHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEt"        ,meNeutralHadronEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfElectronEtFraction"     ,meElectronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfElectronEt"             ,meElectronEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFraction",meChargedHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEt"        ,meChargedHadronEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfMuonEtFraction"         ,meMuonEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfMuonEt"                 ,meMuonEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFraction"     ,meHFHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEt"             ,meHFHadronEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFraction"         ,meHFEMEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEt"                 ,meHFEMEt));

    mePhotonEtFraction_profile        = ibooker.bookProfile("PfPhotonEtFraction_profile",        "pfmet.photonEtFraction()",        nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    mePhotonEt_profile                = ibooker.bookProfile("PfPhotonEt_profile",                "pfmet.photonEt()",                nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meNeutralHadronEtFraction_profile = ibooker.bookProfile("PfNeutralHadronEtFraction_profile", "pfmet.neutralHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meNeutralHadronEt_profile         = ibooker.bookProfile("PfNeutralHadronEt_profile",         "pfmet.neutralHadronEt()",         nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meElectronEtFraction_profile      = ibooker.bookProfile("PfElectronEtFraction_profile",      "pfmet.electronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meElectronEt_profile              = ibooker.bookProfile("PfElectronEt_profile",              "pfmet.electronEt()",              nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meChargedHadronEtFraction_profile = ibooker.bookProfile("PfChargedHadronEtFraction_profile", "pfmet.chargedHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meChargedHadronEt_profile         = ibooker.bookProfile("PfChargedHadronEt_profile",         "pfmet.chargedHadronEt()",         nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meMuonEtFraction_profile          = ibooker.bookProfile("PfMuonEtFraction_profile",          "pfmet.muonEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meMuonEt_profile                  = ibooker.bookProfile("PfMuonEt_profile",                  "pfmet.muonEt()",                  nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meHFHadronEtFraction_profile      = ibooker.bookProfile("PfHFHadronEtFraction_profile",      "pfmet.HFHadronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFHadronEt_profile              = ibooker.bookProfile("PfHFHadronEt_profile",              "pfmet.HFHadronEt()",              nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meHFEMEtFraction_profile          = ibooker.bookProfile("PfHFEMEtFraction_profile",          "pfmet.HFEMEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFEMEt_profile                  = ibooker.bookProfile("PfHFEMEt_profile",                  "pfmet.HFEMEt()",                  nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    
    mePhotonEtFraction_profile       ->setAxisTitle("nvtx", 1);
    mePhotonEt_profile               ->setAxisTitle("nvtx", 1);
    meNeutralHadronEtFraction_profile->setAxisTitle("nvtx", 1);
    meNeutralHadronEt_profile        ->setAxisTitle("nvtx", 1);
    meElectronEtFraction_profile     ->setAxisTitle("nvtx", 1);
    meElectronEt_profile             ->setAxisTitle("nvtx", 1);
    meChargedHadronEtFraction_profile->setAxisTitle("nvtx", 1);
    meChargedHadronEt_profile        ->setAxisTitle("nvtx", 1);
    meMuonEtFraction_profile         ->setAxisTitle("nvtx", 1);
    meMuonEt_profile                 ->setAxisTitle("nvtx", 1);
    meHFHadronEtFraction_profile     ->setAxisTitle("nvtx", 1);
    meHFHadronEt_profile             ->setAxisTitle("nvtx", 1);
    meHFEMEtFraction_profile         ->setAxisTitle("nvtx", 1);
    meHFEMEt_profile                 ->setAxisTitle("nvtx", 1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction_profile"        ,mePhotonEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEt_profile"                ,mePhotonEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFraction_profile" ,meNeutralHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEt_profile"          ,meNeutralHadronEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfElectronEtFraction_profile"      ,meElectronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfElectronEt_profile"              ,meElectronEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFraction_profile" ,meChargedHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEt_profile"         ,meChargedHadronEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfMuonEtFraction_profile"          ,meMuonEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfMuonEt_profile"                  ,meMuonEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFraction_profile"      ,meHFHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEt_profile"              ,meHFHadronEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFraction_profile"          ,meHFEMEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEt_profile"                  ,meHFEMEt_profile));
  }

  if (isCaloMet_){
    if (fill_met_high_level_histo){//now configurable in python file
      hMExLS = ibooker.book2D("MExLS","MEx_LS",200,-200,200,250,0.,2500.);
      hMExLS->setAxisTitle("MEx [GeV]",1);
      hMExLS->setAxisTitle("Lumi Section",2);
      hMExLS->getTH2F()->SetOption("colz");
      hMEyLS = ibooker.book2D("MEyLS","MEy_LS",200,-200,200,250,0.,2500.);
      hMEyLS->setAxisTitle("MEy [GeV]",1);
      hMEyLS->setAxisTitle("Lumi Section",2);
      hMEyLS->getTH2F()->SetOption("colz");
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MExLS",hMExLS));
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MEyLS",hMEyLS));
    }
  }
  
  //if (isTCMet_) {
  //htrkPt    = ibooker.book1D("trackPt", "trackPt", 50, 0, 500);
  //htrkEta   = ibooker.book1D("trackEta", "trackEta", 60, -3.0, 3.0);
  //htrkNhits = ibooker.book1D("trackNhits", "trackNhits", 50, 0, 50);
  //htrkChi2  = ibooker.book1D("trackNormalizedChi2", "trackNormalizedChi2", 20, 0, 20);
  //htrkD0    = ibooker.book1D("trackD0", "trackd0", 50, -1, 1);
  //helePt    = ibooker.book1D("electronPt", "electronPt", 50, 0, 500);
  //heleEta   = ibooker.book1D("electronEta", "electronEta", 60, -3.0, 3.0);
  //heleHoE   = ibooker.book1D("electronHoverE", "electronHoverE", 25, 0, 0.5);
  //hmuPt     = ibooker.book1D("muonPt", "muonPt", 50, 0, 500);
  //hmuEta    = ibooker.book1D("muonEta", "muonEta", 60, -3.0, 3.0);
  //hmuNhits  = ibooker.book1D("muonNhits", "muonNhits", 50, 0, 50);
  //hmuChi2   = ibooker.book1D("muonNormalizedChi2", "muonNormalizedChi2", 20, 0, 20);
  //hmuD0     = ibooker.book1D("muonD0", "muonD0", 50, -1, 1);

  //hMExCorrection       = ibooker.book1D("MExCorrection", "MExCorrection", 100, -500.0,500.0);
  //hMEyCorrection       = ibooker.book1D("MEyCorrection", "MEyCorrection", 100, -500.0,500.0);
  //hMuonCorrectionFlag  = ibooker.book1D("CorrectionFlag","CorrectionFlag", 5, -0.5, 4.5);

  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"trackPt"   ,htrkPt));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"trackEta"  ,htrkEta));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"trackNhits",htrkNhits));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"trackNormalizedChi2" ,htrkChi2));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"trackD0"   ,htrkD0));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"electronPt"   ,helePt));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"electronEta"  ,heleEta));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"electronHoverE",heleHoE));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"muonPt"   ,hmuPt));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"muonEta"  ,hmuEta));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"muonNhits",hmuNhits));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"muonNormalizedChi2" ,hmuChi2));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"muonD0"   ,hmuD0));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MExCorrection"   ,hMExCorrection));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MEyCorrection"   ,hMEyCorrection));
  //map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CorrectionFlag"  ,hMuonCorrectionFlag));
  //}
  
  hMETRate      = ibooker.book1D("METRate",        "METRate",        200,    0, 1000);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METRate",hMETRate));


  ibooker.setCurrentFolder("JetMET");
  lumisecME = ibooker.book1D("lumisec", "lumisec", 2500, 0., 2500.);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>("JetMET/lumisec",lumisecME));

}

// ***********************************************************
void METAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

//  std::cout  << "Run " << iRun.run() << " hltconfig.init " 
//             << hltConfig_.init(iRun,iSetup,triggerResultsLabel_.process(),changed_) << " length: "<<hltConfig_.triggerNames().size()<<" changed "<<changed_<<std::endl; 
  bool changed(true);
  if (hltConfig_.init(iRun,iSetup,triggerResultsLabel_.process(),changed)) {
    if (changed) {
      hltConfig_.dump("ProcessName");
      hltConfig_.dump("GlobalTag");
      hltConfig_.dump("TableName");
//      hltConfig_.dump("Streams");
//      hltConfig_.dump("Datasets");
//      hltConfig_.dump("PrescaleTable");
//      hltConfig_.dump("ProcessPSet");
    }
  } else {
    if (verbose_) std::cout << "HLTEventAnalyzerAOD::analyze:"
              << " config extraction failure with process name "
              << triggerResultsLabel_.process() << std::endl;
  }

  allTriggerNames_.clear();
  for (unsigned i = 0; i<hltConfig_.size();i++) {
    allTriggerNames_.push_back(hltConfig_.triggerName(i));
  }
//  std::cout<<"Length: "<<allTriggerNames_.size()<<std::endl;

  triggerSelectedSubFolders_ = parameters.getParameter<edm::VParameterSet>("triggerSelectedSubFolders");
  for ( std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
    int pos = it - triggerFolderEventFlag_.begin();
    if ((*it)->on()) {
      (*it)->initRun( iRun, iSetup );
      if (triggerSelectedSubFolders_[pos].exists(std::string("hltDBKey"))) {
//        std::cout<<"Looking for hltDBKey for"<<triggerFolderLabels_[pos]<<std::endl;
        if ((*it)->expressionsFromDB((*it)->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
          triggerFolderExpr_[pos] = (*it)->expressionsFromDB((*it)->hltDBKey(), iSetup);
      }
//      for (unsigned j = 0; j<triggerFolderExpr_[pos].size(); j++) std::cout<<"pos "<<pos<<" "<<triggerFolderLabels_[pos]<<" triggerFolderExpr_"<<triggerFolderExpr_[pos][j]<<std::endl;
    }
  }
}

// ***********************************************************
void METAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  
  //
  //--- Check the time length of the Run from the lumi section plots
  

  TH1F* tlumisec;

  MonitorElement *meLumiSec = map_dijet_MEs["aaa"];
  meLumiSec = map_dijet_MEs["JetMET/lumisec"];

  int totlsec=0;
  int totlssecsum=0;
  double totltime=0.;
  if (meLumiSec &&  meLumiSec->getRootObject() ) {
    tlumisec = meLumiSec->getTH1F();
    //check overflow bin (if we have more than 2500 LS in a run)
    //lumisec is filled every time the analyze section is processed
    //we know an LS is present only once in a run: normalize how many events we had on average
    //if lumi fluctuates strongly might be unreliable for overflow bin though
    for (int i=0; i< (tlumisec->GetNbinsX()); i++){
      if (tlumisec->GetBinContent(i)!=0){ 
	totlsec+=1;
	totlssecsum+=tlumisec->GetBinContent(i);
      }
    }
    int num_per_ls=(double)totlssecsum/(double)totlsec;
    totlsec=totlsec+tlumisec->GetBinContent(tlumisec->GetNbinsX()+1)/(double)num_per_ls;
    totltime = double(totlsec*90); // one lumi sec ~ 90 (sec)
  }
  
 if (totltime==0.) totltime=1.;
 
  std::string dirName = FolderName_+metCollectionLabel_.label()+"/";
  //dbe_->setCurrentFolder(dirName);

 

  //below is the original METAnalyzer formulation
  
  for (std::vector<std::string>::const_iterator ic = folderNames_.begin(); ic != folderNames_.end(); ic++) {
    std::string DirName;
    DirName = dirName+*ic;
    makeRatePlot(DirName,totltime);
    for ( std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
      int pos = it - triggerFolderEventFlag_.begin();
      if ((*it)->on()) {
	makeRatePlot(DirName+"/"+triggerFolderLabels_[pos],totltime);
      }
    }
  }
  
}


// ***********************************************************
void METAnalyzer::makeRatePlot(std::string DirName, double totltime)
{
  
  //dbe_->setCurrentFolder(DirName);
  MonitorElement *meMET = map_dijet_MEs[DirName+"/"+"MET"];
  MonitorElement *mMETRate = map_dijet_MEs[DirName+"/"+"METRate"];

  TH1F* tMET;
  TH1F* tMETRate;

  if ( meMET && mMETRate){
    if ( meMET->getRootObject() && mMETRate->getRootObject()) {
      tMET     = meMET->getTH1F();

      // Integral plot & convert number of events to rate (hz)
      tMETRate = (TH1F*) tMET->Clone("METRateHist");
      for (int i = tMETRate->GetNbinsX()-1; i>=0; i--){
	mMETRate->setBinContent(i+1,tMETRate->GetBinContent(i+2)+tMET->GetBinContent(i+1));
      }
      for (int i = 0; i<tMETRate->GetNbinsX(); i++){
	mMETRate->setBinContent(i+1,tMETRate->GetBinContent(i+1)/double(totltime));
      }
    }
  }
  
}
  
// ***********************************************************
void METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {


  // *** Fill lumisection ME
  int myLuminosityBlock;
  myLuminosityBlock = iEvent.luminosityBlock();
  if(fill_met_high_level_histo){
    lumisecME=map_dijet_MEs["JetMET/lumisec"]; if(lumisecME && lumisecME->getRootObject()) lumisecME->Fill(myLuminosityBlock);
  }

  if (myLuminosityBlock<LSBegin_) return;
  if (myLuminosityBlock>LSEnd_ && LSEnd_>0) return;

  if (verbose_) std::cout << "METAnalyzer analyze" << std::endl;

  std::string DirName = FolderName_+metCollectionLabel_.label();
  



  // ==========================================================
  // Trigger information
  //
//  trigJetMB_=0;
//  trigHighPtJet_=0;
//  trigLowPtJet_=0;
//  trigMinBias_=0;
//  trigHighMET_=0;
//  //  _trig_LowMET=0;
//  trigEle_=0;
//  trigMuon_=0;
//  trigPhysDec_=0;
  std::vector<int> triggerFolderDecisions;
  triggerFolderDecisions_ = std::vector<int> (triggerFolderEventFlag_.size(), 0);
  // **** Get the TriggerResults container
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);

  if( triggerResults.isValid()) {
    /////////// Analyzing HLT Trigger Results (TriggerResults) //////////
    // Check how many HLT triggers are in triggerResults
    int ntrigs = (*triggerResults).size();
    if (verbose_) std::cout << "ntrigs=" << ntrigs << std::endl;
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    for (std::vector<GenericTriggerEventFlag *>::const_iterator it =  triggerFolderEventFlag_.begin(); it!=triggerFolderEventFlag_.end();it++) {
      unsigned pos = it - triggerFolderEventFlag_.begin();
      bool fd = (*it)->accept(iEvent, iSetup);
      triggerFolderDecisions_[pos] = fd;
    }
    allTriggerDecisions_.clear();
    for (unsigned i=0;i<allTriggerNames_.size();++i)  {
      allTriggerDecisions_.push_back((*triggerResults).accept(i)); 
//      std::cout<<"TR "<<(*triggerResults).size()<<" "<<(*triggerResults).accept(i)<<" "<<allTriggerNames_[i]<<std::endl;
    }
  }

  // ==========================================================
  // MET information

  // **** Get the MET container
  edm::Handle<reco::METCollection> tcmetcoll;
  edm::Handle<reco::CaloMETCollection> calometcoll;
  edm::Handle<reco::PFMETCollection> pfmetcoll;

  //if(isTCMet_){
  //iEvent.getByToken(tcMetToken_, tcmetcoll);
  //if(!tcmetcoll.isValid()) return;
  //}
  if(isCaloMet_){
    iEvent.getByToken(caloMetToken_, calometcoll);
    if(!calometcoll.isValid()) return;
  }
  if(isPFMet_){
    iEvent.getByToken(pfMetToken_, pfmetcoll);
    if(!pfmetcoll.isValid()) return;
  }

  const MET *met=NULL;
  const PFMET *pfmet=NULL;
  const CaloMET *calomet=NULL;
  //if(isTCMet_){
  //met=&(tcmetcoll->front());
  //}
  if(isPFMet_){
    met=&(pfmetcoll->front());
    pfmet=&(pfmetcoll->front());
  }
  if(isCaloMet_){
    met=&(calometcoll->front());
    calomet=&(calometcoll->front());
  }
  
  LogTrace(metname)<<"[METAnalyzer] Call to the MET analyzer";

  // ==========================================================
  // TCMET

  //if (/*isTCMet_ || */(isCaloMet_ && metCollectionLabel_.label() == "corMetGlobalMuons")) {

    //iEvent.getByToken(MuonToken_, muonHandle_);
    //iEvent.getByToken(TrackToken_, trackHandle_);
    //iEvent.getByToken(ElectronToken_, electronHandle_);
    //iEvent.getByToken(BeamspotToken_, beamSpotHandle_);
    //iEvent.getByToken(tcMETValueMapToken_,tcMetValueMapHandle_);

    //if(!muonHandle_.isValid())     edm::LogInfo("OutputInfo") << "falied to retrieve muon data require by MET Task";
    //if(!trackHandle_.isValid())    edm::LogInfo("OutputInfo") << "falied to retrieve track data require by MET Task";
    //if(!electronHandle_.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve electron data require by MET Task";
    //if(!beamSpotHandle_.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve beam spot data require by MET Task";

    //beamSpot_ = ( beamSpotHandle_.isValid() ) ? beamSpotHandle_->position() : math::XYZPoint(0, 0, 0);
    //}

  // ==========================================================
  //

  edm::Handle<HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByToken(HcalNoiseRBXToken_,HRBXCollection);
  if (!HRBXCollection.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
    if (verbose_) std::cout << "METAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }

  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByToken(hbheNoiseFilterResultToken_, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    if (verbose_) std::cout << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
  }

  // ==========================================================
  bool bJetID = false;
  bool bDiJetID = false;
  // Jet ID -------------------------------------------------------
  //

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection> jptJets;
  edm::Handle<PFJetCollection> pfJets;

  int collsize=-1;

  if (isCaloMet_){
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find calojet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find calojet product" << std::endl;
    }
    collsize=caloJets->size();
  }
  ///*
  //if (isTCMet_){
  //iEvent.getByToken(jptJetsToken_, jptJets);
  //if (!jptJets.isValid()) {
  //  LogDebug("") << "METAnalyzer: Could not find jptjet product" << std::endl;
  //  if (verbose_) std::cout << "METAnalyzer: Could not find jptjet product" << std::endl;
  //}
  //collsize=jptJets->size();
  //}*/

  edm::Handle< edm::ValueMap<reco::JetID> >jetID_ValueMap_Handle;
  if(/*isTCMet_ || */isCaloMet_){
    if(!runcosmics_){
      iEvent.getByToken(jetID_ValueMapToken_,jetID_ValueMap_Handle);
    }
  }

  if (isPFMet_){ iEvent.getByToken(pfJetsToken_, pfJets);
    if (!pfJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find pfjet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find pfjet product" << std::endl;
    }
    collsize=pfJets->size();
  }

  unsigned int ind1=-1;
  double pt1=-1;
  bool pass_jetID1=false;
  unsigned int ind2=-1;
  double pt2=-1;
  bool pass_jetID2=false;

  //do loose jet ID-> check threshold on corrected jets
  for (int ijet=0; ijet<collsize; ijet++) {
    double pt_jet=-10;
    double scale=1.;
    bool iscleaned=false;
    if (!jetCorrectionService_.empty()) {
      const JetCorrector* corrector = JetCorrector::getJetCorrector(jetCorrectionService_, iSetup);	 
      if(isCaloMet_){
	scale = corrector->correction((*caloJets)[ijet], iEvent, iSetup);
      }
      //if(isTCMet_){
      //scale = corrector->correction((*jptJets)[ijet], iEvent, iSetup);
      //}
      if(isPFMet_){
	scale = corrector->correction((*pfJets)[ijet], iEvent, iSetup);
      }
    }
    if(isCaloMet_){
	pt_jet=scale*(*caloJets)[ijet].pt();
	if(pt_jet> ptThreshold_){
	  reco::CaloJetRef calojetref(caloJets, ijet);
	  if(!runcosmics_){
	    reco::JetID jetID = (*jetID_ValueMap_Handle)[calojetref];
	    iscleaned = jetIDFunctorLoose((*caloJets)[ijet], jetID);
	  }else{
	    iscleaned=true;
	  }
	}
    }
    ///*
    //if(isTCMet_){
    //pt_jet=scale*(*jptJets)[ijet].pt();
    //if(pt_jet> ptThreshold_){
    //	const edm::RefToBase<reco::Jet>&  rawJet = (*jptJets)[ijet].getCaloJetRef();
    //	const reco::CaloJet *rawCaloJet = dynamic_cast<const reco::CaloJet*>(&*rawJet);
    //	reco::CaloJetRef const theCaloJetRef = (rawJet).castTo<reco::CaloJetRef>();
    //	if(!runcosmics_){
    //	  reco::JetID jetID = (*jetID_ValueMap_Handle)[theCaloJetRef];
    //	  iscleaned = jetIDFunctorLoose(*rawCaloJet, jetID);
    //	}else{
    //	  iscleaned=true;
    //	}
    //}
    //}*/
    if(isPFMet_){
      pt_jet=scale*(*pfJets)[ijet].pt();
      if(pt_jet> ptThreshold_){
	iscleaned = pfjetIDFunctorLoose((*pfJets)[ijet]);
      }
    }
    if(iscleaned){
      bJetID=true;
    }
    if(pt_jet>pt1){
      pt2=pt1;
      ind2=ind1;
      pass_jetID2=pass_jetID1;
      pt1=pt_jet;
      ind1=ijet;
      pass_jetID1=iscleaned;
    }else if (pt_jet>pt2){
      pt2=pt_jet;
      ind2=ijet;
      pass_jetID2=iscleaned;
    }
  }
  if(pass_jetID1 && pass_jetID2){
    double dphi=-1.0;
    if(isCaloMet_){
      dphi=fabs((*caloJets)[ind1].phi()-(*caloJets)[ind2].phi());
    }
    ///* if(isTCMet_){
    //dphi=fabs((*jptJets)[ind1].phi()-(*jptJets)[ind2].phi());
    //}*/
    if(isPFMet_){
      dphi=fabs((*pfJets)[ind1].phi()-(*pfJets)[ind2].phi());
    }
    if(dphi>acos(-1.)){
      dphi=2*acos(-1.)-dphi;
    }
    if(dphi>2.7){
      bDiJetID=true;
    }
  }

  // ==========================================================
  // HCAL Noise filter

  bool bHBHENoiseFilter = HBHENoiseFilterResult;

  // ==========================================================
  // Get BeamHaloSummary
  edm::Handle<BeamHaloSummary> TheBeamHaloSummary ;
  iEvent.getByToken(beamHaloSummaryToken_, TheBeamHaloSummary) ;

  if (!TheBeamHaloSummary.isValid()) {
    if (verbose_) std::cout << "BeamHaloSummary doesn't exist" << std::endl;
  }

  bool bBeamHaloID = true;

  if(!TheBeamHaloSummary.isValid()) {

  const BeamHaloSummary TheSummary = (*TheBeamHaloSummary.product() );

  if( !TheSummary.EcalTightHaloId()  && !TheSummary.HcalTightHaloId() &&
      !TheSummary.CSCTightHaloId()   && !TheSummary.GlobalTightHaloId() )
    bBeamHaloID = false;
  }

  // ==========================================================
  //Vertex information
  Handle<VertexCollection> vertexHandle;
  iEvent.getByToken(vertexToken_, vertexHandle);

  if (!vertexHandle.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
    if (verbose_) std::cout << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
  }
  numPV_ = 0;
  if ( vertexHandle.isValid() ){
    VertexCollection vertexCollection = *(vertexHandle.product());
    numPV_  = vertexCollection.size();
  }
  bool bPrimaryVertex = (bypassAllPVChecks_ || (numPV_>0));
  // ==========================================================

  edm::Handle< L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  iEvent.getByToken( gtToken_, gtReadoutRecord);

  if (!gtReadoutRecord.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find GT readout record" << std::endl;
    if (verbose_) std::cout << "CaloMETAnalyzer: Could not find GT readout record product" << std::endl;
  }
  // DCS Filter
  bool bDCSFilter = (bypassAllDCSChecks_ || DCSFilter_->filter(iEvent, iSetup));
  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements

  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    if ((*ic=="Uncleaned")  &&(isCaloMet_ || bPrimaryVertex))     fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet,map_dijet_MEs);
    //take two lines out for first check
    if ((*ic=="Cleaned")    &&bDCSFilter&&bHBHENoiseFilter&&bPrimaryVertex&&bBeamHaloID&&bJetID) fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet,map_dijet_MEs);
    if ((*ic=="DiJet" )     &&bDCSFilter&&bHBHENoiseFilter&&bPrimaryVertex&&bBeamHaloID&&bDiJetID) fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet,map_dijet_MEs);
  }
}


// ***********************************************************
void METAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName,
			    const reco::MET& met, const reco::PFMET& pfmet, const reco::CaloMET& calomet,std::map<std::string,MonitorElement*>&  map_of_MEs)
{

  //dbe_->setCurrentFolder(DirName);

  bool bLumiSecPlot=fill_met_high_level_histo;
  //if (DirName.find("Uncleaned")) bLumiSecPlot=true; //now done on configlevel
  fillMonitorElement(iEvent, DirName, std::string(""), met, pfmet, calomet, map_of_MEs,bLumiSecPlot);
  if (DirName.find("Cleaned")) {
    for (unsigned i = 0; i<triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i])  fillMonitorElement(iEvent, DirName, triggerFolderLabels_[i], met, pfmet, calomet, map_of_MEs, false);
    }
  }

  if (DirName.find("DiJet")) {
    for (unsigned i = 0; i<triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i])  fillMonitorElement(iEvent, DirName, triggerFolderLabels_[i], met, pfmet, calomet, map_of_MEs, false);
    }
  }
  






//  if (trigJetMB_)
//    fillMonitorElement(iEvent,DirName,"",met,pfmet,calomet, bLumiSecPlot);
//  if (trigHighPtJet_)
//    fillMonitorElement(iEvent,DirName,"HighPtJet",met,pfmet,calomet,false);
//  if (trigLowPtJet_)
//    fillMonitorElement(iEvent,DirName,"LowPtJet",met,pfmet,calomet,false);
//  if (trigMinBias_)
//    fillMonitorElement(iEvent,DirName,"MinBias",met,pfmet,calomet,false);
//  if (trigHighMET_)
//    fillMonitorElement(iEvent,DirName,"HighMET",met,pfmet,calomet,false);
//  //  if (_trig_LowMET)
//  //    fillMonitorElement(iEvent,DirName,"LowMET",met,pfmet,calomet,false);
//  if (trigEle_)
//    fillMonitorElement(iEvent,DirName,"Ele",met,pfmet,calomet,false);
//  if (trigMuon_)
//    fillMonitorElement(iEvent,DirName,"Muon",met,pfmet,calomet,false);
}

// ***********************************************************
void METAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName,
					 std::string subFolderName,
				     const reco::MET& met, const reco::PFMET & pfmet, const reco::CaloMET &calomet, std::map<std::string,MonitorElement*>&  map_of_MEs,bool bLumiSecPlot)
{

//  if (subFolderName=="HighPtJet") {
//    if (!selectHighPtJetEvent(iEvent)) return;
//  }
//  else if (subFolderName=="LowPtJet") {
//    if (!selectLowPtJetEvent(iEvent)) return;
//  }
//  else if (subFolderName=="HighMET") {
//    if (met.pt()<highMETThreshold_) return;
//  }
//  //  else if (subFolderName=="LowMET") {
//  //    if (met.pt()<_lowMETThreshold) return;
//  //  }
//  else if (subFolderName=="Ele") {
//    if (!selectWElectronEvent(iEvent)) return;
//  }
//  else if (subFolderName=="Muon") {
//    if (!selectWMuonEvent(iEvent)) return;
//  }

// Reconstructed MET Information
  double SumET  = met.sumEt();
  double METSig = met.mEtSig();
  //double Ez     = met.e_longitudinal();
  double MET    = met.pt();
  double MEx    = met.px();
  double MEy    = met.py();
  double METPhi = met.phi();
  //
  int myLuminosityBlock;
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (subFolderName!="") DirName = DirName +"/"+subFolderName;


  hTrigger = map_of_MEs[DirName+"/triggerResults"];
  //    std::cout<<"Hello"<<c++<<":"<<hTrigger <<std::endl;//":"<< hTrigger->getRootObject()<<std::endl;
  if (hTrigger       && hTrigger->getRootObject()) {
    //      std::cout<<"Hello"<<c++<<std::endl;
    for (unsigned int i = 0; i<allTriggerDecisions_.size();i++){ 
      //        std::cout<<"Hello"<<c++<<":"<<i<<":"<< allTriggerDecisions_[i]<<":"<<allTriggerDecisions_[i]<<std::endl;
      if(i<(unsigned int)hTrigger->getNbinsX()){
	hTrigger->Fill(i + .5, allTriggerDecisions_[i]);
	if (!hTriggerLabelsIsSet_) {
	  hTrigger->setBinLabel(i+1, allTriggerNames_[i]);//Can't be done in beginJob (no trigger list). Can't be done in beginRun (would have to anticipate folder structure).FIXME doesn't work
	}
      }
    }
    if (!hTriggerLabelsIsSet_) for (int i = allTriggerDecisions_.size(); i<hTrigger->getNbinsX();i++){ 
	hTrigger->setBinLabel(i+1, "");//Can't be done in beginJob (no trigger list). Can't be done in beginRun (would have to anticipate folder structure).
      }
    hTriggerLabelsIsSet_ = true;
    //        std::cout<<"Filling decision "<<allTriggerNames_[i]<<" "<<allTriggerDecisions_[i]<<std::endl;
  }
  
    
  hMEx    = map_of_MEs[DirName+"/"+"MEx"];     if (hMEx           && hMEx->getRootObject())    hMEx          ->Fill(MEx);
    hMEy    = map_of_MEs[DirName+"/"+"MEy"];     if (hMEy           && hMEy->getRootObject())     hMEy          ->Fill(MEy);
    hMET    = map_of_MEs[DirName+"/"+"MET"];     if (hMET           && hMET->getRootObject())     hMET          ->Fill(MET);
    hMETPhi = map_of_MEs[DirName+"/"+"METPhi"];  if (hMETPhi        && hMETPhi->getRootObject())  hMETPhi       ->Fill(METPhi);
    hSumET  = map_of_MEs[DirName+"/"+"SumET"];   if (hSumET         && hSumET->getRootObject())   hSumET        ->Fill(SumET);
    hMETSig = map_of_MEs[DirName+"/"+"METSig"];  if (hMETSig        && hMETSig->getRootObject())  hMETSig       ->Fill(METSig);
    hMET_logx   = map_of_MEs[DirName+"/"+"MET_logx"];    if (hMET_logx      && hMET_logx->getRootObject())    hMET_logx->Fill(log10(MET));
    hSumET_logx = map_of_MEs[DirName+"/"+"SumET_logx"];  if (hSumET_logx    && hSumET_logx->getRootObject())  hSumET_logx->Fill(log10(SumET));
    
    // Fill NPV profiles
      //--------------------------------------------------------------------------
    meMEx_profile   = map_of_MEs[DirName + "/MEx_profile"];
    meMEy_profile   = map_of_MEs[DirName + "/MEy_profile"];
    meMET_profile   = map_of_MEs[DirName + "/MET_profile"];
    meSumET_profile = map_of_MEs[DirName + "/SumET_profile"];
    
    if (meMEx_profile   && meMEx_profile  ->getRootObject()) meMEx_profile  ->Fill(numPV_, MEx);
    if (meMEy_profile   && meMEy_profile  ->getRootObject()) meMEy_profile  ->Fill(numPV_, MEy);
    if (meMET_profile   && meMET_profile  ->getRootObject()) meMET_profile  ->Fill(numPV_, MET);
    if (meSumET_profile && meSumET_profile->getRootObject()) meSumET_profile->Fill(numPV_, SumET);

    if(isCaloMet_){
      //const reco::CaloMETCollection *calometcol = calometcoll.product();
      //const reco::CaloMET *calomet;
      //calomet = &(calometcol->front());
      
      double caloEtFractionHadronic = calomet.etFractionHadronic();
      double caloEmEtFraction       = calomet.emEtFraction();

      double caloMaxEtInEMTowers    = calomet.maxEtInEmTowers();
      double caloMaxEtInHadTowers   = calomet.maxEtInHadTowers();
      
      double caloHadEtInHB = calomet.hadEtInHB();
      double caloHadEtInHO = calomet.hadEtInHO();
      double caloHadEtInHE = calomet.hadEtInHE();
      double caloHadEtInHF = calomet.hadEtInHF();
      double caloEmEtInEB  = calomet.emEtInEB();
      double caloEmEtInEE  = calomet.emEtInEE();
      double caloEmEtInHF  = calomet.emEtInHF();

      hCaloMaxEtInEmTowers  = map_of_MEs[DirName+"/"+"CaloMaxEtInEmTowers"];   if (hCaloMaxEtInEmTowers  && hCaloMaxEtInEmTowers->getRootObject())   hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
      hCaloMaxEtInHadTowers = map_of_MEs[DirName+"/"+"CaloMaxEtInHadTowers"];  if (hCaloMaxEtInHadTowers && hCaloMaxEtInHadTowers->getRootObject())  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
      
      hCaloHadEtInHB = map_of_MEs[DirName+"/"+"CaloHadEtInHB"];  if (hCaloHadEtInHB  &&  hCaloHadEtInHB->getRootObject())  hCaloHadEtInHB->Fill(caloHadEtInHB);
      hCaloHadEtInHO = map_of_MEs[DirName+"/"+"CaloHadEtInHO"];  if (hCaloHadEtInHO  &&  hCaloHadEtInHO->getRootObject())  hCaloHadEtInHO->Fill(caloHadEtInHO);
      hCaloHadEtInHE = map_of_MEs[DirName+"/"+"CaloHadEtInHE"];  if (hCaloHadEtInHE  &&  hCaloHadEtInHE->getRootObject())  hCaloHadEtInHE->Fill(caloHadEtInHE);
      hCaloHadEtInHF = map_of_MEs[DirName+"/"+"CaloHadEtInHF"];  if (hCaloHadEtInHF  &&  hCaloHadEtInHF->getRootObject())  hCaloHadEtInHF->Fill(caloHadEtInHF);
      hCaloEmEtInEB  = map_of_MEs[DirName+"/"+"CaloEmEtInEB"];   if (hCaloEmEtInEB   &&  hCaloEmEtInEB->getRootObject())   hCaloEmEtInEB->Fill(caloEmEtInEB);
      hCaloEmEtInEE  = map_of_MEs[DirName+"/"+"CaloEmEtInEE"];   if (hCaloEmEtInEE   &&  hCaloEmEtInEE->getRootObject())   hCaloEmEtInEE->Fill(caloEmEtInEE);
      hCaloEmEtInHF  = map_of_MEs[DirName+"/"+"CaloEmEtInHF"];   if (hCaloEmEtInHF   &&  hCaloEmEtInHF->getRootObject())   hCaloEmEtInHF->Fill(caloEmEtInHF);

      hCaloMETPhi020 = map_of_MEs[DirName+"/"+"CaloMETPhi020"];    if (MET> 20. && hCaloMETPhi020  &&  hCaloMETPhi020->getRootObject()) { hCaloMETPhi020->Fill(METPhi);}


      hCaloEtFractionHadronic = map_of_MEs[DirName+"/"+"CaloEtFractionHadronic"]; if (hCaloEtFractionHadronic && hCaloEtFractionHadronic->getRootObject())  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
      hCaloEmEtFraction       = map_of_MEs[DirName+"/"+"CaloEmEtFraction"];       if (hCaloEmEtFraction       && hCaloEmEtFraction->getRootObject())        hCaloEmEtFraction->Fill(caloEmEtFraction);
      hCaloEmEtFraction020 = map_of_MEs[DirName+"/"+"CaloEmEtFraction020"];       if (MET> 20.  &&  hCaloEmEtFraction020    && hCaloEmEtFraction020->getRootObject()) hCaloEmEtFraction020->Fill(caloEmEtFraction);
      //if (metCollectionLabel_.label() == "corMetGlobalMuons" ) {
	
      //for( reco::MuonCollection::const_iterator muonit = muonHandle_->begin(); muonit != muonHandle_->end(); muonit++ ) {
      //  const reco::TrackRef siTrack = muonit->innerTrack();
      //  hCalomuPt    = map_of_MEs[DirName+"/"+"CalomuonPt"];  
      //  if (hCalomuPt    && hCalomuPt->getRootObject())   hCalomuPt->Fill( muonit->p4().pt() );
      //  hCalomuEta   = map_of_MEs[DirName+"/"+"CalomuonEta"];    if (hCalomuEta   && hCalomuEta->getRootObject())    hCalomuEta->Fill( muonit->p4().eta() );
      //  hCalomuNhits = map_of_MEs[DirName+"/"+"CalomuonNhits"];  if (hCalomuNhits && hCalomuNhits->getRootObject())  hCalomuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
      //  hCalomuChi2  = map_of_MEs[DirName+"/"+"CalomuonNormalizedChi2"];   if (hCalomuChi2  && hCalomuChi2->getRootObject())   hCalomuChi2->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
      //  double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( beamSpot_) : -999;
      //  hCalomuD0    = map_of_MEs[DirName+"/"+"CalomuonD0"];     if (hCalomuD0    && hCalomuD0->getRootObject())  hCalomuD0->Fill( d0 );
      //}
	
      //const unsigned int nMuons = muonHandle_->size();
      //for( unsigned int mus = 0; mus < nMuons; mus++ ) {
      //  reco::MuonRef muref( muonHandle_, mus);
      //  reco::MuonMETCorrectionData muCorrData = (*tcMetValueMapHandle_)[muref];
      //  hCaloMExCorrection      = map_of_MEs[DirName+"/"+"CaloMExCorrection"];       if (hCaloMExCorrection      && hCaloMExCorrection->getRootObject())       hCaloMExCorrection-> Fill(muCorrData.corrY());
      //  hCaloMEyCorrection      = map_of_MEs[DirName+"/"+"CaloMEyCorrection"];       if (hCaloMEyCorrection      && hCaloMEyCorrection->getRootObject())       hCaloMEyCorrection-> Fill(muCorrData.corrX());
      //  hCaloMuonCorrectionFlag = map_of_MEs[DirName+"/"+"CaloMuonCorrectionFlag"];  if (hCaloMuonCorrectionFlag && hCaloMuonCorrectionFlag->getRootObject())  hCaloMuonCorrectionFlag-> Fill(muCorrData.type());
      //}
      //} 
    }

    if(isPFMet_){
      // **** Get the MET container
      //const PFMETCollection *pfmetcol = pfmetcoll.product();
      //const PFMET *pfmet;
      //pfmet = &(pfmetcol->front());
      
      // PFMET getters
      //----------------------------------------------------------------------------
      double pfPhotonEtFraction        = pfmet.photonEtFraction();
      double pfPhotonEt                = pfmet.photonEt();
      double pfNeutralHadronEtFraction = pfmet.neutralHadronEtFraction();
      double pfNeutralHadronEt         = pfmet.neutralHadronEt();
      double pfElectronEtFraction      = pfmet.electronEtFraction();
      double pfElectronEt              = pfmet.electronEt();
      double pfChargedHadronEtFraction = pfmet.chargedHadronEtFraction();
      double pfChargedHadronEt         = pfmet.chargedHadronEt();
      double pfMuonEtFraction          = pfmet.muonEtFraction();
      double pfMuonEt                  = pfmet.muonEt();
      double pfHFHadronEtFraction      = pfmet.HFHadronEtFraction();
      double pfHFHadronEt              = pfmet.HFHadronEt();
      double pfHFEMEtFraction          = pfmet.HFEMEtFraction();
      double pfHFEMEt                  = pfmet.HFEMEt();
      
      mePhotonEtFraction        = map_of_MEs[DirName + "/PfPhotonEtFraction"];
      mePhotonEt                = map_of_MEs[DirName + "/PfPhotonEt"];
      meNeutralHadronEtFraction = map_of_MEs[DirName + "/PfNeutralHadronEtFraction"];
      meNeutralHadronEt         = map_of_MEs[DirName + "/PfNeutralHadronEt"];
      meElectronEtFraction      = map_of_MEs[DirName + "/PfElectronEtFraction"];
      meElectronEt              = map_of_MEs[DirName + "/PfElectronEt"];
      meChargedHadronEtFraction = map_of_MEs[DirName + "/PfChargedHadronEtFraction"];
      meChargedHadronEt         = map_of_MEs[DirName + "/PfChargedHadronEt"];
      meMuonEtFraction          = map_of_MEs[DirName + "/PfMuonEtFraction"];
      meMuonEt                  = map_of_MEs[DirName + "/PfMuonEt"];
      meHFHadronEtFraction      = map_of_MEs[DirName + "/PfHFHadronEtFraction"];
      meHFHadronEt              = map_of_MEs[DirName + "/PfHFHadronEt"];
      meHFEMEtFraction          = map_of_MEs[DirName + "/PfHFEMEtFraction"];
      meHFEMEt                  = map_of_MEs[DirName + "/PfHFEMEt"];
      
      if (mePhotonEtFraction        && mePhotonEtFraction       ->getRootObject()) mePhotonEtFraction       ->Fill(pfPhotonEtFraction);
      if (mePhotonEt                && mePhotonEt               ->getRootObject()) mePhotonEt               ->Fill(pfPhotonEt);
      if (meNeutralHadronEtFraction && meNeutralHadronEtFraction->getRootObject()) meNeutralHadronEtFraction->Fill(pfNeutralHadronEtFraction);
      if (meNeutralHadronEt         && meNeutralHadronEt        ->getRootObject()) meNeutralHadronEt        ->Fill(pfNeutralHadronEt);
      if (meElectronEtFraction      && meElectronEtFraction     ->getRootObject()) meElectronEtFraction     ->Fill(pfElectronEtFraction);
      if (meElectronEt              && meElectronEt             ->getRootObject()) meElectronEt             ->Fill(pfElectronEt);
      if (meChargedHadronEtFraction && meChargedHadronEtFraction->getRootObject()) meChargedHadronEtFraction->Fill(pfChargedHadronEtFraction);
      if (meChargedHadronEt         && meChargedHadronEt        ->getRootObject()) meChargedHadronEt        ->Fill(pfChargedHadronEt);
      if (meMuonEtFraction          && meMuonEtFraction         ->getRootObject()) meMuonEtFraction         ->Fill(pfMuonEtFraction);
      if (meMuonEt                  && meMuonEt                 ->getRootObject()) meMuonEt                 ->Fill(pfMuonEt);
      if (meHFHadronEtFraction      && meHFHadronEtFraction     ->getRootObject()) meHFHadronEtFraction     ->Fill(pfHFHadronEtFraction);
      if (meHFHadronEt              && meHFHadronEt             ->getRootObject()) meHFHadronEt             ->Fill(pfHFHadronEt);
      if (meHFEMEtFraction          && meHFEMEtFraction         ->getRootObject()) meHFEMEtFraction         ->Fill(pfHFEMEtFraction);
      if (meHFEMEt                  && meHFEMEt                 ->getRootObject()) meHFEMEt                 ->Fill(pfHFEMEt);

      //NPV profiles     
      
      mePhotonEtFraction_profile        = map_of_MEs[DirName + "/PfPhotonEtFraction_profile"];
      mePhotonEt_profile                = map_of_MEs[DirName + "/PfPhotonEt_profile"];
      meNeutralHadronEtFraction_profile = map_of_MEs[DirName + "/PfNeutralHadronEtFraction_profile"];
      meNeutralHadronEt_profile         = map_of_MEs[DirName + "/PfNeutralHadronEt_profile"];
      meElectronEtFraction_profile      = map_of_MEs[DirName + "/PfElectronEtFraction_profile"];
      meElectronEt_profile              = map_of_MEs[DirName + "/PfElectronEt_profile"];
      meChargedHadronEtFraction_profile = map_of_MEs[DirName + "/PfChargedHadronEtFraction_profile"];
      meChargedHadronEt_profile         = map_of_MEs[DirName + "/PfChargedHadronEt_profile"];
      meMuonEtFraction_profile          = map_of_MEs[DirName + "/PfMuonEtFraction_profile"];
      meMuonEt_profile                  = map_of_MEs[DirName + "/PfMuonEt_profile"];
      meHFHadronEtFraction_profile      = map_of_MEs[DirName + "/PfHFHadronEtFraction_profile"];
      meHFHadronEt_profile              = map_of_MEs[DirName + "/PfHFHadronEt_profile"];
      meHFEMEtFraction_profile          = map_of_MEs[DirName + "/PfHFEMEtFraction_profile"];
      meHFEMEt_profile                  = map_of_MEs[DirName + "/PfHFEMEt_profile"];
      
      if (mePhotonEtFraction_profile        && mePhotonEtFraction_profile       ->getRootObject()) mePhotonEtFraction_profile       ->Fill(numPV_, pfPhotonEtFraction);
      if (mePhotonEt_profile                && mePhotonEt_profile               ->getRootObject()) mePhotonEt_profile               ->Fill(numPV_, pfPhotonEt);
      if (meNeutralHadronEtFraction_profile && meNeutralHadronEtFraction_profile->getRootObject()) meNeutralHadronEtFraction_profile->Fill(numPV_, pfNeutralHadronEtFraction);
      if (meNeutralHadronEt_profile         && meNeutralHadronEt_profile        ->getRootObject()) meNeutralHadronEt_profile        ->Fill(numPV_, pfNeutralHadronEt);
      if (meElectronEtFraction_profile      && meElectronEtFraction_profile     ->getRootObject()) meElectronEtFraction_profile     ->Fill(numPV_, pfElectronEtFraction);
      if (meElectronEt_profile              && meElectronEt_profile             ->getRootObject()) meElectronEt_profile             ->Fill(numPV_, pfElectronEt);
      if (meChargedHadronEtFraction_profile && meChargedHadronEtFraction_profile->getRootObject()) meChargedHadronEtFraction_profile->Fill(numPV_, pfChargedHadronEtFraction);
      if (meChargedHadronEt_profile         && meChargedHadronEt_profile        ->getRootObject()) meChargedHadronEt_profile        ->Fill(numPV_, pfChargedHadronEt);
      if (meMuonEtFraction_profile          && meMuonEtFraction_profile         ->getRootObject()) meMuonEtFraction_profile         ->Fill(numPV_, pfMuonEtFraction);
      if (meMuonEt_profile                  && meMuonEt_profile                 ->getRootObject()) meMuonEt_profile                 ->Fill(numPV_, pfMuonEt);
      if (meHFHadronEtFraction_profile      && meHFHadronEtFraction_profile     ->getRootObject()) meHFHadronEtFraction_profile     ->Fill(numPV_, pfHFHadronEtFraction);
      if (meHFHadronEt_profile              && meHFHadronEt_profile             ->getRootObject()) meHFHadronEt_profile             ->Fill(numPV_, pfHFHadronEt);
      if (meHFEMEtFraction_profile          && meHFEMEtFraction_profile         ->getRootObject()) meHFEMEtFraction_profile         ->Fill(numPV_, pfHFEMEtFraction);
      if (meHFEMEt_profile                  && meHFEMEt_profile                 ->getRootObject()) meHFEMEt_profile                 ->Fill(numPV_, pfHFEMEt);
    }

    if (isCaloMet_){
      //if (bLumiSecPlot){//get from config level
      if (fill_met_high_level_histo){
	hMExLS = map_of_MEs[DirName+"/"+"MExLS"]; if (hMExLS  &&  hMExLS->getRootObject())   hMExLS->Fill(MEx,myLuminosityBlock);
	hMEyLS = map_of_MEs[DirName+"/"+"MEyLS"]; if (hMEyLS  &&  hMEyLS->getRootObject())   hMEyLS->Fill(MEy,myLuminosityBlock);
      }
    }

    ////////////////////////////////////
    ///* if (isTCMet_) {

    //if(trackHandle_.isValid()) {
    //	for( edm::View<reco::Track>::const_iterator trkit = trackHandle_->begin(); trkit != trackHandle_->end(); trkit++ ) {
    //	  htrkPt    = map_of_MEs[DirName+"/"+"trackPt"];     if (htrkPt    && htrkPt->getRootObject())     htrkPt->Fill( trkit->pt() );
    //	  htrkEta   = map_of_MEs[DirName+"/"+"trackEta"];    if (htrkEta   && htrkEta->getRootObject())    htrkEta->Fill( trkit->eta() );
    //	  htrkNhits = map_of_MEs[DirName+"/"+"trackNhits"];  if (htrkNhits && htrkNhits->getRootObject())  htrkNhits->Fill( trkit->numberOfValidHits() );
    //	  htrkChi2  = map_of_MEs[DirName+"/"+"trackNormalizedChi2"];  
    //	  if (htrkChi2  && htrkChi2->getRootObject())   htrkChi2->Fill( trkit->chi2() / trkit->ndof() );
    //	  double d0 = -1 * trkit->dxy( beamSpot_ );
    //	  htrkD0    = map_of_MEs[DirName+"/"+"trackD0"];     if (htrkD0 && htrkD0->getRootObject())        htrkD0->Fill( d0 );
    //	}
    //}else{if (verbose_) std::cout<<"tracks not valid"<<std::endl;}

    //if(electronHandle_.isValid()) {
    //	for( edm::View<reco::GsfElectron>::const_iterator eleit = electronHandle_->begin(); eleit != electronHandle_->end(); eleit++ ) {
    //	  helePt  = map_of_MEs[DirName+"/"+"electronPt"];   if (helePt  && helePt->getRootObject())   helePt->Fill( eleit->p4().pt() );
    //	  heleEta = map_of_MEs[DirName+"/"+"electronEta"];  if (heleEta && heleEta->getRootObject())  heleEta->Fill( eleit->p4().eta() );
    //	  heleHoE = map_of_MEs[DirName+"/"+"electronHoverE"];  if (heleHoE && heleHoE->getRootObject())  heleHoE->Fill( eleit->hadronicOverEm() );
    //	}
    //}else{
    //	if (verbose_) std::cout<<"electrons not valid"<<std::endl;
    //}

    //if(muonHandle_.isValid()) {
    //	for( reco::MuonCollection::const_iterator muonit = muonHandle_->begin(); muonit != muonHandle_->end(); muonit++ ) {
    //	  const reco::TrackRef siTrack = muonit->innerTrack();
    //	  hmuPt    = map_of_MEs[DirName+"/"+"muonPt"];     if (hmuPt    && hmuPt->getRootObject())  hmuPt   ->Fill( muonit->p4().pt() );
    //	  hmuEta   = map_of_MEs[DirName+"/"+"muonEta"];    if (hmuEta   && hmuEta->getRootObject())  hmuEta  ->Fill( muonit->p4().eta() );
    //	  hmuNhits = map_of_MEs[DirName+"/"+"muonNhits"];  if (hmuNhits && hmuNhits->getRootObject())  hmuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
    // hmuChi2  = map_of_MEs[DirName+"/"+"muonNormalizedChi2"];   if (hmuChi2  && hmuChi2->getRootObject())  hmuChi2 ->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
    //	  double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( beamSpot_) : -999;
    //	  hmuD0    = map_of_MEs[DirName+"/"+"muonD0"];     if (hmuD0    && hmuD0->getRootObject())  hmuD0->Fill( d0 );
    //	}
    //	const unsigned int nMuons = muonHandle_->size();
    //	for( unsigned int mus = 0; mus < nMuons; mus++ ) {
    //	  reco::MuonRef muref( muonHandle_, mus);
    //	  reco::MuonMETCorrectionData muCorrData = (*tcMetValueMapHandle_)[muref];
    //	  hMExCorrection      = map_of_MEs[DirName+"/"+"MExCorrection"];       if (hMExCorrection      && hMExCorrection->getRootObject())       hMExCorrection-> Fill(muCorrData.corrY());
    //	  hMEyCorrection      = map_of_MEs[DirName+"/"+"MEyCorrection"];       if (hMEyCorrection      && hMEyCorrection->getRootObject())       hMEyCorrection-> Fill(muCorrData.corrX());
    //hMuonCorrectionFlag = map_of_MEs[DirName+"/"+"CorrectionFlag"];  if (hMuonCorrectionFlag && hMuonCorrectionFlag->getRootObject())  hMuonCorrectionFlag-> Fill(muCorrData.type());
    //	}
    //  }else{
    //if (verbose_)  std::cout<<"muons not valid"<<std::endl;
    //      }
    //  }*/
}

//// ***********************************************************
//bool METAnalyzer::selectHighPtJetEvent(const edm::Event& iEvent){
//
//  bool return_value=false;
//
//  if(isCaloMet_){
//    edm::Handle<reco::CaloJetCollection> caloJets;
//    iEvent.getByToken(caloJetsToken_, caloJets);
//    if (!caloJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    
//    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
//	 cal!=caloJets->end(); ++cal){
//      if (cal->pt()>highPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }/*
//  if(isTCMet_){
//    edm::Handle<reco::JPTJetCollection> jptJets;
//    iEvent.getByToken(jptJetsToken_, jptJets);
//    if (!jptJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    
//    for (reco::JPTJetCollection::const_iterator cal = jptJets->begin();
//	 cal!=jptJets->end(); ++cal){
//      if (cal->pt()>highPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }*/
//  if(isPFMet_){
//    edm::Handle<reco::PFJetCollection> PFJets;
//    iEvent.getByToken(pfJetsToken_, PFJets);
//    if (!PFJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    for (reco::PFJetCollection::const_iterator cal = PFJets->begin();
//	 cal!=PFJets->end(); ++cal){
//      if (cal->pt()>highPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//
//
//  return return_value;
//}
//
//// // ***********************************************************
//bool METAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){
//
//  bool return_value=false;
//  if(isCaloMet_){
//    edm::Handle<reco::CaloJetCollection> caloJets;
//    iEvent.getByToken(caloJetsToken_, caloJets);
//    if (!caloJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    
//    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
//	 cal!=caloJets->end(); ++cal){
//      if (cal->pt()>lowPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//  if(isTCMet_){
//    edm::Handle<reco::JPTJetCollection> jptJets;
//    iEvent.getByToken(jptJetsToken_, jptJets);
//    if (!jptJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    
//    for (reco::JPTJetCollection::const_iterator cal = jptJets->begin();
//	 cal!=jptJets->end(); ++cal){
//      if (cal->pt()>lowPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//  if(isPFMet_){
//    edm::Handle<reco::PFJetCollection> PFJets;
//    iEvent.getByToken(pfJetsToken_, PFJets);
//    if (!PFJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    for (reco::PFJetCollection::const_iterator cal = PFJets->begin();
//	 cal!=PFJets->end(); ++cal){
//      if (cal->pt()>lowPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//  return return_value;
//
//}
//
//
//// ***********************************************************
//bool METAnalyzer::selectWElectronEvent(const edm::Event& iEvent){
//
//  bool return_value=true;
//
//  /*
//    W-electron event selection comes here
//   */
//
//  return return_value;
//
//}
//
//// ***********************************************************
//bool METAnalyzer::selectWMuonEvent(const edm::Event& iEvent){
//
//  bool return_value=true;
//
//  /*
//    W-muon event selection comes here
//   */
//
//  return return_value;
//
//}

