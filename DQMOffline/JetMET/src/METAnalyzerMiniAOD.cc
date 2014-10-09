
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

#include "DQMOffline/JetMET/interface/METAnalyzerMiniAOD.h"
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
#include "math.h"

#include <string>

using namespace edm;
using namespace reco;
using namespace math;

// ***********************************************************
METAnalyzerMiniAOD::METAnalyzerMiniAOD(const edm::ParameterSet& pSet) {
  parameters = pSet;

  outputMEsInRootFile   = parameters.getParameter<bool>("OutputMEsInRootFile");
  mOutputFile_   = parameters.getParameter<std::string>("OutputFile");

  LSBegin_     = pSet.getParameter<int>("LSBegin");
  LSEnd_       = pSet.getParameter<int>("LSEnd");

  MetType_ = parameters.getUntrackedParameter<std::string>("METType");

  triggerResultsLabel_        = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_= consumes<edm::TriggerResults>(edm::InputTag(triggerResultsLabel_));

  isCaloMet_ = (std::string("calo")==MetType_);
  //isTCMet_ = (std::string("tc") ==MetType_);
  isPFMet_ = (std::string("pf") ==MetType_);

  // MET information
  metCollectionLabel_       = parameters.getParameter<edm::InputTag>("METCollectionLabel");
  patMETToken_= consumes<pat::METCollection>(edm::InputTag(metCollectionLabel_));

  if(/*isTCMet_ || */isCaloMet_){
    jetIDFunctorLoose=JetIDSelectionFunctor(JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::LOOSE);
  }
  if(isPFMet_){
    patCandToken_ = consumes<std::vector<pat::PackedCandidate> >(pSet.getParameter<edm::InputTag>("srcPFlow"));
    pfjetIDFunctorLoose=PFJetIDSelectionFunctor(PFJetIDSelectionFunctor::FIRSTDATA, PFJetIDSelectionFunctor::LOOSE);
  }
  ptThreshold_ = parameters.getParameter<double>("ptThreshold");

  
  fill_met_high_level_histo = parameters.getParameter<bool>("fillMetHighLevel");
  
  hTriggerLabelsIsSet_ = false;
  //jet cleanup parameters
  cleaningParameters_ = pSet.getParameter<ParameterSet>("CleaningParameters"); 

  diagnosticsParameters_ = pSet.getParameter<std::vector<edm::ParameterSet> >("METDiagonisticsParameters");
  
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

  // Other data collections
  jetCollectionLabel_       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  patJetsToken_ = consumes< edm::View<pat::Jet> >(jetCollectionLabel_);

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

}

// ***********************************************************
METAnalyzerMiniAOD::~METAnalyzerMiniAOD() {
  for (std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
    delete *it;
  }
  delete DCSFilter_;
}


void METAnalyzerMiniAOD::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				 edm::EventSetup const &) {
  std::string DirName = FolderName_+metCollectionLabel_.label();
  ibooker.setCurrentFolder(DirName);

  if(!folderNames_.empty()){
    folderNames_.clear();
  }

  folderNames_.push_back("Uncleaned");
  folderNames_.push_back("Cleaned");
  folderNames_.push_back("DiJet");

  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    bookMESet(DirName+"/"+*ic, ibooker,map_dijet_MEs);
    }
}


// ***********************************************************
void METAnalyzerMiniAOD::bookMESet(std::string DirName, DQMStore::IBooker & ibooker, std::map<std::string,MonitorElement*>& map_of_MEs)
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
void METAnalyzerMiniAOD::bookMonitorElement(std::string DirName,DQMStore::IBooker & ibooker, std::map<std::string,MonitorElement*>& map_of_MEs, bool bLumiSecPlot=false)
{
  if (verbose_) std::cout << "bookMonitorElement " << DirName << std::endl;

  ibooker.setCurrentFolder(DirName);

  hTrigger    = ibooker.book1D("triggerResults", "triggerResults", 500, 0, 500); 
  hMEx        = ibooker.book1D("MEx",        "MEx",        200, -500,  500);
  hMEy        = ibooker.book1D("MEy",        "MEy",        200, -500,  500);
  hMET        = ibooker.book1D("MET",        "MET",        200,    0, 1000);
  hSumET      = ibooker.book1D("SumET",      "SumET",      400,    0, 4000);
  hMETSig     = ibooker.book1D("METSig",     "METSig",      51,    0,   51);
  hMETPhi     = ibooker.book1D("METPhi",     "METPhi",      60, -M_PI,  M_PI);
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

    hCaloMETPhi020  = ibooker.book1D("CaloMETPhi020",  "CaloMETPhi020",   60, -M_PI,  M_PI);
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
    meNeutralHadronEtFraction = ibooker.book1D("PfNeutralHadronEtFraction", "pfmet.neutralHadronEtFraction()",  50, 0,    1);
    meElectronEtFraction      = ibooker.book1D("PfElectronEtFraction",      "pfmet.electronEtFraction()",       50, 0,    1);
    meChargedHadronEtFraction = ibooker.book1D("PfChargedHadronEtFraction", "pfmet.chargedHadronEtFraction()",  50, 0,    1);
    meMuonEtFraction          = ibooker.book1D("PfMuonEtFraction",          "pfmet.muonEtFraction()",           50, 0,    1);
    meHFHadronEtFraction      = ibooker.book1D("PfHFHadronEtFraction",      "pfmet.HFHadronEtFraction()",       50, 0,    1);
    meHFEMEtFraction          = ibooker.book1D("PfHFEMEtFraction",          "pfmet.HFEMEtFraction()",           50, 0,    1);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction"       ,mePhotonEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFraction",meNeutralHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfElectronEtFraction"     ,meElectronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFraction",meChargedHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfMuonEtFraction"         ,meMuonEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFraction"     ,meHFHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFraction"         ,meHFEMEtFraction));
    
    mePhotonEtFraction_profile        = ibooker.bookProfile("PfPhotonEtFraction_profile",        "pfmet.photonEtFraction()",        nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meNeutralHadronEtFraction_profile = ibooker.bookProfile("PfNeutralHadronEtFraction_profile", "pfmet.neutralHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meElectronEtFraction_profile      = ibooker.bookProfile("PfElectronEtFraction_profile",      "pfmet.electronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meChargedHadronEtFraction_profile = ibooker.bookProfile("PfChargedHadronEtFraction_profile", "pfmet.chargedHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meMuonEtFraction_profile          = ibooker.bookProfile("PfMuonEtFraction_profile",          "pfmet.muonEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFHadronEtFraction_profile      = ibooker.bookProfile("PfHFHadronEtFraction_profile",      "pfmet.HFHadronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFEMEtFraction_profile          = ibooker.bookProfile("PfHFEMEtFraction_profile",          "pfmet.HFEMEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
        
    mePhotonEtFraction_profile       ->setAxisTitle("nvtx", 1);
    meNeutralHadronEtFraction_profile->setAxisTitle("nvtx", 1);
    meElectronEtFraction_profile     ->setAxisTitle("nvtx", 1);
    meChargedHadronEtFraction_profile->setAxisTitle("nvtx", 1);
    meMuonEtFraction_profile         ->setAxisTitle("nvtx", 1);
    meHFHadronEtFraction_profile     ->setAxisTitle("nvtx", 1);
    meHFEMEtFraction_profile         ->setAxisTitle("nvtx", 1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction_profile"        ,mePhotonEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFraction_profile" ,meNeutralHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfElectronEtFraction_profile"      ,meElectronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFraction_profile" ,meChargedHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfMuonEtFraction_profile"          ,meMuonEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFraction_profile"      ,meHFHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFraction_profile"          ,meHFEMEtFraction_profile));

   if(!etaMinPFCand_.empty()){
      etaMinPFCand_.clear();
      etaMaxPFCand_.clear();
      typePFCand_.clear();
      nbinsPFCand_.clear();
      countsPFCand_.clear();
      MExPFCand_.clear();
      MEyPFCand_.clear();
      profilePFCand_x_.clear();
      profilePFCand_y_.clear();
      occupancyPFCand_.clear();
      energyPFCand_.clear();
      ptPFCand_.clear();
      multiplicityPFCand_.clear();
      profilePFCand_x_name_.clear();
      profilePFCand_y_name_.clear();
      occupancyPFCand_name_.clear();
      energyPFCand_name_.clear();
      ptPFCand_name_.clear();
      multiplicityPFCand_name_.clear();
    }
    for (std::vector<edm::ParameterSet>::const_iterator v = diagnosticsParameters_.begin(); v!=diagnosticsParameters_.end(); v++) {
      int etaNBinsPFCand = v->getParameter<int>("etaNBins");
      double etaMinPFCand = v->getParameter<double>("etaMin");
      double etaMaxPFCand = v->getParameter<double>("etaMax");
      int phiNBinsPFCand = v->getParameter<int>("phiNBins");
      double phiMinPFCand = v->getParameter<double>("phiMin");
      double phiMaxPFCand = v->getParameter<double>("phiMax");
      int nMinPFCand = v->getParameter<int>("nMin");
      int nMaxPFCand = v->getParameter<int>("nMax");
      int nbinsPFCand = v->getParameter<double>("nbins");

      // etaNBins_.push_back(etaNBins);
      etaMinPFCand_.push_back(etaMinPFCand);
      etaMaxPFCand_.push_back(etaMaxPFCand);
      nbinsPFCand_.push_back(nbinsPFCand);
      typePFCand_.push_back(v->getParameter<int>("type"));
      countsPFCand_.push_back(0);
      MExPFCand_.push_back(0.);
      MEyPFCand_.push_back(0.);
      //std::cout<<" n/min/maxPFCand "<<nbinsPFCand<<" "<<etaMinPFCand<<" "<<etaMaxPFCand<<std::endl;

      profilePFCand_x_.push_back(ibooker.bookProfile(std::string(v->getParameter<std::string>("name")).append("_Px_").c_str(),     "Px",       nbinsPFCand, nMinPFCand, nMaxPFCand, -300,300));
      profilePFCand_x_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_Px_").c_str());
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+profilePFCand_x_name_[profilePFCand_x_name_.size()-1], profilePFCand_x_[profilePFCand_x_.size()-1]));
      profilePFCand_y_.push_back(ibooker.bookProfile(std::string(v->getParameter<std::string>("name")).append("_Py_").c_str(),     "Py",       nbinsPFCand, nMinPFCand, nMaxPFCand, -300,300));
      profilePFCand_y_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_Py_").c_str());
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+profilePFCand_y_name_[profilePFCand_y_name_.size()-1], profilePFCand_y_[profilePFCand_y_.size()-1]));
      occupancyPFCand_.push_back(ibooker.book2D(std::string(v->getParameter<std::string>("name")).append("_occupancy_").c_str(),"occupancy", etaNBinsPFCand, etaMinPFCand, etaMaxPFCand, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand));
      occupancyPFCand_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_occupancy_").c_str());
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+ occupancyPFCand_name_[occupancyPFCand_name_.size()-1], occupancyPFCand_[occupancyPFCand_.size()-1]));
      energyPFCand_.push_back(ibooker.book2D(std::string(v->getParameter<std::string>("name")).append("_energy_").c_str(),"energy", etaNBinsPFCand, etaMinPFCand, etaMaxPFCand, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand));
      energyPFCand_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_energy_").c_str());
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+ energyPFCand_name_[energyPFCand_name_.size()-1], energyPFCand_[energyPFCand_.size()-1]));
      ptPFCand_.push_back(ibooker.book2D(std::string(v->getParameter<std::string>("name")).append("_pt_").c_str(),"pt", etaNBinsPFCand, etaMinPFCand, etaMaxPFCand, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand));
      ptPFCand_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_pt_").c_str());
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+ ptPFCand_name_[ptPFCand_name_.size()-1], ptPFCand_[ptPFCand_.size()-1]));
      multiplicityPFCand_.push_back(ibooker.book1D(std::string(v->getParameter<std::string>("name")).append("_multiplicity_").c_str(),"multiplicity", nbinsPFCand, nMinPFCand, nMaxPFCand));
      multiplicityPFCand_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_multiplicity_").c_str());
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+ multiplicityPFCand_name_[multiplicityPFCand_name_.size()-1], multiplicityPFCand_[multiplicityPFCand_.size()-1]));
    }

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
  
  hMETRate      = ibooker.book1D("METRate",        "METRate",        200,    0, 1000);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"METRate",hMETRate));


  ibooker.setCurrentFolder("JetMET");
  lumisecME = ibooker.book1D("lumisec", "lumisec", 2500, 0., 2500.);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>("JetMET/lumisec",lumisecME));

}

// ***********************************************************
void METAnalyzerMiniAOD::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
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
void METAnalyzerMiniAOD::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
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

 

  //below is the original METAnalyzerMiniAOD formulation
  
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
void METAnalyzerMiniAOD::makeRatePlot(std::string DirName, double totltime)
{
  
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
void METAnalyzerMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {


  // *** Fill lumisection ME
  int myLuminosityBlock;
  myLuminosityBlock = iEvent.luminosityBlock();
  if(fill_met_high_level_histo){
    lumisecME=map_dijet_MEs["JetMET/lumisec"]; if(lumisecME && lumisecME->getRootObject()) lumisecME->Fill(myLuminosityBlock);
  }

  if (myLuminosityBlock<LSBegin_) return;
  if (myLuminosityBlock>LSEnd_ && LSEnd_>0) return;

  if (verbose_) std::cout << "METAnalyzerMiniAOD analyze" << std::endl;

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

  
  LogTrace(metname)<<"[METAnalyzerMiniAOD] Call to the MET analyzer";

  // ==========================================================
  //

  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByToken(hbheNoiseFilterResultToken_, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "METAnalyzerMiniAOD: Could not find HBHENoiseFilterResult" << std::endl;
    if (verbose_) std::cout << "METAnalyzerMiniAOD: Could not find HBHENoiseFilterResult" << std::endl;
  }

  // ==========================================================
  bool bJetID = false;
  bool bDiJetID = false;
  // Jet ID -------------------------------------------------------
  //


  // **** Get the MET container
  edm::Handle<pat::METCollection> metcoll;
  iEvent.getByToken(patMETToken_, metcoll);
  if(!metcoll.isValid()) return;

  const pat::MET *met=NULL;
  met=&(metcoll->front());


  edm::Handle<edm::View<pat::Jet> >jets;
  iEvent.getByToken(patJetsToken_,jets);

  edm::View<pat::Jet> patJets = *jets;

  unsigned int ind1=-1;
  double pt1=-1;
  bool pass_jetID1=false;
  unsigned int ind2=-1;
  double pt2=-1;
  bool pass_jetID2=false;

  //do loose jet ID-> check threshold on corrected jets
  int index=0;
  if(jets.isValid()){
    for(edm::View<pat::Jet>::const_iterator i_jet = patJets.begin();i_jet != patJets.end(); ++i_jet) {
      double pt_jet=-10;
      bool iscleaned=false;
      if(isCaloMet_&& i_jet->isCaloJet()){
	pt_jet=i_jet->pt();
	if(pt_jet> ptThreshold_){
	  iscleaned = jetIDFunctorLoose((*i_jet));
	}else{
	  iscleaned=true;
	}
      }
      if(isPFMet_&& i_jet->isPFJet()){
	pt_jet=i_jet->pt();
	if(pt_jet> ptThreshold_){
	  iscleaned = pfjetIDFunctorLoose((*i_jet));
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
	ind1=index;
	pass_jetID1=iscleaned;
      }else if (pt_jet>pt2){
	pt2=pt_jet;
	ind2=index;
	pass_jetID2=iscleaned;
      }
      index+=1;
    }
  }
  if(pass_jetID1 && pass_jetID2){
    edm::View<pat::Jet>::const_iterator i_jet1 = patJets.begin()+ind1;
    edm::View<pat::Jet>::const_iterator i_jet2 = patJets.begin()+ind2;
    double dphi=-1.0;
    dphi=fabs(i_jet1->phi()- i_jet2->phi());
    
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
  //Vertex information
  Handle<VertexCollection> vertexHandle;
  iEvent.getByToken(vertexToken_, vertexHandle);

  if (!vertexHandle.isValid()) {
    LogDebug("") << "CaloMETAnalyzerMiniAOD: Could not find vertex collection" << std::endl;
    if (verbose_) std::cout << "CaloMETAnalyzerMiniAOD: Could not find vertex collection" << std::endl;
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
    LogDebug("") << "CaloMETAnalyzerMiniAOD: Could not find GT readout record" << std::endl;
    if (verbose_) std::cout << "CaloMETAnalyzerMiniAOD: Could not find GT readout record product" << std::endl;
  }
  // DCS Filter
  bool bDCSFilter = (bypassAllDCSChecks_ || DCSFilter_->filter(iEvent, iSetup));
  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements

  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    if ((*ic=="Uncleaned")  &&(isCaloMet_ || bPrimaryVertex))     fillMESet(iEvent, DirName+"/"+*ic, *met,map_dijet_MEs);
    //take two lines out for first check
    if ((*ic=="Cleaned")    &&bDCSFilter&&bHBHENoiseFilter&&bPrimaryVertex&&bJetID) fillMESet(iEvent, DirName+"/"+*ic, *met,map_dijet_MEs);
    if ((*ic=="DiJet" )     &&bDCSFilter&&bHBHENoiseFilter&&bPrimaryVertex&&bDiJetID) fillMESet(iEvent, DirName+"/"+*ic, *met,map_dijet_MEs);
  }
}


// ***********************************************************
void METAnalyzerMiniAOD::fillMESet(const edm::Event& iEvent, std::string DirName,
				   const pat::MET& met,std::map<std::string,MonitorElement*>&  map_of_MEs)
{

  //dbe_->setCurrentFolder(DirName);

  bool bLumiSecPlot=fill_met_high_level_histo;
  //if (DirName.find("Uncleaned")) bLumiSecPlot=true; //now done on configlevel
  fillMonitorElement(iEvent, DirName, std::string(""), met, map_of_MEs,bLumiSecPlot);
  if (DirName.find("Cleaned")) {
    for (unsigned i = 0; i<triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i])  fillMonitorElement(iEvent, DirName, triggerFolderLabels_[i], met, map_of_MEs, false);
    }
  }

  if (DirName.find("DiJet")) {
    for (unsigned i = 0; i<triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i])  fillMonitorElement(iEvent, DirName, triggerFolderLabels_[i], met, map_of_MEs, false);
    }
  }
  

}

// ***********************************************************
void METAnalyzerMiniAOD::fillMonitorElement(const edm::Event& iEvent, std::string DirName,
					 std::string subFolderName,
					    const pat::MET& met, std::map<std::string,MonitorElement*>&  map_of_MEs,bool bLumiSecPlot)
{
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

    if(isCaloMet_ && met.isCaloMET() ){
      //const reco::CaloMETCollection *calometcol = calometcoll.product();
      //const reco::CaloMET *calomet;
      //calomet = &(calometcol->front());
      
      double caloEtFractionHadronic = met.etFractionHadronic();
      double caloEmEtFraction       = met.emEtFraction();

      double caloMaxEtInEMTowers    = met.maxEtInEmTowers();
      double caloMaxEtInHadTowers   = met.maxEtInHadTowers();
      
      double caloHadEtInHB = met.hadEtInHB();
      double caloHadEtInHO = met.hadEtInHO();
      double caloHadEtInHE = met.hadEtInHE();
      double caloHadEtInHF = met.hadEtInHF();
      double caloEmEtInEB  = met.emEtInEB();
      double caloEmEtInEE  = met.emEtInEE();
      double caloEmEtInHF  = met.emEtInHF();

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
    }

    if(isPFMet_ && met.isPFMET()){


      for (unsigned i=0;i<countsPFCand_.size();i++) {
	countsPFCand_[i]=0;
	MExPFCand_[i]=0.;
	MEyPFCand_[i]=0.;
      }



      // typedef std::vector<reco::PFCandidate> pfCand;
      edm::Handle<std::vector<pat::PackedCandidate> > particleFlow;
      iEvent.getByToken(patCandToken_, particleFlow);
      for (unsigned int i = 0; i < particleFlow->size(); ++i) {
	const pat::PackedCandidate& c = particleFlow->at(i);
	for (unsigned int j=0; j<typePFCand_.size(); j++) {
	  if (abs(c.pdgId())==typePFCand_[j]) {
	    if ((c.eta()>etaMinPFCand_[j]) and(c.eta()<etaMaxPFCand_[j])) {
	      countsPFCand_[j]+=1;
	      MExPFCand_[j]-=c.px();
	      MEyPFCand_[j]-=c.py();
	      ptPFCand_[j]   = map_of_MEs[DirName + "/"+ptPFCand_name_[j]];
	      if ( ptPFCand_[j]       && ptPFCand_[j]->getRootObject()) ptPFCand_[j]->Fill(c.eta(), c.phi(), c.pt());
	      energyPFCand_[j]   = map_of_MEs[DirName + "/"+energyPFCand_name_[j]];
	      if ( energyPFCand_[j]       && energyPFCand_[j]->getRootObject()) energyPFCand_[j]->Fill(c.eta(), c.phi(), c.energy());
	      occupancyPFCand_[j]   = map_of_MEs[DirName + "/"+occupancyPFCand_name_[j]];
	      if ( occupancyPFCand_[j]       && occupancyPFCand_[j]->getRootObject()) occupancyPFCand_[j]->Fill(c.eta(), c.phi());
	    }
	  }
	}
      }
      for (unsigned int j=0; j<countsPFCand_.size(); j++) {
	profilePFCand_x_[j]   = map_of_MEs[DirName + "/"+profilePFCand_x_name_[j]];
	if(profilePFCand_x_[j] && profilePFCand_x_[j]->getRootObject())	profilePFCand_x_[j]->Fill(countsPFCand_[j], MExPFCand_[j]);
	profilePFCand_y_[j]   = map_of_MEs[DirName + "/"+profilePFCand_y_name_[j]];
	if(profilePFCand_y_[j] && profilePFCand_y_[j]->getRootObject())	profilePFCand_y_[j]->Fill(countsPFCand_[j], MEyPFCand_[j]);
	multiplicityPFCand_[j]   = map_of_MEs[DirName + "/"+multiplicityPFCand_name_[j]];
	if(multiplicityPFCand_[j] && multiplicityPFCand_[j]->getRootObject())	multiplicityPFCand_[j]->Fill(countsPFCand_[j]);
      }


      
      // PFMET getters
      //----------------------------------------------------------------------------
      double pfPhotonEtFraction        = met.NeutralEMFraction();
      double pfNeutralHadronEtFraction = met.NeutralHadEtFraction();
      double pfElectronEtFraction      = met.ChargedEMEtFraction();
      double pfChargedHadronEtFraction = met.ChargedHadEtFraction();
      double pfMuonEtFraction          = met.MuonEtFraction();
      double pfHFHadronEtFraction      = met.Type6EtFraction();//HFHadrons
      double pfHFEMEtFraction          = met.Type7EtFraction();//HFEMEt
      
      mePhotonEtFraction        = map_of_MEs[DirName + "/PfPhotonEtFraction"];
      meNeutralHadronEtFraction = map_of_MEs[DirName + "/PfNeutralHadronEtFraction"];
      meElectronEtFraction      = map_of_MEs[DirName + "/PfElectronEtFraction"];
      meChargedHadronEtFraction = map_of_MEs[DirName + "/PfChargedHadronEtFraction"];
      meMuonEtFraction          = map_of_MEs[DirName + "/PfMuonEtFraction"];
      meHFHadronEtFraction      = map_of_MEs[DirName + "/PfHFHadronEtFraction"];
      meHFEMEtFraction          = map_of_MEs[DirName + "/PfHFEMEtFraction"];
      
      if (mePhotonEtFraction        && mePhotonEtFraction       ->getRootObject()) mePhotonEtFraction       ->Fill(pfPhotonEtFraction);
      if (meNeutralHadronEtFraction && meNeutralHadronEtFraction->getRootObject()) meNeutralHadronEtFraction->Fill(pfNeutralHadronEtFraction);
      if (meElectronEtFraction      && meElectronEtFraction     ->getRootObject()) meElectronEtFraction     ->Fill(pfElectronEtFraction);
      if (meChargedHadronEtFraction && meChargedHadronEtFraction->getRootObject()) meChargedHadronEtFraction->Fill(pfChargedHadronEtFraction);
      if (meMuonEtFraction          && meMuonEtFraction         ->getRootObject()) meMuonEtFraction         ->Fill(pfMuonEtFraction);
      if (meHFHadronEtFraction      && meHFHadronEtFraction     ->getRootObject()) meHFHadronEtFraction     ->Fill(pfHFHadronEtFraction);
      if (meHFEMEtFraction          && meHFEMEtFraction         ->getRootObject()) meHFEMEtFraction         ->Fill(pfHFEMEtFraction);

      //NPV profiles     
      
      mePhotonEtFraction_profile        = map_of_MEs[DirName + "/PfPhotonEtFraction_profile"];
      meNeutralHadronEtFraction_profile = map_of_MEs[DirName + "/PfNeutralHadronEtFraction_profile"];
      meElectronEtFraction_profile      = map_of_MEs[DirName + "/PfElectronEtFraction_profile"];
      meChargedHadronEtFraction_profile = map_of_MEs[DirName + "/PfChargedHadronEtFraction_profile"];
      meMuonEtFraction_profile          = map_of_MEs[DirName + "/PfMuonEtFraction_profile"];
      meHFHadronEtFraction_profile      = map_of_MEs[DirName + "/PfHFHadronEtFraction_profile"];
      meHFEMEtFraction_profile          = map_of_MEs[DirName + "/PfHFEMEtFraction_profile"];
      
      if (mePhotonEtFraction_profile        && mePhotonEtFraction_profile       ->getRootObject()) mePhotonEtFraction_profile       ->Fill(numPV_, pfPhotonEtFraction);
      if (meNeutralHadronEtFraction_profile && meNeutralHadronEtFraction_profile->getRootObject()) meNeutralHadronEtFraction_profile->Fill(numPV_, pfNeutralHadronEtFraction);
      if (meElectronEtFraction_profile      && meElectronEtFraction_profile     ->getRootObject()) meElectronEtFraction_profile     ->Fill(numPV_, pfElectronEtFraction);
      if (meChargedHadronEtFraction_profile && meChargedHadronEtFraction_profile->getRootObject()) meChargedHadronEtFraction_profile->Fill(numPV_, pfChargedHadronEtFraction);
      if (meMuonEtFraction_profile          && meMuonEtFraction_profile         ->getRootObject()) meMuonEtFraction_profile         ->Fill(numPV_, pfMuonEtFraction);
      if (meHFHadronEtFraction_profile      && meHFHadronEtFraction_profile     ->getRootObject()) meHFHadronEtFraction_profile     ->Fill(numPV_, pfHFHadronEtFraction);
      if (meHFEMEtFraction_profile          && meHFEMEtFraction_profile         ->getRootObject()) meHFEMEtFraction_profile         ->Fill(numPV_, pfHFEMEtFraction);
    }

    if (isCaloMet_){
      //if (bLumiSecPlot){//get from config level
      if (fill_met_high_level_histo){
	hMExLS = map_of_MEs[DirName+"/"+"MExLS"]; if (hMExLS  &&  hMExLS->getRootObject())   hMExLS->Fill(MEx,myLuminosityBlock);
	hMEyLS = map_of_MEs[DirName+"/"+"MEyLS"]; if (hMEyLS  &&  hMEyLS->getRootObject())   hMEyLS->Fill(MEy,myLuminosityBlock);
      }
    }

}
