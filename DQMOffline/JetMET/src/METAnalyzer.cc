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
#include "math.h"
#include "TH2F.h"
#include "TH2.h"

#include <string>

using namespace edm;
using namespace reco;
using namespace math;

// ***********************************************************
METAnalyzer::METAnalyzer(const edm::ParameterSet& pSet) {
  parameters = pSet;

  m_l1algoname_ = pSet.getParameter<std::string>("l1algoname");
  m_bitAlgTechTrig_=-1;

  LSBegin_     = pSet.getParameter<int>("LSBegin");
  LSEnd_       = pSet.getParameter<int>("LSEnd");
 // Smallest track pt
  ptMinCand_ = pSet.getParameter<double>("ptMinCand");

  // Smallest raw HCAL energy linked to the track
  hcalMin_ = pSet.getParameter<double>("hcalMin");

  MetType_ = parameters.getUntrackedParameter<std::string>("METType");

  triggerResultsLabel_        = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_= consumes<edm::TriggerResults>(edm::InputTag(triggerResultsLabel_));

  isCaloMet_ = (std::string("calo")==MetType_);
  //isTCMet_ = (std::string("tc") ==MetType_);
  isPFMet_ = (std::string("pf") ==MetType_);
  isMiniAODMet_ = (std::string("miniaod") ==MetType_);
  if(!isMiniAODMet_){
    jetCorrectorToken_ = consumes<reco::JetCorrector>(pSet.getParameter<edm::InputTag>("JetCorrections"));
  }

  // MET information
  metCollectionLabel_       = parameters.getParameter<edm::InputTag>("METCollectionLabel");

  if(/*isTCMet_ || */isCaloMet_){
    inputJetIDValueMap      = pSet.getParameter<edm::InputTag>("InputJetIDValueMap");
    jetID_ValueMapToken_= consumes< edm::ValueMap<reco::JetID> >(inputJetIDValueMap);
    jetIDFunctorLoose=JetIDSelectionFunctor(JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::LOOSE);
  }
  if(isPFMet_ || isMiniAODMet_){
    pflowToken_ = consumes<std::vector<reco::PFCandidate> >(pSet.getParameter<edm::InputTag>("srcPFlow"));
    pfjetIDFunctorLoose=PFJetIDSelectionFunctor(PFJetIDSelectionFunctor::FIRSTDATA, PFJetIDSelectionFunctor::LOOSE);
  }
  ptThreshold_ = parameters.getParameter<double>("ptThreshold");


  if(isPFMet_){
    pfMetToken_= consumes<reco::PFMETCollection>(edm::InputTag(metCollectionLabel_));
  }
  if(isCaloMet_){
    caloMetToken_= consumes<reco::CaloMETCollection>(edm::InputTag(metCollectionLabel_));
  }
  if(isMiniAODMet_){
    patMetToken_= consumes<pat::METCollection>(edm::InputTag(metCollectionLabel_));
  }
  //if(isTCMet_){
  // tcMetToken_= consumes<reco::METCollection>(edm::InputTag(metCollectionLabel_));
  //}
  
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
  onlyCleaned_ =  parameters.getUntrackedParameter<bool>("onlyCleaned");  
  vertexTag_    = cleaningParameters_.getParameter<edm::InputTag>("vertexCollection");
  vertexToken_  = consumes<std::vector<reco::Vertex> >(edm::InputTag(vertexTag_));

  //Trigger parameters
  gtTag_          = cleaningParameters_.getParameter<edm::InputTag>("gtLabel");
  gtToken_= consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(gtTag_));

  // Other data collections
  jetCollectionLabel_       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  if (isCaloMet_) caloJetsToken_ = consumes<reco::CaloJetCollection>(jetCollectionLabel_);
  //if (isTCMet_)   jptJetsToken_ = consumes<reco::JPTJetCollection>(jetCollectionLabel_);
  if (isPFMet_)   pfJetsToken_ = consumes<reco::PFJetCollection>(jetCollectionLabel_);
  if (isMiniAODMet_)   patJetsToken_ = consumes<pat::JetCollection>(jetCollectionLabel_);

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

 nCh = std::vector<unsigned int>(10,static_cast<unsigned int>(0));
 nEv = std::vector<unsigned int>(2,static_cast<unsigned int>(0));

}

// ***********************************************************
METAnalyzer::~METAnalyzer() {
  for (std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
    delete *it;
  }
  delete DCSFilter_;
}


void METAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				 edm::EventSetup const &) {
  std::string DirName = FolderName_+metCollectionLabel_.label();
  ibooker.setCurrentFolder(DirName);

  if(!folderNames_.empty()){
    folderNames_.clear();
  }
  if(runcosmics_){
    folderNames_.push_back("Uncleaned");
  }else{
    if(!onlyCleaned_){
      folderNames_.push_back("Uncleaned");
    }
    folderNames_.push_back("Cleaned");
    folderNames_.push_back("DiJet");
  }
  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    bookMESet(DirName+"/"+*ic, ibooker,map_dijet_MEs);
    }
}


// ***********************************************************
void METAnalyzer::bookMESet(std::string DirName, DQMStore::IBooker & ibooker, std::map<std::string,MonitorElement*>& map_of_MEs)
{
  bool bLumiSecPlot=fill_met_high_level_histo;
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

  }

  if(isPFMet_){
    mePhotonEtFraction_EmptyBunch          = ibooker.book1D("PfPhotonEtFraction_BXm2BXm1Empty",        "pfmet.photonEtFraction() prev empty 2 bunches",         50, 0,    1);
    mePhotonEtFraction_noEmptyBunch        = ibooker.book1D("PfPhotonEtFraction_BXm2BXm1Filled",      "pfmet.photonEtFraction() prev filled 2 bunches",         50, 0,    1);
    meNeutralHadronEtFraction_EmptyBunch   = ibooker.book1D("PfNeutralHadronEtFraction_BXm2BXm1Empty",   "pfmet.neutralHadronEtFraction() prev empty 2 bunches",         50, 0,    1);
    meNeutralHadronEtFraction_noEmptyBunch = ibooker.book1D("PfNeutralHadronEtFraction_BXm2BXm1Filled", "pfmet.neutralHadronEtFraction() prev filled 2 bunches",         50, 0,    1);
    meChargedHadronEtFraction_EmptyBunch   = ibooker.book1D("PfChargedHadronEtFraction_BXm2BXm1Empty",   "pfmet.chargedHadronEtFraction() prev empty 2 bunches",         50, 0,    1);
    meChargedHadronEtFraction_noEmptyBunch = ibooker.book1D("PfChargedHadronEtFraction_BXm2BXm1Filled", "pfmet.chargedHadronEtFraction() prev filled 2 bunches",         50, 0,    1);
    meMET_EmptyBunch                       = ibooker.book1D("MET_EmptyBunch",   "MET prev empty 2 bunches",        200,    0, 1000);
    meMET_noEmptyBunch                     = ibooker.book1D("MET_BXm2BXm1Filled", "MET prev filled 2 bunches",       200,    0, 1000);
    meSumET_EmptyBunch                     = ibooker.book1D("SumET_EmptyBunch",   "SumET prev empty 2 bunches",    400,    0, 4000);
    meSumET_noEmptyBunch                   = ibooker.book1D("SumET_BXm2BXm1Filled", "SumET prev filled 2 bunches",   400,    0, 4000);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction_BXm2BXm1Empty"       ,mePhotonEtFraction_EmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction_BXm2BXm1Filled"     ,mePhotonEtFraction_noEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFraction_BXm2BXm1Empty"  ,meNeutralHadronEtFraction_EmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralEtFraction_BXm2BXm1Filled"      ,meNeutralHadronEtFraction_noEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFraction_BXm2BXm1Empty"  ,meChargedHadronEtFraction_EmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedEtFraction_BXm2BXm1Filled"      ,meChargedHadronEtFraction_noEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MET_BXm2BXm1Empty"    ,meMET_EmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MET_BXm2BXm1Filled"  ,meMET_noEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"SumET_BXm2BXm1Empty"  ,meSumET_EmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"SumET_BXm2BXm1Filled",meSumET_noEmptyBunch));

    mePhotonEtFraction_oneEmptyBunch          = ibooker.book1D("PfPhotonEtFraction_BXm1Empty",        "pfmet.photonEtFraction() prev empty bunch",         50, 0,    1);
    mePhotonEtFraction_oneFullBunch        = ibooker.book1D("PfPhotonEtFraction_BXm1Filled",      "pfmet.photonEtFraction() prev filled bunch",         50, 0,    1);
    meNeutralHadronEtFraction_oneEmptyBunch   = ibooker.book1D("PfNeutralHadronEtFraction_BXm1Empty",   "pfmet.neutralHadronEtFraction() prev empty bunch",         50, 0,    1);
    meNeutralHadronEtFraction_oneFullBunch = ibooker.book1D("PfNeutralHadronEtFraction_BXm1Filled", "pfmet.neutralHadronEtFraction() prev filled bunch",         50, 0,    1);
    meChargedHadronEtFraction_oneEmptyBunch   = ibooker.book1D("PfChargedHadronEtFraction_BXm1Empty",   "pfmet.chargedHadronEtFraction() prev empty bunch",         50, 0,    1);
    meChargedHadronEtFraction_oneFullBunch = ibooker.book1D("PfChargedHadronEtFraction_BXm1Filled", "pfmet.chargedHadronEtFraction() prev filled bunch",         50, 0,    1);
    meMET_oneEmptyBunch                       = ibooker.book1D("MET_BXm1Empty",   "MET prev empty bunch",        200,    0, 1000);
    meMET_oneFullBunch                     = ibooker.book1D("MET_BXm1Filled", "MET prev filled bunch",       200,    0, 1000);
    meSumET_oneEmptyBunch                     = ibooker.book1D("SumET_BXm1Empty",   "SumET prev empty bunch",    400,    0, 4000);
    meSumET_oneFullBunch                   = ibooker.book1D("SumET_BXm1Filled", "SumET prev filled bunch",   400,    0, 4000);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction_BXm1Empty"       ,mePhotonEtFraction_oneEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction_BXm1Filled"     ,mePhotonEtFraction_oneFullBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFraction_BXm1Empty"  ,meNeutralHadronEtFraction_oneEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralEtFraction_BXm1Filled"      ,meNeutralHadronEtFraction_oneFullBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFraction_BXm1Empty"  ,meChargedHadronEtFraction_oneEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedEtFraction_BXm1Filled"      ,meChargedHadronEtFraction_oneFullBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MET_BXm1Empty"    ,meMET_oneEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MET_BXm1Filled"  ,meMET_oneFullBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"SumET_BXm1Empty"  ,meSumET_oneEmptyBunch));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"SumET_BXm1Filled",meSumET_oneFullBunch));

    mePhotonEt                = ibooker.book1D("PfPhotonEt",                "pfmet.photonEt()",                100, 0, 1000);
    meNeutralHadronEt         = ibooker.book1D("PfNeutralHadronEt",         "pfmet.neutralHadronEt()",         100, 0, 1000);
    meElectronEt              = ibooker.book1D("PfElectronEt",              "pfmet.electronEt()",              100, 0, 100);
    meChargedHadronEt         = ibooker.book1D("PfChargedHadronEt",         "pfmet.chargedHadronEt()",         100, 0, 1000);
    meMuonEt                  = ibooker.book1D("PfMuonEt",                  "pfmet.muonEt()",                  100, 0, 100);
    meHFHadronEt              = ibooker.book1D("PfHFHadronEt",              "pfmet.HFHadronEt()",              100, 0, 1000);
    meHFEMEt                  = ibooker.book1D("PfHFEMEt",                  "pfmet.HFEMEt()",                  100, 0, 1000);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEt"               ,mePhotonEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEt"        ,meNeutralHadronEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfElectronEt"             ,meElectronEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEt"        ,meChargedHadronEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfMuonEt"                 ,meMuonEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEt"             ,meHFHadronEt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEt"                 ,meHFEMEt));

    mePhotonEt_profile                = ibooker.bookProfile("PfPhotonEt_profile",                "pfmet.photonEt()",                nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meNeutralHadronEt_profile         = ibooker.bookProfile("PfNeutralHadronEt_profile",         "pfmet.neutralHadronEt()",         nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meChargedHadronEt_profile         = ibooker.bookProfile("PfChargedHadronEt_profile",         "pfmet.chargedHadronEt()",         nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meHFHadronEt_profile              = ibooker.bookProfile("PfHFHadronEt_profile",              "pfmet.HFHadronEt()",              nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meHFEMEt_profile                  = ibooker.bookProfile("PfHFEMEt_profile",                  "pfmet.HFEMEt()",                  nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    
    mePhotonEt_profile               ->setAxisTitle("nvtx", 1);
    meNeutralHadronEt_profile        ->setAxisTitle("nvtx", 1);
    meChargedHadronEt_profile        ->setAxisTitle("nvtx", 1);
    meHFHadronEt_profile             ->setAxisTitle("nvtx", 1);
    meHFEMEt_profile                 ->setAxisTitle("nvtx", 1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEt_profile"                ,mePhotonEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEt_profile"         ,meNeutralHadronEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEt_profile"         ,meChargedHadronEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEt_profile"              ,meHFHadronEt_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEt_profile"                  ,meHFEMEt_profile));

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
      multiplicityPFCand_.push_back(ibooker.book1D(std::string(v->getParameter<std::string>("name")).append("_multiplicity_").c_str(),"multiplicity", nbinsPFCand, nMinPFCand, nMaxPFCand));
      multiplicityPFCand_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_multiplicity_").c_str());
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+ multiplicityPFCand_name_[multiplicityPFCand_name_.size()-1], multiplicityPFCand_[multiplicityPFCand_.size()-1]));

      //push back names first, we need to create histograms with the name and fill it for endcap plots later
      occupancyPFCand_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_occupancy_").c_str());
      energyPFCand_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_energy_").c_str());
      ptPFCand_name_.push_back(std::string(v->getParameter<std::string>("name")).append("_pt_").c_str());
      //special booking for endcap plots, merge plots for eminus and eplus into one plot, using variable binning
      //barrel plots have eta-boundaries on minus and plus side
      //parameters start on minus side
      if(etaMinPFCand*etaMaxPFCand<0){//barrel plots, plot only in barrel region
	occupancyPFCand_.push_back(ibooker.book2D(std::string(v->getParameter<std::string>("name")).append("_occupancy_").c_str(),"occupancy", etaNBinsPFCand, etaMinPFCand, etaMaxPFCand, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand));
	energyPFCand_.push_back(ibooker.book2D(std::string(v->getParameter<std::string>("name")).append("_energy_").c_str(),"energy", etaNBinsPFCand, etaMinPFCand, etaMaxPFCand, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand));
	ptPFCand_.push_back(ibooker.book2D(std::string(v->getParameter<std::string>("name")).append("_pt_").c_str(),"pt", etaNBinsPFCand, etaMinPFCand, etaMaxPFCand, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand));
      }else{//endcap or forward plots, 
	const int nbins_eta_endcap=2*(etaNBinsPFCand+1);
	double eta_limits_endcap[nbins_eta_endcap];
	for(int i=0;i<nbins_eta_endcap;i++){
	  if(i<(etaNBinsPFCand+1)){
	    eta_limits_endcap[i]=etaMinPFCand+i*(etaMaxPFCand-etaMinPFCand)/(double)etaNBinsPFCand;
	  }else{
	    eta_limits_endcap[i]= -etaMaxPFCand +(i- (etaNBinsPFCand+1) )*(etaMaxPFCand-etaMinPFCand)/(double)etaNBinsPFCand;
	  }
	}
	TH2F* hist_temp_occup = new TH2F((occupancyPFCand_name_[occupancyPFCand_name_.size()-1]).c_str(),"occupancy",nbins_eta_endcap-1, eta_limits_endcap, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand);
	occupancyPFCand_.push_back(ibooker.book2D(occupancyPFCand_name_[occupancyPFCand_name_.size()-1],hist_temp_occup));
	TH2F* hist_temp_energy = new TH2F((energyPFCand_name_[energyPFCand_name_.size()-1]).c_str(),"energy",nbins_eta_endcap-1, eta_limits_endcap, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand);
	energyPFCand_.push_back(ibooker.book2D(energyPFCand_name_[energyPFCand_name_.size()-1],hist_temp_energy));
	TH2F* hist_temp_pt = new TH2F((ptPFCand_name_[ptPFCand_name_.size()-1]).c_str(),"pt",nbins_eta_endcap-1, eta_limits_endcap, phiNBinsPFCand, phiMinPFCand, phiMaxPFCand);
	ptPFCand_.push_back(ibooker.book2D(ptPFCand_name_[ptPFCand_name_.size()-1], hist_temp_pt));
      }
 
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+ occupancyPFCand_name_[occupancyPFCand_name_.size()-1], occupancyPFCand_[occupancyPFCand_.size()-1]));
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+ energyPFCand_name_[energyPFCand_name_.size()-1], energyPFCand_[energyPFCand_.size()-1]));
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+ ptPFCand_name_[ptPFCand_name_.size()-1], ptPFCand_[ptPFCand_.size()-1]));
    }
 
    mProfileIsoPFChHad_TrackOccupancy=ibooker.book2D("IsoPfChHad_Track_profile","Isolated PFChHadron Tracker_occupancy", 108, -2.7, 2.7, 160, -M_PI,M_PI);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_Track_profile"        ,mProfileIsoPFChHad_TrackOccupancy));
    mProfileIsoPFChHad_TrackPt=ibooker.book2D("IsoPfChHad_TrackPt","Isolated PFChHadron TrackPt", 108, -2.7, 2.7, 160, -M_PI,M_PI);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_TrackPt"        ,mProfileIsoPFChHad_TrackPt));


    mProfileIsoPFChHad_EcalOccupancyCentral  = ibooker.book2D("IsoPfChHad_ECAL_profile_central","IsolatedPFChHa ECAL occupancy (Barrel)", 180, -1.479, 1.479, 125, -M_PI,M_PI);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_ECAL_profile_central"        ,mProfileIsoPFChHad_EcalOccupancyCentral));
    mProfileIsoPFChHad_EMPtCentral=ibooker.book2D("IsoPfChHad_EMPt_central","Isolated PFChHadron HadPt_central", 180, -1.479, 1.479, 360, -M_PI,M_PI);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_EMPt_central"        ,mProfileIsoPFChHad_EMPtCentral));

    mProfileIsoPFChHad_EcalOccupancyEndcap  = ibooker.book2D("IsoPfChHad_ECAL_profile_endcap","IsolatedPFChHa ECAL occupancy (Endcap)", 110, -2.75, 2.75, 125, -M_PI,M_PI);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_ECAL_profile_endcap"        ,mProfileIsoPFChHad_EcalOccupancyEndcap));
    mProfileIsoPFChHad_EMPtEndcap=ibooker.book2D("IsoPfChHad_EMPt_endcap","Isolated PFChHadron EMPt_endcap", 110, -2.75, 2.75, 125, -M_PI,M_PI);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_EMPt_endcap"        ,mProfileIsoPFChHad_EMPtEndcap));

    const int nbins_eta=16;

    double eta_limits[nbins_eta]={-2.650,-2.500,-2.322,-2.172,-2.043,-1.930,-1.830,-1.740,1.740,1.830,1.930,2.043,2.172,2.3122,2.500,2.650};

    TH2F* hist_temp_HCAL =new TH2F("IsoPfChHad_HCAL_profile_endcap","IsolatedPFChHa HCAL occupancy (outer endcap)",nbins_eta-1,eta_limits, 36, -M_PI,M_PI);
    TH2F* hist_tempPt_HCAL=(TH2F*)hist_temp_HCAL->Clone("Isolated PFCHHadron HadPt (outer endcap)");

    mProfileIsoPFChHad_HcalOccupancyCentral  = ibooker.book2D("IsoPfChHad_HCAL_profile_central","IsolatedPFChHa HCAL occupancy (Central Part)", 40, -1.740, 1.740, 72, -M_PI,M_PI);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_HCAL_profile_central"        ,mProfileIsoPFChHad_HcalOccupancyCentral));
    mProfileIsoPFChHad_HadPtCentral=ibooker.book2D("IsoPfChHad_HadPt_central","Isolated PFChHadron HadPt_central", 40, -1.740, 1.740, 72, -M_PI,M_PI);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_HadPt_central"        ,mProfileIsoPFChHad_HadPtCentral));

    mProfileIsoPFChHad_HcalOccupancyEndcap  = ibooker.book2D("IsoPfChHad_HCAL_profile_endcap",hist_temp_HCAL);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_HCAL_profile_endcap"        ,mProfileIsoPFChHad_HcalOccupancyEndcap));
    mProfileIsoPFChHad_HadPtEndcap=ibooker.book2D("IsoPfChHad_HadPt_endcap",hist_tempPt_HCAL);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"IsoPfChHad_HadPt_endcap"        ,mProfileIsoPFChHad_HadPtEndcap));


  }

  if(isPFMet_ || isMiniAODMet_){
    mePhotonEtFraction        = ibooker.book1D("PfPhotonEtFraction",        "pfmet.photonEtFraction()",         50, 0,    1);
    meNeutralHadronEtFraction = ibooker.book1D("PfNeutralHadronEtFraction", "pfmet.neutralHadronEtFraction()",  50, 0,    1);
    meChargedHadronEtFraction = ibooker.book1D("PfChargedHadronEtFraction", "pfmet.chargedHadronEtFraction()",  50, 0,    1);
    meHFHadronEtFraction      = ibooker.book1D("PfHFHadronEtFraction",      "pfmet.HFHadronEtFraction()",       50, 0,    1);
    meHFEMEtFraction          = ibooker.book1D("PfHFEMEtFraction",          "pfmet.HFEMEtFraction()",           50, 0,    1);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction"       ,mePhotonEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFraction",meNeutralHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFraction",meChargedHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFraction"     ,meHFHadronEtFraction));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFraction"     ,meHFEMEtFraction));

    mePhotonEtFraction_profile        = ibooker.bookProfile("PfPhotonEtFraction_profile",        "pfmet.photonEtFraction()",        nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meNeutralHadronEtFraction_profile = ibooker.bookProfile("PfNeutralHadronEtFraction_profile", "pfmet.neutralHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meChargedHadronEtFraction_profile = ibooker.bookProfile("PfChargedHadronEtFraction_profile", "pfmet.chargedHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFHadronEtFraction_profile      = ibooker.bookProfile("PfHFHadronEtFraction_profile",      "pfmet.HFHadronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFEMEtFraction_profile          = ibooker.bookProfile("PfHFEMEtFraction_profile",          "pfmet.HFEMEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    mePhotonEtFraction_profile       ->setAxisTitle("nvtx", 1);
    meNeutralHadronEtFraction_profile->setAxisTitle("nvtx", 1);
    meChargedHadronEtFraction_profile->setAxisTitle("nvtx", 1);
    meHFHadronEtFraction_profile     ->setAxisTitle("nvtx", 1);
    meHFEMEtFraction_profile         ->setAxisTitle("nvtx", 1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfPhotonEtFraction_profile"        ,mePhotonEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfNeutralHadronEtFraction_profile" ,meNeutralHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfChargedHadronEtFraction_profile" ,meChargedHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFHadronEtFraction_profile"      ,meHFHadronEtFraction_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PfHFEMEtFraction_profile"          ,meHFEMEtFraction_profile));
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
  lumisecME = ibooker.book1D("lumisec", "lumisec", 2501, -1., 2500.);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>("JetMET/lumisec",lumisecME));

}

// ***********************************************************
void METAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  edm::ESHandle<L1GtTriggerMenu> menuRcd;
  iSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
  const L1GtTriggerMenu* menu = menuRcd.product();
  for (CItAlgo techTrig = menu->gtTechnicalTriggerMap().begin(); techTrig != menu->gtTechnicalTriggerMap().end(); ++techTrig) {
    if ((techTrig->second).algoName() == m_l1algoname_) {
      m_bitAlgTechTrig_=(techTrig->second).algoBitNumber();
      break;
    }
  }

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
  for (unsigned int i = 0; i<hltConfig_.size();i++) {
    allTriggerNames_.push_back(hltConfig_.triggerName(i));
  }
//  std::cout<<"Length: "<<allTriggerNames_.size()<<std::endl;

  triggerSelectedSubFolders_ = parameters.getParameter<edm::VParameterSet>("triggerSelectedSubFolders");
  for ( std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
    int pos = it - triggerFolderEventFlag_.begin();
    if ((*it)->on()) {
      (*it)->initRun( iRun, iSetup );
      if (triggerSelectedSubFolders_[pos].exists(std::string("hltDBKey"))) {
        if ((*it)->expressionsFromDB((*it)->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
          triggerFolderExpr_[pos] = (*it)->expressionsFromDB((*it)->hltDBKey(), iSetup);
      }
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
      unsigned int pos = it - triggerFolderEventFlag_.begin();
      bool fd = (*it)->accept(iEvent, iSetup);
      triggerFolderDecisions_[pos] = fd;
    }
    allTriggerDecisions_.clear();
    for (unsigned int i=0;i<allTriggerNames_.size();++i)  {
      allTriggerDecisions_.push_back((*triggerResults).accept(i)); 
      //std::cout<<"TR "<<(*triggerResults).size()<<" "<<(*triggerResults).accept(i)<<" "<<allTriggerNames_[i]<<std::endl;
    }
  }

  // ==========================================================
  // MET information

  // **** Get the MET container
  edm::Handle<reco::METCollection> tcmetcoll;
  edm::Handle<reco::CaloMETCollection> calometcoll;
  edm::Handle<reco::PFMETCollection> pfmetcoll;
  edm::Handle<pat::METCollection> patmetcoll;

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
  if(isMiniAODMet_){
    iEvent.getByToken(patMetToken_, patmetcoll);
    if(!patmetcoll.isValid()) return;
  }

  const MET *met=NULL;
  const pat::MET *patmet=NULL;
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
  if(isMiniAODMet_){
    met=&(patmetcoll->front());
    patmet=&(patmetcoll->front());
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

  edm::Handle<bool> HBHENoiseFilterResultHandle;
  bool HBHENoiseFilterResult=true;
  if(!isMiniAODMet_){//not checked for MiniAOD
    iEvent.getByToken(hbheNoiseFilterResultToken_, HBHENoiseFilterResultHandle);
    if (!HBHENoiseFilterResultHandle.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    }
    HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  }
  // ==========================================================
  bool bJetID = false;
  bool bDiJetID = false;
  // Jet ID -------------------------------------------------------
  //

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection> jptJets;
  edm::Handle<PFJetCollection> pfJets;
  edm::Handle<pat::JetCollection> patJets;

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

  if (isMiniAODMet_){ iEvent.getByToken(patJetsToken_, patJets);
    if (!patJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find patjet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find patjet product" << std::endl;
    }
    collsize=patJets->size();
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

  edm::Handle<reco::JetCorrector> jetCorr;
  bool pass_correction_flag=false;
  if(!isMiniAODMet_){
    iEvent.getByToken(jetCorrectorToken_, jetCorr);
    if (jetCorr.isValid()){
      pass_correction_flag=true;
    }
  }else{
    pass_correction_flag=true;
  }
  //do loose jet ID-> check threshold on corrected jets
  for (int ijet=0; ijet<collsize; ijet++) {
    double pt_jet=-10;
    double scale=1.;
    bool iscleaned=false;
    if (pass_correction_flag) {
      if(isCaloMet_){
	scale = jetCorr->correction((*caloJets)[ijet]);
      }
      //if(isTCMet_){
      //scale = jetCorr->correction((*jptJets)[ijet]);
      //}
      if(isPFMet_){
	scale = jetCorr->correction((*pfJets)[ijet]);
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
   if(isMiniAODMet_){
      pt_jet=(*patJets)[ijet].pt();
      if(pt_jet> ptThreshold_){
	pat::strbitset stringbitset=pfjetIDFunctorLoose.getBitTemplate();
	iscleaned = pfjetIDFunctorLoose((*patJets)[ijet],stringbitset);
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
    if(isMiniAODMet_){
      dphi=fabs((*patJets)[0].phi()-(*patJets)[1].phi());
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

  bool techTriggerResultBxE = false;
  bool techTriggerResultBxF = false;
  bool techTriggerResultBx0 = false;

  if (!gtReadoutRecord.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find GT readout record" << std::endl;
    if (verbose_) std::cout << "METAnalyzer: Could not find GT readout record product" << std::endl;
  }else{
    // trigger results before mask for BxInEvent -2 (E), -1 (F), 0 (L1A), 1, 2 
    const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxE = gtReadoutRecord->technicalTriggerWord(-2);
    const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxF = gtReadoutRecord->technicalTriggerWord(-1);
    const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBx0 = gtReadoutRecord->technicalTriggerWord();
    //const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxG = gtReadoutRecord->technicalTriggerWord(1);
    //const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxH = gtReadoutRecord->technicalTriggerWord(2);
    if (m_bitAlgTechTrig_ > -1) {
      techTriggerResultBx0 = technicalTriggerWordBeforeMaskBx0.at(m_bitAlgTechTrig_);
      if(techTriggerResultBx0!=0){
	techTriggerResultBxE = technicalTriggerWordBeforeMaskBxE.at(m_bitAlgTechTrig_);
	techTriggerResultBxF = technicalTriggerWordBeforeMaskBxF.at(m_bitAlgTechTrig_);
      }	
    }
  }

  // DCS Filter
  bool bDCSFilter = (bypassAllDCSChecks_ || DCSFilter_->filter(iEvent, iSetup));
  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements
  std::string DirName_old=DirName;
  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    bool pass_selection = false;
    if ((*ic=="Uncleaned")  &&(isCaloMet_ || bPrimaryVertex)){
      fillMESet(iEvent, DirName_old+"/"+*ic, *met,*patmet, *pfmet,*calomet,map_dijet_MEs);
      pass_selection =true;
    }
    //take two lines out for first check
    if ((*ic=="Cleaned")    &&bDCSFilter&&bHBHENoiseFilter&&bPrimaryVertex&&bJetID){
      fillMESet(iEvent, DirName_old+"/"+*ic, *met,*patmet,*pfmet,*calomet,map_dijet_MEs);
      pass_selection=true;
    }
    if ((*ic=="DiJet" )     &&bDCSFilter&&bHBHENoiseFilter&& bPrimaryVertex&& bDiJetID){
      fillMESet(iEvent, DirName_old+"/"+*ic, *met,*patmet,*pfmet,*calomet,map_dijet_MEs);
      pass_selection=true;
    }
    if(pass_selection && isPFMet_){
      DirName=DirName_old+"/"+*ic;
      if(techTriggerResultBx0 && techTriggerResultBxE && techTriggerResultBxF){
	mePhotonEtFraction_noEmptyBunch    = map_dijet_MEs[DirName+"/"+"PfPhotonEtFraction_BXm2BXm1Filled"];     if (  mePhotonEtFraction_noEmptyBunch  && mePhotonEtFraction_noEmptyBunch ->getRootObject())  mePhotonEtFraction_noEmptyBunch  ->Fill((*pfmet).photonEtFraction());
	meNeutralHadronEtFraction_noEmptyBunch    = map_dijet_MEs[DirName+"/"+"PfNeutralHadronEtFraction_BXm2BXm1Filled"];     if (  meNeutralHadronEtFraction_noEmptyBunch  && meNeutralHadronEtFraction_noEmptyBunch ->getRootObject())  meNeutralHadronEtFraction_noEmptyBunch  ->Fill((*pfmet).neutralHadronEtFraction());
	meChargedHadronEtFraction_noEmptyBunch    = map_dijet_MEs[DirName+"/"+"PfChargedHadronEtFraction_BXm2BXm1Filled"];     if (  meChargedHadronEtFraction_noEmptyBunch  && meChargedHadronEtFraction_noEmptyBunch ->getRootObject())  meChargedHadronEtFraction_noEmptyBunch  ->Fill((*pfmet).chargedHadronEtFraction());
	meMET_noEmptyBunch    = map_dijet_MEs[DirName+"/"+"MET_BXm2BXm1Filled"];     if (  meMET_noEmptyBunch  && meMET_noEmptyBunch ->getRootObject())  meMET_noEmptyBunch  ->Fill((*pfmet).pt());
	meSumET_noEmptyBunch    = map_dijet_MEs[DirName+"/"+"SumET_BXm2BXm1Filled"];     if (  meSumET_noEmptyBunch  && meSumET_noEmptyBunch ->getRootObject())  meSumET_noEmptyBunch  ->Fill((*pfmet).sumEt());
      }
      if(techTriggerResultBx0 && techTriggerResultBxF){
	mePhotonEtFraction_oneFullBunch    = map_dijet_MEs[DirName+"/"+"PfPhotonEtFraction_BXm1Filled"];     if (  mePhotonEtFraction_oneFullBunch  && mePhotonEtFraction_oneFullBunch ->getRootObject())  mePhotonEtFraction_oneFullBunch  ->Fill((*pfmet).photonEtFraction());
	meNeutralHadronEtFraction_oneFullBunch    = map_dijet_MEs[DirName+"/"+"PfNeutralHadronEtFraction_BXm1Filled"];     if (  meNeutralHadronEtFraction_oneFullBunch  && meNeutralHadronEtFraction_oneFullBunch ->getRootObject())  meNeutralHadronEtFraction_oneFullBunch  ->Fill((*pfmet).neutralHadronEtFraction());
	meChargedHadronEtFraction_oneFullBunch    = map_dijet_MEs[DirName+"/"+"PfChargedHadronEtFraction_BXm1Filled"];     if (  meChargedHadronEtFraction_oneFullBunch  && meChargedHadronEtFraction_oneFullBunch ->getRootObject())  meChargedHadronEtFraction_oneFullBunch  ->Fill((*pfmet).chargedHadronEtFraction());
	meMET_oneFullBunch    = map_dijet_MEs[DirName+"/"+"MET_BXm1Filled"];     if (  meMET_oneFullBunch  && meMET_oneFullBunch ->getRootObject())  meMET_oneFullBunch  ->Fill((*pfmet).pt());
	meSumET_oneFullBunch    = map_dijet_MEs[DirName+"/"+"SumET_BXm1Filled"];     if (  meSumET_oneFullBunch  && meSumET_oneFullBunch ->getRootObject())  meSumET_oneFullBunch  ->Fill((*pfmet).sumEt());
      }
      if(techTriggerResultBx0 && !techTriggerResultBxE && !techTriggerResultBxF){
	mePhotonEtFraction_EmptyBunch    = map_dijet_MEs[DirName+"/"+"PfPhotonEtFraction_BXm2BXm1Empty"];     if (  mePhotonEtFraction_EmptyBunch  && mePhotonEtFraction_EmptyBunch ->getRootObject())  mePhotonEtFraction_EmptyBunch  ->Fill((*pfmet).photonEtFraction());
	meNeutralHadronEtFraction_EmptyBunch    = map_dijet_MEs[DirName+"/"+"PfNeutralHadronEtFraction_BXm2BXm1Empty"];     if (  meNeutralHadronEtFraction_EmptyBunch  && meNeutralHadronEtFraction_EmptyBunch ->getRootObject())  meNeutralHadronEtFraction_EmptyBunch  ->Fill((*pfmet).neutralHadronEtFraction());
	meChargedHadronEtFraction_EmptyBunch    = map_dijet_MEs[DirName+"/"+"PfChargedHadronEtFraction_BXm2BXm1Empty"];     if (  meChargedHadronEtFraction_EmptyBunch  && meChargedHadronEtFraction_EmptyBunch ->getRootObject())  meChargedHadronEtFraction_EmptyBunch  ->Fill((*pfmet).chargedHadronEtFraction());
	meMET_EmptyBunch    = map_dijet_MEs[DirName+"/"+"MET_BXm2BXm1Empty"];     if (  meMET_EmptyBunch  && meMET_EmptyBunch ->getRootObject())  meMET_EmptyBunch  ->Fill((*pfmet).pt());
	meSumET_EmptyBunch    = map_dijet_MEs[DirName+"/"+"SumET_BXm2BXm1Empty"];     if (  meSumET_EmptyBunch  && meSumET_EmptyBunch ->getRootObject())  meSumET_EmptyBunch  ->Fill((*pfmet).sumEt());
      }
      if(techTriggerResultBx0 && !techTriggerResultBxF){
	mePhotonEtFraction_oneEmptyBunch    = map_dijet_MEs[DirName+"/"+"PfPhotonEtFraction_BXm1Empty"];     if (  mePhotonEtFraction_oneEmptyBunch  && mePhotonEtFraction_oneEmptyBunch ->getRootObject())  mePhotonEtFraction_oneEmptyBunch  ->Fill((*pfmet).photonEtFraction());
	meNeutralHadronEtFraction_oneEmptyBunch    = map_dijet_MEs[DirName+"/"+"PfNeutralHadronEtFraction_BXm1Empty"];     if (  meNeutralHadronEtFraction_oneEmptyBunch  && meNeutralHadronEtFraction_oneEmptyBunch ->getRootObject())  meNeutralHadronEtFraction_oneEmptyBunch  ->Fill((*pfmet).neutralHadronEtFraction());
	meChargedHadronEtFraction_oneEmptyBunch    = map_dijet_MEs[DirName+"/"+"PfChargedHadronEtFraction_BXm1Empty"];     if (  meChargedHadronEtFraction_oneEmptyBunch  && meChargedHadronEtFraction_oneEmptyBunch ->getRootObject())  meChargedHadronEtFraction_oneEmptyBunch  ->Fill((*pfmet).chargedHadronEtFraction());
	meMET_oneEmptyBunch    = map_dijet_MEs[DirName+"/"+"MET_BXm1Empty"];     if (  meMET_oneEmptyBunch  && meMET_oneEmptyBunch ->getRootObject())  meMET_oneEmptyBunch  ->Fill((*pfmet).pt());
	meSumET_oneEmptyBunch    = map_dijet_MEs[DirName+"/"+"SumET_BXm1Empty"];     if (  meSumET_oneEmptyBunch  && meSumET_oneEmptyBunch ->getRootObject())  meSumET_oneEmptyBunch  ->Fill((*pfmet).sumEt());
      }
    } 
  }
}


// ***********************************************************
void METAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName,
			    const reco::MET& met, const pat::MET& patmet, const reco::PFMET& pfmet, const reco::CaloMET& calomet,std::map<std::string,MonitorElement*>&  map_of_MEs)
{
  bool bLumiSecPlot=fill_met_high_level_histo;
  if (DirName.find("Uncleaned")) bLumiSecPlot=true; //now done on configlevel
  fillMonitorElement(iEvent, DirName, std::string(""), met, patmet, pfmet, calomet, map_of_MEs,bLumiSecPlot);
  if (DirName.find("Cleaned")) {
    for (unsigned int i = 0; i<triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i]){  fillMonitorElement(iEvent, DirName, triggerFolderLabels_[i], met, patmet, pfmet, calomet, map_of_MEs, false);
      }
    }
  }

  if (DirName.find("DiJet")) {
    for (unsigned int i = 0; i<triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i])  fillMonitorElement(iEvent, DirName, triggerFolderLabels_[i], met, patmet, pfmet, calomet, map_of_MEs, false);
    }
  }
  

}

// ***********************************************************
void METAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName,
					 std::string subFolderName,
				     const reco::MET& met,const pat::MET & patmet, const reco::PFMET & pfmet, const reco::CaloMET &calomet, std::map<std::string,MonitorElement*>&  map_of_MEs,bool bLumiSecPlot)
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

  if (subFolderName!=""){
    DirName = DirName +"/"+subFolderName;
  }

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
      
      double caloHadEtInHB = calomet.hadEtInHB();
      double caloHadEtInHO = calomet.hadEtInHO();
      double caloHadEtInHE = calomet.hadEtInHE();
      double caloHadEtInHF = calomet.hadEtInHF();
      double caloEmEtInEB  = calomet.emEtInEB();
      double caloEmEtInEE  = calomet.emEtInEE();
      double caloEmEtInHF  = calomet.emEtInHF();
      
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

      for (unsigned int i=0;i<countsPFCand_.size();i++) {
	countsPFCand_[i]=0;
	MExPFCand_[i]=0.;
	MEyPFCand_[i]=0.;
      }

      // typedef std::vector<reco::PFCandidate> pfCand;
      edm::Handle<std::vector<reco::PFCandidate> > particleFlow;
      iEvent.getByToken(pflowToken_, particleFlow);
      for (unsigned int i = 0; i < particleFlow->size(); ++i) {
	const reco::PFCandidate& c = particleFlow->at(i);
	for (unsigned int j=0; j<typePFCand_.size(); j++) {
	  if (c.particleId()==typePFCand_[j]) {
	    //second check for endcap, if inside barrel Max and Min symmetric around 0
	    if ( ((c.eta()>etaMinPFCand_[j]) && (c.eta()<etaMaxPFCand_[j])) || ((c.eta()> (-etaMaxPFCand_[j])) && (c.eta()< (-etaMinPFCand_[j]))) ){
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
	//fill quantities for isolated charged hadron quantities
	//only for charged hadrons
	if ( c.particleId() == 1 &&  c.pt() > ptMinCand_ ){
	  // At least 1 GeV in HCAL
	  double ecalRaw = c.rawEcalEnergy();
	  double hcalRaw = c.rawHcalEnergy();
	  if ( (ecalRaw + hcalRaw) > hcalMin_ ){
	    const PFCandidate::ElementsInBlocks& theElements = c.elementsInBlocks();
	    if( theElements.empty() ) continue;
	    unsigned int iTrack=-999;
	    std::vector<unsigned int> iECAL;// =999;
	    std::vector<unsigned int> iHCAL;// =999;
	    const reco::PFBlockRef blockRef = theElements[0].first;
	    const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
	    // Check that there is only one track in the block.
	    unsigned int nTracks = 0;
	    for(unsigned int iEle=0; iEle<elements.size(); iEle++) {	         
	      // Find the tracks in the block
	      PFBlockElement::Type type = elements[iEle].type();
	      switch( type ) {
	      case PFBlockElement::TRACK:
		iTrack = iEle;
		nTracks++;
		break;
	      case PFBlockElement::ECAL:
		iECAL.push_back( iEle );
		break;
	      case PFBlockElement::HCAL:
		iHCAL.push_back( iEle );
		break;
	      default:
		continue;
	      } 
	    }
	    if ( nTracks == 1 ){
	      // Characteristics of the track
	      const reco::PFBlockElementTrack& et = dynamic_cast<const reco::PFBlockElementTrack &>( elements[iTrack] );
	      mProfileIsoPFChHad_TrackOccupancy=map_of_MEs[DirName+"/"+"IsoPfChHad_Track_profile"];
	      if (mProfileIsoPFChHad_TrackOccupancy  && mProfileIsoPFChHad_TrackOccupancy->getRootObject()) mProfileIsoPFChHad_TrackOccupancy->Fill(et.trackRef()->eta(),et.trackRef()->phi());
	      mProfileIsoPFChHad_TrackPt=map_of_MEs[DirName+"/"+"IsoPfChHad_TrackPt"];
	      if (mProfileIsoPFChHad_TrackPt  && mProfileIsoPFChHad_TrackPt->getRootObject()) mProfileIsoPFChHad_TrackPt->Fill(et.trackRef()->eta(),et.trackRef()->phi(),et.trackRef()->pt());
	      //ECAL element
	      
	      for(unsigned int ii=0;ii<iECAL.size();ii++) {
		const reco::PFBlockElementCluster& eecal = dynamic_cast<const reco::PFBlockElementCluster &>( elements[ iECAL[ii] ] );
		if(fabs(eecal.clusterRef()->eta())<1.479){
		  mProfileIsoPFChHad_EcalOccupancyCentral=map_of_MEs[DirName+"/"+"IsoPfChHad_ECAL_profile_central"];
		  if (mProfileIsoPFChHad_EcalOccupancyCentral  && mProfileIsoPFChHad_EcalOccupancyCentral->getRootObject()) mProfileIsoPFChHad_EcalOccupancyCentral->Fill(eecal.clusterRef()->eta(),eecal.clusterRef()->phi());
		  mProfileIsoPFChHad_EMPtCentral=map_of_MEs[DirName+"/"+"IsoPfChHad_EMPt_central"];
		  if (mProfileIsoPFChHad_EMPtCentral  && mProfileIsoPFChHad_EMPtCentral->getRootObject()) mProfileIsoPFChHad_EMPtCentral->Fill(eecal.clusterRef()->eta(),eecal.clusterRef()->phi(),eecal.clusterRef()->pt());
		}else{
		  mProfileIsoPFChHad_EcalOccupancyEndcap=map_of_MEs[DirName+"/"+"IsoPfChHad_ECAL_profile_endcap"];
		  if (mProfileIsoPFChHad_EcalOccupancyEndcap  && mProfileIsoPFChHad_EcalOccupancyEndcap->getRootObject()) mProfileIsoPFChHad_EcalOccupancyEndcap->Fill(eecal.clusterRef()->eta(),eecal.clusterRef()->phi());
		  mProfileIsoPFChHad_EMPtEndcap=map_of_MEs[DirName+"/"+"IsoPfChHad_EMPt_endcap"];
		  if (mProfileIsoPFChHad_EMPtEndcap  && mProfileIsoPFChHad_EMPtEndcap->getRootObject()) mProfileIsoPFChHad_EMPtEndcap->Fill(eecal.clusterRef()->eta(),eecal.clusterRef()->phi(),eecal.clusterRef()->pt());
		}
	      }
	      //HCAL element
	      for(unsigned int ii=0;ii<iHCAL.size();ii++) {
		const reco::PFBlockElementCluster& ehcal = dynamic_cast<const reco::PFBlockElementCluster &>( elements[ iHCAL[ii] ] );
		if(fabs(ehcal.clusterRef()->eta())<1.740){
		  mProfileIsoPFChHad_HcalOccupancyCentral=map_of_MEs[DirName+"/"+"IsoPfChHad_HCAL_profile_central"];
		  if (mProfileIsoPFChHad_HcalOccupancyCentral  && mProfileIsoPFChHad_HcalOccupancyCentral->getRootObject()) mProfileIsoPFChHad_HcalOccupancyCentral->Fill(ehcal.clusterRef()->eta(),ehcal.clusterRef()->phi());
		  mProfileIsoPFChHad_HadPtCentral=map_of_MEs[DirName+"/"+"IsoPfChHad_HadPt_central"];
		  if (mProfileIsoPFChHad_HadPtCentral  && mProfileIsoPFChHad_HadPtCentral->getRootObject()) mProfileIsoPFChHad_HadPtCentral->Fill(ehcal.clusterRef()->eta(),ehcal.clusterRef()->phi(),ehcal.clusterRef()->pt());
		}else{
		  mProfileIsoPFChHad_HcalOccupancyEndcap=map_of_MEs[DirName+"/"+"IsoPfChHad_HCAL_profile_endcap"];
		  if (mProfileIsoPFChHad_HcalOccupancyEndcap  && mProfileIsoPFChHad_HcalOccupancyEndcap->getRootObject()) mProfileIsoPFChHad_HcalOccupancyEndcap->Fill(ehcal.clusterRef()->eta(),ehcal.clusterRef()->phi());
		  mProfileIsoPFChHad_HadPtEndcap=map_of_MEs[DirName+"/"+"IsoPfChHad_HadPt_endcap"];
		  if (mProfileIsoPFChHad_HadPtEndcap  && mProfileIsoPFChHad_HadPtEndcap->getRootObject()) mProfileIsoPFChHad_HadPtEndcap->Fill(ehcal.clusterRef()->eta(),ehcal.clusterRef()->phi(),ehcal.clusterRef()->pt());
		}	
	      }      
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
      double pfPhotonEtFraction        = pfmet.photonEtFraction();
      double pfPhotonEt                = pfmet.photonEt();
      double pfNeutralHadronEtFraction = pfmet.neutralHadronEtFraction();
      double pfNeutralHadronEt         = pfmet.neutralHadronEt();
      double pfElectronEt              = pfmet.electronEt();
      double pfChargedHadronEtFraction = pfmet.chargedHadronEtFraction();
      double pfChargedHadronEt         = pfmet.chargedHadronEt();
      double pfMuonEt                  = pfmet.muonEt();
      double pfHFHadronEtFraction      = pfmet.HFHadronEtFraction();
      double pfHFHadronEt              = pfmet.HFHadronEt();
      double pfHFEMEtFraction          = pfmet.HFEMEtFraction();
      double pfHFEMEt                  = pfmet.HFEMEt();
      
      mePhotonEtFraction        = map_of_MEs[DirName + "/PfPhotonEtFraction"];
      mePhotonEt                = map_of_MEs[DirName + "/PfPhotonEt"];
      meNeutralHadronEtFraction = map_of_MEs[DirName + "/PfNeutralHadronEtFraction"];
      meNeutralHadronEt         = map_of_MEs[DirName + "/PfNeutralHadronEt"];
      meElectronEt              = map_of_MEs[DirName + "/PfElectronEt"];
      meChargedHadronEtFraction = map_of_MEs[DirName + "/PfChargedHadronEtFraction"];
      meChargedHadronEt         = map_of_MEs[DirName + "/PfChargedHadronEt"];
      meMuonEt                  = map_of_MEs[DirName + "/PfMuonEt"];
      meHFHadronEtFraction      = map_of_MEs[DirName + "/PfHFHadronEtFraction"];
      meHFHadronEt              = map_of_MEs[DirName + "/PfHFHadronEt"];
      meHFEMEtFraction          = map_of_MEs[DirName + "/PfHFEMEtFraction"];
      meHFEMEt                  = map_of_MEs[DirName + "/PfHFEMEt"];
      
      if (mePhotonEtFraction        && mePhotonEtFraction       ->getRootObject()) mePhotonEtFraction       ->Fill(pfPhotonEtFraction);
      if (mePhotonEt                && mePhotonEt               ->getRootObject()) mePhotonEt               ->Fill(pfPhotonEt);
      if (meNeutralHadronEtFraction && meNeutralHadronEtFraction->getRootObject()) meNeutralHadronEtFraction->Fill(pfNeutralHadronEtFraction);
      if (meNeutralHadronEt         && meNeutralHadronEt        ->getRootObject()) meNeutralHadronEt        ->Fill(pfNeutralHadronEt);
      if (meElectronEt              && meElectronEt             ->getRootObject()) meElectronEt             ->Fill(pfElectronEt);
      if (meChargedHadronEtFraction && meChargedHadronEtFraction->getRootObject()) meChargedHadronEtFraction->Fill(pfChargedHadronEtFraction);
      if (meChargedHadronEt         && meChargedHadronEt        ->getRootObject()) meChargedHadronEt        ->Fill(pfChargedHadronEt);
      if (meMuonEt                  && meMuonEt                 ->getRootObject()) meMuonEt                 ->Fill(pfMuonEt);
      if (meHFHadronEtFraction      && meHFHadronEtFraction     ->getRootObject()) meHFHadronEtFraction     ->Fill(pfHFHadronEtFraction);
      if (meHFHadronEt              && meHFHadronEt             ->getRootObject()) meHFHadronEt             ->Fill(pfHFHadronEt);
      if (meHFEMEtFraction          && meHFEMEtFraction         ->getRootObject()) meHFEMEtFraction         ->Fill(pfHFEMEtFraction);
      if (meHFEMEt                  && meHFEMEt                 ->getRootObject()) meHFEMEt                 ->Fill(pfHFEMEt);

      //std::cout<<"fraction sum "<<pfPhotonEtFraction+pfNeutralHadronEtFraction+pfElectronEtFraction+pfChargedHadronEtFraction+pfMuonEtFraction+pfHFHadronEtFraction+pfHFEMEtFraction<<std::endl;


      //NPV profiles     
      
      mePhotonEtFraction_profile        = map_of_MEs[DirName + "/PfPhotonEtFraction_profile"];
      mePhotonEt_profile                = map_of_MEs[DirName + "/PfPhotonEt_profile"];
      meNeutralHadronEtFraction_profile = map_of_MEs[DirName + "/PfNeutralHadronEtFraction_profile"];
      meNeutralHadronEt_profile         = map_of_MEs[DirName + "/PfNeutralHadronEt_profile"];
      meChargedHadronEtFraction_profile = map_of_MEs[DirName + "/PfChargedHadronEtFraction_profile"];
      meChargedHadronEt_profile         = map_of_MEs[DirName + "/PfChargedHadronEt_profile"];
      meHFHadronEtFraction_profile      = map_of_MEs[DirName + "/PfHFHadronEtFraction_profile"];
      meHFHadronEt_profile              = map_of_MEs[DirName + "/PfHFHadronEt_profile"];
      meHFEMEtFraction_profile          = map_of_MEs[DirName + "/PfHFEMEtFraction_profile"];
      meHFEMEt_profile                  = map_of_MEs[DirName + "/PfHFEMEt_profile"];
      
      if (mePhotonEtFraction_profile        && mePhotonEtFraction_profile       ->getRootObject()) mePhotonEtFraction_profile       ->Fill(numPV_, pfPhotonEtFraction);
      if (mePhotonEt_profile                && mePhotonEt_profile               ->getRootObject()) mePhotonEt_profile               ->Fill(numPV_, pfPhotonEt);
      if (meNeutralHadronEtFraction_profile && meNeutralHadronEtFraction_profile->getRootObject()) meNeutralHadronEtFraction_profile->Fill(numPV_, pfNeutralHadronEtFraction);
      if (meNeutralHadronEt_profile         && meNeutralHadronEt_profile        ->getRootObject()) meNeutralHadronEt_profile        ->Fill(numPV_, pfNeutralHadronEt);
      if (meChargedHadronEtFraction_profile && meChargedHadronEtFraction_profile->getRootObject()) meChargedHadronEtFraction_profile->Fill(numPV_, pfChargedHadronEtFraction);
      if (meChargedHadronEt_profile         && meChargedHadronEt_profile        ->getRootObject()) meChargedHadronEt_profile        ->Fill(numPV_, pfChargedHadronEt);
      if (meHFHadronEtFraction_profile      && meHFHadronEtFraction_profile     ->getRootObject()) meHFHadronEtFraction_profile     ->Fill(numPV_, pfHFHadronEtFraction);
      if (meHFHadronEt_profile              && meHFHadronEt_profile             ->getRootObject()) meHFHadronEt_profile             ->Fill(numPV_, pfHFHadronEt);
      if (meHFEMEtFraction_profile          && meHFEMEtFraction_profile         ->getRootObject()) meHFEMEtFraction_profile         ->Fill(numPV_, pfHFEMEtFraction);
      if (meHFEMEt_profile                  && meHFEMEt_profile                 ->getRootObject()) meHFEMEt_profile                 ->Fill(numPV_, pfHFEMEt);
    }

    if(isMiniAODMet_){
      mePhotonEtFraction        = map_of_MEs[DirName + "/PfPhotonEtFraction"];
      meNeutralHadronEtFraction = map_of_MEs[DirName + "/PfNeutralHadronEtFraction"];
      meChargedHadronEtFraction = map_of_MEs[DirName + "/PfChargedHadronEtFraction"];
      meHFHadronEtFraction      = map_of_MEs[DirName + "/PfHFHadronEtFraction"];
      meHFEMEtFraction          = map_of_MEs[DirName + "/PfHFEMEtFraction"];
      
      if (mePhotonEtFraction        && mePhotonEtFraction       ->getRootObject()) mePhotonEtFraction       ->Fill(patmet.NeutralEMFraction());
      if (meNeutralHadronEtFraction && meNeutralHadronEtFraction->getRootObject()) meNeutralHadronEtFraction->Fill(patmet.NeutralHadEtFraction());
      if (meChargedHadronEtFraction && meChargedHadronEtFraction->getRootObject()) meChargedHadronEtFraction->Fill(patmet.ChargedHadEtFraction());
      if (meHFHadronEtFraction      && meHFHadronEtFraction     ->getRootObject()) meHFHadronEtFraction     ->Fill(patmet.Type6EtFraction());//HFHadrons
      if (meHFEMEtFraction          && meHFEMEtFraction         ->getRootObject()) meHFEMEtFraction         ->Fill(patmet.Type7EtFraction());

      //NPV profiles     
      mePhotonEtFraction_profile        = map_of_MEs[DirName + "/PfPhotonEtFraction_profile"];
      meNeutralHadronEtFraction_profile = map_of_MEs[DirName + "/PfNeutralHadronEtFraction_profile"];
      meChargedHadronEtFraction_profile = map_of_MEs[DirName + "/PfChargedHadronEtFraction_profile"];
      meHFHadronEtFraction_profile      = map_of_MEs[DirName + "/PfHFHadronEtFraction_profile"];
      meHFEMEtFraction_profile          = map_of_MEs[DirName + "/PfHFEMEtFraction_profile"];
      
      if (mePhotonEtFraction_profile        && mePhotonEtFraction_profile       ->getRootObject()) mePhotonEtFraction_profile       ->Fill(numPV_, patmet.NeutralEMFraction());
      if (meNeutralHadronEtFraction_profile && meNeutralHadronEtFraction_profile->getRootObject()) meNeutralHadronEtFraction_profile->Fill(numPV_, patmet.NeutralHadEtFraction());
      if (meChargedHadronEtFraction_profile && meChargedHadronEtFraction_profile->getRootObject()) meChargedHadronEtFraction_profile->Fill(numPV_, patmet.ChargedHadEtFraction());
      if (meHFHadronEtFraction_profile      && meHFHadronEtFraction_profile     ->getRootObject()) meHFHadronEtFraction_profile     ->Fill(numPV_, patmet.Type6EtFraction());
      if (meHFEMEtFraction_profile          && meHFEMEtFraction_profile         ->getRootObject()) meHFEMEtFraction_profile         ->Fill(numPV_, patmet.Type7EtFraction());
    }

    if (isCaloMet_){
      //if (bLumiSecPlot){//get from config level
      if (fill_met_high_level_histo){
	hMExLS = map_of_MEs[DirName+"/"+"MExLS"]; if (hMExLS  &&  hMExLS->getRootObject())   hMExLS->Fill(MEx,myLuminosityBlock);
	hMEyLS = map_of_MEs[DirName+"/"+"MEyLS"]; if (hMEyLS  &&  hMEyLS->getRootObject())   hMEyLS->Fill(MEy,myLuminosityBlock);
      }
    }
}
