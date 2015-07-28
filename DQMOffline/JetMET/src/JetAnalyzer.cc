/** \class JetAnalyzer
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

#include "DQMOffline/JetMET/interface/JetAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRCalo.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include <string>

#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;

// ***********************************************************
JetAnalyzer::JetAnalyzer(const edm::ParameterSet& pSet)
//: trackPropagator_(new jetAnalysis::TrackPropagatorToCalo)//,
    //sOverNCalculator_(new jetAnalysis::StripSignalOverNoiseCalculator)
{

  parameters_ = pSet.getParameter<edm::ParameterSet>("jetAnalysis");
  mInputCollection_           =    pSet.getParameter<edm::InputTag>       ("jetsrc");
  m_l1algoname_ = pSet.getParameter<std::string>("l1algoname");
  m_bitAlgTechTrig_=-1;

  jetType_ = pSet.getParameter<std::string>("JetType");
  m_l1algoname_ = pSet.getParameter<std::string>("l1algoname");

  fill_jet_high_level_histo=pSet.getParameter<bool>("filljetHighLevel"),
  
  isCaloJet_ = (std::string("calo")==jetType_);
  //isJPTJet_  = (std::string("jpt") ==jetType_);
  isPFJet_   = (std::string("pf") ==jetType_);
  isMiniAODJet_   = (std::string("miniaod") ==jetType_);
  jetCorrectorTag_=pSet.getParameter<edm::InputTag>("JetCorrections");
  if(!isMiniAODJet_){//in MiniAOD jet is already corrected
    jetCorrectorToken_ = consumes<reco::JetCorrector>(jetCorrectorTag_);
  }
  
  if (isCaloJet_){ 
    caloJetsToken_ = consumes<reco::CaloJetCollection>(mInputCollection_);
  // MET information
    caloMetToken_= consumes<reco::CaloMETCollection>(edm::InputTag(pSet.getParameter<edm::InputTag>("METCollectionLabel")));
  }
  //if (isJPTJet_)   jptJetsToken_ = consumes<reco::JPTJetCollection>(mInputCollection_);
  if (isPFJet_){    pfJetsToken_ = consumes<reco::PFJetCollection>(mInputCollection_);
    MuonsToken_ = consumes<reco::MuonCollection>(pSet.getParameter<edm::InputTag>       ("muonsrc"));
    pfMetToken_= consumes<reco::PFMETCollection>(edm::InputTag(pSet.getParameter<edm::InputTag>("METCollectionLabel")));
  }
  if (isMiniAODJet_){
    patJetsToken_ = consumes<pat::JetCollection>(mInputCollection_);
    patMetToken_= consumes<pat::METCollection>(edm::InputTag(pSet.getParameter<edm::InputTag>("METCollectionLabel")));
  }
  cutBasedPUDiscriminantToken_ = consumes< edm::ValueMap<float> >(pSet.getParameter<edm::InputTag>("InputCutPUIDDiscriminant"));
  cutBasedPUIDToken_ = consumes< edm::ValueMap<int> >(pSet.getParameter<edm::InputTag>("InputCutPUIDValue"));
  mvaPUIDToken_ = consumes< edm::ValueMap<int> >(pSet.getParameter<edm::InputTag>("InputMVAPUIDValue"));
  mvaFullPUDiscriminantToken_ = consumes< edm::ValueMap<float> >(pSet.getParameter<edm::InputTag>("InputMVAPUIDDiscriminant"));

  qgMultiplicityToken_= consumes< edm::ValueMap<int> >(pSet.getParameter<edm::InputTag>("InputQGMultiplicity"));
  qgLikelihoodToken_= consumes< edm::ValueMap<float> >(pSet.getParameter<edm::InputTag>("InputQGLikelihood"));
  qgptDToken_= consumes< edm::ValueMap<float> >(pSet.getParameter<edm::InputTag>("InputQGPtDToken"));
  qgaxis2Token_= consumes< edm::ValueMap<float> >(pSet.getParameter<edm::InputTag>("InputQGAxis2"));

  fill_CHS_histos= pSet.getParameter<bool>("fillCHShistos");
  
  JetIDQuality_  = pSet.getParameter<string>("JetIDQuality");
  JetIDVersion_  = pSet.getParameter<string>("JetIDVersion");

  // JetID definitions for Calo and JPT Jets
  if(isCaloJet_){
    inputJetIDValueMap      = pSet.getParameter<edm::InputTag>("InputJetIDValueMap");
    jetID_ValueMapToken_= consumes< edm::ValueMap<reco::JetID> >(inputJetIDValueMap);
    if(JetIDVersion_== "PURE09"){
      jetidversion = JetIDSelectionFunctor::PURE09;
    }else if (JetIDVersion_== "DQM09"){
      jetidversion = JetIDSelectionFunctor::DQM09;
    }else if (JetIDVersion_=="CRAFT08"){
      jetidversion = JetIDSelectionFunctor::CRAFT08;
    }else{
      if (verbose_) std::cout<<"no Valid JetID version given"<<std::endl;
    }
    if(JetIDQuality_== "MINIMAL"){
      jetidquality = JetIDSelectionFunctor::MINIMAL;
    }else if (JetIDQuality_== "LOOSE_AOD"){
      jetidquality = JetIDSelectionFunctor::LOOSE_AOD;
    }else if (JetIDQuality_=="LOOSE"){
      jetidquality = JetIDSelectionFunctor::LOOSE;
    }else if (JetIDQuality_=="TIGHT"){
      jetidquality = JetIDSelectionFunctor::TIGHT;
    }else{
      if (verbose_) std::cout<<"no Valid JetID quality given"<<std::endl;
    }
    jetIDFunctor=JetIDSelectionFunctor( jetidversion, jetidquality);

  }

  //Jet ID definitions for PFJets
  if(isPFJet_ || isMiniAODJet_){
    if(JetIDVersion_== "FIRSTDATA"){
      pfjetidversion = PFJetIDSelectionFunctor::FIRSTDATA;
    }else{
      if (verbose_) std::cout<<"no valid PF JetID version given"<<std::endl;
    }
    if (JetIDQuality_=="LOOSE"){
      pfjetidquality = PFJetIDSelectionFunctor::LOOSE;
    }else if (JetIDQuality_=="TIGHT"){
      pfjetidquality = PFJetIDSelectionFunctor::TIGHT;
    }else{
     if (verbose_)  std::cout<<"no Valid PFJetID quality given"<<std::endl;
    }
    pfjetIDFunctor=PFJetIDSelectionFunctor( pfjetidversion, pfjetidquality);
  }
  //check later if some of those are also needed for PFJets
  leadJetFlag_ = 0;
  jetLoPass_   = 0;
  jetHiPass_   = 0;
  ptThreshold_ = 20.;
  ptThresholdUnc_ = 20.;
  asymmetryThirdJetCut_ = 5.;
  balanceThirdJetCut_   = 0.2; 

  theTriggerResultsLabel_        = pSet.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_          = consumes<edm::TriggerResults>(edm::InputTag(theTriggerResultsLabel_));
  //
  runcosmics_          = pSet.getUntrackedParameter<bool>("runcosmics", false);
  jetCleaningFlag_            = pSet.getUntrackedParameter<bool>("JetCleaningFlag", true);

  if(runcosmics_){
    jetCleaningFlag_ =false;
  }

 
  // ==========================================================
  //DCS information
  // ==========================================================
  edm::ConsumesCollector iC  = consumesCollector();
  DCSFilterForJetMonitoring_  = new JetMETDQMDCSFilter(pSet.getParameter<ParameterSet>("DCSFilterForJetMonitoring"), iC);
  DCSFilterForDCSMonitoring_  = new JetMETDQMDCSFilter("ecal:hbhe:hf:ho:pixel:sistrip:es:muon", iC);
  
  //Trigger selectoin
  edm::ParameterSet highptjetparms = pSet.getParameter<edm::ParameterSet>("highPtJetTrigger");
  edm::ParameterSet lowptjetparms  = pSet.getParameter<edm::ParameterSet>("lowPtJetTrigger" );
  
  highPtJetEventFlag_ = new GenericTriggerEventFlag( highptjetparms, consumesCollector(), *this );
  lowPtJetEventFlag_  = new GenericTriggerEventFlag( lowptjetparms , consumesCollector(), *this );
  
  highPtJetExpr_ = highptjetparms.getParameter<std::vector<std::string> >("hltPaths");
  lowPtJetExpr_  = lowptjetparms .getParameter<std::vector<std::string> >("hltPaths");
  
  processname_ = pSet.getParameter<std::string>("processname");
  
  //jet cleanup parameters
  cleaningParameters_ = pSet.getParameter<ParameterSet>("CleaningParameters");

  bypassAllPVChecks_= cleaningParameters_.getParameter<bool>("bypassAllPVChecks");
  vertexLabel_      = cleaningParameters_.getParameter<edm::InputTag>("vertexCollection");
  vertexToken_      = consumes<std::vector<reco::Vertex> >(edm::InputTag(vertexLabel_));

  gtLabel_          = cleaningParameters_.getParameter<edm::InputTag>("gtLabel");
  gtToken_          = consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(gtLabel_));
  
  std::string inputCollectionLabel(mInputCollection_.label());
  verbose_= parameters_.getParameter<int>("verbose");
   // monitoring of eta parameter
  etaBin_ = parameters_.getParameter<int>("etaBin");
  etaMin_ = parameters_.getParameter<double>("etaMin");
  etaMax_ = parameters_.getParameter<double>("etaMax");
  // monitoring of phi paramater
  phiBin_ = parameters_.getParameter<int>("phiBin");
  phiMin_ = parameters_.getParameter<double>("phiMin");
  phiMax_ = parameters_.getParameter<double>("phiMax");
  // monitoring of the transverse momentum
  ptBin_ = parameters_.getParameter<int>("ptBin");
  ptMin_ = parameters_.getParameter<double>("ptMin");
  ptMax_ = parameters_.getParameter<double>("ptMax");
  // 
  eBin_ = parameters_.getParameter<int>("eBin");
  eMin_ = parameters_.getParameter<double>("eMin");
  eMax_ = parameters_.getParameter<double>("eMax");
  // 
  pBin_ = parameters_.getParameter<int>("pBin");
  pMin_ = parameters_.getParameter<double>("pMin");
  pMax_ = parameters_.getParameter<double>("pMax");
  // 
  nbinsPV_ = parameters_.getParameter<int>("pVBin");
  nPVlow_   = parameters_.getParameter<double>("pVMin");
  nPVhigh_  = parameters_.getParameter<double>("pVMax");
  //
  ptThreshold_ = parameters_.getParameter<double>("ptThreshold");
  ptThresholdUnc_=parameters_.getParameter<double>("ptThresholdUnc");
  asymmetryThirdJetCut_ = parameters_.getParameter<double>("asymmetryThirdJetCut");
  balanceThirdJetCut_   = parameters_.getParameter<double>("balanceThirdJetCut");
}  
  

// ***********************************************************
JetAnalyzer::~JetAnalyzer() {
  
  delete highPtJetEventFlag_;
  delete lowPtJetEventFlag_;

  delete DCSFilterForDCSMonitoring_;
  delete DCSFilterForJetMonitoring_;
  LogTrace("JetAnalyzer")<<"[JetAnalyzer] Saving the histos";
}

// ***********************************************************
void JetAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				     edm::EventSetup const & ) {
  
  //  dbe_ = edm::Service<DQMStore>().operator->();
  if(jetCleaningFlag_){
    ibooker.setCurrentFolder("JetMET/Jet/Cleaned"+mInputCollection_.label());
    DirName = "JetMET/Jet/Cleaned"+mInputCollection_.label();
  }else{
    ibooker.setCurrentFolder("JetMET/Jet/Uncleaned"+mInputCollection_.label());
    DirName = "JetMET/Jet/Uncleaned"+mInputCollection_.label();
  }

  jetME = ibooker.book1D("jetReco", "jetReco", 4, 1, 5);
  jetME->setBinLabel(1,"CaloJets",1);
  jetME->setBinLabel(2,"PFJets",1);
  jetME->setBinLabel(3,"JPTJets",1);
  jetME->setBinLabel(4,"MiniAODJets",1);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"jetReco",jetME));


  mPt           = ibooker.book1D("Pt",           "pt",                 ptBin_,  ptMin_,  ptMax_);
  mEta          = ibooker.book1D("Eta",          "eta",               etaBin_, etaMin_, etaMax_);
  mPhi          = ibooker.book1D("Phi",          "phi",               phiBin_, phiMin_, phiMax_);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt" ,mPt));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta",mEta));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi",mPhi));

  //if(!isJPTJet_){
  mConstituents = ibooker.book1D("Constituents", "# of constituents",     50,      0,    100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents",mConstituents));
  //}
  mJetEnergyCorr= ibooker.book1D("JetEnergyCorr", "jet energy correction factor", 50, 0.0,3.0);
  mJetEnergyCorrVSEta= ibooker.bookProfile("JetEnergyCorrVSEta", "jet energy correction factor VS eta", etaBin_, etaMin_,etaMax_, 0.0,3.0);
  mJetEnergyCorrVSPt= ibooker.bookProfile("JetEnergyCorrVSPt", "jet energy correction factor VS pt", ptBin_, ptMin_,ptMax_, 0.0,3.0);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorr" ,mJetEnergyCorr));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorrVSEta" ,mJetEnergyCorrVSEta));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorrVSPt" ,mJetEnergyCorrVSPt));
  
  mPt_uncor           = ibooker.book1D("Pt_uncor",           "pt for uncorrected jets",                 ptBin_,  ptThresholdUnc_,  ptMax_);
  mEta_uncor          = ibooker.book1D("Eta_uncor",          "eta for uncorrected jets",               etaBin_, etaMin_, etaMax_);
  mPhi_uncor          = ibooker.book1D("Phi_uncor",          "phi for uncorrected jets",               phiBin_, phiMin_, phiMax_);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_uncor" ,mPt_uncor));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_uncor",mEta_uncor));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_uncor",mPhi_uncor));
  //if(!isJPTJet_){
  mConstituents_uncor = ibooker.book1D("Constituents_uncor", "# of constituents for uncorrected jets",     50,      0,    100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_uncor",mConstituents_uncor));
  //}

  mDPhi                   = ibooker.book1D("DPhi", "dPhi btw the two leading jets", 100, 0., acos(-1.));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DPhi",mDPhi));

  // Book NPV profiles
  //----------------------------------------------------------------------------
  mPt_profile           = ibooker.bookProfile("Pt_profile",           "pt",                nbinsPV_, nPVlow_, nPVhigh_,   ptBin_,  ptMin_,  ptMax_);
  mEta_profile          = ibooker.bookProfile("Eta_profile",          "eta",               nbinsPV_, nPVlow_, nPVhigh_,  etaBin_, etaMin_, etaMax_);
  mPhi_profile          = ibooker.bookProfile("Phi_profile",          "phi",               nbinsPV_, nPVlow_, nPVhigh_,  phiBin_, phiMin_, phiMax_);
  //if(!isJPTJet_){
  mConstituents_profile = ibooker.bookProfile("Constituents_profile", "# of constituents", nbinsPV_, nPVlow_, nPVhigh_,      50,      0,    100);
  //}
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_profile" ,mPt_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_profile",mEta_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_profile",mPhi_profile));



  if(!runcosmics_){//JIDPassFrac_ defines a collection of cleaned jets, for which we will want to fill the cleaning passing fraction
    mLooseJIDPassFractionVSeta      = ibooker.bookProfile("JetIDPassFractionVSeta","JetIDPassFractionVSeta",etaBin_, etaMin_, etaMax_,0.,1.2);
    mLooseJIDPassFractionVSpt       = ibooker.bookProfile("JetIDPassFractionVSpt","JetIDPassFractionVSpt",ptBin_, ptMin_, ptMax_,0.,1.2);
    mLooseJIDPassFractionVSptNoHF   = ibooker.bookProfile("JetIDPassFractionVSptNoHF","JetIDPassFractionVSptNoHF",ptBin_, ptMin_, ptMax_,0.,1.2);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetIDPassFractionVSeta"   ,mLooseJIDPassFractionVSeta));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetIDPassFractionVSpt"    ,mLooseJIDPassFractionVSpt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetIDPassFractionVSptNoHF",mLooseJIDPassFractionVSptNoHF));
  }

  mNJets_profile = ibooker.bookProfile("NJets_profile", "number of jets", nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
 

  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  mPt_profile          ->setAxisTitle("nvtx",1);
  mEta_profile         ->setAxisTitle("nvtx",1);
  mPhi_profile         ->setAxisTitle("nvtx",1);
  //if(!isJPTJet_){
  mConstituents_profile->setAxisTitle("nvtx",1);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_profile",mConstituents_profile));
  //}
  mNJets_profile->setAxisTitle("nvtx",1);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_profile" ,mPt_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_profile",mEta_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_profile",mPhi_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NJets_profile" ,mNJets_profile));


  mPhiVSEta                     = ibooker.book2D("PhiVSEta", "PhiVSEta", 50, etaMin_, etaMax_, 24, phiMin_, phiMax_);
  mPhiVSEta->getTH2F()->SetOption("colz");
  mPhiVSEta->setAxisTitle("#eta",1);
  mPhiVSEta->setAxisTitle("#phi",2);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhiVSEta" ,mPhiVSEta));

  mPt_1                    = ibooker.book1D("Pt_1", "Pt spectrum of jets - range 1", 20, 0, 100);   
  mPt_2                    = ibooker.book1D("Pt_2", "Pt spectrum of jets - range 2", 60, 0, 300);   
  mPt_3                    = ibooker.book1D("Pt_3", "Pt spectrum of jets - range 3", 100, 0, 5000);
  // Low and high pt trigger paths
  mPt_Lo                  = ibooker.book1D("Pt_Lo", "Pt (Pass Low Pt Jet Trigger)", 20, 0, 100);   
  //mEta_Lo                 = ibooker.book1D("Eta_Lo", "Eta (Pass Low Pt Jet Trigger)", etaBin_, etaMin_, etaMax_);
  mPhi_Lo                 = ibooker.book1D("Phi_Lo", "Phi (Pass Low Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  
  mPt_Hi                  = ibooker.book1D("Pt_Hi", "Pt (Pass Hi Pt Jet Trigger)", 60, 0, 300);   
  mEta_Hi                 = ibooker.book1D("Eta_Hi", "Eta (Pass Hi Pt Jet Trigger)", etaBin_, etaMin_, etaMax_);
  mPhi_Hi                 = ibooker.book1D("Phi_Hi", "Phi (Pass Hi Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  mNJets                   = ibooker.book1D("NJets", "number of jets", 100, 0, 100);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_1" ,mPt_1));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_2" ,mPt_2));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_3" ,mPt_3));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_Lo" ,mPt_Lo));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_Lo" ,mPhi_Lo));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_Hi" ,mPt_Hi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_Hi" ,mEta_Hi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_Hi" ,mPhi_Hi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NJets" ,mNJets));
  
  //mPt_Barrel_Lo            = ibooker.book1D("Pt_Barrel_Lo", "Pt Barrel (Pass Low Pt Jet Trigger)", 20, 0, 100);   
  //mPhi_Barrel_Lo           = ibooker.book1D("Phi_Barrel_Lo", "Phi Barrel (Pass Low Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  //if(!isJPTJet_){
  mConstituents_Barrel     = ibooker.book1D("Constituents_Barrel", "Constituents Barrel", 50, 0, 100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_Barrel",mConstituents_Barrel));
  //}
  
  //mPt_EndCap_Lo            = ibooker.book1D("Pt_EndCap_Lo", "Pt EndCap (Pass Low Pt Jet Trigger)", 20, 0, 100);   
  //mPhi_EndCap_Lo           = ibooker.book1D("Phi_EndCap_Lo", "Phi EndCap (Pass Low Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  //if(!isJPTJet_){
  mConstituents_EndCap     = ibooker.book1D("Constituents_EndCap", "Constituents EndCap", 50, 0, 100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_EndCap",mConstituents_EndCap));
  //}
 

  //mPt_Forward_Lo           = ibooker.book1D("Pt_Forward_Lo", "Pt Forward (Pass Low Pt Jet Trigger)", 20, 0, 100);  
  //mPhi_Forward_Lo          = ibooker.book1D("Phi_Forward_Lo", "Phi Forward (Pass Low Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  //if(!isJPTJet_){
  mConstituents_Forward    = ibooker.book1D("Constituents_Forward", "Constituents Forward", 50, 0, 100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_Forward",mConstituents_Forward));
  //}



  mPt_Barrel_Hi            = ibooker.book1D("Pt_Barrel_Hi", "Pt Barrel (Pass Hi Pt Jet Trigger)", 60, 0, 300);   
  mPhi_Barrel_Hi           = ibooker.book1D("Phi_Barrel_Hi", "Phi Barrel (Pass Hi Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  
  mPt_EndCap_Hi            = ibooker.book1D("Pt_EndCap_Hi", "Pt EndCap (Pass Hi Pt Jet Trigger)", 60, 0, 300);  
  mPhi_EndCap_Hi           = ibooker.book1D("Phi_EndCap_Hi", "Phi EndCap (Pass Hi Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  
  mPt_Forward_Hi           = ibooker.book1D("Pt_Forward_Hi", "Pt Forward (Pass Hi Pt Jet Trigger)", 60, 0, 300);  
  mPhi_Forward_Hi          = ibooker.book1D("Phi_Forward_Hi", "Phi Forward (Pass Hi Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_Barrel_Hi" ,mPt_Barrel_Hi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_Barrel_Hi",mPhi_Barrel_Hi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_EndCap_Hi" ,mPt_EndCap_Hi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_EndCap_Hi",mPhi_EndCap_Hi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_Forward_Hi" ,mPt_Forward_Hi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_Forward_Hi",mPhi_Forward_Hi));
  
  mPhi_Barrel              = ibooker.book1D("Phi_Barrel", "Phi_Barrel", phiBin_, phiMin_, phiMax_);
  mPt_Barrel               = ibooker.book1D("Pt_Barrel", "Pt_Barrel", ptBin_, ptMin_, ptMax_);
  
  mPhi_EndCap              = ibooker.book1D("Phi_EndCap", "Phi_EndCap", phiBin_, phiMin_, phiMax_);
  mPt_EndCap               = ibooker.book1D("Pt_EndCap", "Pt_EndCap", ptBin_, ptMin_, ptMax_);
  
  mPhi_Forward             = ibooker.book1D("Phi_Forward", "Phi_Forward", phiBin_, phiMin_, phiMax_);
  mPt_Forward              = ibooker.book1D("Pt_Forward", "Pt_Forward", ptBin_, ptMin_, ptMax_);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_Barrel" ,mPt_Barrel));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_Barrel",mPhi_Barrel));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_EndCap" ,mPt_EndCap));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_EndCap",mPhi_EndCap));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_Forward" ,mPt_Forward));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_Forward",mPhi_Forward));
  
  // Leading Jet Parameters
  mEtaFirst                = ibooker.book1D("EtaFirst", "EtaFirst", 50, -5, 5);
  mPhiFirst                = ibooker.book1D("PhiFirst", "PhiFirst", 70, phiMin_, phiMax_);
  mPtFirst                 = ibooker.book1D("PtFirst", "PtFirst", ptBin_, ptMin_, ptMax_);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EtaFirst" ,mEtaFirst));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtFirst"  ,mPtFirst));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhiFirst" ,mPhiFirst));
  
  //--- Calo jet selection only
  if(isCaloJet_) {

    // CaloJet specific
    mHadEnergyInHO          = ibooker.book1D("HadEnergyInHO", "HadEnergyInHO", 50, 0, 20);
    mHadEnergyInHB          = ibooker.book1D("HadEnergy5InHB", "HadEnergyInHB", 50, 0, 100);
    mHadEnergyInHF          = ibooker.book1D("HadEnergyInHF", "HadEnergyInHF", 50, 0, 100);
    mHadEnergyInHE          = ibooker.book1D("HadEnergyInHE", "HadEnergyInHE", 50, 0, 200);
    mEmEnergyInEB           = ibooker.book1D("EmEnergyInEB", "EmEnergyInEB", 50, 0, 100);
    mEmEnergyInEE           = ibooker.book1D("EmEnergyInEE", "EmEnergyInEE", 50, 0, 100);
    mEmEnergyInHF           = ibooker.book1D("EmEnergyInHF", "EmEnergyInHF", 60, -20, 200);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HadEnergyInHO"  ,mHadEnergyInHO));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HadEnergyInHB"  ,mHadEnergyInHB));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HadEnergyInHF"  ,mHadEnergyInHF));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HadEnergyInHE"  ,mHadEnergyInHE));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EmEnergyInEB" ,mEmEnergyInEB));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EmEnergyInEE" ,mEmEnergyInEE));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EmEnergyInHF" ,mEmEnergyInHF));
    
    //JetID variables
    mresEMF                 = ibooker.book1D("resEMF", "resEMF", 50, 0., 1.);
    mN90Hits                = ibooker.book1D("N90Hits", "N90Hits", 50, 0., 50);
    mfHPD                   = ibooker.book1D("fHPD", "fHPD", 50, 0., 1.);
    mfRBX                   = ibooker.book1D("fRBX", "fRBX", 50, 0., 1.);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"resEMF" ,mresEMF));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"N90Hits" ,mN90Hits));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"fHPD" ,mfHPD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"fRBX" ,mfRBX));

    mHFrac        = ibooker.book1D("HFrac",        "HFrac",                70,   -0.2,    1.2);
    mEFrac        = ibooker.book1D("EFrac",        "EFrac",           52,   -0.02,    1.02);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac" ,mHFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac" ,mEFrac));
    mHFrac_profile        = ibooker.bookProfile("HFrac_profile",        "HFrac",             nbinsPV_, nPVlow_, nPVhigh_,     70,   -0.2,    1.2);
    mEFrac_profile        = ibooker.bookProfile("EFrac_profile",        "EFrac",             nbinsPV_, nPVlow_, nPVhigh_,     52,   -0.02,    1.02);
    mHFrac_profile       ->setAxisTitle("nvtx",1);
    mEFrac_profile       ->setAxisTitle("nvtx",1);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_profile",mHFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_profile",mEFrac_profile));
    mHFrac_Barrel            = ibooker.book1D("HFrac_Barrel", "HFrac Barrel", 50, 0, 1);
    mEFrac_Barrel            = ibooker.book1D("EFrac_Barrel", "EFrac Barrel", 52, -0.02, 1.02);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_Barrel" ,mHFrac_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_Barrel" ,mEFrac_Barrel));
    mHFrac_EndCap            = ibooker.book1D("HFrac_EndCap", "HFrac EndCap", 50, 0, 1);
    mEFrac_EndCap            = ibooker.book1D("EFrac_EndCap", "EFrac EndCap", 52, -0.02, 1.02);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_EndCap" ,mHFrac_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_EndCap" ,mEFrac_EndCap));
    mHFrac_Forward           = ibooker.book1D("HFrac_Forward", "HFrac Forward", 70, -0.2, 1.2);
    mEFrac_Forward           = ibooker.book1D("EFrac_Forward", "EFrac Forward", 52, -0.02, 1.02);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_Forward" ,mHFrac_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_Forward" ,mEFrac_Forward));
  }

  if(isPFJet_) {
    /* remove quark gluon plots for default jet selection, but select physics signatures which monitor gluon and quark
    if(jetCleaningFlag_){ 
      //gluon quark separation axis  
      if(fill_CHS_histos){
	mAxis2_lowPt_Barrel = ibooker.book1D("qg_Axis2_lowPt_Barrel","qg Axis2 #sigma_{2} lowPt Barrel",50,0.,0.20);
	mpTD_lowPt_Barrel= ibooker.book1D("qg_pTD_lowPt_Barrel","qg fragmentation function p_{T}^{D} lowPt Barrel",50,0.15,1.05);
	mMultiplicityQG_lowPt_Barrel= ibooker.book1D("qg_multiplicity_lowPt_Barrel","qg multiplicity lowPt Barrel",50,0,50);
	mqgLikelihood_lowPt_Barrel= ibooker.book1D("qg_Likelihood_lowPt_Barrel","qg likelihood lowPt Barrel",50,-1.1,1.1);
	mAxis2_lowPt_EndCap = ibooker.book1D("qg_Axis2_lowPt_EndCap","qg Axis2 #sigma_{2} lowPt EndCap",50,0.,0.20);
	mpTD_lowPt_EndCap= ibooker.book1D("qg_pTD_lowPt_EndCap","qg fragmentation function p_{T}^{D} lowPt EndCap",50,0.15,1.05);
	mMultiplicityQG_lowPt_EndCap= ibooker.book1D("qg_multiplicity_lowPt_EndCap","qg multiplicity lowPt EndCap",50,0,100);
	mqgLikelihood_lowPt_EndCap= ibooker.book1D("qg_Likelihood_lowPt_EndCap","qg likelihood lowPt EndCap",50,-1.1,1.1);
	mAxis2_lowPt_Forward = ibooker.book1D("qg_Axis2_lowPt_Forward","qg Axis2 #sigma_{2} lowPt Forward",50,0.,0.20);
	mpTD_lowPt_Forward= ibooker.book1D("qg_pTD_lowPt_Forward","qg fragmentation function p_{T}^{D} lowPt Forward",50,0.15,1.05);
	mMultiplicityQG_lowPt_Forward= ibooker.book1D("qg_multiplicity_lowPt_Forward","qg multiplicity lowPt Forward",50,0,100);
	mqgLikelihood_lowPt_Forward= ibooker.book1D("qg_Likelihood_lowPt_Forward","qg likelihood lowPt Forward",50,-1.1,1.1);
	
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_lowPt_Barrel" ,mAxis2_lowPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_lowPt_Barrel" ,mpTD_lowPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_lowPt_Barrel" ,mMultiplicityQG_lowPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_lowPt_Barrel" ,mqgLikelihood_lowPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_lowPt_EndCap" ,mAxis2_lowPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_lowPt_EndCap" ,mpTD_lowPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_lowPt_EndCap" ,mMultiplicityQG_lowPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_lowPt_EndCap" ,mqgLikelihood_lowPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_lowPt_Forward" ,mAxis2_lowPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_lowPt_Forward" ,mpTD_lowPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_lowPt_Forward" ,mMultiplicityQG_lowPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_lowPt_Forward" ,mqgLikelihood_lowPt_Forward));

	mAxis2_mediumPt_Barrel = ibooker.book1D("qg_Axis2_mediumPt_Barrel","qg Axis2 #sigma_{2} mediumPt Barrel",50,0.,0.20);
	mpTD_mediumPt_Barrel= ibooker.book1D("qg_pTD_mediumPt_Barrel","qg fragmentation function p_{T}^{D} mediumPt Barrel",50,0.15,1.05);
	mMultiplicityQG_mediumPt_Barrel= ibooker.book1D("qg_multiplicity_mediumPt_Barrel","qg multiplicity mediumPt Barrel",50,0,100);
	mqgLikelihood_mediumPt_Barrel= ibooker.book1D("qg_Likelihood_mediumPt_Barrel","qg likelihood mediumPt Barrel",50,-1.1,1.1);
	mAxis2_mediumPt_EndCap = ibooker.book1D("qg_Axis2_mediumPt_EndCap","qg Axis2 #sigma_{2} mediumPt EndCap",50,0.,0.20);
	mpTD_mediumPt_EndCap= ibooker.book1D("qg_pTD_mediumPt_EndCap","qg fragmentation function p_{T}^{D} mediumPt EndCap",50,0.15,1.05);
	mMultiplicityQG_mediumPt_EndCap= ibooker.book1D("qg_multiplicity_mediumPt_EndCap","qg multiplicity mediumPt EndCap",50,0,100);
	mqgLikelihood_mediumPt_EndCap= ibooker.book1D("qg_Likelihood_mediumPt_EndCap","qg likelihood mediumPt EndCap",50,-1.1,1.1);
	mAxis2_mediumPt_Forward = ibooker.book1D("qg_Axis2_mediumPt_Forward","qg Axis2 #sigma_{2} mediumPt Forward",50,0.,0.20);
	mpTD_mediumPt_Forward= ibooker.book1D("qg_pTD_mediumPt_Forward","qg fragmentation function p_{T}^{D} mediumPt Forward",50,0.15,1.05);
	mMultiplicityQG_mediumPt_Forward= ibooker.book1D("qg_multiplicity_mediumPt_Forward","qg multiplicity mediumPt Forward",50,0,100);
	mqgLikelihood_mediumPt_Forward= ibooker.book1D("qg_Likelihood_mediumPt_Forward","qg likelihood mediumPt Forward",50,-1.1,1.1);
	
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_mediumPt_Barrel" ,mAxis2_mediumPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_mediumPt_Barrel" ,mpTD_mediumPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_mediumPt_Barrel" ,mMultiplicityQG_mediumPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_mediumPt_Barrel" ,mqgLikelihood_mediumPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_mediumPt_EndCap" ,mAxis2_mediumPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_mediumPt_EndCap" ,mpTD_mediumPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_mediumPt_EndCap" ,mMultiplicityQG_mediumPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_mediumPt_EndCap" ,mqgLikelihood_mediumPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_mediumPt_Forward" ,mAxis2_mediumPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_mediumPt_Forward" ,mpTD_mediumPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_mediumPt_Forward" ,mMultiplicityQG_mediumPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_mediumPt_Forward" ,mqgLikelihood_mediumPt_Forward));

	mAxis2_highPt_Barrel = ibooker.book1D("qg_Axis2_highPt_Barrel","qg Axis2 #sigma_{2} highPt Barrel",50,0.,0.20);
	mpTD_highPt_Barrel= ibooker.book1D("qg_pTD_highPt_Barrel","qg fragmentation function p_{T}^{D} highPt Barrel",50,0.15,1.05);
	mMultiplicityQG_highPt_Barrel= ibooker.book1D("qg_multiplicity_highPt_Barrel","qg multiplicity highPt Barrel",50,0,100);
	mqgLikelihood_highPt_Barrel= ibooker.book1D("qg_Likelihood_highPt_Barrel","qg likelihood highPt Barrel",50,-1.1,1.1);
	mAxis2_highPt_EndCap = ibooker.book1D("qg_Axis2_highPt_EndCap","qg Axis2 #sigma_{2} highPt EndCap",50,0.,0.20);
	mpTD_highPt_EndCap= ibooker.book1D("qg_pTD_highPt_EndCap","qg fragmentation function p_{T}^{D} highPt EndCap",50,0.15,1.05);
	mMultiplicityQG_highPt_EndCap= ibooker.book1D("qg_multiplicity_highPt_EndCap","qg multiplicity highPt EndCap",50,0,100);
	mqgLikelihood_highPt_EndCap= ibooker.book1D("qg_Likelihood_highPt_EndCap","qg likelihood highPt EndCap",50,-1.1,1.1);
	mAxis2_highPt_Forward = ibooker.book1D("qg_Axis2_highPt_Forward","qg Axis2 #sigma_{2} highPt Forward",50,0.,0.20);
	mpTD_highPt_Forward= ibooker.book1D("qg_pTD_highPt_Forward","qg fragmentation function p_{T}^{D} highPt Forward",50,0.15,1.05);
	mMultiplicityQG_highPt_Forward= ibooker.book1D("qg_multiplicity_highPt_Forward","qg multiplicity highPt Forward",50,0,100);
	mqgLikelihood_highPt_Forward= ibooker.book1D("qg_Likelihood_highPt_Forward","qg likelihood highPt Forward",50,-1.1,1.1);
	
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_highPt_Barrel" ,mAxis2_highPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_highPt_Barrel" ,mpTD_highPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_highPt_Barrel" ,mMultiplicityQG_highPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_highPt_Barrel" ,mqgLikelihood_highPt_Barrel));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_highPt_EndCap" ,mAxis2_highPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_highPt_EndCap" ,mpTD_highPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_highPt_EndCap" ,mMultiplicityQG_highPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_highPt_EndCap" ,mqgLikelihood_highPt_EndCap));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_highPt_Forward" ,mAxis2_highPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_highPt_Forward" ,mpTD_highPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_highPt_Forward" ,mMultiplicityQG_highPt_Forward));
	map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_highPt_Forward" ,mqgLikelihood_highPt_Forward));
      }
    }*/
    //PFJet specific histograms
    mCHFracVSeta_lowPt= ibooker.bookProfile("CHFracVSeta_lowPt","CHFracVSeta_lowPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mNHFracVSeta_lowPt= ibooker.bookProfile("NHFacVSeta_lowPt","NHFracVSeta_lowPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mPhFracVSeta_lowPt= ibooker.bookProfile("PhFracVSeta_lowPt","PhFracVSeta_lowPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mCHFracVSeta_mediumPt= ibooker.bookProfile("CHFracVSeta_mediumPt","CHFracVSeta_mediumPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mNHFracVSeta_mediumPt= ibooker.bookProfile("NHFracVSeta_mediumPt","NHFracVSeta_mediumPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mPhFracVSeta_mediumPt= ibooker.bookProfile("PhFracVSeta_mediumPt","PhFracVSeta_mediumPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mCHFracVSeta_highPt= ibooker.bookProfile("CHFracVSeta_highPt","CHFracVSeta_highPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mNHFracVSeta_highPt= ibooker.bookProfile("NHFracVSeta_highPt","NHFracVSeta_highPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mPhFracVSeta_highPt= ibooker.bookProfile("PhFracVSeta_highPt","PhFracVSeta_highPt",etaBin_, etaMin_, etaMax_,0.,1.2);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSeta_lowPt" ,mCHFracVSeta_lowPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSeta_lowPt" ,mNHFracVSeta_lowPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSeta_lowPt" ,mPhFracVSeta_lowPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSeta_mediumPt" ,mCHFracVSeta_mediumPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSeta_mediumPt" ,mNHFracVSeta_mediumPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSeta_mediumPt" ,mPhFracVSeta_mediumPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSeta_highPt" ,mCHFracVSeta_highPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSeta_highPt" ,mNHFracVSeta_highPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSeta_highPt" ,mPhFracVSeta_highPt));

    mLooseMVAPUJIDPassFractionVSeta  = ibooker.bookProfile("LooseMVAPUIDPassFractionVSeta","LooseMVAPUIDPassFractionVSeta",etaBin_, etaMin_, etaMax_,0.,1.2);
    mLooseMVAPUJIDPassFractionVSpt   = ibooker.bookProfile("LooseMVAPUIDPassFractionVSpt","LooseMVAPUIDPassFractionVSpt",ptBin_, ptMin_, ptMax_,0.,1.2);
    mMediumMVAPUJIDPassFractionVSeta = ibooker.bookProfile("MediumMVAPUIDPassFractionVSeta","MediumMVAPUIDPassFractionVSeta",etaBin_, etaMin_, etaMax_,0.,1.2);
    mMediumMVAPUJIDPassFractionVSpt  = ibooker.bookProfile("MediumMVAPUIDPassFractionVSpt","MediumMVAPUIDPassFractionVSpt",ptBin_, ptMin_, ptMax_,0.,1.2);
    mTightMVAPUJIDPassFractionVSeta  = ibooker.bookProfile("TightMVAPUIDPassFractionVSeta","TightMVAPUIDPassFractionVSeta",etaBin_, etaMin_, etaMax_,0.,1.2);
    mTightMVAPUJIDPassFractionVSpt   = ibooker.bookProfile("TightMVAPUIDPassFractionVSpt","TightMVAPUIDPassFractionVSpt",ptBin_, ptMin_, ptMax_,0.,1.2);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"LooseMVAPUIDPassFractionVSeta",mLooseMVAPUJIDPassFractionVSeta));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"LooseMVAPUIDPassFractionVSpt",mLooseMVAPUJIDPassFractionVSpt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MediumMVAPUIDPassFractionVSeta",mMediumMVAPUJIDPassFractionVSeta));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MediumMVAPUIDPassFractionVSpt",mMediumMVAPUJIDPassFractionVSpt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"TightMVAPUIDPassFractionVSeta",mTightMVAPUJIDPassFractionVSeta)); 
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"TightMVAPUIDPassFractionVSpt",mTightMVAPUJIDPassFractionVSpt));

    mLooseCutPUJIDPassFractionVSeta  = ibooker.bookProfile("LooseCutPUIDPassFractionVSeta","LooseCutPUIDPassFractionVSeta",etaBin_, etaMin_, etaMax_,0.,1.2);
    mLooseCutPUJIDPassFractionVSpt   = ibooker.bookProfile("LooseCutPUIDPassFractionVSpt","LooseCutPUIDPassFractionVSpt",ptBin_, ptMin_, ptMax_,0.,1.2);
    mMediumCutPUJIDPassFractionVSeta = ibooker.bookProfile("MediumCutPUIDPassFractionVSeta","MediumCutPUIDPassFractionVSeta",etaBin_, etaMin_, etaMax_,0.,1.2);
    mMediumCutPUJIDPassFractionVSpt  = ibooker.bookProfile("MediumCutPUIDPassFractionVSpt","MediumCutPUIDPassFractionVSpt",ptBin_, ptMin_, ptMax_,0.,1.2);
    mTightCutPUJIDPassFractionVSeta  = ibooker.bookProfile("TightCutPUIDPassFractionVSeta","TightCutPUIDPassFractionVSeta",etaBin_, etaMin_, etaMax_,0.,1.2);
    mTightCutPUJIDPassFractionVSpt   = ibooker.bookProfile("TightCutPUIDPassFractionVSpt","TightCutPUIDPassFractionVSpt",ptBin_, ptMin_, ptMax_,0.,1.2);
    mCutPUJIDDiscriminant_lowPt_Barrel   = ibooker.book1D("CutPUJIDDiscriminant_lowPt_Barrel","CutPUJIDDiscriminant_lowPt_Barrel",50, -1.00, 1.00);
    mCutPUJIDDiscriminant_lowPt_EndCap   = ibooker.book1D("CutPUJIDDiscriminant_lowPt_EndCap","CutPUJIDDiscriminant_lowPt_EndCap",50, -1.00, 1.00);
    mCutPUJIDDiscriminant_lowPt_Forward   = ibooker.book1D("CutPUJIDDiscriminant_lowPt_Forward","CutPUJIDDiscriminant_lowPt_Forward",50, -1.00, 1.00);
    mCutPUJIDDiscriminant_mediumPt_Barrel   = ibooker.book1D("CutPUJIDDiscriminant_mediumPt_Barrel","CutPUJIDDiscriminant_mediumPt_Barrel",50, -1.00, 1.00);
    mCutPUJIDDiscriminant_mediumPt_EndCap   = ibooker.book1D("CutPUJIDDiscriminant_mediumPt_EndCap","CutPUJIDDiscriminant_mediumPt_EndCap",50, -1.00, 1.00);
    mCutPUJIDDiscriminant_mediumPt_Forward   = ibooker.book1D("CutPUJIDDiscriminant_mediumPt_Forward","CutPUJIDDiscriminant_mediumPt_Forward",50, -1.00, 1.00);
    mCutPUJIDDiscriminant_highPt_Barrel   = ibooker.book1D("CutPUJIDDiscriminant_highPt_Barrel","CutPUJIDDiscriminant_highPt_Barrel",50, -1.00, 1.00);
    mCutPUJIDDiscriminant_highPt_EndCap   = ibooker.book1D("CutPUJIDDiscriminant_highPt_EndCap","CutPUJIDDiscriminant_highPt_EndCap",50, -1.00, 1.00);
    mCutPUJIDDiscriminant_highPt_Forward   = ibooker.book1D("CutPUJIDDiscriminant_highPt_Forward","CutPUJIDDiscriminant_highPt_Forward",50, -1.00, 1.00);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"LooseCutPUIDPassFractionVSeta",mLooseCutPUJIDPassFractionVSeta));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"LooseCutPUIDPassFractionVSpt",mLooseCutPUJIDPassFractionVSpt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MediumCutPUIDPassFractionVSeta",mMediumCutPUJIDPassFractionVSeta));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MediumCutPUIDPassFractionVSpt",mMediumCutPUJIDPassFractionVSpt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"TightCutPUIDPassFractionVSeta",mTightCutPUJIDPassFractionVSeta)); 
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"TightCutPUIDPassFractionVSpt",mTightCutPUJIDPassFractionVSpt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_lowPt_Barrel",mCutPUJIDDiscriminant_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_lowPt_EndCap",mCutPUJIDDiscriminant_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_lowPt_Forward",mCutPUJIDDiscriminant_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_mediumPt_Barrel",mCutPUJIDDiscriminant_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_mediumPt_EndCap",mCutPUJIDDiscriminant_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_mediumPt_Forward",mCutPUJIDDiscriminant_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_highPt_Barrel",mCutPUJIDDiscriminant_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_highPt_EndCap",mCutPUJIDDiscriminant_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CutPUJIDDiscriminant_highPt_Forward",mCutPUJIDDiscriminant_highPt_Forward));
    //barrel histograms for PFJets
    // energy fractions
    mCHFrac_lowPt_Barrel     = ibooker.book1D("CHFrac_lowPt_Barrel", "CHFrac_lowPt_Barrel", 120, -0.1, 1.1);
    mNHFrac_lowPt_Barrel     = ibooker.book1D("NHFrac_lowPt_Barrel", "NHFrac_lowPt_Barrel", 120, -0.1, 1.1);
    mPhFrac_lowPt_Barrel     = ibooker.book1D("PhFrac_lowPt_Barrel", "PhFrac_lowPt_Barrel", 120, -0.1, 1.1);
    mCHFrac_mediumPt_Barrel  = ibooker.book1D("CHFrac_mediumPt_Barrel", "CHFrac_mediumPt_Barrel", 120, -0.1, 1.1);
    mNHFrac_mediumPt_Barrel  = ibooker.book1D("NHFrac_mediumPt_Barrel", "NHFrac_mediumPt_Barrel", 120, -0.1, 1.1);
    mPhFrac_mediumPt_Barrel  = ibooker.book1D("PhFrac_mediumPt_Barrel", "PhFrac_mediumPt_Barrel", 120, -0.1, 1.1);
    mCHFrac_highPt_Barrel    = ibooker.book1D("CHFrac_highPt_Barrel", "CHFrac_highPt_Barrel", 120, -0.1, 1.1);
    mNHFrac_highPt_Barrel    = ibooker.book1D("NHFrac_highPt_Barrel", "NHFrac_highPt_Barrel", 120, -0.1, 1.1);
    mPhFrac_highPt_Barrel    = ibooker.book1D("PhFrac_highPt_Barrel", "PhFrac_highPt_Barrel", 120, -0.1, 1.1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_lowPt_Barrel" ,mCHFrac_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_lowPt_Barrel" ,mNHFrac_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_lowPt_Barrel" ,mPhFrac_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_mediumPt_Barrel" ,mCHFrac_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_mediumPt_Barrel" ,mNHFrac_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_mediumPt_Barrel" ,mPhFrac_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_highPt_Barrel" ,mCHFrac_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_highPt_Barrel" ,mNHFrac_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_highPt_Barrel" ,mPhFrac_highPt_Barrel));
   
    mMass_lowPt_Barrel     = ibooker.book1D("JetMass_lowPt_Barrel", "JetMass_lowPt_Barrel", 50, 0, 150);
    mMass_lowPt_EndCap     = ibooker.book1D("JetMass_lowPt_EndCap", "JetMass_lowPt_EndCap", 50, 0, 150);
    mMass_lowPt_Forward     = ibooker.book1D("JetMass_lowPt_Forward", "JetMass_lowPt_Forward", 50, 0, 150);
    mMass_mediumPt_Barrel     = ibooker.book1D("JetMass_mediumPt_Barrel", "JetMass_mediumPt_Barrel", 50, 0, 150);
    mMass_mediumPt_EndCap     = ibooker.book1D("JetMass_mediumPt_EndCap", "JetMass_mediumPt_EndCap", 50, 0, 150);
    mMass_mediumPt_Forward     = ibooker.book1D("JetMass_mediumPt_Forward", "JetMass_mediumPt_Forward", 75, 0, 150);
    mMass_highPt_Barrel     = ibooker.book1D("JetMass_highPt_Barrel", "JetMass_highPt_Barrel", 50, 0, 150);
    mMass_highPt_EndCap     = ibooker.book1D("JetMass_highPt_EndCap", "JetMass_highPt_EndCap", 50, 0, 150);
    mMass_highPt_Forward     = ibooker.book1D("JetMass_highPt_Forward", "JetMass_highPt_Forward", 50, 0, 150);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_lowPt_Barrel" , mMass_lowPt_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_lowPt_EndCap" , mMass_lowPt_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_lowPt_Forward" , mMass_lowPt_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_mediumPt_Barrel" , mMass_mediumPt_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_mediumPt_EndCap" , mMass_mediumPt_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_mediumPt_Forward" , mMass_mediumPt_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_highPt_Barrel" , mMass_highPt_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_highPt_EndCap" , mMass_highPt_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetMass_highPt_Forward" , mMass_highPt_Forward ));


    //energies
    mCHEn_lowPt_Barrel     = ibooker.book1D("CHEn_lowPt_Barrel", "CHEn_lowPt_Barrel", ptBin_, 0., ptMax_);
    mNHEn_lowPt_Barrel     = ibooker.book1D("NHEn_lowPt_Barrel", "NHEn_lowPt_Barrel", ptBin_, 0., ptMax_);
    mPhEn_lowPt_Barrel     = ibooker.book1D("PhEn_lowPt_Barrel", "PhEn_lowPt_Barrel", ptBin_, 0., ptMax_);
    mElEn_lowPt_Barrel     = ibooker.book1D("ElEn_lowPt_Barrel", "ElEn_lowPt_Barrel", ptBin_, 0., 100);
    mMuEn_lowPt_Barrel     = ibooker.book1D("MuEn_lowPt_Barrel", "MuEn_lowPt_Barrel", ptBin_, 0., 100);
    mCHEn_mediumPt_Barrel  = ibooker.book1D("CHEn_mediumPt_Barrel", "CHEn_mediumPt_Barrel", ptBin_, 0., ptMax_);
    mNHEn_mediumPt_Barrel  = ibooker.book1D("NHEn_mediumPt_Barrel", "NHEn_mediumPt_Barrel", ptBin_, 0., ptMax_);
    mPhEn_mediumPt_Barrel  = ibooker.book1D("PhEn_mediumPt_Barrel", "PhEn_mediumPt_Barrel", ptBin_, 0., ptMax_);
    mElEn_mediumPt_Barrel  = ibooker.book1D("ElEn_mediumPt_Barrel", "ElEn_mediumPt_Barrel", ptBin_, 0., 100);
    mMuEn_mediumPt_Barrel  = ibooker.book1D("MuEn_mediumPt_Barrel", "MuEn_mediumPt_Barrel", ptBin_, 0., 100);
    mCHEn_highPt_Barrel    = ibooker.book1D("CHEn_highPt_Barrel", "CHEn_highPt_Barrel", ptBin_, 0., 1.1*ptMax_);
    mNHEn_highPt_Barrel    = ibooker.book1D("NHEn_highPt_Barrel", "NHEn_highPt_Barrel", ptBin_, 0., ptMax_);
    mPhEn_highPt_Barrel    = ibooker.book1D("PhEn_highPt_Barrel", "PhEn_highPt_Barrel", ptBin_, 0., ptMax_);
    mElEn_highPt_Barrel    = ibooker.book1D("ElEn_highPt_Barrel", "ElEn_highPt_Barrel", ptBin_, 0., 100);
    mMuEn_highPt_Barrel    = ibooker.book1D("MuEn_highPt_Barrel", "MuEn_highPt_Barrel", ptBin_, 0., 100);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHEn_lowPt_Barrel" ,mCHEn_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHEn_lowPt_Barrel" ,mNHEn_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhEn_lowPt_Barrel" ,mPhEn_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElEn_lowPt_Barrel" ,mElEn_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuEn_lowPt_Barrel" ,mMuEn_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHEn_mediumPt_Barrel" ,mCHEn_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHEn_mediumPt_Barrel" ,mNHEn_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhEn_mediumPt_Barrel" ,mPhEn_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElEn_mediumPt_Barrel" ,mElEn_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuEn_mediumPt_Barrel" ,mMuEn_mediumPt_Barrel)); 
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHEn_highPt_Barrel" ,mCHEn_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHEn_highPt_Barrel" ,mNHEn_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhEn_highPt_Barrel" ,mPhEn_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElEn_highPt_Barrel" ,mElEn_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuEn_highPt_Barrel" ,mMuEn_highPt_Barrel));

    //multiplicities
    mChMultiplicity_lowPt_Barrel    = ibooker.book1D("ChMultiplicity_lowPt_Barrel", "ChMultiplicity_lowPt_Barrel", 60,0,60);
    mNeutMultiplicity_lowPt_Barrel   = ibooker.book1D("NeutMultiplicity_lowPt_Barrel", "NeutMultiplicity_lowPt_Barrel", 60,0,60);
    mMuMultiplicity_lowPt_Barrel    = ibooker.book1D("MuMultiplicity_lowPt_Barrel", "MuMultiplicity_lowPt_Barrel", 10,0,10);
    mChMultiplicity_mediumPt_Barrel    = ibooker.book1D("ChMultiplicity_mediumPt_Barrel", "ChMultiplicity_mediumPt_Barrel", 60,0,60);
    mNeutMultiplicity_mediumPt_Barrel   = ibooker.book1D("NeutMultiplicity_mediumPt_Barrel", "NeutMultiplicity_mediumPt_Barrel", 60,0,60);
    mMuMultiplicity_mediumPt_Barrel    = ibooker.book1D("MuMultiplicity_mediumPt_Barrel", "MuMultiplicity_mediumPt_Barrel", 10,0,10);
    mChMultiplicity_highPt_Barrel    = ibooker.book1D("ChMultiplicity_highPt_Barrel", "ChMultiplicity_highPt_Barrel", 60,0,60);
    mNeutMultiplicity_highPt_Barrel   = ibooker.book1D("NeutMultiplicity_highPt_Barrel", "NeutMultiplicity_highPt_Barrel", 60,0,60);
    mMuMultiplicity_highPt_Barrel    = ibooker.book1D("MuMultiplicity_highPt_Barrel", "MuMultiplicity_highPt_Barrel", 10,0,10);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_lowPt_Barrel" ,mChMultiplicity_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_lowPt_Barrel" ,mNeutMultiplicity_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuMultiplicity_lowPt_Barrel" ,mMuMultiplicity_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_mediumPt_Barrel" ,mChMultiplicity_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_mediumPt_Barrel" ,mNeutMultiplicity_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuMultiplicity_mediumPt_Barrel" ,mMuMultiplicity_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_highPt_Barrel" ,mChMultiplicity_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_highPt_Barrel" ,mNeutMultiplicity_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuMultiplicity_highPt_Barrel" ,mMuMultiplicity_highPt_Barrel));
  }
  //
  if(isMiniAODJet_ || isPFJet_){
    mMVAPUJIDDiscriminant_lowPt_Barrel   = ibooker.book1D("MVAPUJIDDiscriminant_lowPt_Barrel","MVAPUJIDDiscriminant_lowPt_Barrel",50, -1.00, 1.00);
    mMVAPUJIDDiscriminant_lowPt_EndCap   = ibooker.book1D("MVAPUJIDDiscriminant_lowPt_EndCap","MVAPUJIDDiscriminant_lowPt_EndCap",50, -1.00, 1.00);
    mMVAPUJIDDiscriminant_lowPt_Forward   = ibooker.book1D("MVAPUJIDDiscriminant_lowPt_Forward","MVAPUJIDDiscriminant_lowPt_Forward",50, -1.00, 1.00);
    mMVAPUJIDDiscriminant_mediumPt_Barrel   = ibooker.book1D("MVAPUJIDDiscriminant_mediumPt_Barrel","MVAPUJIDDiscriminant_mediumPt_Barrel",50, -1.00, 1.00);
    mMVAPUJIDDiscriminant_mediumPt_EndCap   = ibooker.book1D("MVAPUJIDDiscriminant_mediumPt_EndCap","MVAPUJIDDiscriminant_mediumPt_EndCap",50, -1.00, 1.00);
    mMVAPUJIDDiscriminant_mediumPt_Forward   = ibooker.book1D("MVAPUJIDDiscriminant_mediumPt_Forward","MVAPUJIDDiscriminant_mediumPt_Forward",50, -1.00, 1.00);
    mMVAPUJIDDiscriminant_highPt_Barrel   = ibooker.book1D("MVAPUJIDDiscriminant_highPt_Barrel","MVAPUJIDDiscriminant_highPt_Barrel",50, -1.00, 1.00);
    mMVAPUJIDDiscriminant_highPt_EndCap   = ibooker.book1D("MVAPUJIDDiscriminant_highPt_EndCap","MVAPUJIDDiscriminant_highPt_EndCap",50, -1.00, 1.00);
    mMVAPUJIDDiscriminant_highPt_Forward   = ibooker.book1D("MVAPUJIDDiscriminant_highPt_Forward","MVAPUJIDDiscriminant_highPt_Forward",50, -1.00, 1.00);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_lowPt_Barrel",mMVAPUJIDDiscriminant_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_lowPt_EndCap",mMVAPUJIDDiscriminant_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_lowPt_Forward",mMVAPUJIDDiscriminant_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_Barrel",mMVAPUJIDDiscriminant_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_EndCap",mMVAPUJIDDiscriminant_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_Forward",mMVAPUJIDDiscriminant_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_highPt_Barrel",mMVAPUJIDDiscriminant_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_highPt_EndCap",mMVAPUJIDDiscriminant_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MVAPUJIDDiscriminant_highPt_Forward",mMVAPUJIDDiscriminant_highPt_Forward));

    mCHFracVSpT_Barrel= ibooker.bookProfile("CHFracVSpT_Barrel","CHFracVSpT_Barrel",ptBin_, ptMin_, ptMax_,0.,1.2);
    mNHFracVSpT_Barrel= ibooker.bookProfile("NHFracVSpT_Barrel","NHFracVSpT_Barrel",ptBin_, ptMin_, ptMax_,0.,1.2);
    mPhFracVSpT_Barrel= ibooker.bookProfile("PhFracVSpT_Barrel","PhFracVSpT_Barrel",ptBin_, ptMin_, ptMax_,0.,1.2);
    mCHFracVSpT_EndCap= ibooker.bookProfile("CHFracVSpT_EndCap","CHFracVSpT_EndCap",ptBin_, ptMin_, ptMax_,0.,1.2);
    mNHFracVSpT_EndCap= ibooker.bookProfile("NHFracVSpT_EndCap","NHFracVSpT_EndCap",ptBin_, ptMin_, ptMax_,0.,1.2);
    mPhFracVSpT_EndCap= ibooker.bookProfile("PhFracVSpT_EndCap","PhFracVSpT_EndCap",ptBin_, ptMin_, ptMax_,0.,1.2);
    mHFHFracVSpT_Forward= ibooker.bookProfile("HFHFracVSpT_Forward","HFHFracVSpT_Forward",ptBin_, ptMin_, ptMax_,-0.2,1.2);
    mHFEFracVSpT_Forward= ibooker.bookProfile("HFEFracVSpT_Forward","HFEFracVSpT_Forward",ptBin_, ptMin_, ptMax_,-0.2,1.2);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSpT_Barrel" ,mCHFracVSpT_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSpT_Barrel" ,mNHFracVSpT_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSpT_Barrel" ,mPhFracVSpT_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSpT_EndCap" ,mCHFracVSpT_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSpT_EndCap" ,mNHFracVSpT_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSpT_EndCap" ,mPhFracVSpT_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracVSpT_Forward" ,mHFHFracVSpT_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEFracVSpT_Forward" ,mHFEFracVSpT_Forward));
  }
  if(isPFJet_){
    //endcap monitoring
    //energy fractions
    mCHFrac_lowPt_EndCap     = ibooker.book1D("CHFrac_lowPt_EndCap", "CHFrac_lowPt_EndCap", 120, -0.1, 1.1);
    mNHFrac_lowPt_EndCap     = ibooker.book1D("NHFrac_lowPt_EndCap", "NHFrac_lowPt_EndCap", 120, -0.1, 1.1);
    mPhFrac_lowPt_EndCap     = ibooker.book1D("PhFrac_lowPt_EndCap", "PhFrac_lowPt_EndCap", 120, -0.1, 1.1);
    mCHFrac_mediumPt_EndCap  = ibooker.book1D("CHFrac_mediumPt_EndCap", "CHFrac_mediumPt_EndCap", 120, -0.1, 1.1);
    mNHFrac_mediumPt_EndCap  = ibooker.book1D("NHFrac_mediumPt_EndCap", "NHFrac_mediumPt_EndCap", 120, -0.1, 1.1);
    mPhFrac_mediumPt_EndCap  = ibooker.book1D("PhFrac_mediumPt_EndCap", "PhFrac_mediumPt_EndCap", 120, -0.1, 1.1);
    mCHFrac_highPt_EndCap    = ibooker.book1D("CHFrac_highPt_EndCap", "CHFrac_highPt_EndCap", 120, -0.1, 1.1);
    mNHFrac_highPt_EndCap    = ibooker.book1D("NHFrac_highPt_EndCap", "NHFrac_highPt_EndCap", 120, -0.1, 1.1);
    mPhFrac_highPt_EndCap    = ibooker.book1D("PhFrac_highPt_EndCap", "PhFrac_highPt_EndCap", 120, -0.1, 1.1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_lowPt_EndCap" ,mCHFrac_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_lowPt_EndCap" ,mNHFrac_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_lowPt_EndCap" ,mPhFrac_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_mediumPt_EndCap" ,mCHFrac_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_mediumPt_EndCap" ,mNHFrac_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_mediumPt_EndCap" ,mPhFrac_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_highPt_EndCap" ,mCHFrac_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_highPt_EndCap" ,mNHFrac_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_highPt_EndCap" ,mPhFrac_highPt_EndCap));

    //energies
    mCHEn_lowPt_EndCap     = ibooker.book1D("CHEn_lowPt_EndCap", "CHEn_lowPt_EndCap", ptBin_, 0., ptMax_);
    mNHEn_lowPt_EndCap     = ibooker.book1D("NHEn_lowPt_EndCap", "NHEn_lowPt_EndCap", ptBin_, 0., ptMax_);
    mPhEn_lowPt_EndCap     = ibooker.book1D("PhEn_lowPt_EndCap", "PhEn_lowPt_EndCap", ptBin_, 0., ptMax_);
    mElEn_lowPt_EndCap     = ibooker.book1D("ElEn_lowPt_EndCap", "ElEn_lowPt_EndCap", ptBin_, 0., 100);
    mMuEn_lowPt_EndCap     = ibooker.book1D("MuEn_lowPt_EndCap", "MuEn_lowPt_EndCap", ptBin_, 0., 100);
    mCHEn_mediumPt_EndCap  = ibooker.book1D("CHEn_mediumPt_EndCap", "CHEn_mediumPt_EndCap", ptBin_, 0., ptMax_);
    mNHEn_mediumPt_EndCap  = ibooker.book1D("NHEn_mediumPt_EndCap", "NHEn_mediumPt_EndCap", ptBin_, 0., ptMax_);
    mPhEn_mediumPt_EndCap  = ibooker.book1D("PhEn_mediumPt_EndCap", "PhEn_mediumPt_EndCap", ptBin_, 0., ptMax_);
    mElEn_mediumPt_EndCap  = ibooker.book1D("ElEn_mediumPt_EndCap", "ElEn_mediumPt_EndCap", ptBin_, 0., 100);
    mMuEn_mediumPt_EndCap  = ibooker.book1D("MuEn_mediumPt_EndCap", "MuEn_mediumPt_EndCap", ptBin_, 0., 100);
    mCHEn_highPt_EndCap    = ibooker.book1D("CHEn_highPt_EndCap", "CHEn_highPt_EndCap", ptBin_, 0., 1.5*ptMax_);
    mNHEn_highPt_EndCap    = ibooker.book1D("NHEn_highPt_EndCap", "NHEn_highPt_EndCap", ptBin_, 0., 1.5*ptMax_);
    mPhEn_highPt_EndCap    = ibooker.book1D("PhEn_highPt_EndCap", "PhEn_highPt_EndCap", ptBin_, 0., 1.5*ptMax_);
    mElEn_highPt_EndCap    = ibooker.book1D("ElEn_highPt_EndCap", "ElEn_highPt_EndCap", ptBin_, 0., 100);
    mMuEn_highPt_EndCap    = ibooker.book1D("MuEn_highPt_EndCap", "MuEn_highPt_EndCap", ptBin_, 0., 100);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHEn_lowPt_EndCap" ,mCHEn_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHEn_lowPt_EndCap" ,mNHEn_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhEn_lowPt_EndCap" ,mPhEn_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElEn_lowPt_EndCap" ,mElEn_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuEn_lowPt_EndCap" ,mMuEn_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHEn_mediumPt_EndCap" ,mCHEn_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHEn_mediumPt_EndCap" ,mNHEn_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhEn_mediumPt_EndCap" ,mPhEn_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElEn_mediumPt_EndCap" ,mElEn_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuEn_mediumPt_EndCap" ,mMuEn_mediumPt_EndCap)); 
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHEn_highPt_EndCap" ,mCHEn_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHEn_highPt_EndCap" ,mNHEn_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhEn_highPt_EndCap" ,mPhEn_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElEn_highPt_EndCap" ,mElEn_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuEn_highPt_EndCap" ,mMuEn_highPt_EndCap));

    //now get handle on OOT PU
    mePhFracBarrel_BXm2BXm1Empty          = ibooker.book1D("PhFracBarrel_BXm2BXm1Empty",        "PHFrac prev empty 2 bunches (Barrel)",         50, 0,    1);
    mePhFracBarrel_BXm2BXm1Filled        = ibooker.book1D("PhFracBarrel_BXm2BXm1Filled",      "PHFrac prev filled 2 bunches (Barrel)",         50, 0,    1);
    meNHFracBarrel_BXm2BXm1Empty   = ibooker.book1D("NHFracBarrel_BXm2BXm1Empty",   "NHFrac prev empty 2 bunches (Barrel)",         50, 0,    1);
    meNHFracBarrel_BXm2BXm1Filled = ibooker.book1D("NHFracBarrel_BXm2BXm1Filled", "NHFrac prev filled 2 bunches (Barrel)",         50, 0,    1);
    meCHFracBarrel_BXm2BXm1Empty   = ibooker.book1D("CHFracBarrel_BXm2BXm1Empty",   "CHFrac prev empty 2 bunches (Barrel)",         50, 0,    1);
    meCHFracBarrel_BXm2BXm1Filled = ibooker.book1D("CHFracBarrel_BXm2BXm1Filled", "CHFrac prev filled 2 bunches (Barrel)",         50, 0,    1);
    mePtBarrel_BXm2BXm1Empty                       = ibooker.book1D("PtBarrel_BXm2BXm1Empty",   "pT prev empty 2 bunches (Barrel)", ptBin_, ptMin_, ptMax_);
    mePtBarrel_BXm2BXm1Filled                     = ibooker.book1D("PtBarrel_BXm2BXm1Filled", "pT prev filled 2 bunches (Barrel)", ptBin_, ptMin_, ptMax_);
    mePhFracEndCapPlus_BXm2BXm1Empty          = ibooker.book1D("PhFracEndCapPlus_BXm2BXm1Empty",        "PHFrac prev empty 2 bunches (EndCapPlus)",         50, 0,    1);
    mePhFracEndCapPlus_BXm2BXm1Filled        = ibooker.book1D("PhFracEndCapPlus_BXm2BXm1Filled",      "PHFrac prev filled 2 bunches (EndCapPlus)",         50, 0,    1);
    meNHFracEndCapPlus_BXm2BXm1Empty   = ibooker.book1D("NHFracEndCapPlus_BXm2BXm1Empty",   "NHFrac prev empty 2 bunches (EndCapPlus)",         50, 0,    1);
    meNHFracEndCapPlus_BXm2BXm1Filled = ibooker.book1D("NHFracEndCapPlus_BXm2BXm1Filled", "NHFrac prev filled 2 bunches (EndCapPlus)",         50, 0,    1);
    meCHFracEndCapPlus_BXm2BXm1Empty   = ibooker.book1D("CHFracEndCapPlus_BXm2BXm1Empty",   "CHFrac prev empty 2 bunches (EndCapPlus)",         50, 0,    1);
    meCHFracEndCapPlus_BXm2BXm1Filled = ibooker.book1D("CHFracEndCapPlus_BXm2BXm1Filled", "CHFrac prev filled 2 bunches (EndCapPlus)",         50, 0,    1);
    mePtEndCapPlus_BXm2BXm1Empty                       = ibooker.book1D("PtEndCapPlus_BXm2BXm1Empty",   "pT prev empty 2 bunches (EndCapPlus)", ptBin_, ptMin_, ptMax_);
    mePtEndCapPlus_BXm2BXm1Filled                     = ibooker.book1D("PtEndCapPlus_BXm2BXm1Filled", "pT prev filled 2 bunches (EndCapPlus)", ptBin_, ptMin_, ptMax_);
    meHFHFracPlus_BXm2BXm1Empty   = ibooker.book1D("HFHFracPlus_BXm2BXm1Empty",   "HFHFrac prev empty 2 bunches (EndCapPlus)",         50, 0,    1);
    meHFHFracPlus_BXm2BXm1Filled = ibooker.book1D("HFHFracPlus_BXm2BXm1Filled", "HFHFrac prev filled 2 bunches (EndCapPlus)",         50, 0,    1);
    meHFEMFracPlus_BXm2BXm1Empty   = ibooker.book1D("HFEMFracPlus_BXm2BXm1Empty",   "HFEMFrac prev empty 2 bunches (EndCapPlus)",         50, 0,    1);
    meHFEMFracPlus_BXm2BXm1Filled = ibooker.book1D("HFEMFracPlus_BXm2BXm1Filled", "HFEMFrac prev filled 2 bunches (EndCapPlus)",         50, 0,    1);
    mePtForwardPlus_BXm2BXm1Empty                       = ibooker.book1D("PtForwardPlus_BXm2BXm1Empty",   "pT prev empty 2 bunches (ForwardPlus)", ptBin_, ptMin_, ptMax_);
    mePtForwardPlus_BXm2BXm1Filled                     = ibooker.book1D("PtForwardPlus_BXm2BXm1Filled", "pT prev filled 2 bunches (ForwardPlus)", ptBin_, ptMin_, ptMax_);
    mePhFracEndCapMinus_BXm2BXm1Empty          = ibooker.book1D("PhFracEndCapMinus_BXm2BXm1Empty",        "PHFrac prev empty 2 bunches (EndCapMinus)",         50, 0,    1);
    mePhFracEndCapMinus_BXm2BXm1Filled        = ibooker.book1D("PhFracEndCapMinus_BXm2BXm1Filled",      "PHFrac prev filled 2 bunches (EndCapMinus)",         50, 0,    1);
    meNHFracEndCapMinus_BXm2BXm1Empty   = ibooker.book1D("NHFracEndCapMinus_BXm2BXm1Empty",   "NHFrac prev empty 2 bunches (EndCapMinus)",         50, 0,    1);
    meNHFracEndCapMinus_BXm2BXm1Filled = ibooker.book1D("NHFracEndCapMinus_BXm2BXm1Filled", "NHFrac prev filled 2 bunches (EndCapMinus)",         50, 0,    1);
    meCHFracEndCapMinus_BXm2BXm1Empty   = ibooker.book1D("CHFracEndCapMinus_BXm2BXm1Empty",   "CHFrac prev empty 2 bunches (EndCapMinus)",         50, 0,    1);
    meCHFracEndCapMinus_BXm2BXm1Filled = ibooker.book1D("CHFracEndCapMinus_BXm2BXm1Filled", "CHFrac prev filled 2 bunches (EndCapMinus)",         50, 0,    1);
    mePtEndCapMinus_BXm2BXm1Empty                       = ibooker.book1D("PtEndCapMinus_BXm2BXm1Empty",   "pT prev empty 2 bunches (EndCapMinus)", ptBin_, ptMin_, ptMax_);
    mePtEndCapMinus_BXm2BXm1Filled                     = ibooker.book1D("PtEndCapMinus_BXm2BXm1Filled", "pT prev filled 2 bunches (EndCapMinus)", ptBin_, ptMin_, ptMax_);
    meHFHFracMinus_BXm2BXm1Empty   = ibooker.book1D("HFHFracMinus_BXm2BXm1Empty",   "HFHFrac prev empty 2 bunches (EndCapMinus)",         50, 0,    1);
    meHFHFracMinus_BXm2BXm1Filled = ibooker.book1D("HFHFracMinus_BXm2BXm1Filled", "HFHFrac prev filled 2 bunches (EndCapMinus)",         50, 0,    1);
    meHFEMFracMinus_BXm2BXm1Empty   = ibooker.book1D("HFEMFracMinus_BXm2BXm1Empty",   "HFEMFrac prev empty 2 bunches (EndCapMinus)",         50, 0,    1);
    meHFEMFracMinus_BXm2BXm1Filled = ibooker.book1D("HFEMFracMinus_BXm2BXm1Filled", "HFEMFrac prev filled 2 bunches (EndCapMinus)",         50, 0,    1);
    mePtForwardMinus_BXm2BXm1Empty                       = ibooker.book1D("PtForwardMinus_BXm2BXm1Empty",   "pT prev empty 2 bunches (ForwardMinus)", ptBin_, ptMin_, ptMax_);
    mePtForwardMinus_BXm2BXm1Filled                     = ibooker.book1D("PtForwardMinus_BXm2BXm1Filled", "pT prev filled 2 bunches (ForwardMinus)", ptBin_, ptMin_, ptMax_);
    meEta_BXm2BXm1Empty                     = ibooker.book1D("Eta_BXm2BXm1Empty",   "eta prev empty 2 bunches",  etaBin_, etaMin_, etaMax_);
    meEta_BXm2BXm1Filled                   = ibooker.book1D("Eta_BXm2BXm1Filled", "eta prev filled 2 bunches", etaBin_, etaMin_, etaMax_);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracBarrel_BXm2BXm1Empty"       ,mePhFracBarrel_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracBarrel_BXm2BXm1Filled"     ,mePhFracBarrel_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracBarrel_BXm2BXm1Empty"  ,meNHFracBarrel_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracBarrel_BXm2BXm1Filled"      ,meNHFracBarrel_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracBarrel_BXm2BXm1Empty"  ,meCHFracBarrel_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracBarrel_BXm2BXm1Filled"      ,meCHFracBarrel_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtBarrel_BXm2BXm1Empty"    ,mePtBarrel_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtBarrel_BXm2BXm1Filled"  ,mePtBarrel_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracEndCapPlus_BXm2BXm1Empty"       ,mePhFracEndCapPlus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracEndCapPlus_BXm2BXm1Filled"     ,mePhFracEndCapPlus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracEndCapPlus_BXm2BXm1Empty"  ,meNHFracEndCapPlus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracEndCapPlus_BXm2BXm1Filled"      ,meNHFracEndCapPlus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracEndCapPlus_BXm2BXm1Empty"  ,meCHFracEndCapPlus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracEndCapPlus_BXm2BXm1Filled"      ,meCHFracEndCapPlus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtEndCapPlus_BXm2BXm1Empty"    ,mePtEndCapPlus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtEndCapPlus_BXm2BXm1Filled"  ,mePtEndCapPlus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracPlus_BXm2BXm1Empty"  ,meHFHFracPlus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracPlus_BXm2BXm1Filled"      ,meHFHFracPlus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFracPlus_BXm2BXm1Empty"  ,meHFEMFracPlus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFracPlus_BXm2BXm1Filled"      ,meHFEMFracPlus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtForwardPlus_BXm2BXm1Empty"    ,mePtForwardPlus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtForwardPlus_BXm2BXm1Filled"  ,mePtForwardPlus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracEndCapMinus_BXm2BXm1Empty"       ,mePhFracEndCapMinus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracEndCapMinus_BXm2BXm1Filled"     ,mePhFracEndCapMinus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracEndCapMinus_BXm2BXm1Empty"  ,meNHFracEndCapMinus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracEndCapMinus_BXm2BXm1Filled"      ,meNHFracEndCapMinus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracEndCapMinus_BXm2BXm1Empty"  ,meCHFracEndCapMinus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracEndCapMinus_BXm2BXm1Filled"      ,meCHFracEndCapMinus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtEndCapMinus_BXm2BXm1Empty"    ,mePtEndCapMinus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtEndCapMinus_BXm2BXm1Filled"  ,mePtEndCapMinus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracMinus_BXm2BXm1Empty"  ,meHFHFracMinus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracMinus_BXm2BXm1Filled"      ,meHFHFracMinus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFracMinus_BXm2BXm1Empty"  ,meHFEMFracMinus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFracMinus_BXm2BXm1Filled"      ,meHFEMFracMinus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtForwardMinus_BXm2BXm1Empty"    ,mePtForwardMinus_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtForwardMinus_BXm2BXm1Filled"  ,mePtForwardMinus_BXm2BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_BXm2BXm1Empty"  ,meEta_BXm2BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_BXm2BXm1Filled",meEta_BXm2BXm1Filled));
 
    mePhFracBarrel_BXm1Empty          = ibooker.book1D("PhFracBarrel_BXm1Empty",        "PHFrac prev empty 1 bunch (Barrel)",         50, 0,    1);
    mePhFracBarrel_BXm1Filled        = ibooker.book1D("PhFracBarrel_BXm1Filled",      "PHFrac prev filled 1 bunch (Barrel)",         50, 0,    1);
    meNHFracBarrel_BXm1Empty   = ibooker.book1D("NHFracBarrel_BXm1Empty",   "NHFrac prev empty 1 bunch (Barrel)",         50, 0,    1);
    meNHFracBarrel_BXm1Filled = ibooker.book1D("NHFracBarrel_BXm1Filled", "NHFrac prev filled 1 bunch (Barrel)",         50, 0,    1);
    meCHFracBarrel_BXm1Empty   = ibooker.book1D("CHFracBarrel_BXm1Empty",   "CHFrac prev empty 1 bunch (Barrel)",         50, 0,    1);
    meCHFracBarrel_BXm1Filled = ibooker.book1D("CHFracBarrel_BXm1Filled", "CHFrac prev filled 1 bunch (Barrel)",         50, 0,    1);
    mePtBarrel_BXm1Empty                       = ibooker.book1D("PtBarrel_BXm1Empty",   "pT prev empty 1 bunch (Barrel)", ptBin_, ptMin_, ptMax_);
    mePtBarrel_BXm1Filled                     = ibooker.book1D("PtBarrel_BXm1Filled", "pT prev filled 1 bunch (Barrel)", ptBin_, ptMin_, ptMax_);
    mePhFracEndCapPlus_BXm1Empty          = ibooker.book1D("PhFracEndCapPlus_BXm1Empty",        "PHFrac prev empty 1 bunch (EndCapPlus)",         50, 0,    1);
    mePhFracEndCapPlus_BXm1Filled        = ibooker.book1D("PhFracEndCapPlus_BXm1Filled",      "PHFrac prev filled 1 bunch (EndCapPlus)",         50, 0,    1);
    meNHFracEndCapPlus_BXm1Empty   = ibooker.book1D("NHFracEndCapPlus_BXm1Empty",   "NHFrac prev empty 1 bunch (EndCapPlus)",         50, 0,    1);
    meNHFracEndCapPlus_BXm1Filled = ibooker.book1D("NHFracEndCapPlus_BXm1Filled", "NHFrac prev filled 1 bunch (EndCapPlus)",         50, 0,    1);
    meCHFracEndCapPlus_BXm1Empty   = ibooker.book1D("CHFracEndCapPlus_BXm1Empty",   "CHFrac prev empty 1 bunch (EndCapPlus)",         50, 0,    1);
    meCHFracEndCapPlus_BXm1Filled = ibooker.book1D("CHFracEndCapPlus_BXm1Filled", "CHFrac prev filled 1 bunch (EndCapPlus)",         50, 0,    1);
    mePtEndCapPlus_BXm1Empty                       = ibooker.book1D("PtEndCapPlus_BXm1Empty",   "pT prev empty 1 bunch (EndCapPlus)", ptBin_, ptMin_, ptMax_);
    mePtEndCapPlus_BXm1Filled                     = ibooker.book1D("PtEndCapPlus_BXm1Filled", "pT prev filled 1 bunch (EndCapPlus)", ptBin_, ptMin_, ptMax_);
    meHFHFracPlus_BXm1Empty   = ibooker.book1D("HFHFracPlus_BXm1Empty",   "HFHFrac prev empty 1 bunch (EndCapPlus)",         50, 0,    1);
    meHFHFracPlus_BXm1Filled = ibooker.book1D("HFHFracPlus_BXm1Filled", "HFHFrac prev filled 1 bunch (EndCapPlus)",         50, 0,    1);
    meHFEMFracPlus_BXm1Empty   = ibooker.book1D("HFEMFracPlus_BXm1Empty",   "HFEMFrac prev empty 1 bunch (EndCapPlus)",         50, 0,    1);
    meHFEMFracPlus_BXm1Filled = ibooker.book1D("HFEMFracPlus_BXm1Filled", "HFEMFrac prev filled 1 bunch (EndCapPlus)",         50, 0,    1);
    mePtForwardPlus_BXm1Empty                       = ibooker.book1D("PtForwardPlus_BXm1Empty",   "pT prev empty 1 bunch (ForwardPlus)", ptBin_, ptMin_, ptMax_);
    mePtForwardPlus_BXm1Filled                     = ibooker.book1D("PtForwardPlus_BXm1Filled", "pT prev filled 1 bunch (ForwardPlus)", ptBin_, ptMin_, ptMax_);
    mePhFracEndCapMinus_BXm1Empty          = ibooker.book1D("PhFracEndCapMinus_BXm1Empty",        "PHFrac prev empty 1 bunch (EndCapMinus)",         50, 0,    1);
    mePhFracEndCapMinus_BXm1Filled        = ibooker.book1D("PhFracEndCapMinus_BXm1Filled",      "PHFrac prev filled 1 bunch (EndCapMinus)",         50, 0,    1);
    meNHFracEndCapMinus_BXm1Empty   = ibooker.book1D("NHFracEndCapMinus_BXm1Empty",   "NHFrac prev empty 1 bunch (EndCapMinus)",         50, 0,    1);
    meNHFracEndCapMinus_BXm1Filled = ibooker.book1D("NHFracEndCapMinus_BXm1Filled", "NHFrac prev filled 1 bunch (EndCapMinus)",         50, 0,    1);
    meCHFracEndCapMinus_BXm1Empty   = ibooker.book1D("CHFracEndCapMinus_BXm1Empty",   "CHFrac prev empty 1 bunch (EndCapMinus)",         50, 0,    1);
    meCHFracEndCapMinus_BXm1Filled = ibooker.book1D("CHFracEndCapMinus_BXm1Filled", "CHFrac prev filled 1 bunch (EndCapMinus)",         50, 0,    1);
    mePtEndCapMinus_BXm1Empty                       = ibooker.book1D("PtEndCapMinus_BXm1Empty",   "pT prev empty 1 bunch (EndCapMinus)", ptBin_, ptMin_, ptMax_);
    mePtEndCapMinus_BXm1Filled                     = ibooker.book1D("PtEndCapMinus_BXm1Filled", "pT prev filled 1 bunch (EndCapMinus)", ptBin_, ptMin_, ptMax_);
    meHFHFracMinus_BXm1Empty   = ibooker.book1D("HFHFracMinus_BXm1Empty",   "HFHFrac prev empty 1 bunch (EndCapMinus)",         50, 0,    1);
    meHFHFracMinus_BXm1Filled = ibooker.book1D("HFHFracMinus_BXm1Filled", "HFHFrac prev filled 1 bunch (EndCapMinus)",         50, 0,    1);
    meHFEMFracMinus_BXm1Empty   = ibooker.book1D("HFEMFracMinus_BXm1Empty",   "HFEMFrac prev empty 1 bunch (EndCapMinus)",         50, 0,    1);
    meHFEMFracMinus_BXm1Filled = ibooker.book1D("HFEMFracMinus_BXm1Filled", "HFEMFrac prev filled 1 bunch (EndCapMinus)",         50, 0,    1);
    mePtForwardMinus_BXm1Empty                       = ibooker.book1D("PtForwardMinus_BXm1Empty",   "pT prev empty 1 bunch (ForwardMinus)", ptBin_, ptMin_, ptMax_);
    mePtForwardMinus_BXm1Filled                     = ibooker.book1D("PtForwardMinus_BXm1Filled", "pT prev filled 1 bunch (ForwardMinus)", ptBin_, ptMin_, ptMax_);
    meEta_BXm1Empty                     = ibooker.book1D("Eta_BXm1Empty",   "eta prev empty 1 bunch",  etaBin_, etaMin_, etaMax_);
    meEta_BXm1Filled                   = ibooker.book1D("Eta_BXm1Filled", "eta prev filled 1 bunch", etaBin_, etaMin_, etaMax_);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracBarrel_BXm1Empty"       ,mePhFracBarrel_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracBarrel_BXm1Filled"     ,mePhFracBarrel_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracBarrel_BXm1Empty"  ,meNHFracBarrel_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracBarrel_BXm1Filled"      ,meNHFracBarrel_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracBarrel_BXm1Empty"  ,meCHFracBarrel_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracBarrel_BXm1Filled"      ,meCHFracBarrel_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtBarrel_BXm1Empty"    ,mePtBarrel_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtBarrel_BXm1Filled"  ,mePtBarrel_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracEndCapPlus_BXm1Empty"       ,mePhFracEndCapPlus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracEndCapPlus_BXm1Filled"     ,mePhFracEndCapPlus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracEndCapPlus_BXm1Empty"  ,meNHFracEndCapPlus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracEndCapPlus_BXm1Filled"      ,meNHFracEndCapPlus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracEndCapPlus_BXm1Empty"  ,meCHFracEndCapPlus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracEndCapPlus_BXm1Filled"      ,meCHFracEndCapPlus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtEndCapPlus_BXm1Empty"    ,mePtEndCapPlus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtEndCapPlus_BXm1Filled"  ,mePtEndCapPlus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracPlus_BXm1Empty"  ,meHFHFracPlus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracPlus_BXm1Filled"      ,meHFHFracPlus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFracPlus_BXm1Empty"  ,meHFEMFracPlus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFracPlus_BXm1Filled"      ,meHFEMFracPlus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtForwardPlus_BXm1Empty"    ,mePtForwardPlus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtForwardPlus_BXm1Filled"  ,mePtForwardPlus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracEndCapMinus_BXm1Empty"       ,mePhFracEndCapMinus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracEndCapMinus_BXm1Filled"     ,mePhFracEndCapMinus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracEndCapMinus_BXm1Empty"  ,meNHFracEndCapMinus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracEndCapMinus_BXm1Filled"      ,meNHFracEndCapMinus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracEndCapMinus_BXm1Empty"  ,meCHFracEndCapMinus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracEndCapMinus_BXm1Filled"      ,meCHFracEndCapMinus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtEndCapMinus_BXm1Empty"    ,mePtEndCapMinus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtEndCapMinus_BXm1Filled"  ,mePtEndCapMinus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracMinus_BXm1Empty"  ,meHFHFracMinus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracMinus_BXm1Filled"      ,meHFHFracMinus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFracMinus_BXm1Empty"  ,meHFEMFracMinus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFracMinus_BXm1Filled"      ,meHFEMFracMinus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtForwardMinus_BXm1Empty"    ,mePtForwardMinus_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtForwardMinus_BXm1Filled"  ,mePtForwardMinus_BXm1Filled));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_BXm1Empty"  ,meEta_BXm1Empty));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_BXm1Filled",meEta_BXm1Filled));

    //multiplicities
    mChMultiplicity_lowPt_EndCap    = ibooker.book1D("ChMultiplicity_lowPt_EndCap", "ChMultiplicity_lowPt_EndCap", 60,0,60);
    mNeutMultiplicity_lowPt_EndCap   = ibooker.book1D("NeutMultiplicity_lowPt_EndCap", "NeutMultiplicity_lowPt_EndCap", 60,0,60);
    mMuMultiplicity_lowPt_EndCap    = ibooker.book1D("MuMultiplicity_lowPt_EndCap", "MuMultiplicity_lowPt_EndCap", 10,0,10);
    mChMultiplicity_mediumPt_EndCap    = ibooker.book1D("ChMultiplicity_mediumPt_EndCap", "ChMultiplicity_mediumPt_EndCap", 60,0,60);
    mNeutMultiplicity_mediumPt_EndCap   = ibooker.book1D("NeutMultiplicity_mediumPt_EndCap", "NeutMultiplicity_mediumPt_EndCap", 60,0,60);
    mMuMultiplicity_mediumPt_EndCap    = ibooker.book1D("MuMultiplicity_mediumPt_EndCap", "MuMultiplicity_mediumPt_EndCap", 10,0,10);
    mChMultiplicity_highPt_EndCap    = ibooker.book1D("ChMultiplicity_highPt_EndCap", "ChMultiplicity_highPt_EndCap", 60,0,60);
    mNeutMultiplicity_highPt_EndCap   = ibooker.book1D("NeutMultiplicity_highPt_EndCap", "NeutMultiplicity_highPt_EndCap", 60,0,60);
    mMuMultiplicity_highPt_EndCap    = ibooker.book1D("MuMultiplicity_highPt_EndCap", "MuMultiplicity_highPt_EndCap", 10,0,10);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_lowPt_EndCap" ,mChMultiplicity_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_lowPt_EndCap" ,mNeutMultiplicity_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuMultiplicity_lowPt_EndCap" ,mMuMultiplicity_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_mediumPt_EndCap" ,mChMultiplicity_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_mediumPt_EndCap" ,mNeutMultiplicity_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuMultiplicity_mediumPt_EndCap" ,mMuMultiplicity_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_highPt_EndCap" ,mChMultiplicity_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_highPt_EndCap" ,mNeutMultiplicity_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuMultiplicity_highPt_EndCap" ,mMuMultiplicity_highPt_EndCap));

    //forward monitoring
    //energy fraction
    mHFEFrac_lowPt_Forward    = ibooker.book1D("HFEFrac_lowPt_Forward", "HFEFrac_lowPt_Forward", 70, -0.2, 1.2);
    mHFHFrac_lowPt_Forward    = ibooker.book1D("HFHFrac_lowPt_Forward", "HFHFrac_lowPt_Forward", 70, -0.2, 1.2);
    mHFEFrac_mediumPt_Forward = ibooker.book1D("HFEFrac_mediumPt_Forward", "HFEFrac_mediumPt_Forward", 70, -0.2, 1.2);
    mHFHFrac_mediumPt_Forward = ibooker.book1D("HFHFrac_mediumPt_Forward", "HFHFrac_mediumPt_Forward", 70, -0.2, 1.2);
    mHFEFrac_highPt_Forward   = ibooker.book1D("HFEFrac_highPt_Forward", "HFEFrac_highPt_Forward", 70, -0.2, 1.2);
    mHFHFrac_highPt_Forward   = ibooker.book1D("HFHFrac_highPt_Forward", "HFHFrac_highPt_Forward", 70, -0.2, 1.2);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFrac_lowPt_Forward"    ,mHFHFrac_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEFrac_lowPt_Forward"    ,mHFEFrac_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFrac_mediumPt_Forward" ,mHFHFrac_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEFrac_mediumPt_Forward" ,mHFEFrac_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFrac_highPt_Forward"   ,mHFHFrac_highPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEFrac_highPt_Forward"   ,mHFEFrac_highPt_Forward));

    //energies
    mHFEEn_lowPt_Forward    = ibooker.book1D("HFEEn_lowPt_Forward", "HFEEn_lowPt_Forward", ptBin_, 0., ptMax_);
    mHFHEn_lowPt_Forward    = ibooker.book1D("HFHEn_lowPt_Forward", "HFHEn_lowPt_Forward", ptBin_, 0., 2.0*ptMax_);
    mHFEEn_mediumPt_Forward = ibooker.book1D("HFEEn_mediumPt_Forward", "HFEEn_mediumPt_Forward", ptBin_, 0., 1.5*ptMax_);
    mHFHEn_mediumPt_Forward = ibooker.book1D("HFHEn_mediumPt_Forward", "HFHEn_mediumPt_Forward", ptBin_, 0., 2.5*ptMax_);
    mHFEEn_highPt_Forward   = ibooker.book1D("HFEEn_highPt_Forward", "HFEEn_highPt_Forward", ptBin_, 0., 1.5*ptMax_);
    mHFHEn_highPt_Forward   = ibooker.book1D("HFHEn_highPt_Forward", "HFHEn_highPt_Forward", ptBin_, 0., 5.0*ptMax_);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHEn_lowPt_Forward"    ,mHFHEn_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEEn_lowPt_Forward"    ,mHFEEn_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHEn_mediumPt_Forward" ,mHFHEn_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEEn_mediumPt_Forward" ,mHFEEn_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHEn_highPt_Forward"   ,mHFHEn_highPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEEn_highPt_Forward"   ,mHFEEn_highPt_Forward));
    //multiplicities
    mNeutMultiplicity_lowPt_Forward    = ibooker.book1D("NeutMultiplicity_lowPt_Forward", "NeutMultiplicity_lowPt_Forward", 60,0,60);
    mNeutMultiplicity_mediumPt_Forward = ibooker.book1D("NeutMultiplicity_mediumPt_Forward", "NeutMultiplicity_mediumPt_Forward", 60,0,60);
    mNeutMultiplicity_highPt_Forward   = ibooker.book1D("NeutMultiplicity_highPt_Forward", "NeutMultiplicity_highPt_Forward", 60,0,60);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_lowPt_Forward" ,mNeutMultiplicity_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_mediumPt_Forward" ,mNeutMultiplicity_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_highPt_Forward" ,mNeutMultiplicity_highPt_Forward));
    
    mChargedHadronEnergy = ibooker.book1D("ChargedHadronEnergy", "charged HAD energy",    50, 0, 100);
    mNeutralHadronEnergy = ibooker.book1D("NeutralHadronEnergy", "neutral HAD energy",    50, 0, 100);
    mChargedEmEnergy     = ibooker.book1D("ChargedEmEnergy",    "charged EM energy ",    50, 0, 100);
    mChargedMuEnergy     = ibooker.book1D("ChargedMuEnergy",     "charged Mu energy",     50, 0, 100);
    mNeutralEmEnergy     = ibooker.book1D("NeutralEmEnergy",     "neutral EM energy",     50, 0, 100);
    mChargedMultiplicity = ibooker.book1D("ChargedMultiplicity", "charged multiplicity ", 50, 0, 100);
    mNeutralMultiplicity = ibooker.book1D("NeutralMultiplicity", "neutral multiplicity",  50, 0, 100);
    mMuonMultiplicity    = ibooker.book1D("MuonMultiplicity",    "muon multiplicity",     50, 0, 100);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedHadronEnergy" ,mChargedHadronEnergy));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralHadronEnergy" ,mNeutralHadronEnergy));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedEmEnergy"     ,mChargedEmEnergy));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedMuEnergy"     ,mChargedMuEnergy));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralEmEnergy"     ,mNeutralEmEnergy));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedMultiplicity" ,mChargedMultiplicity));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralMultiplicity" ,mNeutralMultiplicity));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuonMultiplicity"    ,mMuonMultiplicity));

    // Book NPV profiles
    //----------------------------------------------------------------------------
    mChargedHadronEnergy_profile = ibooker.bookProfile("ChargedHadronEnergy_profile", "charged HAD energy",   nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
    mNeutralHadronEnergy_profile = ibooker.bookProfile("NeutralHadronEnergy_profile", "neutral HAD energy",   nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
    mChargedEmEnergy_profile     = ibooker.bookProfile("ChargedEmEnergy_profile",     "charged EM energy",    nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
    mChargedMuEnergy_profile     = ibooker.bookProfile("ChargedMuEnergy_profile",     "charged Mu energy",    nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
    mNeutralEmEnergy_profile     = ibooker.bookProfile("NeutralEmEnergy_profile",     "neutral EM energy",    nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
    mChargedMultiplicity_profile = ibooker.bookProfile("ChargedMultiplicity_profile", "charged multiplicity", nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
    mNeutralMultiplicity_profile = ibooker.bookProfile("NeutralMultiplicity_profile", "neutral multiplicity", nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
    mMuonMultiplicity_profile    = ibooker.bookProfile("MuonMultiplicity_profile",    "muon multiplicity",    nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
    
    // Set NPV profiles x-axis title
    //----------------------------------------------------------------------------
    mChargedHadronEnergy_profile->setAxisTitle("nvtx",1);
    mNeutralHadronEnergy_profile->setAxisTitle("nvtx",1);
    mChargedEmEnergy_profile    ->setAxisTitle("nvtx",1);
    mChargedMuEnergy_profile    ->setAxisTitle("nvtx",1);
    mNeutralEmEnergy_profile    ->setAxisTitle("nvtx",1);
    mChargedMultiplicity_profile->setAxisTitle("nvtx",1);
    mNeutralMultiplicity_profile->setAxisTitle("nvtx",1);
    mMuonMultiplicity_profile   ->setAxisTitle("nvtx",1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedHadronEnergy_profile" ,mChargedHadronEnergy_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralHadronEnergy_profile" ,mNeutralHadronEnergy_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedEmEnergy_profile"     ,mChargedEmEnergy_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedMuEnergy_profile"     ,mChargedMuEnergy_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralEmEnergy_profile"     ,mNeutralEmEnergy_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedMultiplicity_profile" ,mChargedMultiplicity_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralMultiplicity_profile" ,mNeutralMultiplicity_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuonMultiplicity_profile"    ,mMuonMultiplicity_profile));
    
    mNeutralFraction     = ibooker.book1D("NeutralConstituentsFraction","Neutral Constituents Fraction",100,0,1);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralConstituentsFraction" ,mNeutralFraction));

  }

  if(jetCleaningFlag_){
    //so far we have only one additional selection -> implement to make it expandable
    folderNames_.push_back("DiJet");
    if(isPFJet_){//check for now only for PFJets
      folderNames_.push_back("ZJets");
    }
    //book for each of these selection default histograms
    for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
	 ic != folderNames_.end(); ic++){
      bookMESetSelection(DirName+"/"+*ic, ibooker);
    }
  }

  ibooker.setCurrentFolder("JetMET");
  cleanupME = ibooker.book1D("cleanup", "cleanup", 10, 0., 10.);
  cleanupME->setBinLabel(1,"Primary Vertex");
  cleanupME->setBinLabel(2,"DCS::Pixel");
  cleanupME->setBinLabel(3,"DCS::SiStrip");
  cleanupME->setBinLabel(4,"DCS::ECAL");
  cleanupME->setBinLabel(5,"DCS::ES");
  cleanupME->setBinLabel(6,"DCS::HBHE");
  cleanupME->setBinLabel(7,"DCS::HF");
  cleanupME->setBinLabel(8,"DCS::HO");
  cleanupME->setBinLabel(9,"DCS::Muon");
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>("JetMET/cleanup" ,cleanupME));

  verticesME = ibooker.book1D("vertices", "vertices", 100, 0, 100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>("JetMET/vertices" ,verticesME));



}

void JetAnalyzer::bookMESetSelection(std::string DirName, DQMStore::IBooker & ibooker)
{
  ibooker.setCurrentFolder(DirName);
  // Generic jet parameters
  mPt           = ibooker.book1D("Pt",           "pt",                 ptBin_,  ptMin_,  ptMax_);
  mEta          = ibooker.book1D("Eta",          "eta",               etaBin_, etaMin_, etaMax_);
  mPhi          = ibooker.book1D("Phi",          "phi",               phiBin_, phiMin_, phiMax_);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt"  ,mPt));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta" ,mEta));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi" ,mPhi));
  //if(!isJPTJet_){
  mConstituents = ibooker.book1D("Constituents", "# of constituents",     50,      0,    100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents" ,mConstituents));
  //}
  mJetEnergyCorr= ibooker.book1D("JetEnergyCorr", "jet energy correction factor", 50, 0.0,3.0);
  mJetEnergyCorrVSEta= ibooker.bookProfile("JetEnergyCorrVSEta", "jet energy correction factor VS eta", etaBin_, etaMin_,etaMax_, 0.0,3.0);
  mJetEnergyCorrVSPt= ibooker.bookProfile("JetEnergyCorrVSPt", "jet energy correction factor VS pt", ptBin_, ptMin_,ptMax_, 0.0,3.0);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorr" ,mJetEnergyCorr));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorrVSEta" ,mJetEnergyCorrVSEta));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorrVSPt" ,mJetEnergyCorrVSPt));
  
  //fill for Dijets: concentrates on gluon jets -> fill leading two jets
  //fill for ZJets: concentrates on quark jets -> fill leading jet
  if(fill_CHS_histos && isPFJet_){
    mAxis2_lowPt_Barrel = ibooker.book1D("qg_Axis2_lowPt_Barrel","qg Axis2 #sigma_{2} lowPt Barrel",50,0.,0.20);
    mpTD_lowPt_Barrel= ibooker.book1D("qg_pTD_lowPt_Barrel","qg fragmentation function p_{T}^{D} lowPt Barrel",50,0.15,1.05);
    mMultiplicityQG_lowPt_Barrel= ibooker.book1D("qg_multiplicity_lowPt_Barrel","qg multiplicity lowPt Barrel",50,0,50);
    mqgLikelihood_lowPt_Barrel= ibooker.book1D("qg_Likelihood_lowPt_Barrel","qg likelihood lowPt Barrel",50,-1.1,1.1);
    mAxis2_lowPt_EndCap = ibooker.book1D("qg_Axis2_lowPt_EndCap","qg Axis2 #sigma_{2} lowPt EndCap",50,0.,0.20);
    mpTD_lowPt_EndCap= ibooker.book1D("qg_pTD_lowPt_EndCap","qg fragmentation function p_{T}^{D} lowPt EndCap",50,0.15,1.05);
    mMultiplicityQG_lowPt_EndCap= ibooker.book1D("qg_multiplicity_lowPt_EndCap","qg multiplicity lowPt EndCap",50,0,100);
    mqgLikelihood_lowPt_EndCap= ibooker.book1D("qg_Likelihood_lowPt_EndCap","qg likelihood lowPt EndCap",50,-1.1,1.1);
    mAxis2_lowPt_Forward = ibooker.book1D("qg_Axis2_lowPt_Forward","qg Axis2 #sigma_{2} lowPt Forward",50,0.,0.20);
    mpTD_lowPt_Forward= ibooker.book1D("qg_pTD_lowPt_Forward","qg fragmentation function p_{T}^{D} lowPt Forward",50,0.15,1.05);
    mMultiplicityQG_lowPt_Forward= ibooker.book1D("qg_multiplicity_lowPt_Forward","qg multiplicity lowPt Forward",50,0,100);
    mqgLikelihood_lowPt_Forward= ibooker.book1D("qg_Likelihood_lowPt_Forward","qg likelihood lowPt Forward",50,-1.1,1.1);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_lowPt_Barrel" ,mAxis2_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_lowPt_Barrel" ,mpTD_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_lowPt_Barrel" ,mMultiplicityQG_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_lowPt_Barrel" ,mqgLikelihood_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_lowPt_EndCap" ,mAxis2_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_lowPt_EndCap" ,mpTD_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_lowPt_EndCap" ,mMultiplicityQG_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_lowPt_EndCap" ,mqgLikelihood_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_lowPt_Forward" ,mAxis2_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_lowPt_Forward" ,mpTD_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_lowPt_Forward" ,mMultiplicityQG_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_lowPt_Forward" ,mqgLikelihood_lowPt_Forward));
    
    mAxis2_mediumPt_Barrel = ibooker.book1D("qg_Axis2_mediumPt_Barrel","qg Axis2 #sigma_{2} mediumPt Barrel",50,0.,0.20);
    mpTD_mediumPt_Barrel= ibooker.book1D("qg_pTD_mediumPt_Barrel","qg fragmentation function p_{T}^{D} mediumPt Barrel",50,0.15,1.05);
    mMultiplicityQG_mediumPt_Barrel= ibooker.book1D("qg_multiplicity_mediumPt_Barrel","qg multiplicity mediumPt Barrel",50,0,100);
    mqgLikelihood_mediumPt_Barrel= ibooker.book1D("qg_Likelihood_mediumPt_Barrel","qg likelihood mediumPt Barrel",50,-1.1,1.1);
    mAxis2_mediumPt_EndCap = ibooker.book1D("qg_Axis2_mediumPt_EndCap","qg Axis2 #sigma_{2} mediumPt EndCap",50,0.,0.20);
    mpTD_mediumPt_EndCap= ibooker.book1D("qg_pTD_mediumPt_EndCap","qg fragmentation function p_{T}^{D} mediumPt EndCap",50,0.15,1.05);
    mMultiplicityQG_mediumPt_EndCap= ibooker.book1D("qg_multiplicity_mediumPt_EndCap","qg multiplicity mediumPt EndCap",50,0,100);
    mqgLikelihood_mediumPt_EndCap= ibooker.book1D("qg_Likelihood_mediumPt_EndCap","qg likelihood mediumPt EndCap",50,-1.1,1.1);
    mAxis2_mediumPt_Forward = ibooker.book1D("qg_Axis2_mediumPt_Forward","qg Axis2 #sigma_{2} mediumPt Forward",50,0.,0.20);
    mpTD_mediumPt_Forward= ibooker.book1D("qg_pTD_mediumPt_Forward","qg fragmentation function p_{T}^{D} mediumPt Forward",50,0.15,1.05);
    mMultiplicityQG_mediumPt_Forward= ibooker.book1D("qg_multiplicity_mediumPt_Forward","qg multiplicity mediumPt Forward",50,0,100);
    mqgLikelihood_mediumPt_Forward= ibooker.book1D("qg_Likelihood_mediumPt_Forward","qg likelihood mediumPt Forward",50,-1.1,1.1);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_mediumPt_Barrel" ,mAxis2_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_mediumPt_Barrel" ,mpTD_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_mediumPt_Barrel" ,mMultiplicityQG_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_mediumPt_Barrel" ,mqgLikelihood_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_mediumPt_EndCap" ,mAxis2_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_mediumPt_EndCap" ,mpTD_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_mediumPt_EndCap" ,mMultiplicityQG_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_mediumPt_EndCap" ,mqgLikelihood_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_mediumPt_Forward" ,mAxis2_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_mediumPt_Forward" ,mpTD_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_mediumPt_Forward" ,mMultiplicityQG_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_mediumPt_Forward" ,mqgLikelihood_mediumPt_Forward));
    
    mAxis2_highPt_Barrel = ibooker.book1D("qg_Axis2_highPt_Barrel","qg Axis2 #sigma_{2} highPt Barrel",50,0.,0.20);
    mpTD_highPt_Barrel= ibooker.book1D("qg_pTD_highPt_Barrel","qg fragmentation function p_{T}^{D} highPt Barrel",50,0.15,1.05);
    mMultiplicityQG_highPt_Barrel= ibooker.book1D("qg_multiplicity_highPt_Barrel","qg multiplicity highPt Barrel",50,0,100);
    mqgLikelihood_highPt_Barrel= ibooker.book1D("qg_Likelihood_highPt_Barrel","qg likelihood highPt Barrel",50,-1.1,1.1);
    mAxis2_highPt_EndCap = ibooker.book1D("qg_Axis2_highPt_EndCap","qg Axis2 #sigma_{2} highPt EndCap",50,0.,0.20);
    mpTD_highPt_EndCap= ibooker.book1D("qg_pTD_highPt_EndCap","qg fragmentation function p_{T}^{D} highPt EndCap",50,0.15,1.05);
    mMultiplicityQG_highPt_EndCap= ibooker.book1D("qg_multiplicity_highPt_EndCap","qg multiplicity highPt EndCap",50,0,100);
    mqgLikelihood_highPt_EndCap= ibooker.book1D("qg_Likelihood_highPt_EndCap","qg likelihood highPt EndCap",50,-1.1,1.1);
    mAxis2_highPt_Forward = ibooker.book1D("qg_Axis2_highPt_Forward","qg Axis2 #sigma_{2} highPt Forward",50,0.,0.20);
    mpTD_highPt_Forward= ibooker.book1D("qg_pTD_highPt_Forward","qg fragmentation function p_{T}^{D} highPt Forward",50,0.15,1.05);
    mMultiplicityQG_highPt_Forward= ibooker.book1D("qg_multiplicity_highPt_Forward","qg multiplicity highPt Forward",50,0,100);
    mqgLikelihood_highPt_Forward= ibooker.book1D("qg_Likelihood_highPt_Forward","qg likelihood highPt Forward",50,-1.1,1.1);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_highPt_Barrel" ,mAxis2_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_highPt_Barrel" ,mpTD_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_highPt_Barrel" ,mMultiplicityQG_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_highPt_Barrel" ,mqgLikelihood_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_highPt_EndCap" ,mAxis2_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_highPt_EndCap" ,mpTD_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_highPt_EndCap" ,mMultiplicityQG_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_highPt_EndCap" ,mqgLikelihood_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Axis2_highPt_Forward" ,mAxis2_highPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_pTD_highPt_Forward" ,mpTD_highPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_multiplicity_highPt_Forward" ,mMultiplicityQG_highPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"qg_Likelihood_highPt_Forward" ,mqgLikelihood_highPt_Forward));
  }

  if(DirName.find("DiJet")!=std::string::npos){
    mDPhi                   = ibooker.book1D("DPhi", "dPhi btw the two leading jets", 100, 0., acos(-1.));
    mDijetAsymmetry                   = ibooker.book1D("DijetAsymmetry", "DijetAsymmetry", 100, -1., 1.);
    mDijetBalance                     = ibooker.book1D("DijetBalance",   "DijetBalance",   100, -2., 2.);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DPhi" ,mDPhi));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DijetAsymmetry" ,mDijetAsymmetry));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DijetBalance"   ,mDijetBalance));
    
    if(isPFJet_|| isMiniAODJet_){ 
      mChargedMultiplicity = ibooker.book1D("ChargedMultiplicity", "charged multiplicity ", 50, 0, 100);
      mNeutralMultiplicity = ibooker.book1D("NeutralMultiplicity", "neutral multiplicity",  50, 0, 100);
      mMuonMultiplicity    = ibooker.book1D("MuonMultiplicity",    "muon multiplicity",     50, 0, 100);
      
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedMultiplicity" ,mChargedMultiplicity));
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralMultiplicity" ,mNeutralMultiplicity));
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuonMultiplicity"    ,mMuonMultiplicity));

      mChargedMultiplicity_profile = ibooker.bookProfile("ChargedMultiplicity_profile", "charged multiplicity", nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
      mNeutralMultiplicity_profile = ibooker.bookProfile("NeutralMultiplicity_profile", "neutral multiplicity", nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
      mMuonMultiplicity_profile    = ibooker.bookProfile("MuonMultiplicity_profile",    "muon multiplicity",    nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 100);
      mChargedMultiplicity_profile->setAxisTitle("nvtx",1);
      mNeutralMultiplicity_profile->setAxisTitle("nvtx",1);
      mMuonMultiplicity_profile   ->setAxisTitle("nvtx",1);

      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedMultiplicity_profile" ,mChargedMultiplicity_profile));
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralMultiplicity_profile" ,mNeutralMultiplicity_profile));
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuonMultiplicity_profile"    ,mMuonMultiplicity_profile));

      mNeutralFraction     = ibooker.book1D("NeutralConstituentsFraction","Neutral Constituents Fraction",100,0,1);
      map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralConstituentsFraction" ,mNeutralFraction));
    }
  }


  if(DirName.find("ZJets")!=std::string::npos){
    mZMass                          = ibooker.book1D("DiMuonMass", "DiMuonMass", 50, 71., 111.);
    mDPhiZJet                        = ibooker.book1D("DPhiZJ", "dPhi btw Z and Jet1", 100, 0., acos(-1.));
    mZJetAsymmetry                   = ibooker.book1D("ZJetAsymmetry", "ZJetAsymmetry", 100, -1., 1.);
    mJetZBalance_lowZPt_J_Barrel     = ibooker.book1D("JZB_lowZPt_J_Barrel",   "ZJetBalance (pTJet1-pTZ) (30<pTZ<90), |#eta_{jet}|<1.3",   50, -75.,75);
    mJetZBalance_mediumZPt_J_Barrel  = ibooker.book1D("JZB_mediumZPt_J_Barrel",   "ZJetBalance (90<pTZ<140), |#eta_{jet}|<1.3",   50, -75.,75);
    mJetZBalance_highZPt_J_Barrel    = ibooker.book1D("JZB_highZPt_J_Barrel",   "ZJetBalance (pTZ>140), |#eta_{jet}|<1.3",   50, -75., 75.);
    mJetZBalance_lowZPt_J_EndCap     = ibooker.book1D("JZB_lowZPt_J_EndCap",   "ZJetBalance (30<pTZ<90), 1.3<|#eta_{jet}|<3.0",   50, -75.,75);
    mJetZBalance_mediumZPt_J_EndCap  = ibooker.book1D("JZB_mediumZPt_J_EndCap",   "ZJetBalance (90<pTZ<140), 1.3<|#eta_{jet}|<3.0",   50, -75.,75);
    mJetZBalance_highZPt_J_EndCap    = ibooker.book1D("JZB_highZPt_J_EndCap",   "ZJetBalance (pTZ>140), 1.3<|#eta_{jet}|<3.0",   50, -75., 75.);
    mJetZBalance_lowZPt_J_Forward     = ibooker.book1D("JZB_lowZPt_J_Forward",   "ZJetBalance (30<pTZ<90), |#eta_{jet}|>3.0",   50, -75.,75);
    mJetZBalance_mediumZPt_J_Forward = ibooker.book1D("JZB_mediumZPt_J_Forward",   "ZJetBalance (90<pTZ<140), |#eta_{jet}|>3.0",   50, -75.,75);
    mJetZBalance_highZPt_J_Forward   = ibooker.book1D("JZB_highZPt_J_Forward",   "ZJetBalance (pTZ>140), |#eta_{jet}|>3.0",   50, -75., 75.);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DiMuonMass" ,mZMass));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DPhiZJ" ,mDPhiZJet ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ZJetAsymmetry" ,mZJetAsymmetry ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_lowZPt_J_Barrel" ,mJetZBalance_lowZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_mediumZPt_J_Barrel" ,mJetZBalance_mediumZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_highZPt_J_Barrel" ,mJetZBalance_highZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_lowZPt_J_EndCap" ,mJetZBalance_lowZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_mediumZPt_J_EndCap" ,mJetZBalance_mediumZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_highZPt_J_EndCap" ,mJetZBalance_highZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_lowZPt_J_Forward" ,mJetZBalance_lowZPt_J_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_mediumZPt_J_Forward" ,mJetZBalance_mediumZPt_J_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JZB_highZPt_J_Forward" ,mJetZBalance_highZPt_J_Forward ));

    mJ1Pt_over_ZPt_J_Barrel      = ibooker.book1D("J1Pt_over_ZPt_J_Barrel",   "Jet1_Pt/ZPt, Barrel",   50, 0.,3.0);
    mJ1Pt_over_ZPt_J_EndCap   = ibooker.book1D("J1Pt_over_ZPt_J_EndCap",   "Jet1_Pt/ZPt, EndCap",   50, 0.,3.0);
    mJ1Pt_over_ZPt_J_Forward     = ibooker.book1D("J1Pt_over_ZPt_J_Forward",   "Jet1_Pt/ZPt, Forward",   50, 0.,3.0);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_J_Barrel" ,mJ1Pt_over_ZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_J_EndCap" ,mJ1Pt_over_ZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_J_Forward" ,mJ1Pt_over_ZPt_J_Forward ));

    mJ1Pt_over_ZPt_lowZPt_J_Barrel      = ibooker.book1D("J1Pt_over_ZPt_lowZPt_J_Barrel",   "Jet1_Pt/ZPt (30<pTZ<90), |#eta_{jet}|<1.3",   50, 0.,3.0);
    mJ1Pt_over_ZPt_mediumZPt_J_Barrel   = ibooker.book1D("J1Pt_over_ZPt_mediumZPt_J_Barrel",   "Jet1_Pt/ZPt (90<pTZ<140), |#eta_{jet}|<1.3",   50, 0.,3.0);
    mJ1Pt_over_ZPt_highZPt_J_Barrel     = ibooker.book1D("J1Pt_over_ZPt_highPt_J_Barrel",   "Jet1_Pt/ZPt (pTZ>140), |#eta_{jet}|<1.3",   50, 0.,3.0);
    mJ1Pt_over_ZPt_lowZPt_J_EndCap      = ibooker.book1D("J1Pt_over_ZPt_lowZPt_J_EndCap",   "Jet1_Pt/ZPt (30<pTZ<90), 1.3<|#eta_{jet}|<3.0",   50, 0.,3.0);
    mJ1Pt_over_ZPt_mediumZPt_J_EndCap   = ibooker.book1D("J1Pt_over_ZPt_mediumZPt_J_EndCap",   "Jet1_Pt/ZPt (90<pTZ<140), 1.3<|#eta_{jet}|<3.0",   50, 0.,3.0);
    mJ1Pt_over_ZPt_highZPt_J_EndCap     = ibooker.book1D("J1Pt_over_ZPt_highZPt_J_EndCap",   "Jet1_Pt/ZPt (pTZ>140), 1.3<|#eta_{jet}|<3.0",   50, 0.,3.0);
    mJ1Pt_over_ZPt_lowZPt_J_Forward      = ibooker.book1D("J1Pt_over_ZPt_lowZPt_J_Forward",  "Jet1_Pt/ZPt (30<pTZ<90), |#eta_{jet}|>3.0",   50, 0.,3.0);
    mJ1Pt_over_ZPt_mediumZPt_J_Forward  = ibooker.book1D("J1Pt_over_ZPt_mediumPt_J_Forward",  "Jet1_Pt/ZPt (90<pTZ<140), |#eta_{jet}|>3.0",   50, 0.,3.0);
    mJ1Pt_over_ZPt_highZPt_J_Forward    = ibooker.book1D("J1Pt_over_ZPt_highZPt_J_Forward",  "Jet1_Pt/ZPt (pTZ>140), |#eta_{jet}|>3.0",   50, 0.,3.0);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_lowZPt_J_Barrel" ,mJ1Pt_over_ZPt_lowZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_mediumZPt_J_Barrel" ,mJ1Pt_over_ZPt_mediumZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_highZPt_J_Barrel" ,mJ1Pt_over_ZPt_highZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_lowZPt_J_EndCap" ,mJ1Pt_over_ZPt_lowZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_mediumZPt_J_EndCap" ,mJ1Pt_over_ZPt_mediumZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_highZPt_J_EndCap" ,mJ1Pt_over_ZPt_highZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_lowZPt_J_Forward" ,mJ1Pt_over_ZPt_lowZPt_J_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_mediumZPt_J_Forward" ,mJ1Pt_over_ZPt_mediumZPt_J_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"J1Pt_over_ZPt_highZPt_J_Forward" ,mJ1Pt_over_ZPt_highZPt_J_Forward ));


    mMPF_J_Barrel      = ibooker.book1D("MPF_J_Barrel",   "Jet1_Pt/ZPt, Barrel",   50, 0.,2.0);
    mMPF_J_EndCap   = ibooker.book1D("MPF_J_EndCap",   "Jet1_Pt/ZPt, EndCap",   50, 0.,2.0);
    mMPF_J_Forward     = ibooker.book1D("MPF_J_Forward",   "Jet1_Pt/ZPt, Forward",   50, 0.,2.0);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_J_Barrel" ,mMPF_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_J_EndCap" ,mMPF_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_J_Forward" ,mMPF_J_Forward ));

    mMPF_lowZPt_J_Barrel      = ibooker.book1D("MPF_lowZPt_J_Barrel",   "Jet1_Pt/ZPt (30<pTZ<90), |#eta_{jet}|<1.3",   50, 0.,2.0);
    mMPF_mediumZPt_J_Barrel   = ibooker.book1D("MPF_mediumZPt_J_Barrel",   "Jet1_Pt/ZPt (90<pTZ<140), |#eta_{jet}|<1.3",   50, 0.,2.0);
    mMPF_highZPt_J_Barrel     = ibooker.book1D("MPF_highPt_J_Barrel",   "Jet1_Pt/ZPt (pTZ>140), |#eta_{jet}|<1.3",   50, 0.,2.0);
    mMPF_lowZPt_J_EndCap      = ibooker.book1D("MPF_lowZPt_J_EndCap",   "Jet1_Pt/ZPt (30<pTZ<90), 1.3<|#eta_{jet}|<3.0",   50, 0.,2.0);
    mMPF_mediumZPt_J_EndCap   = ibooker.book1D("MPF_mediumZPt_J_EndCap",   "Jet1_Pt/ZPt (90<pTZ<140), 1.3<|#eta_{jet}|<3.0",   50, 0.,2.0);
    mMPF_highZPt_J_EndCap     = ibooker.book1D("MPF_highZPt_J_EndCap",   "Jet1_Pt/ZPt (pTZ>140), 1.3<|#eta_{jet}|<3.0",   50, 0.,2.0);
    mMPF_lowZPt_J_Forward      = ibooker.book1D("MPF_lowZPt_J_Forward",  "Jet1_Pt/ZPt (30<pTZ<90), |#eta_{jet}|>3.0",   50, 0.,2.0);
    mMPF_mediumZPt_J_Forward  = ibooker.book1D("MPF_mediumPt_J_Forward",  "Jet1_Pt/ZPt (90<pTZ<140), |#eta_{jet}|>3.0",   50, 0.,2.0);
    mMPF_highZPt_J_Forward    = ibooker.book1D("MPF_highZPt_J_Forward",  "Jet1_Pt/ZPt (pTZ>140), |#eta_{jet}|>3.0",   50, 0.,2.0);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_lowZPt_J_Barrel" ,mMPF_lowZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_mediumZPt_J_Barrel" ,mMPF_mediumZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_highZPt_J_Barrel" ,mMPF_highZPt_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_lowZPt_J_EndCap" ,mMPF_lowZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_mediumZPt_J_EndCap" ,mMPF_mediumZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_highZPt_J_EndCap" ,mMPF_highZPt_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_lowZPt_J_Forward" ,mMPF_lowZPt_J_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_mediumZPt_J_Forward" ,mMPF_mediumZPt_J_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MPF_highZPt_J_Forward" ,mMPF_highZPt_J_Forward ));


    mDeltaPt_Z_j1_over_ZPt_30_55_J_Barrel    = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_30_55_J_Barrel",   "DeltaPt_Z_j1_over_ZPt_30_55_J_Barrel",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_55_75_J_Barrel    = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_55_75_J_Barrel",   "DeltaPt_Z_j1_over_ZPt_55_75_J_Barrel",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_75_150_J_Barrel   = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_75_150_J_Barrel",  "DeltaPt_Z_j1_over_ZPt_75_150_J_Barrel",   50, -1.00,1.00); 
    mDeltaPt_Z_j1_over_ZPt_150_290_J_Barrel  = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_150_290_J_Barrel", "DeltaPt_Z_j1_over_ZPt_150_290_J_Barrel",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_290_J_Barrel      = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_290_J_Barrel",     "DeltaPt_Z_j1_over_ZPt_290_J_Barrel",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_30_55_J_EndCap    = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_30_55_J_EndCap",   "DeltaPt_Z_j1_over_ZPt_30_55_J_EndCap",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_55_75_J_EndCap    = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_55_75_J_EndCap",   "DeltaPt_Z_j1_over_ZPt_55_75_J_EndCap",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_75_150_J_EndCap   = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_75_150_J_EndCap",  "DeltaPt_Z_j1_over_ZPt_75_150_J_EndCap",   50, -1.00,1.00); 
    mDeltaPt_Z_j1_over_ZPt_150_290_J_EndCap  = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_150_290_J_EndCap", "DeltaPt_Z_j1_over_ZPt_150_290_J_EndCap",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_290_J_EndCap      = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_290_J_EndCap",     "DeltaPt_Z_j1_over_ZPt_290_J_EndCap",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_30_55_J_Forward  = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_30_55_J_Forward",   "DeltaPt_Z_j1_over_ZPt_30_55_J_Forward",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_55_100_J_Forward = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_55_100_J_Forward",  "DeltaPt_Z_j1_over_ZPt_55_100_J_Forward",   50, -1.00,1.00);
    mDeltaPt_Z_j1_over_ZPt_100_J_Forward    = ibooker.book1D("DeltaPt_Z_j1_over_ZPt_100_J_Forward",     "DeltaPt_Z_j1_over_ZPt_100_J_Forward",   50, -1.00,1.00); 

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_30_55_J_Barrel" ,mDeltaPt_Z_j1_over_ZPt_30_55_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_55_75_J_Barrel" ,mDeltaPt_Z_j1_over_ZPt_55_75_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_75_150_J_Barrel" ,mDeltaPt_Z_j1_over_ZPt_75_150_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_150_290_J_Barrel" ,mDeltaPt_Z_j1_over_ZPt_150_290_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_290_J_Barrel" ,mDeltaPt_Z_j1_over_ZPt_290_J_Barrel ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_30_55_J_EndCap" ,mDeltaPt_Z_j1_over_ZPt_30_55_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_55_75_J_EndCap" ,mDeltaPt_Z_j1_over_ZPt_55_75_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_75_150_J_EndCap" ,mDeltaPt_Z_j1_over_ZPt_75_150_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_150_290_J_EndCap" ,mDeltaPt_Z_j1_over_ZPt_150_290_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_290_J_EndCap" ,mDeltaPt_Z_j1_over_ZPt_290_J_EndCap ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_30_55_J_Forward" ,mDeltaPt_Z_j1_over_ZPt_30_55_J_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_55_100_J_Forward" ,mDeltaPt_Z_j1_over_ZPt_55_100_J_Forward ));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DeltaPt_Z_j1_over_ZPt_100_J_Forward" ,mDeltaPt_Z_j1_over_ZPt_100_J_Forward ));
  }
  // Book NPV profiles
  //----------------------------------------------------------------------------
  mPt_profile           = ibooker.bookProfile("Pt_profile",           "pt",                nbinsPV_, nPVlow_, nPVhigh_,   ptBin_,  ptMin_,  ptMax_);
  mEta_profile          = ibooker.bookProfile("Eta_profile",          "eta",               nbinsPV_, nPVlow_, nPVhigh_,  etaBin_, etaMin_, etaMax_);
  mPhi_profile          = ibooker.bookProfile("Phi_profile",          "phi",               nbinsPV_, nPVlow_, nPVhigh_,  phiBin_, phiMin_, phiMax_);
  //if(!isJPTJet_){
  mConstituents_profile = ibooker.bookProfile("Constituents_profile", "# of constituents", nbinsPV_, nPVlow_, nPVhigh_,      50,      0,    100);
  //}
  // met NPV profiles x-axis title
  //----------------------------------------------------------------------------
  mPt_profile          ->setAxisTitle("nvtx",1);
  mEta_profile         ->setAxisTitle("nvtx",1);
  mPhi_profile         ->setAxisTitle("nvtx",1);
  //if(!isJPTJet_){
  mConstituents_profile->setAxisTitle("nvtx",1);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_profile",mConstituents_profile));
  //}

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_profile" ,mPt_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_profile",mEta_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_profile",mPhi_profile));
  //
  //--- Calo jet melection only

  if(isCaloJet_) {
    mHFrac        = ibooker.book1D("HFrac",        "HFrac",                140,   -0.2,    1.2);
    mEFrac        = ibooker.book1D("EFrac",        "EFrac",                140,   -0.2,    1.2);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac" ,mHFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac" ,mEFrac));
    
    mHFrac_profile        = ibooker.bookProfile("HFrac_profile",        "HFrac",             nbinsPV_, nPVlow_, nPVhigh_,     140,   -0.2,    1.2);
    mEFrac_profile        = ibooker.bookProfile("EFrac_profile",        "EFrac",             nbinsPV_, nPVlow_, nPVhigh_,     140,   -0.2,    1.2);
    mHFrac_profile       ->setAxisTitle("nvtx",1);
    mEFrac_profile       ->setAxisTitle("nvtx",1);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_profile",mHFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_profile",mEFrac_profile));
    
    // CaloJet specific
    //JetID variables
    mresEMF                 = ibooker.book1D("resEMF", "resEMF", 50, 0., 1.);
    mN90Hits                = ibooker.book1D("N90Hits", "N90Hits", 50, 0., 100);
    mfHPD                   = ibooker.book1D("fHPD", "fHPD", 50, 0., 1.);
    mfRBX                   = ibooker.book1D("fRBX", "fRBX", 50, 0., 1.);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"resEMF" ,mresEMF));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"N90Hits",mN90Hits));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"fHPD" ,mfHPD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"fRBX" ,mfRBX));
    
  }
    
  if(isPFJet_|| isMiniAODJet_){ 
    //barrel histograms for PFJets
    // energy fractions
    mCHFrac     = ibooker.book1D("CHFrac", "CHFrac", 120, -0.1, 1.1);
    mNHFrac     = ibooker.book1D("NHFrac", "NHFrac", 120, -0.1, 1.1);
    mPhFrac     = ibooker.book1D("PhFrac", "PhFrac", 120, -0.1, 1.1);
    mHFEMFrac   = ibooker.book1D("HFEMFrac","HFEMFrac", 120, -0.1, 1.1);
    mHFHFrac   = ibooker.book1D("HFHFrac", "HFHFrac", 120, -0.1, 1.1);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac"  ,mCHFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac"  ,mNHFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac"  ,mPhFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFrac",mHFEMFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFrac" ,mHFHFrac));

    // Book NPV profiles
    //----------------------------------------------------------------------------
    mCHFrac_profile = ibooker.bookProfile("CHFrac_profile", "charged HAD fraction profile",   nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 1);
    mNHFrac_profile = ibooker.bookProfile("NHFrac_profile", "neutral HAD fraction profile",   nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 1);
    mPhFrac_profile    = ibooker.bookProfile("PhFrac_profile",     "Photon Fraction Profile",    nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 1);
    mHFEMFrac_profile  = ibooker.bookProfile("HFEMFrac_profile","HF electomagnetic fraction Profile", nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 1);
    mHFHFrac_profile   = ibooker.bookProfile("HFHFrac_profile", "HF hadronic fraction profile", nbinsPV_, nPVlow_, nPVhigh_, 50, 0, 1);

    
    // met NPV profiles x-axis title
    //----------------------------------------------------------------------------
    mCHFrac_profile    ->setAxisTitle("nvtx",1);
    mNHFrac_profile    ->setAxisTitle("nvtx",1);
    mPhFrac_profile    ->setAxisTitle("nvtx",1);
    mHFEMFrac_profile  ->setAxisTitle("nvtx",1);
    mHFHFrac_profile   ->setAxisTitle("nvtx",1);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_profile"  ,mCHFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_profile"  ,mNHFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_profile"  ,mPhFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFrac_profile",mHFEMFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFrac_profile" ,mHFHFrac_profile));
    
  }
}

// ***********************************************************
void JetAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  //LogDebug("JetAnalyzer") << "beginRun, run " << run.id();
  //

  if ( highPtJetEventFlag_->on() ) highPtJetEventFlag_->initRun( iRun, iSetup );
  if ( lowPtJetEventFlag_ ->on() ) lowPtJetEventFlag_ ->initRun( iRun, iSetup );

  if (highPtJetEventFlag_->on() && highPtJetEventFlag_->expressionsFromDB(highPtJetEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highPtJetExpr_ = highPtJetEventFlag_->expressionsFromDB(highPtJetEventFlag_->hltDBKey(), iSetup);
  if (lowPtJetEventFlag_->on() && lowPtJetEventFlag_->expressionsFromDB(lowPtJetEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    lowPtJetExpr_  = lowPtJetEventFlag_->expressionsFromDB(lowPtJetEventFlag_->hltDBKey(),   iSetup);
//  if (!jetCorrectionService_.empty()){
//    energycorrected=true;
//  }
  //--- htlConfig_
  //processname_="HLT";
  bool changed(true);
  hltInitialized_ = hltConfig_.init(iRun,iSetup,processname_,changed);
  if (!hltInitialized_) {
  //if (!hltConfig_.init(iRun,iSetup,processname_,changed)) {
    processname_ = "FU";
    hltInitialized_ = hltConfig_.init(iRun,iSetup,processname_,changed);
    if(!hltInitialized_){
      //if (!hltConfig_.init(iRun,iSetup,processname_,changed)){
      LogDebug("JetAnalyzer") << "HLTConfigProvider failed to initialize.";
    }
  }

  edm::ESHandle<L1GtTriggerMenu> menuRcd;
  iSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
  const L1GtTriggerMenu* menu = menuRcd.product();
  for (CItAlgo techTrig = menu->gtTechnicalTriggerMap().begin(); techTrig != menu->gtTechnicalTriggerMap().end(); ++techTrig) {
    if ((techTrig->second).algoName() == m_l1algoname_) {
      m_bitAlgTechTrig_=(techTrig->second).algoBitNumber();
      break;
    }
  }
 
}

// ***********************************************************
void JetAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
}

// ***********************************************************
void JetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {


  //set general folders first --> change later on for different folders
  if(jetCleaningFlag_){
    //dbe_->setCurrentFolder("JetMET/Jet/Cleaned"+mInputCollection_.label());
    DirName = "JetMET/Jet/Cleaned"+mInputCollection_.label();
  }else{
    //dbe_->setCurrentFolder("JetMET/Jet/Uncleaned"+mInputCollection_.label());
    DirName = "JetMET/Jet/Uncleaned"+mInputCollection_.label();
  }
 

  Handle<ValueMap<float> > puJetIdMva;
  Handle<ValueMap<int> > puJetIdFlagMva;
  Handle<ValueMap<float> > puJetId;
  Handle<ValueMap<int> > puJetIdFlag;

  Handle<ValueMap<int> > qgMultiplicity;
  Handle<ValueMap<float> > qgLikelihood;
  Handle<ValueMap<float> > qgptD;
  Handle<ValueMap<float> > qgaxis2;

  //should insure we have a PFJet in with CHS
  if(fill_CHS_histos){
    iEvent.getByToken(qgMultiplicityToken_,qgMultiplicity);
    iEvent.getByToken(qgLikelihoodToken_,qgLikelihood);
    iEvent.getByToken(qgptDToken_,qgptD);
    iEvent.getByToken(qgaxis2Token_,qgaxis2);
  }

  if(!isMiniAODJet_){  
    iEvent.getByToken(mvaPUIDToken_,puJetIdFlagMva);
    iEvent.getByToken(cutBasedPUDiscriminantToken_,puJetId);
    iEvent.getByToken(cutBasedPUIDToken_,puJetIdFlag);
    iEvent.getByToken(mvaFullPUDiscriminantToken_ ,puJetIdMva);
  }

  // **** Get the TriggerResults container
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  
  Int_t JetLoPass = 0;
  Int_t JetHiPass = 0;

  if (triggerResults.isValid()){
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);

    const unsigned int nTrig(triggerNames.size());
    for (unsigned int i=0;i<nTrig;++i)
      {
        if (triggerNames.triggerName(i).find(highPtJetExpr_[0].substr(0,highPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults->accept(i))
	  JetHiPass=1;
        else if (triggerNames.triggerName(i).find(lowPtJetExpr_[0].substr(0,lowPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults->accept(i))
	  JetLoPass=1;
      }

  }

  if (verbose_)  std::cout << "trigger label " << theTriggerResultsLabel_ << std::endl;


  if (verbose_) {
    std::cout << ">>> Trigger  Lo = " <<  JetLoPass
	      <<             " Hi = " <<  JetHiPass
	      << std::endl;
  }

  // ==========================================================
  //Vertex information
  Handle<VertexCollection> vertexHandle;
  iEvent.getByToken(vertexToken_, vertexHandle);

  if (!vertexHandle.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
    if (verbose_) std::cout << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
  }
  int numPV = 0;
  if ( vertexHandle.isValid() ){
    VertexCollection vertexCollection = *(vertexHandle.product());
    numPV  = vertexCollection.size();
  }
  bool bPrimaryVertex = (bypassAllPVChecks_ || (numPV>0));
  if(fill_jet_high_level_histo){//should be filled for all events, no selection up to this point
    verticesME = map_of_MEs["JetMET/vertices"]; if(verticesME && verticesME->getRootObject())verticesME->Fill(numPV);
  }
  // ==========================================================
  edm::Handle<L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  iEvent.getByToken(gtToken_, gtReadoutRecord);

  if (!gtReadoutRecord.isValid()) {
    LogInfo("JetAnalyzer") << "JetAnalyzer: Could not find GT readout record" << std::endl;
    if (verbose_) std::cout << "JetAnalyzer: Could not find GT readout record product" << std::endl;
  }

  bool techTriggerResultBxE = false;
  bool techTriggerResultBxF = false;
  bool techTriggerResultBx0 = false;

  if (!gtReadoutRecord.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find GT readout record" << std::endl;
    if (verbose_) std::cout << "CaloMETAnalyzer: Could not find GT readout record product" << std::endl;
  }else{
    // trigger results before mask for BxInEvent -2 (E), -1 (F), 0 (L1A), 1, 2 
    const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxE = gtReadoutRecord->technicalTriggerWord(-2);
    const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxF = gtReadoutRecord->technicalTriggerWord(-1);
    const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBx0 = gtReadoutRecord->technicalTriggerWord();
    //const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxG = gtReadoutRecord->technicalTriggerWord(1);
    //const TechnicalTriggerWord&  technicalTriggerWordBeforeMaskBxH = gtReadoutRecord->technicalTriggerWord(2);
    if (m_bitAlgTechTrig_ > -1 && technicalTriggerWordBeforeMaskBx0.size() > 0) {
      techTriggerResultBx0 = technicalTriggerWordBeforeMaskBx0.at(m_bitAlgTechTrig_);
      if(techTriggerResultBx0!=0){
	techTriggerResultBxE = technicalTriggerWordBeforeMaskBxE.at(m_bitAlgTechTrig_);
	techTriggerResultBxF = technicalTriggerWordBeforeMaskBxF.at(m_bitAlgTechTrig_);
      }	
    }
  }


  DCSFilterForDCSMonitoring_->filter(iEvent, iSetup);
  if(fill_jet_high_level_histo){//should be filled only once
    cleanupME = map_of_MEs["JetMET/cleanup"]; if(cleanupME && cleanupME->getRootObject()){
      if (bPrimaryVertex) cleanupME->Fill(0.5);
      if ( DCSFilterForDCSMonitoring_->passPIX      ) cleanupME->Fill(1.5);
      if ( DCSFilterForDCSMonitoring_->passSiStrip  ) cleanupME->Fill(2.5);
      if ( DCSFilterForDCSMonitoring_->passECAL     ) cleanupME->Fill(3.5);
      if ( DCSFilterForDCSMonitoring_->passES       ) cleanupME->Fill(4.5);
      if ( DCSFilterForDCSMonitoring_->passHBHE     ) cleanupME->Fill(5.5);
      if ( DCSFilterForDCSMonitoring_->passHF       ) cleanupME->Fill(6.5);
      if ( DCSFilterForDCSMonitoring_->passHO       ) cleanupME->Fill(7.5);
      if ( DCSFilterForDCSMonitoring_->passMuon     ) cleanupME->Fill(8.5);
    }
  }
  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection> jptJets;
  edm::Handle<PFJetCollection> pfJets;
  edm::Handle<pat::JetCollection> patJets;

  edm::Handle<MuonCollection> Muons;

  bool pass_Z_selection=false;
  reco::Candidate::PolarLorentzVector zCand;

  int mu_index0=-1;
  int mu_index1=-1;

  if (isCaloJet_) iEvent.getByToken(caloJetsToken_, caloJets);
  //if (isJPTJet_) iEvent.getByToken(jptJetsToken_, jptJets);
  if (isPFJet_){ iEvent.getByToken(pfJetsToken_, pfJets);
    iEvent.getByToken(MuonsToken_, Muons);
    double pt0=-1;
    double pt1=-1;
    //fill it only for cleaned jets
    if(jetCleaningFlag_ && Muons.isValid() && Muons->size()>1){
      for (unsigned int i=0;i<Muons->size();i++){
	bool pass_muon_id=false;
	bool pass_muon_iso=false;
	double dxy=fabs((*Muons)[i].muonBestTrack()->dxy());
	double dz=fabs((*Muons)[i].muonBestTrack()->dz());
	if (numPV>0){
	  dxy=fabs((*Muons)[i].muonBestTrack()->dxy((*vertexHandle)[0].position()));
	  dz=fabs((*Muons)[i].muonBestTrack()->dz((*vertexHandle)[0].position()));
	}
	if((*Muons)[i].pt()>20 && fabs((*Muons)[i].eta())<2.3){
	  if((*Muons)[i].isGlobalMuon() && (*Muons)[i].isPFMuon() && (*Muons)[i].globalTrack()->hitPattern().numberOfValidMuonHits() > 0  && (*Muons)[i].numberOfMatchedStations() > 1 &&  dxy < 0.2 && (*Muons)[i].numberOfMatchedStations() > 1 && dz<0.5 && (*Muons)[i].innerTrack()->hitPattern().numberOfValidPixelHits() > 0 && (*Muons)[i].innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5){
	    pass_muon_id=true;
	  }
	  // Muon pf isolation DB corrected
	  float muonIsoPFdb  = ((*Muons)[i].pfIsolationR04().sumChargedHadronPt 
				+ std::max(0., (*Muons)[i].pfIsolationR04().sumNeutralHadronEt + (*Muons)[i].pfIsolationR04().sumPhotonEt - 0.5*(*Muons)[i].pfIsolationR04().sumPUPt))/(*Muons)[i].pt();    
	  if(muonIsoPFdb<0.12){
	    pass_muon_iso=true;
	  }

	  if(pass_muon_id && pass_muon_iso){
	    if((*Muons)[i].pt()>pt0){
	      mu_index1=mu_index0;
	      pt1=pt0;
	      mu_index0=i;
	      pt0=(*Muons)[i].pt();
	    }else if ((*Muons)[i].pt()>pt1){
	      mu_index1=i;
	      pt1=(*Muons)[i].pt();
	    }
	  }
	}
      }
      if(mu_index0>=0 && mu_index1>=0){
	if((*Muons)[mu_index0].charge()*(*Muons)[mu_index1].charge()<0){
	  zCand=(*Muons)[mu_index0].polarP4()+(*Muons)[mu_index1].polarP4();
	  if(fabs(zCand.M()-91.)<20 && zCand.Pt()>30){
	    pass_Z_selection=true;
	  }
	}
      }
    }
  }
  if(isMiniAODJet_) iEvent.getByToken(patJetsToken_,patJets);

  edm::Handle< edm::ValueMap<reco::JetID> >jetID_ValueMap_Handle;
  if(/*isJPTJet_ || */isCaloJet_){
    if(!runcosmics_){
      iEvent.getByToken(jetID_ValueMapToken_,jetID_ValueMap_Handle);
    }
  }

  //check for collections AND DCS filters
  bool dcsDecision = DCSFilterForJetMonitoring_->filter(iEvent, iSetup);
  bool jetCollectionIsValid = false;
  if (isCaloJet_)  jetCollectionIsValid = caloJets.isValid();
  //if (isJPTJet_)   jetCollectionIsValid = jptJets.isValid();
  if (isPFJet_)    jetCollectionIsValid = pfJets.isValid();
  if (isMiniAODJet_) jetCollectionIsValid = patJets.isValid();


  if (jetCleaningFlag_ && (!jetCollectionIsValid || !bPrimaryVertex || !dcsDecision)) return;

  unsigned int collSize=-1;
  if (isCaloJet_)  collSize = caloJets->size();
  //if (isJPTJet_) {
  //collSize=jptJets->size();
  //if(collSize>0){
  //  //update the track propagator and strip noise calculator
  //  trackPropagator_->update(iSetup);
  //  //sOverNCalculator_->update(iSetup);
  //}
  //}
  if (isPFJet_) collSize=pfJets->size();
  if (isMiniAODJet_) collSize=patJets->size();

  double scale=-1;
  //now start changes for jets
  std::vector<Jet> corJets;
  corJets.clear();

  //maybe not most elegant solution, but works for sure
  int ind1=-1;
  double pt1=-1;
  int ind2=-1;
  double pt2=-1;
  int ind3=-1;
  double pt3=-1;

  bool cleaned_first_jet=false;
  bool cleaned_second_jet=false;
  //bool cleaned_third_jet=false;
  //for ZJets selection check for muon jet overlap
  int ind1_mu_vetoed=-1;
  double pt1_mu_vetoed=-1;
  int ind2_mu_vetoed=-1;
  double pt2_mu_vetoed=-1;
  bool cleaned_first_jet_mu_vetoed=false;
  bool cleaned_second_jet_mu_vetoed=false;

  //now start changes for jets
  std::vector<Jet> recoJets;
  recoJets.clear();

  int numofjets=0;

  edm::Handle<reco::JetCorrector> jetCorr;
  bool pass_correction_flag=false;
  if(!isMiniAODJet_ && !jetCorrectorTag_.label().empty()){
    iEvent.getByToken(jetCorrectorToken_, jetCorr);
    if (jetCorr.isValid()){
      pass_correction_flag=true;
    }
  }
  if(isMiniAODJet_){
    pass_correction_flag=true;
  }

  for (unsigned int ijet=0; ijet<collSize; ijet++) {
    //bool thiscleaned=false;
    Jet correctedJet;
    bool pass_uncorrected=false;
    bool pass_corrected=false;
    if (isCaloJet_){
      correctedJet=(*caloJets)[ijet];
    }
    //if (isJPTJet_){
    //correctedJet=(*jptJets)[ijet];
    //}
    if (isPFJet_){
      correctedJet=(*pfJets)[ijet];
    }
    if (isMiniAODJet_){
      correctedJet=(*patJets)[ijet];
    }
    if(!isMiniAODJet_ && correctedJet.pt()>ptThresholdUnc_){
      pass_uncorrected=true;
    }
    if(isMiniAODJet_ && (correctedJet.pt()*(*patJets)[ijet].jecFactor("Uncorrected"))>ptThresholdUnc_){
      pass_uncorrected=true;
    }
    if (pass_correction_flag && !isMiniAODJet_) {
      if (isCaloJet_){
        scale = jetCorr->correction((*caloJets)[ijet]);
      }
      if (isPFJet_){ 
        scale = jetCorr->correction((*pfJets)[ijet]);
      }
      correctedJet.scaleEnergy(scale);	    
    }
    if(correctedJet.pt()> ptThreshold_){
      pass_corrected=true;
    }
    //if (!pass_corrected && !pass_uncorrected) continue;
    //remove the continue line, for physics selections we might losen the pt-thresholds as we care only about leading jets
    //fill only corrected jets -> check ID for uncorrected jets
    if(pass_corrected){
      recoJets.push_back(correctedJet);
    }
    bool jetpassid=true;
    bool Thiscleaned=true;
    bool JetIDWPU=true;
    //jet ID for calojets
    if (isCaloJet_) {
      reco::CaloJetRef calojetref(caloJets, ijet);
      if(!runcosmics_){
	reco::JetID jetID = (*jetID_ValueMap_Handle)[calojetref];
	jetpassid = jetIDFunctor((*caloJets)[ijet], jetID);
	JetIDWPU=jetpassid;
	if(jetCleaningFlag_){
	  Thiscleaned=jetpassid;
	}
	if(Thiscleaned && pass_corrected){//if cleaning requested->jet passes a loose ID
	  mN90Hits = map_of_MEs[DirName+"/"+"N90Hits"]; if (mN90Hits && mN90Hits->getRootObject()) mN90Hits->Fill (jetID.n90Hits);
	  mfHPD = map_of_MEs[DirName+"/"+"fHPD"]; if (mfHPD && mfHPD->getRootObject())             mfHPD->Fill (jetID.fHPD);
	  mresEMF = map_of_MEs[DirName+"/"+"resEMF"]; if (mresEMF && mresEMF->getRootObject())     mresEMF->Fill (jetID.restrictedEMF);
	  mfRBX = map_of_MEs[DirName+"/"+"fRBX"]; if (mfRBX && mfRBX->getRootObject())             mfRBX->Fill (jetID.fRBX);
	}
      }
      if(jetCleaningFlag_){
	Thiscleaned=jetpassid;
      }
      if(Thiscleaned && pass_uncorrected){
	mPt_uncor = map_of_MEs[DirName+"/"+"Pt_uncor"]; if (mPt_uncor && mPt_uncor->getRootObject())   mPt_uncor->Fill ((*caloJets)[ijet].pt());
	mEta_uncor = map_of_MEs[DirName+"/"+"Eta_uncor"]; if (mEta_uncor && mEta_uncor->getRootObject()) mEta_uncor->Fill ((*caloJets)[ijet].eta());
	mPhi_uncor = map_of_MEs[DirName+"/"+"Phi_uncor"]; if (mPhi_uncor && mPhi_uncor->getRootObject()) mPhi_uncor->Fill ((*caloJets)[ijet].phi());
	mConstituents_uncor = map_of_MEs[DirName+"/"+"Constituents_uncor"]; if (mConstituents_uncor && mConstituents_uncor->getRootObject()) mConstituents_uncor->Fill ((*caloJets)[ijet].nConstituents());
      }
      //now do calojet specific fractions and histograms ->H and E fracs
      if(Thiscleaned && pass_corrected){//if cleaning requested->jet passes a loose ID
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject())        mHFrac->Fill ((*caloJets)[ijet].energyFractionHadronic());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject())        mEFrac->Fill ((*caloJets)[ijet].emEnergyFraction());
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject())        mHFrac_profile       ->Fill(numPV, (*caloJets)[ijet].energyFractionHadronic());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject())        mEFrac_profile       ->Fill(numPV, (*caloJets)[ijet].emEnergyFraction());
	if (fabs((*caloJets)[ijet].eta()) <= 1.3) {	
	  mHFrac_Barrel = map_of_MEs[DirName+"/"+"HFrac_Barrel"]; if (mHFrac_Barrel && mHFrac_Barrel->getRootObject())           mHFrac_Barrel->Fill((*caloJets)[ijet].energyFractionHadronic());	
	  mEFrac_Barrel = map_of_MEs[DirName+"/"+"EFrac_Barrel"]; if (mEFrac_Barrel && mEFrac_Barrel->getRootObject())           mEFrac_Barrel->Fill((*caloJets)[ijet].emEnergyFraction());	
	}else if(fabs((*caloJets)[ijet].eta()) <3.0){
	  mHFrac_EndCap = map_of_MEs[DirName+"/"+"HFrac_EndCap"]; if (mHFrac_EndCap && mHFrac_EndCap->getRootObject())           mHFrac_EndCap->Fill((*caloJets)[ijet].energyFractionHadronic());	
	  mEFrac_EndCap = map_of_MEs[DirName+"/"+"EFrac_EndCap"]; if (mEFrac_EndCap && mEFrac_EndCap->getRootObject())           mEFrac_EndCap->Fill((*caloJets)[ijet].emEnergyFraction());
	}else{
	  mHFrac_Forward = map_of_MEs[DirName+"/"+"HFrac_Forward"]; if (mHFrac_Forward && mHFrac_Forward->getRootObject())           mHFrac_Forward->Fill((*caloJets)[ijet].energyFractionHadronic());	
	  mEFrac_Forward = map_of_MEs[DirName+"/"+"EFrac_Forward"]; if (mEFrac_Forward && mEFrac_Forward->getRootObject())           mEFrac_Forward->Fill((*caloJets)[ijet].emEnergyFraction());
	}	
	mHadEnergyInHO = map_of_MEs[DirName+"/"+"HadEnergyInHO"]; if (mHadEnergyInHO && mHadEnergyInHO->getRootObject())   mHadEnergyInHO->Fill ((*caloJets)[ijet].hadEnergyInHO());
	mHadEnergyInHB = map_of_MEs[DirName+"/"+"HadEnergyInHB"]; if (mHadEnergyInHB && mHadEnergyInHB->getRootObject())   mHadEnergyInHB->Fill ((*caloJets)[ijet].hadEnergyInHB());
	mHadEnergyInHF = map_of_MEs[DirName+"/"+"HadEnergyInHF"]; if (mHadEnergyInHF && mHadEnergyInHF->getRootObject())   mHadEnergyInHF->Fill ((*caloJets)[ijet].hadEnergyInHF());
	mHadEnergyInHE = map_of_MEs[DirName+"/"+"HadEnergyInHE"]; if (mHadEnergyInHE && mHadEnergyInHE->getRootObject())   mHadEnergyInHE->Fill ((*caloJets)[ijet].hadEnergyInHE());
	mEmEnergyInEB = map_of_MEs[DirName+"/"+"EmEnergyInEB"]; if (mEmEnergyInEB && mEmEnergyInEB->getRootObject())    mEmEnergyInEB->Fill ((*caloJets)[ijet].emEnergyInEB());
	mEmEnergyInEE = map_of_MEs[DirName+"/"+"EmEnergyInEE"]; if (mEmEnergyInEE && mEmEnergyInEE->getRootObject())    mEmEnergyInEE->Fill ((*caloJets)[ijet].emEnergyInEE());
	mEmEnergyInHF = map_of_MEs[DirName+"/"+"EmEnergyInHF"]; if (mEmEnergyInHF && mEmEnergyInHF->getRootObject())    mEmEnergyInHF->Fill ((*caloJets)[ijet].emEnergyInHF());
	
      }
    }
    if(isMiniAODJet_ && (*patJets)[ijet].isPFJet()){
      pat::strbitset stringbitset=pfjetIDFunctor.getBitTemplate();
      jetpassid = pfjetIDFunctor((*patJets)[ijet],stringbitset);
      if(jetCleaningFlag_){
	Thiscleaned = jetpassid;
	JetIDWPU = jetpassid;
      }
      if(Thiscleaned && pass_uncorrected){
	mPt_uncor = map_of_MEs[DirName+"/"+"Pt_uncor"]; if (mPt_uncor && mPt_uncor->getRootObject()) if (mPt_uncor)   mPt_uncor->Fill ((*patJets)[ijet].pt()*(*patJets)[ijet].jecFactor("Uncorrected"));
	mEta_uncor = map_of_MEs[DirName+"/"+"Eta_uncor"]; if (mEta_uncor && mEta_uncor->getRootObject()) if (mEta_uncor)  mEta_uncor->Fill ((*patJets)[ijet].eta());
	mPhi_uncor = map_of_MEs[DirName+"/"+"Phi_uncor"]; if (mPhi_uncor && mPhi_uncor->getRootObject()) if (mPhi_uncor)  mPhi_uncor->Fill ((*patJets)[ijet].phi());
	mConstituents_uncor = map_of_MEs[DirName+"/"+"Constituents_uncor"]; if (mConstituents_uncor && mConstituents_uncor->getRootObject()) if (mConstituents_uncor) mConstituents_uncor->Fill ((*patJets)[ijet].nConstituents());
      }
      if(Thiscleaned && pass_corrected){
	if(fabs(correctedJet.eta()) <= 1.3) {
	  if(correctedJet.pt()<=50.){
	    mMVAPUJIDDiscriminant_lowPt_Barrel=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_lowPt_Barrel"]; if(mMVAPUJIDDiscriminant_lowPt_Barrel && mMVAPUJIDDiscriminant_lowPt_Barrel->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_lowPt_Barrel->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant")); } 
	  }
	  if (correctedJet.pt()>50. && correctedJet.pt()<=140.) {
	    mMVAPUJIDDiscriminant_mediumPt_Barrel=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_Barrel"]; if(mMVAPUJIDDiscriminant_mediumPt_Barrel && mMVAPUJIDDiscriminant_mediumPt_Barrel->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_mediumPt_Barrel->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant")); }
	  }
	  if(correctedJet.pt()>140.){
	    mMVAPUJIDDiscriminant_highPt_Barrel=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_highPt_Barrel"]; if(mMVAPUJIDDiscriminant_highPt_Barrel && mMVAPUJIDDiscriminant_highPt_Barrel->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_highPt_Barrel->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant")); }
	  }
	  mCHFracVSpT_Barrel = map_of_MEs[DirName+"/"+"CHFracVSpT_Barrel"]; if(mCHFracVSpT_Barrel && mCHFracVSpT_Barrel->getRootObject()) mCHFracVSpT_Barrel->Fill(correctedJet.pt(),(*patJets)[ijet].chargedHadronEnergyFraction());
	  mNHFracVSpT_Barrel = map_of_MEs[DirName+"/"+"NHFracVSpT_Barrel"];if (mNHFracVSpT_Barrel && mNHFracVSpT_Barrel->getRootObject()) mNHFracVSpT_Barrel->Fill(correctedJet.pt(),(*patJets)[ijet].neutralHadronEnergyFraction());
	  mPhFracVSpT_Barrel = map_of_MEs[DirName+"/"+"PhFracVSpT_Barrel"];if (mPhFracVSpT_Barrel && mPhFracVSpT_Barrel->getRootObject()) mPhFracVSpT_Barrel->Fill(correctedJet.pt(),(*patJets)[ijet].neutralEmEnergyFraction());
	}else if(fabs(correctedJet.eta()) <= 3) {
	  if(correctedJet.pt()<=50.){
	    mMVAPUJIDDiscriminant_lowPt_EndCap=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_lowPt_EndCap"]; if(mMVAPUJIDDiscriminant_lowPt_EndCap && mMVAPUJIDDiscriminant_lowPt_EndCap->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_lowPt_EndCap->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant")); } 
	  }
	  if (correctedJet.pt()>50. && correctedJet.pt()<=140.) {
	    mMVAPUJIDDiscriminant_mediumPt_EndCap=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_EndCap"]; if(mMVAPUJIDDiscriminant_mediumPt_EndCap && mMVAPUJIDDiscriminant_mediumPt_EndCap->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_mediumPt_EndCap->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant")); }
	  }
	  if(correctedJet.pt()>140.){
	    mMVAPUJIDDiscriminant_highPt_EndCap=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_highPt_EndCap"]; if(mMVAPUJIDDiscriminant_highPt_EndCap && mMVAPUJIDDiscriminant_highPt_EndCap->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_highPt_EndCap->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant")); }
	  }
	  mCHFracVSpT_EndCap = map_of_MEs[DirName+"/"+"CHFracVSpT_EndCap"]; if(mCHFracVSpT_EndCap && mCHFracVSpT_EndCap->getRootObject()) mCHFracVSpT_EndCap->Fill(correctedJet.pt(),(*patJets)[ijet].chargedHadronEnergyFraction());
	  mNHFracVSpT_EndCap = map_of_MEs[DirName+"/"+"NHFracVSpT_EndCap"];if (mNHFracVSpT_EndCap && mNHFracVSpT_EndCap->getRootObject()) mNHFracVSpT_EndCap->Fill(correctedJet.pt(),(*patJets)[ijet].neutralHadronEnergyFraction());
	  mPhFracVSpT_EndCap = map_of_MEs[DirName+"/"+"PhFracVSpT_EndCap"];if (mPhFracVSpT_EndCap && mPhFracVSpT_EndCap->getRootObject()) mPhFracVSpT_EndCap->Fill(correctedJet.pt(),(*patJets)[ijet].neutralEmEnergyFraction());
	}else if(fabs(correctedJet.eta()) <= 5) {
	  if(correctedJet.pt()<=50.){
	    mMVAPUJIDDiscriminant_lowPt_Forward=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_lowPt_Forward"]; if(mMVAPUJIDDiscriminant_lowPt_Forward && mMVAPUJIDDiscriminant_lowPt_Forward->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_lowPt_Forward->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant")); } 
	  }
	  if (correctedJet.pt()>50. && correctedJet.pt()<=140.) {
	    mMVAPUJIDDiscriminant_mediumPt_Forward=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_Forward"]; if(mMVAPUJIDDiscriminant_mediumPt_Forward && mMVAPUJIDDiscriminant_mediumPt_Forward->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_mediumPt_Forward->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant"));} 
	  }
	  if(correctedJet.pt()>140.){
	    mMVAPUJIDDiscriminant_highPt_Forward=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_highPt_Forward"]; if(mMVAPUJIDDiscriminant_highPt_Forward && mMVAPUJIDDiscriminant_highPt_Forward->getRootObject()){if((*patJets)[ijet].hasUserFloat("pileupJetId:fullDiscriminant"))  mMVAPUJIDDiscriminant_highPt_Forward->Fill( (*patJets)[ijet].userFloat("pileupJetId:fullDiscriminant"));} 
	  }
	  mHFHFracVSpT_Forward = map_of_MEs[DirName+"/"+"HFHFracVSpT_Forward"]; if (mHFHFracVSpT_Forward && mHFHFracVSpT_Forward->getRootObject())    mHFHFracVSpT_Forward->Fill(correctedJet.pt(),(*patJets)[ijet].HFHadronEnergyFraction ());	
	  mHFEFracVSpT_Forward = map_of_MEs[DirName+"/"+"HFEFracVSpT_Forward"]; if (mHFEFracVSpT_Forward && mHFEFracVSpT_Forward->getRootObject())    mHFEFracVSpT_Forward->Fill (correctedJet.pt(),(*patJets)[ijet].HFEMEnergyFraction ());
	}
      }
    }
    if(isPFJet_){
      reco::PFJetRef pfjetref(pfJets, ijet);
      float puidmva=-1;
      float puidcut=-1;
      int puidmvaflag=-10;
      int puidcutflag=-10;
      puidmva=(*puJetIdMva)[pfjetref];
      puidcut=(*puJetId)[pfjetref];
      puidmvaflag=(*puJetIdFlagMva)[pfjetref];
      puidcutflag=(*puJetIdFlag)[pfjetref];
      jetpassid = pfjetIDFunctor((*pfJets)[ijet]);
      if((*pfJets)[ijet].muonEnergyFraction()>0.8){
	jetpassid =false;
      }
      //int QGmulti=-1;
      //float QGLikelihood=-10;
      //float QGptD=-10;
      //float QGaxis2=-10;
      //if(fill_CHS_histos){
      //QGmulti=(*qgMultiplicity)[pfjetref];
      //QGLikelihood=(*qgLikelihood)[pfjetref];
      //QGptD=(*qgptD)[pfjetref];
      //QGaxis2=(*qgaxis2)[pfjetref];
      //}
      if(jetCleaningFlag_){
	Thiscleaned = jetpassid;
	JetIDWPU= (jetpassid && PileupJetIdentifier::passJetId( puidmvaflag, PileupJetIdentifier::kLoose ));
      }
      if(Thiscleaned && pass_uncorrected){
	mPt_uncor = map_of_MEs[DirName+"/"+"Pt_uncor"]; if (mPt_uncor && mPt_uncor->getRootObject())  mPt_uncor->Fill ((*pfJets)[ijet].pt());
	mEta_uncor = map_of_MEs[DirName+"/"+"Eta_uncor"]; if (mEta_uncor && mEta_uncor->getRootObject()) mEta_uncor->Fill ((*pfJets)[ijet].eta());
	mPhi_uncor = map_of_MEs[DirName+"/"+"Phi_uncor"]; if (mPhi_uncor && mPhi_uncor->getRootObject())  mPhi_uncor->Fill ((*pfJets)[ijet].phi());
	mConstituents_uncor = map_of_MEs[DirName+"/"+"Constituents_uncor"]; if (mConstituents_uncor && mConstituents_uncor->getRootObject()) mConstituents_uncor->Fill ((*pfJets)[ijet].nConstituents());
      }
      if(Thiscleaned && pass_corrected){
	if(PileupJetIdentifier::passJetId( puidcutflag, PileupJetIdentifier::kLoose )) {
	  mLooseCutPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"LooseCutPUIDPassFractionVSeta"];if(mLooseCutPUJIDPassFractionVSeta && mLooseCutPUJIDPassFractionVSeta->getRootObject()) mLooseCutPUJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	  mLooseCutPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"LooseCutPUIDPassFractionVSpt"];if(mLooseCutPUJIDPassFractionVSpt && mLooseCutPUJIDPassFractionVSpt->getRootObject()) mLooseCutPUJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	}else{
	  mLooseCutPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"LooseCutPUIDPassFractionVSeta"];if(mLooseCutPUJIDPassFractionVSeta && mLooseCutPUJIDPassFractionVSeta->getRootObject()) mLooseCutPUJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	  mLooseCutPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"LooseCutPUIDPassFractionVSpt"];if(mLooseCutPUJIDPassFractionVSpt && mLooseCutPUJIDPassFractionVSpt->getRootObject()) mLooseCutPUJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	}
	if(PileupJetIdentifier::passJetId( puidcutflag, PileupJetIdentifier::kMedium )) {
	  mMediumCutPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"MediumCutPUIDPassFractionVSeta"];if(mMediumCutPUJIDPassFractionVSeta && mMediumCutPUJIDPassFractionVSeta->getRootObject()) mMediumCutPUJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	  mMediumCutPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"MediumCutPUIDPassFractionVSpt"];if(mMediumCutPUJIDPassFractionVSpt && mMediumCutPUJIDPassFractionVSpt->getRootObject()) mMediumCutPUJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	}else{
	  mMediumCutPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"MediumCutPUIDPassFractionVSeta"];if(mMediumCutPUJIDPassFractionVSeta && mMediumCutPUJIDPassFractionVSeta->getRootObject()) mMediumCutPUJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	  mMediumCutPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"MediumCutPUIDPassFractionVSpt"];if(mMediumCutPUJIDPassFractionVSpt && mMediumCutPUJIDPassFractionVSpt->getRootObject()) mMediumCutPUJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	}
	if(PileupJetIdentifier::passJetId( puidcutflag, PileupJetIdentifier::kTight )) {
	  mTightCutPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"TightCutPUIDPassFractionVSeta"];if(mTightCutPUJIDPassFractionVSeta && mTightCutPUJIDPassFractionVSeta->getRootObject()) mTightCutPUJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	  mTightCutPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"TightCutPUIDPassFractionVSpt"];if(mTightCutPUJIDPassFractionVSpt && mTightCutPUJIDPassFractionVSpt->getRootObject()) mTightCutPUJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	}else{
	  mTightCutPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"TightCutPUIDPassFractionVSeta"];if(mTightCutPUJIDPassFractionVSeta && mTightCutPUJIDPassFractionVSeta->getRootObject()) mTightCutPUJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	  mTightCutPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"TightCutPUIDPassFractionVSpt"];if(mTightCutPUJIDPassFractionVSpt && mTightCutPUJIDPassFractionVSpt->getRootObject()) mTightCutPUJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	}
	if(PileupJetIdentifier::passJetId( puidmvaflag, PileupJetIdentifier::kLoose )) {
	  mLooseMVAPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"LooseMVAPUIDPassFractionVSeta"];if(mLooseMVAPUJIDPassFractionVSeta && mLooseMVAPUJIDPassFractionVSeta->getRootObject()) mLooseMVAPUJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	  mLooseMVAPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"LooseMVAPUIDPassFractionVSpt"];if(mLooseMVAPUJIDPassFractionVSpt && mLooseMVAPUJIDPassFractionVSpt->getRootObject()) mLooseMVAPUJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	}else{
	  mLooseMVAPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"LooseMVAPUIDPassFractionVSeta"];if(mLooseMVAPUJIDPassFractionVSeta && mLooseMVAPUJIDPassFractionVSeta->getRootObject()) mLooseMVAPUJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	  mLooseMVAPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"LooseMVAPUIDPassFractionVSpt"];if(mLooseMVAPUJIDPassFractionVSpt && mLooseMVAPUJIDPassFractionVSpt->getRootObject()) mLooseMVAPUJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	}
	if(PileupJetIdentifier::passJetId( puidmvaflag, PileupJetIdentifier::kMedium )) {
	  mMediumMVAPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"MediumMVAPUIDPassFractionVSeta"];if(mMediumMVAPUJIDPassFractionVSeta && mMediumMVAPUJIDPassFractionVSeta->getRootObject()) mMediumMVAPUJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	  mMediumMVAPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"MediumMVAPUIDPassFractionVSpt"];if(mMediumMVAPUJIDPassFractionVSpt && mMediumMVAPUJIDPassFractionVSpt->getRootObject()) mMediumMVAPUJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	}else{
	  mMediumMVAPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"MediumMVAPUIDPassFractionVSeta"];if(mMediumMVAPUJIDPassFractionVSeta && mMediumMVAPUJIDPassFractionVSeta->getRootObject()) mMediumMVAPUJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	  mMediumMVAPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"MediumMVAPUIDPassFractionVSpt"];if(mMediumMVAPUJIDPassFractionVSpt && mMediumMVAPUJIDPassFractionVSpt->getRootObject()) mMediumMVAPUJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	}
	if(PileupJetIdentifier::passJetId( puidmvaflag, PileupJetIdentifier::kTight )) {
	  mTightMVAPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"TightMVAPUIDPassFractionVSeta"];if(mTightMVAPUJIDPassFractionVSeta && mTightMVAPUJIDPassFractionVSeta->getRootObject()) mTightMVAPUJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	  mTightMVAPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"TightMVAPUIDPassFractionVSpt"];if(mTightMVAPUJIDPassFractionVSpt && mTightMVAPUJIDPassFractionVSpt->getRootObject()) mTightMVAPUJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	}else{
	  mTightMVAPUJIDPassFractionVSeta  = map_of_MEs[DirName+"/"+"TightMVAPUIDPassFractionVSeta"];if(mTightMVAPUJIDPassFractionVSeta && mTightMVAPUJIDPassFractionVSeta->getRootObject()) mTightMVAPUJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	  mTightMVAPUJIDPassFractionVSpt  = map_of_MEs[DirName+"/"+"TightMVAPUIDPassFractionVSpt"];if(mTightMVAPUJIDPassFractionVSpt && mTightMVAPUJIDPassFractionVSpt->getRootObject()) mTightMVAPUJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	}
	if (correctedJet.pt()<= 50) {
	  mCHFracVSeta_lowPt = map_of_MEs[DirName+"/"+"CHFracVSeta_lowPt"]; if (mCHFracVSeta_lowPt &&  mCHFracVSeta_lowPt->getRootObject()) mCHFracVSeta_lowPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	  mNHFracVSeta_lowPt = map_of_MEs[DirName+"/"+"NHFracVSeta_lowPt"]; if (mNHFracVSeta_lowPt &&  mNHFracVSeta_lowPt->getRootObject()) mNHFracVSeta_lowPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	  mPhFracVSeta_lowPt = map_of_MEs[DirName+"/"+"PhFracVSeta_lowPt"]; if (mPhFracVSeta_lowPt &&  mPhFracVSeta_lowPt->getRootObject()) mPhFracVSeta_lowPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralEmEnergyFraction());
	}
	if (correctedJet.pt()>50. && correctedJet.pt()<=140.) {
	  mCHFracVSeta_mediumPt = map_of_MEs[DirName+"/"+"CHFracVSeta_mediumPt"]; if (mCHFracVSeta_mediumPt &&  mCHFracVSeta_mediumPt->getRootObject()) mCHFracVSeta_mediumPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	  mNHFracVSeta_mediumPt = map_of_MEs[DirName+"/"+"NHFracVSeta_mediumPt"]; if (mNHFracVSeta_mediumPt &&  mNHFracVSeta_mediumPt->getRootObject()) mNHFracVSeta_mediumPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	  mPhFracVSeta_mediumPt = map_of_MEs[DirName+"/"+"PhFracVSeta_mediumPt"]; if (mPhFracVSeta_mediumPt &&  mPhFracVSeta_mediumPt->getRootObject()) mPhFracVSeta_mediumPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralEmEnergyFraction());
	}
	if (correctedJet.pt()>140.) {
	  mCHFracVSeta_highPt = map_of_MEs[DirName+"/"+"CHFracVSeta_highPt"]; if (mCHFracVSeta_highPt &&  mCHFracVSeta_highPt->getRootObject()) mCHFracVSeta_highPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	  mNHFracVSeta_highPt = map_of_MEs[DirName+"/"+"NHFracVSeta_highPt"]; if (mNHFracVSeta_highPt &&  mNHFracVSeta_highPt->getRootObject()) mNHFracVSeta_highPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	  mPhFracVSeta_highPt = map_of_MEs[DirName+"/"+"PhFracVSeta_highPt"]; if (mPhFracVSeta_highPt &&  mPhFracVSeta_highPt->getRootObject()) mPhFracVSeta_highPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralEmEnergyFraction());
	}
	if (fabs(correctedJet.eta()) <= 1.3) {
	  //fractions for barrel
	  if (correctedJet.pt()<=50.) {
	    //mAxis2_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_Barrel"];if(mAxis2_lowPt_Barrel && mAxis2_lowPt_Barrel->getRootObject()) mAxis2_lowPt_Barrel->Fill(QGaxis2);
	    //mpTD_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_Barrel"]; if(mpTD_lowPt_Barrel && mpTD_lowPt_Barrel->getRootObject()) mpTD_lowPt_Barrel->Fill(QGptD);
	    //mMultiplicityQG_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_Barrel"]; if(mMultiplicityQG_lowPt_Barrel && mMultiplicityQG_lowPt_Barrel->getRootObject()) mMultiplicityQG_lowPt_Barrel->Fill(QGmulti);
	    //mqgLikelihood_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_Barrel"]; if(mqgLikelihood_lowPt_Barrel && mqgLikelihood_lowPt_Barrel->getRootObject()) mqgLikelihood_lowPt_Barrel->Fill(QGLikelihood);
	    mMass_lowPt_Barrel=map_of_MEs[DirName+"/"+"JetMass_lowPt_Barrel"]; if(mMass_lowPt_Barrel && mMass_lowPt_Barrel->getRootObject())mMass_lowPt_Barrel->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_lowPt_Barrel=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_lowPt_Barrel"]; if(mMVAPUJIDDiscriminant_lowPt_Barrel && mMVAPUJIDDiscriminant_lowPt_Barrel->getRootObject()) mMVAPUJIDDiscriminant_lowPt_Barrel->Fill(puidmva); 
	    mCutPUJIDDiscriminant_lowPt_Barrel=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_lowPt_Barrel"]; if(mCutPUJIDDiscriminant_lowPt_Barrel && mCutPUJIDDiscriminant_lowPt_Barrel->getRootObject()) mCutPUJIDDiscriminant_lowPt_Barrel->Fill(puidcut); 
	    mCHFrac_lowPt_Barrel = map_of_MEs[DirName+"/"+"CHFrac_lowPt_Barrel"]; if (mCHFrac_lowPt_Barrel &&  mCHFrac_lowPt_Barrel->getRootObject()) mCHFrac_lowPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mNHFrac_lowPt_Barrel = map_of_MEs[DirName+"/"+"NHFrac_lowPt_Barrel"]; if (mNHFrac_lowPt_Barrel &&  mNHFrac_lowPt_Barrel->getRootObject()) mNHFrac_lowPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    mPhFrac_lowPt_Barrel = map_of_MEs[DirName+"/"+"PhFrac_lowPt_Barrel"]; if (mPhFrac_lowPt_Barrel &&  mPhFrac_lowPt_Barrel->getRootObject()) mPhFrac_lowPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	    mCHEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"CHEn_lowPt_Barrel"]; if (mCHEn_lowPt_Barrel &&  mCHEn_lowPt_Barrel->getRootObject()) mCHEn_lowPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergy());
	    mNHEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"NHEn_lowPt_Barrel"]; if (mNHEn_lowPt_Barrel &&  mNHEn_lowPt_Barrel->getRootObject()) mNHEn_lowPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergy());
	    mPhEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"PhEn_lowPt_Barrel"]; if (mPhEn_lowPt_Barrel &&  mPhEn_lowPt_Barrel->getRootObject()) mPhEn_lowPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergy());
	    mElEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"ElEn_lowPt_Barrel"]; if (mElEn_lowPt_Barrel &&  mElEn_lowPt_Barrel->getRootObject()) mElEn_lowPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergy());
	    mMuEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"MuEn_lowPt_Barrel"]; if (mMuEn_lowPt_Barrel &&  mMuEn_lowPt_Barrel->getRootObject()) mMuEn_lowPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergy());
	    mChMultiplicity_lowPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_lowPt_Barrel"]; if(mChMultiplicity_lowPt_Barrel && mChMultiplicity_lowPt_Barrel->getRootObject())  mChMultiplicity_lowPt_Barrel->Fill((*pfJets)[ijet].chargedMultiplicity());
	    mNeutMultiplicity_lowPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_lowPt_Barrel"]; if(mNeutMultiplicity_lowPt_Barrel && mNeutMultiplicity_lowPt_Barrel->getRootObject())  mNeutMultiplicity_lowPt_Barrel->Fill((*pfJets)[ijet].neutralMultiplicity());
	    mMuMultiplicity_lowPt_Barrel = map_of_MEs[DirName+"/"+"MuMultiplicity_lowPt_Barrel"]; if(mMuMultiplicity_lowPt_Barrel && mMuMultiplicity_lowPt_Barrel->getRootObject())  mMuMultiplicity_lowPt_Barrel->Fill((*pfJets)[ijet].muonMultiplicity());
	  }
	  if (correctedJet.pt()>50. && correctedJet.pt()<=140.) {
	    //mAxis2_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_Barrel"];if(mAxis2_mediumPt_Barrel && mAxis2_mediumPt_Barrel->getRootObject()) mAxis2_mediumPt_Barrel->Fill(QGaxis2);
	    //mpTD_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_Barrel"]; if(mpTD_mediumPt_Barrel && mpTD_mediumPt_Barrel->getRootObject()) mpTD_mediumPt_Barrel->Fill(QGptD);
	    //mMultiplicityQG_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_Barrel"]; if(mMultiplicityQG_mediumPt_Barrel && mMultiplicityQG_mediumPt_Barrel->getRootObject()) mMultiplicityQG_mediumPt_Barrel->Fill(QGmulti);
	    //mqgLikelihood_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_Barrel"]; if(mqgLikelihood_mediumPt_Barrel && mqgLikelihood_mediumPt_Barrel->getRootObject()) mqgLikelihood_mediumPt_Barrel->Fill(QGLikelihood);
	    mMass_mediumPt_Barrel=map_of_MEs[DirName+"/"+"JetMass_mediumPt_Barrel"]; if(mMass_mediumPt_Barrel && mMass_mediumPt_Barrel->getRootObject())mMass_mediumPt_Barrel->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_mediumPt_Barrel=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_Barrel"]; if(mMVAPUJIDDiscriminant_mediumPt_Barrel && mMVAPUJIDDiscriminant_mediumPt_Barrel->getRootObject()) mMVAPUJIDDiscriminant_mediumPt_Barrel->Fill(puidmva); 
	    mCutPUJIDDiscriminant_mediumPt_Barrel=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_mediumPt_Barrel"]; if(mCutPUJIDDiscriminant_mediumPt_Barrel && mCutPUJIDDiscriminant_mediumPt_Barrel->getRootObject()) mCutPUJIDDiscriminant_mediumPt_Barrel->Fill(puidcut); 
	    mCHFrac_mediumPt_Barrel = map_of_MEs[DirName+"/"+"CHFrac_mediumPt_Barrel"]; if (mCHFrac_mediumPt_Barrel &&  mCHFrac_mediumPt_Barrel->getRootObject()) mCHFrac_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mNHFrac_mediumPt_Barrel = map_of_MEs[DirName+"/"+"NHFrac_mediumPt_Barrel"]; if (mNHFrac_mediumPt_Barrel &&  mNHFrac_mediumPt_Barrel->getRootObject()) mNHFrac_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    mPhFrac_mediumPt_Barrel = map_of_MEs[DirName+"/"+"PhFrac_mediumPt_Barrel"]; if (mPhFrac_mediumPt_Barrel &&  mPhFrac_mediumPt_Barrel->getRootObject()) mPhFrac_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	    mCHEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"CHEn_mediumPt_Barrel"]; if (mCHEn_mediumPt_Barrel &&  mCHEn_mediumPt_Barrel->getRootObject()) mCHEn_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergy());
	    mNHEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"NHEn_mediumPt_Barrel"]; if (mNHEn_mediumPt_Barrel &&  mNHEn_mediumPt_Barrel->getRootObject()) mNHEn_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergy());
	    mPhEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"PhEn_mediumPt_Barrel"]; if (mPhEn_mediumPt_Barrel &&  mPhEn_mediumPt_Barrel->getRootObject()) mPhEn_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergy());
	    mElEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"ElEn_mediumPt_Barrel"]; if (mElEn_mediumPt_Barrel &&  mElEn_mediumPt_Barrel->getRootObject()) mElEn_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergy());
	    mMuEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"MuEn_mediumPt_Barrel"]; if (mMuEn_mediumPt_Barrel &&  mMuEn_mediumPt_Barrel->getRootObject()) mMuEn_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergy());
	    mChMultiplicity_mediumPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_mediumPt_Barrel"]; if(mChMultiplicity_mediumPt_Barrel && mChMultiplicity_mediumPt_Barrel->getRootObject())  mChMultiplicity_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedMultiplicity());
	    mNeutMultiplicity_mediumPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_mediumPt_Barrel"]; if(mNeutMultiplicity_mediumPt_Barrel && mNeutMultiplicity_mediumPt_Barrel->getRootObject())  mNeutMultiplicity_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralMultiplicity());
	    mMuMultiplicity_mediumPt_Barrel = map_of_MEs[DirName+"/"+"MuMultiplicity_mediumPt_Barrel"]; if(mMuMultiplicity_mediumPt_Barrel && mMuMultiplicity_mediumPt_Barrel->getRootObject())  mMuMultiplicity_mediumPt_Barrel->Fill((*pfJets)[ijet].muonMultiplicity());
	  }
	  if (correctedJet.pt()>140.) {
	    //mAxis2_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_Barrel"];if(mAxis2_highPt_Barrel && mAxis2_highPt_Barrel->getRootObject()) mAxis2_highPt_Barrel->Fill(QGaxis2);
	    //mpTD_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_highPt_Barrel"]; if(mpTD_highPt_Barrel && mpTD_highPt_Barrel->getRootObject()) mpTD_highPt_Barrel->Fill(QGptD);
	    //mMultiplicityQG_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_Barrel"]; if(mMultiplicityQG_highPt_Barrel && mMultiplicityQG_highPt_Barrel->getRootObject()) mMultiplicityQG_highPt_Barrel->Fill(QGmulti);
	    //mqgLikelihood_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_Barrel"]; if(mqgLikelihood_highPt_Barrel && mqgLikelihood_highPt_Barrel->getRootObject()) mqgLikelihood_highPt_Barrel->Fill(QGLikelihood);
	    mMass_highPt_Barrel=map_of_MEs[DirName+"/"+"JetMass_highPt_Barrel"]; if(mMass_highPt_Barrel && mMass_highPt_Barrel->getRootObject())mMass_highPt_Barrel->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_highPt_Barrel=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_highPt_Barrel"]; if(mMVAPUJIDDiscriminant_highPt_Barrel && mMVAPUJIDDiscriminant_highPt_Barrel->getRootObject()) mMVAPUJIDDiscriminant_highPt_Barrel->Fill(puidmva); 
	    mCutPUJIDDiscriminant_highPt_Barrel=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_highPt_Barrel"]; if(mCutPUJIDDiscriminant_highPt_Barrel && mCutPUJIDDiscriminant_highPt_Barrel->getRootObject()) mCutPUJIDDiscriminant_highPt_Barrel->Fill(puidcut); 
	    mCHFrac_highPt_Barrel = map_of_MEs[DirName+"/"+"CHFrac_highPt_Barrel"]; if (mCHFrac_highPt_Barrel &&  mCHFrac_highPt_Barrel->getRootObject()) mCHFrac_highPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mNHFrac_highPt_Barrel = map_of_MEs[DirName+"/"+"NHFrac_highPt_Barrel"]; if (mNHFrac_highPt_Barrel &&  mNHFrac_highPt_Barrel->getRootObject()) mNHFrac_highPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    mPhFrac_highPt_Barrel = map_of_MEs[DirName+"/"+"PhFrac_highPt_Barrel"]; if (mPhFrac_highPt_Barrel &&  mPhFrac_highPt_Barrel->getRootObject()) mPhFrac_highPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	    mCHEn_highPt_Barrel = map_of_MEs[DirName+"/"+"CHEn_highPt_Barrel"]; if (mCHEn_highPt_Barrel &&  mCHEn_highPt_Barrel->getRootObject()) mCHEn_highPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergy());
	    mNHEn_highPt_Barrel = map_of_MEs[DirName+"/"+"NHEn_highPt_Barrel"]; if (mNHEn_highPt_Barrel &&  mNHEn_highPt_Barrel->getRootObject()) mNHEn_highPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergy());
	    mPhEn_highPt_Barrel = map_of_MEs[DirName+"/"+"PhEn_highPt_Barrel"]; if (mPhEn_highPt_Barrel &&  mPhEn_highPt_Barrel->getRootObject()) mPhEn_highPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergy());
	    mElEn_highPt_Barrel = map_of_MEs[DirName+"/"+"ElEn_highPt_Barrel"]; if (mElEn_highPt_Barrel &&  mElEn_highPt_Barrel->getRootObject()) mElEn_highPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergy());
	    mMuEn_highPt_Barrel = map_of_MEs[DirName+"/"+"MuEn_highPt_Barrel"]; if (mMuEn_highPt_Barrel &&  mMuEn_highPt_Barrel->getRootObject()) mMuEn_highPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergy());
	    mChMultiplicity_highPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_highPt_Barrel"]; if(mChMultiplicity_highPt_Barrel && mChMultiplicity_highPt_Barrel->getRootObject())  mChMultiplicity_highPt_Barrel->Fill((*pfJets)[ijet].chargedMultiplicity());
	    mNeutMultiplicity_highPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_highPt_Barrel"]; if(mNeutMultiplicity_highPt_Barrel && mNeutMultiplicity_highPt_Barrel->getRootObject())  mNeutMultiplicity_highPt_Barrel->Fill((*pfJets)[ijet].neutralMultiplicity());
	    mMuMultiplicity_highPt_Barrel = map_of_MEs[DirName+"/"+"MuMultiplicity_highPt_Barrel"]; if(mMuMultiplicity_highPt_Barrel && mMuMultiplicity_highPt_Barrel->getRootObject())  mMuMultiplicity_highPt_Barrel->Fill((*pfJets)[ijet].muonMultiplicity());
	  }
	  mCHFracVSpT_Barrel = map_of_MEs[DirName+"/"+"CHFracVSpT_Barrel"]; if(mCHFracVSpT_Barrel && mCHFracVSpT_Barrel->getRootObject()) mCHFracVSpT_Barrel->Fill(correctedJet.pt(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	  mNHFracVSpT_Barrel = map_of_MEs[DirName+"/"+"NHFracVSpT_Barrel"];if (mNHFracVSpT_Barrel && mNHFracVSpT_Barrel->getRootObject()) mNHFracVSpT_Barrel->Fill(correctedJet.pt(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	  mPhFracVSpT_Barrel = map_of_MEs[DirName+"/"+"PhFracVSpT_Barrel"];if (mPhFracVSpT_Barrel && mPhFracVSpT_Barrel->getRootObject()) mPhFracVSpT_Barrel->Fill(correctedJet.pt(),(*pfJets)[ijet].neutralEmEnergyFraction());
	}else if(fabs(correctedJet.eta()) <= 3) {
	  //fractions for endcap
	  if (correctedJet.pt()<=50.) {
	    //mAxis2_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_EndCap"];if(mAxis2_lowPt_EndCap && mAxis2_lowPt_EndCap->getRootObject()) mAxis2_lowPt_EndCap->Fill(QGaxis2);
	    //mpTD_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_EndCap"]; if(mpTD_lowPt_EndCap && mpTD_lowPt_EndCap->getRootObject()) mpTD_lowPt_EndCap->Fill(QGptD);
	    //mMultiplicityQG_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_EndCap"]; if(mMultiplicityQG_lowPt_EndCap && mMultiplicityQG_lowPt_EndCap->getRootObject()) mMultiplicityQG_lowPt_EndCap->Fill(QGmulti);
	    //mqgLikelihood_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_EndCap"]; if(mqgLikelihood_lowPt_EndCap && mqgLikelihood_lowPt_EndCap->getRootObject()) mqgLikelihood_lowPt_EndCap->Fill(QGLikelihood);
	    mMass_lowPt_EndCap=map_of_MEs[DirName+"/"+"JetMass_lowPt_EndCap"]; if(mMass_lowPt_EndCap && mMass_lowPt_EndCap->getRootObject())mMass_lowPt_EndCap->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_lowPt_EndCap=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_lowPt_EndCap"]; if(mMVAPUJIDDiscriminant_lowPt_EndCap && mMVAPUJIDDiscriminant_lowPt_EndCap->getRootObject()) mMVAPUJIDDiscriminant_lowPt_EndCap->Fill(puidmva); 
	    mCutPUJIDDiscriminant_lowPt_EndCap=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_lowPt_EndCap"]; if(mCutPUJIDDiscriminant_lowPt_EndCap && mCutPUJIDDiscriminant_lowPt_EndCap->getRootObject()) mCutPUJIDDiscriminant_lowPt_EndCap->Fill(puidcut); 
	    mCHFrac_lowPt_EndCap = map_of_MEs[DirName+"/"+"CHFrac_lowPt_EndCap"]; if (mCHFrac_lowPt_EndCap &&  mCHFrac_lowPt_EndCap->getRootObject()) mCHFrac_lowPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mNHFrac_lowPt_EndCap = map_of_MEs[DirName+"/"+"NHFrac_lowPt_EndCap"]; if (mNHFrac_lowPt_EndCap &&  mNHFrac_lowPt_EndCap->getRootObject()) mNHFrac_lowPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    mPhFrac_lowPt_EndCap = map_of_MEs[DirName+"/"+"PhFrac_lowPt_EndCap"]; if (mPhFrac_lowPt_EndCap &&  mPhFrac_lowPt_EndCap->getRootObject()) mPhFrac_lowPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	    mCHEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"CHEn_lowPt_EndCap"]; if (mCHEn_lowPt_EndCap &&  mCHEn_lowPt_EndCap->getRootObject()) mCHEn_lowPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergy());
	    mNHEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"NHEn_lowPt_EndCap"]; if (mNHEn_lowPt_EndCap &&  mNHEn_lowPt_EndCap->getRootObject()) mNHEn_lowPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergy());
	    mPhEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"PhEn_lowPt_EndCap"]; if (mPhEn_lowPt_EndCap &&  mPhEn_lowPt_EndCap->getRootObject()) mPhEn_lowPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergy());
	    mElEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"ElEn_lowPt_EndCap"]; if (mElEn_lowPt_EndCap &&  mElEn_lowPt_EndCap->getRootObject()) mElEn_lowPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergy());
	    mMuEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"MuEn_lowPt_EndCap"]; if (mMuEn_lowPt_EndCap &&  mMuEn_lowPt_EndCap->getRootObject()) mMuEn_lowPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergy());
	    mChMultiplicity_lowPt_EndCap = map_of_MEs[DirName+"/"+"ChMultiplicity_lowPt_EndCap"]; if(mChMultiplicity_lowPt_EndCap && mChMultiplicity_lowPt_EndCap->getRootObject())  mChMultiplicity_lowPt_EndCap->Fill((*pfJets)[ijet].chargedMultiplicity());
	    mNeutMultiplicity_lowPt_EndCap = map_of_MEs[DirName+"/"+"NeutMultiplicity_lowPt_EndCap"]; if(mNeutMultiplicity_lowPt_EndCap && mNeutMultiplicity_lowPt_EndCap->getRootObject())  mNeutMultiplicity_lowPt_EndCap->Fill((*pfJets)[ijet].neutralMultiplicity());
	    mMuMultiplicity_lowPt_EndCap = map_of_MEs[DirName+"/"+"MuMultiplicity_lowPt_EndCap"]; if(mMuMultiplicity_lowPt_EndCap && mMuMultiplicity_lowPt_EndCap->getRootObject())  mMuMultiplicity_lowPt_EndCap->Fill((*pfJets)[ijet].muonMultiplicity());
	  }
	  if (correctedJet.pt()>50. && correctedJet.pt()<=140.) {
	    //mAxis2_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_EndCap"];if(mAxis2_mediumPt_EndCap && mAxis2_mediumPt_EndCap->getRootObject()) mAxis2_mediumPt_EndCap->Fill(QGaxis2);
	    //mpTD_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_EndCap"]; if(mpTD_mediumPt_EndCap && mpTD_mediumPt_EndCap->getRootObject()) mpTD_mediumPt_EndCap->Fill(QGptD);
	    //mMultiplicityQG_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_EndCap"]; if(mMultiplicityQG_mediumPt_EndCap && mMultiplicityQG_mediumPt_EndCap->getRootObject()) mMultiplicityQG_mediumPt_EndCap->Fill(QGmulti);
	    //mqgLikelihood_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_EndCap"]; if(mqgLikelihood_mediumPt_EndCap && mqgLikelihood_mediumPt_EndCap->getRootObject()) mqgLikelihood_mediumPt_EndCap->Fill(QGLikelihood);
	    mMass_mediumPt_EndCap=map_of_MEs[DirName+"/"+"JetMass_mediumPt_EndCap"]; if(mMass_mediumPt_EndCap && mMass_mediumPt_EndCap->getRootObject())mMass_mediumPt_EndCap->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_mediumPt_EndCap=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_EndCap"]; if(mMVAPUJIDDiscriminant_mediumPt_EndCap && mMVAPUJIDDiscriminant_mediumPt_EndCap->getRootObject()) mMVAPUJIDDiscriminant_mediumPt_EndCap->Fill(puidmva); 
	    mCutPUJIDDiscriminant_mediumPt_EndCap=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_mediumPt_EndCap"]; if(mCutPUJIDDiscriminant_mediumPt_EndCap && mCutPUJIDDiscriminant_mediumPt_EndCap->getRootObject()) mCutPUJIDDiscriminant_mediumPt_EndCap->Fill(puidcut); 
	    mCHFrac_mediumPt_EndCap = map_of_MEs[DirName+"/"+"CHFrac_mediumPt_EndCap"]; if (mCHFrac_mediumPt_EndCap &&  mCHFrac_mediumPt_EndCap->getRootObject()) mCHFrac_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mNHFrac_mediumPt_EndCap = map_of_MEs[DirName+"/"+"NHFrac_mediumPt_EndCap"]; if (mNHFrac_mediumPt_EndCap &&  mNHFrac_mediumPt_EndCap->getRootObject()) mNHFrac_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    mPhFrac_mediumPt_EndCap = map_of_MEs[DirName+"/"+"PhFrac_mediumPt_EndCap"]; if (mPhFrac_mediumPt_EndCap &&  mPhFrac_mediumPt_EndCap->getRootObject()) mPhFrac_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	    mCHEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"CHEn_mediumPt_EndCap"]; if (mCHEn_mediumPt_EndCap &&  mCHEn_mediumPt_EndCap->getRootObject()) mCHEn_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergy());
	    mNHEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"NHEn_mediumPt_EndCap"]; if (mNHEn_mediumPt_EndCap &&  mNHEn_mediumPt_EndCap->getRootObject()) mNHEn_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergy());
	    mPhEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"PhEn_mediumPt_EndCap"]; if (mPhEn_mediumPt_EndCap &&  mPhEn_mediumPt_EndCap->getRootObject()) mPhEn_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergy());
	    mElEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"ElEn_mediumPt_EndCap"]; if (mElEn_mediumPt_EndCap &&  mElEn_mediumPt_EndCap->getRootObject()) mElEn_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergy());
	    mMuEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"MuEn_mediumPt_EndCap"]; if (mMuEn_mediumPt_EndCap &&  mMuEn_mediumPt_EndCap->getRootObject()) mMuEn_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergy());
	    mChMultiplicity_mediumPt_EndCap = map_of_MEs[DirName+"/"+"ChMultiplicity_mediumPt_EndCap"]; if(mChMultiplicity_mediumPt_EndCap && mChMultiplicity_mediumPt_EndCap->getRootObject())  mChMultiplicity_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedMultiplicity());
	    mNeutMultiplicity_mediumPt_EndCap = map_of_MEs[DirName+"/"+"NeutMultiplicity_mediumPt_EndCap"]; if(mNeutMultiplicity_mediumPt_EndCap && mNeutMultiplicity_mediumPt_EndCap->getRootObject())  mNeutMultiplicity_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralMultiplicity());
	    mMuMultiplicity_mediumPt_EndCap = map_of_MEs[DirName+"/"+"MuMultiplicity_mediumPt_EndCap"]; if(mMuMultiplicity_mediumPt_EndCap && mMuMultiplicity_mediumPt_EndCap->getRootObject())  mMuMultiplicity_mediumPt_EndCap->Fill((*pfJets)[ijet].muonMultiplicity());
	  }
	  if (correctedJet.pt()>140.) {
	    //mAxis2_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_EndCap"];if(mAxis2_highPt_EndCap && mAxis2_highPt_EndCap->getRootObject()) mAxis2_highPt_EndCap->Fill(QGaxis2);
	    //mpTD_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_highPt_EndCap"]; if(mpTD_highPt_EndCap && mpTD_highPt_EndCap->getRootObject()) mpTD_highPt_EndCap->Fill(QGptD);
	    //mMultiplicityQG_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_EndCap"]; if(mMultiplicityQG_highPt_EndCap && mMultiplicityQG_highPt_EndCap->getRootObject()) mMultiplicityQG_highPt_EndCap->Fill(QGmulti);
	    //mqgLikelihood_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_EndCap"]; if(mqgLikelihood_highPt_EndCap && mqgLikelihood_highPt_EndCap->getRootObject()) mqgLikelihood_highPt_EndCap->Fill(QGLikelihood);
	    mMass_highPt_EndCap=map_of_MEs[DirName+"/"+"JetMass_highPt_EndCap"]; if(mMass_highPt_EndCap && mMass_highPt_EndCap->getRootObject())mMass_highPt_EndCap->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_highPt_EndCap=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_highPt_EndCap"]; if(mMVAPUJIDDiscriminant_highPt_EndCap && mMVAPUJIDDiscriminant_highPt_EndCap->getRootObject()) mMVAPUJIDDiscriminant_highPt_EndCap->Fill(puidmva); 
	    mCutPUJIDDiscriminant_highPt_EndCap=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_highPt_EndCap"]; if(mCutPUJIDDiscriminant_highPt_EndCap && mCutPUJIDDiscriminant_highPt_EndCap->getRootObject()) mCutPUJIDDiscriminant_highPt_EndCap->Fill(puidcut); 
	    mCHFrac_highPt_EndCap = map_of_MEs[DirName+"/"+"CHFrac_highPt_EndCap"]; if (mCHFrac_highPt_EndCap &&  mCHFrac_highPt_EndCap->getRootObject()) mCHFrac_highPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mNHFrac_highPt_EndCap = map_of_MEs[DirName+"/"+"NHFrac_highPt_EndCap"]; if (mNHFrac_highPt_EndCap &&  mNHFrac_highPt_EndCap->getRootObject()) mNHFrac_highPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    mPhFrac_highPt_EndCap = map_of_MEs[DirName+"/"+"PhFrac_highPt_EndCap"]; if (mPhFrac_highPt_EndCap &&  mPhFrac_highPt_EndCap->getRootObject()) mPhFrac_highPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	    mCHEn_highPt_EndCap = map_of_MEs[DirName+"/"+"CHEn_highPt_EndCap"]; if (mCHEn_highPt_EndCap &&  mCHEn_highPt_EndCap->getRootObject()) mCHEn_highPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergy());
	    mNHEn_highPt_EndCap = map_of_MEs[DirName+"/"+"NHEn_highPt_EndCap"]; if (mNHEn_highPt_EndCap &&  mNHEn_highPt_EndCap->getRootObject()) mNHEn_highPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergy());
	    mPhEn_highPt_EndCap = map_of_MEs[DirName+"/"+"PhEn_highPt_EndCap"]; if (mPhEn_highPt_EndCap &&  mPhEn_highPt_EndCap->getRootObject()) mPhEn_highPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergy());
	    mElEn_highPt_EndCap = map_of_MEs[DirName+"/"+"ElEn_highPt_EndCap"]; if (mElEn_highPt_EndCap &&  mElEn_highPt_EndCap->getRootObject()) mElEn_highPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergy());
	    mMuEn_highPt_EndCap = map_of_MEs[DirName+"/"+"MuEn_highPt_EndCap"]; if (mMuEn_highPt_EndCap &&  mMuEn_highPt_EndCap->getRootObject()) mMuEn_highPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergy());
	    mChMultiplicity_highPt_EndCap = map_of_MEs[DirName+"/"+"ChMultiplicity_highPt_EndCap"]; if(mChMultiplicity_highPt_EndCap && mChMultiplicity_highPt_EndCap->getRootObject())  mChMultiplicity_highPt_EndCap->Fill((*pfJets)[ijet].chargedMultiplicity());
	    mNeutMultiplicity_highPt_EndCap = map_of_MEs[DirName+"/"+"NeutMultiplicity_highPt_EndCap"]; if(mNeutMultiplicity_highPt_EndCap && mNeutMultiplicity_highPt_EndCap->getRootObject())  mNeutMultiplicity_highPt_EndCap->Fill((*pfJets)[ijet].neutralMultiplicity());
	    mMuMultiplicity_highPt_EndCap = map_of_MEs[DirName+"/"+"MuMultiplicity_highPt_EndCap"]; if(mMuMultiplicity_highPt_EndCap && mMuMultiplicity_highPt_EndCap->getRootObject())  mMuMultiplicity_highPt_EndCap->Fill((*pfJets)[ijet].muonMultiplicity());
	  }
	  mCHFracVSpT_EndCap = map_of_MEs[DirName+"/"+"CHFracVSpT_EndCap"]; if(mCHFracVSpT_EndCap && mCHFracVSpT_EndCap->getRootObject()) mCHFracVSpT_EndCap->Fill(correctedJet.pt(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	  mNHFracVSpT_EndCap = map_of_MEs[DirName+"/"+"NHFracVSpT_EndCap"];if (mNHFracVSpT_EndCap && mNHFracVSpT_EndCap->getRootObject()) mNHFracVSpT_EndCap->Fill(correctedJet.pt(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	  mPhFracVSpT_EndCap = map_of_MEs[DirName+"/"+"PhFracVSpT_EndCap"];if (mPhFracVSpT_EndCap && mPhFracVSpT_EndCap->getRootObject()) mPhFracVSpT_EndCap->Fill(correctedJet.pt(),(*pfJets)[ijet].neutralEmEnergyFraction());
	}else{
	  mHFHFracVSpT_Forward = map_of_MEs[DirName+"/"+"HFHFracVSpT_Forward"]; if (mHFHFracVSpT_Forward && mHFHFracVSpT_Forward->getRootObject())    mHFHFracVSpT_Forward->Fill(correctedJet.pt(),(*pfJets)[ijet].HFHadronEnergyFraction ());	
	  mHFEFracVSpT_Forward = map_of_MEs[DirName+"/"+"HFEFracVSpT_Forward"]; if (mHFEFracVSpT_Forward && mHFEFracVSpT_Forward->getRootObject())    mHFEFracVSpT_Forward->Fill (correctedJet.pt(),(*pfJets)[ijet].HFEMEnergyFraction ());
	  //fractions
	  if (correctedJet.pt()<=50.) {
	    //mAxis2_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_Forward"];if(mAxis2_lowPt_Forward && mAxis2_lowPt_Forward->getRootObject()) mAxis2_lowPt_Forward->Fill(QGaxis2);
	    //mpTD_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_Forward"]; if(mpTD_lowPt_Forward && mpTD_lowPt_Forward->getRootObject()) mpTD_lowPt_Forward->Fill(QGptD);
	    //mMultiplicityQG_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_Forward"]; if(mMultiplicityQG_lowPt_Forward && mMultiplicityQG_lowPt_Forward->getRootObject()) mMultiplicityQG_lowPt_Forward->Fill(QGmulti);
	    //mqgLikelihood_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_Forward"]; if(mqgLikelihood_lowPt_Forward && mqgLikelihood_lowPt_Forward->getRootObject()) mqgLikelihood_lowPt_Forward->Fill(QGLikelihood);
	    mMass_lowPt_Forward=map_of_MEs[DirName+"/"+"JetMass_lowPt_Forward"]; if(mMass_lowPt_Forward && mMass_lowPt_Forward->getRootObject())mMass_lowPt_Forward->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_lowPt_Forward=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_lowPt_Forward"]; if(mMVAPUJIDDiscriminant_lowPt_Forward && mMVAPUJIDDiscriminant_lowPt_Forward->getRootObject()) mMVAPUJIDDiscriminant_lowPt_Forward->Fill(puidmva); 
	    mCutPUJIDDiscriminant_lowPt_Forward=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_lowPt_Forward"]; if(mCutPUJIDDiscriminant_lowPt_Forward && mCutPUJIDDiscriminant_lowPt_Forward->getRootObject()) mCutPUJIDDiscriminant_lowPt_Forward->Fill(puidcut); 
	    mHFEFrac_lowPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_lowPt_Forward"]; if(mHFEFrac_lowPt_Forward && mHFEFrac_lowPt_Forward->getRootObject()) mHFEFrac_lowPt_Forward->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	    mHFHFrac_lowPt_Forward = map_of_MEs[DirName+"/"+"HFHFrac_lowPt_Forward"]; if(mHFHFrac_lowPt_Forward && mHFHFrac_lowPt_Forward->getRootObject()) mHFHFrac_lowPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    mHFEEn_lowPt_Forward = map_of_MEs[DirName+"/"+"HFEEn_lowPt_Forward"];     if(mHFEEn_lowPt_Forward && mHFEEn_lowPt_Forward->getRootObject())     mHFEEn_lowPt_Forward->Fill((*pfJets)[ijet].HFEMEnergy());
	    mHFHEn_lowPt_Forward = map_of_MEs[DirName+"/"+"HFHEn_lowPt_Forward"];    if(mHFHEn_lowPt_Forward && mHFHEn_lowPt_Forward->getRootObject())     mHFHEn_lowPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergy());
	    mNeutMultiplicity_lowPt_Forward = map_of_MEs[DirName+"/"+"NeutMultiplicity_lowPt_Forward"]; if(mNeutMultiplicity_lowPt_Forward && mNeutMultiplicity_lowPt_Forward->getRootObject())  mNeutMultiplicity_lowPt_Forward->Fill((*pfJets)[ijet].neutralMultiplicity());
	  }
	  if (correctedJet.pt()>50. && correctedJet.pt()<=140.) {
	    //mAxis2_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_Forward"];if(mAxis2_mediumPt_Forward && mAxis2_mediumPt_Forward->getRootObject()) mAxis2_mediumPt_Forward->Fill(QGaxis2);
	    //mpTD_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_Forward"]; if(mpTD_mediumPt_Forward && mpTD_mediumPt_Forward->getRootObject()) mpTD_mediumPt_Forward->Fill(QGptD);
	    //mMultiplicityQG_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_Forward"]; if(mMultiplicityQG_mediumPt_Forward && mMultiplicityQG_mediumPt_Forward->getRootObject()) mMultiplicityQG_mediumPt_Forward->Fill(QGmulti);
	    //mqgLikelihood_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_Forward"]; if(mqgLikelihood_mediumPt_Forward && mqgLikelihood_mediumPt_Forward->getRootObject()) mqgLikelihood_mediumPt_Forward->Fill(QGLikelihood);
	    mMass_mediumPt_Forward=map_of_MEs[DirName+"/"+"JetMass_mediumPt_Forward"]; if(mMass_mediumPt_Forward && mMass_mediumPt_Forward->getRootObject())mMass_mediumPt_Forward->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_mediumPt_Forward=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_mediumPt_Forward"]; if(mMVAPUJIDDiscriminant_mediumPt_Forward && mMVAPUJIDDiscriminant_mediumPt_Forward->getRootObject()) mMVAPUJIDDiscriminant_mediumPt_Forward->Fill(puidmva); 
	    mCutPUJIDDiscriminant_mediumPt_Forward=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_mediumPt_Forward"]; if(mCutPUJIDDiscriminant_mediumPt_Forward && mCutPUJIDDiscriminant_mediumPt_Forward->getRootObject()) mCutPUJIDDiscriminant_mediumPt_Forward->Fill(puidcut); 
	    mHFEFrac_mediumPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_mediumPt_Forward"]; if(mHFEFrac_mediumPt_Forward && mHFEFrac_mediumPt_Forward->getRootObject()) mHFEFrac_mediumPt_Forward->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	    mHFHFrac_mediumPt_Forward = map_of_MEs[DirName+"/"+"HFHFrac_mediumPt_Forward"]; if(mHFHFrac_mediumPt_Forward && mHFHFrac_mediumPt_Forward->getRootObject()) mHFHFrac_mediumPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    mHFEEn_mediumPt_Forward = map_of_MEs[DirName+"/"+"HFEEn_mediumPt_Forward"];     if(mHFEEn_mediumPt_Forward && mHFEEn_mediumPt_Forward->getRootObject())     mHFEEn_mediumPt_Forward->Fill((*pfJets)[ijet].HFEMEnergy());
	    mHFHEn_mediumPt_Forward = map_of_MEs[DirName+"/"+"HFHEn_mediumPt_Forward"];    if(mHFHEn_mediumPt_Forward && mHFHEn_mediumPt_Forward->getRootObject())     mHFHEn_mediumPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergy());
	    mNeutMultiplicity_mediumPt_Forward = map_of_MEs[DirName+"/"+"NeutMultiplicity_mediumPt_Forward"]; if(mNeutMultiplicity_mediumPt_Forward && mNeutMultiplicity_mediumPt_Forward->getRootObject())  mNeutMultiplicity_mediumPt_Forward->Fill((*pfJets)[ijet].neutralMultiplicity());
	  }
	  if (correctedJet.pt()>140.) {
	    //mAxis2_highPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_Forward"];if(mAxis2_highPt_Forward && mAxis2_highPt_Forward->getRootObject()) mAxis2_highPt_Forward->Fill(QGaxis2);
	    //mpTD_highPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_highPt_Forward"]; if(mpTD_highPt_Forward && mpTD_highPt_Forward->getRootObject()) mpTD_highPt_Forward->Fill(QGptD);
	    //mMultiplicityQG_highPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_Forward"]; if(mMultiplicityQG_highPt_Forward && mMultiplicityQG_highPt_Forward->getRootObject()) mMultiplicityQG_highPt_Forward->Fill(QGmulti);
	    //mqgLikelihood_highPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_Forward"]; if(mqgLikelihood_highPt_Forward && mqgLikelihood_highPt_Forward->getRootObject()) mqgLikelihood_highPt_Forward->Fill(QGLikelihood);
	    mMass_highPt_Forward=map_of_MEs[DirName+"/"+"JetMass_highPt_Forward"]; if(mMass_highPt_Forward && mMass_highPt_Forward->getRootObject())mMass_highPt_Forward->Fill((*pfJets)[ijet].mass());
	    mMVAPUJIDDiscriminant_highPt_Forward=map_of_MEs[DirName+"/"+"MVAPUJIDDiscriminant_highPt_Forward"]; if(mMVAPUJIDDiscriminant_highPt_Forward && mMVAPUJIDDiscriminant_highPt_Forward->getRootObject()) mMVAPUJIDDiscriminant_highPt_Forward->Fill(puidmva); 
	    mCutPUJIDDiscriminant_highPt_Forward=map_of_MEs[DirName+"/"+"CutPUJIDDiscriminant_highPt_Forward"]; if(mCutPUJIDDiscriminant_highPt_Forward && mCutPUJIDDiscriminant_highPt_Forward->getRootObject()) mCutPUJIDDiscriminant_highPt_Forward->Fill(puidcut); 
	    mHFEFrac_highPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_highPt_Forward"]; if(mHFEFrac_highPt_Forward && mHFEFrac_highPt_Forward->getRootObject()) mHFEFrac_highPt_Forward->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	    mHFHFrac_highPt_Forward = map_of_MEs[DirName+"/"+"HFHFrac_highPt_Forward"]; if(mHFHFrac_highPt_Forward && mHFHFrac_highPt_Forward->getRootObject()) mHFHFrac_highPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    mHFEEn_highPt_Forward = map_of_MEs[DirName+"/"+"HFEEn_highPt_Forward"];     if(mHFEEn_highPt_Forward && mHFEEn_highPt_Forward->getRootObject())     mHFEEn_highPt_Forward->Fill((*pfJets)[ijet].HFEMEnergy());
	    mHFHEn_highPt_Forward = map_of_MEs[DirName+"/"+"HFHEn_highPt_Forward"];    if(mHFHEn_highPt_Forward && mHFHEn_highPt_Forward->getRootObject())     mHFHEn_highPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergy());
	    mNeutMultiplicity_highPt_Forward = map_of_MEs[DirName+"/"+"NeutMultiplicity_highPt_Forward"]; if(mNeutMultiplicity_highPt_Forward && mNeutMultiplicity_highPt_Forward->getRootObject())  mNeutMultiplicity_highPt_Forward->Fill((*pfJets)[ijet].neutralMultiplicity());
	  }
	}
	//OOT plots
	if(techTriggerResultBx0 && techTriggerResultBxE && techTriggerResultBxF){
	  meEta_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"Eta_BXm2BXm1Filled"];     if (  meEta_BXm2BXm1Filled  && meEta_BXm2BXm1Filled ->getRootObject())  meEta_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].eta());
	  if(fabs(correctedJet.eta()) <= 1.3) {
	    mePhFracBarrel_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PhFracBarrel_BXm2BXm1Filled"];     if (  mePhFracBarrel_BXm2BXm1Filled  && mePhFracBarrel_BXm2BXm1Filled ->getRootObject())  mePhFracBarrel_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracBarrel_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"NHFracBarrel_BXm2BXm1Filled"];     if (  meNHFracBarrel_BXm2BXm1Filled  && meNHFracBarrel_BXm2BXm1Filled ->getRootObject())  meNHFracBarrel_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracBarrel_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"CHFracBarrel_BXm2BXm1Filled"];     if (  meCHFracBarrel_BXm2BXm1Filled  && meCHFracBarrel_BXm2BXm1Filled ->getRootObject())  meCHFracBarrel_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtBarrel_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PtBarrel_BXm2BXm1Filled"];     if (  mePtBarrel_BXm2BXm1Filled  && mePtBarrel_BXm2BXm1Filled ->getRootObject())  mePtBarrel_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() > -3.0 && correctedJet.eta() <= -1.3) {
	    mePhFracEndCapMinus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PhFracEndCapMinus_BXm2BXm1Filled"];     if (  mePhFracEndCapMinus_BXm2BXm1Filled  && mePhFracEndCapMinus_BXm2BXm1Filled ->getRootObject())  mePhFracEndCapMinus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracEndCapMinus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"NHFracEndCapMinus_BXm2BXm1Filled"];     if (  meNHFracEndCapMinus_BXm2BXm1Filled  && meNHFracEndCapMinus_BXm2BXm1Filled ->getRootObject())  meNHFracEndCapMinus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracEndCapMinus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"CHFracEndCapMinus_BXm2BXm1Filled"];     if (  meCHFracEndCapMinus_BXm2BXm1Filled  && meCHFracEndCapMinus_BXm2BXm1Filled ->getRootObject())  meCHFracEndCapMinus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtEndCapMinus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PtEndCapMinus_BXm2BXm1Filled"];     if (  mePtEndCapMinus_BXm2BXm1Filled  && mePtEndCapMinus_BXm2BXm1Filled ->getRootObject())  mePtEndCapMinus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() >= 1.3 && correctedJet.eta() < 3.0) {
	    mePhFracEndCapPlus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PhFracEndCapPlus_BXm2BXm1Filled"];     if (  mePhFracEndCapPlus_BXm2BXm1Filled  && mePhFracEndCapPlus_BXm2BXm1Filled ->getRootObject())  mePhFracEndCapPlus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracEndCapPlus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"NHFracEndCapPlus_BXm2BXm1Filled"];     if (  meNHFracEndCapPlus_BXm2BXm1Filled  && meNHFracEndCapPlus_BXm2BXm1Filled ->getRootObject())  meNHFracEndCapPlus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracEndCapPlus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"CHFracEndCapPlus_BXm2BXm1Filled"];     if (  meCHFracEndCapPlus_BXm2BXm1Filled  && meCHFracEndCapPlus_BXm2BXm1Filled ->getRootObject())  meCHFracEndCapPlus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtEndCapPlus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PtEndCapPlus_BXm2BXm1Filled"];     if (  mePtEndCapPlus_BXm2BXm1Filled  && mePtEndCapPlus_BXm2BXm1Filled ->getRootObject())  mePtEndCapPlus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() > -5.0 && correctedJet.eta() <= -3.0) {
	    mePtForwardMinus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PtForwardMinus_BXm2BXm1Filled"];     if (  mePtForwardMinus_BXm2BXm1Filled  && mePtForwardMinus_BXm2BXm1Filled ->getRootObject())  mePtForwardMinus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	    meHFHFracMinus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"HFHFracMinus_BXm2BXm1Filled"];     if (  meHFHFracMinus_BXm2BXm1Filled  && meHFHFracMinus_BXm2BXm1Filled ->getRootObject())  meHFHFracMinus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    meHFEMFracMinus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"HFEMFracMinus_BXm2BXm1Filled"];     if (  meHFEMFracMinus_BXm2BXm1Filled  && meHFEMFracMinus_BXm2BXm1Filled ->getRootObject())  meHFEMFracMinus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	  }else if (correctedJet.eta() >= 3.0 && correctedJet.eta() < 5.0) {
	    mePtForwardPlus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"PtForwardPlus_BXm2BXm1Filled"];     if (  mePtForwardPlus_BXm2BXm1Filled  && mePtForwardPlus_BXm2BXm1Filled ->getRootObject())  mePtForwardPlus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	    meHFHFracPlus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"HFHFracPlus_BXm2BXm1Filled"];     if (  meHFHFracPlus_BXm2BXm1Filled  && meHFHFracPlus_BXm2BXm1Filled ->getRootObject())  meHFHFracPlus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    meHFEMFracPlus_BXm2BXm1Filled    = map_of_MEs[DirName+"/"+"HFEMFracPlus_BXm2BXm1Filled"];     if (  meHFEMFracPlus_BXm2BXm1Filled  && meHFEMFracPlus_BXm2BXm1Filled ->getRootObject())  meHFEMFracPlus_BXm2BXm1Filled  ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	  }
	}
	if(techTriggerResultBx0 && techTriggerResultBxF){
	  meEta_BXm1Filled    = map_of_MEs[DirName+"/"+"Eta_BXm1Filled"];     if (  meEta_BXm1Filled  && meEta_BXm1Filled ->getRootObject())  meEta_BXm1Filled  ->Fill((*pfJets)[ijet].eta());
	  if(fabs(correctedJet.eta()) <= 1.3) {
	    mePhFracBarrel_BXm1Filled    = map_of_MEs[DirName+"/"+"PhFracBarrel_BXm1Filled"];     if (  mePhFracBarrel_BXm1Filled  && mePhFracBarrel_BXm1Filled ->getRootObject())  mePhFracBarrel_BXm1Filled  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracBarrel_BXm1Filled    = map_of_MEs[DirName+"/"+"NHFracBarrel_BXm1Filled"];     if (  meNHFracBarrel_BXm1Filled  && meNHFracBarrel_BXm1Filled ->getRootObject())  meNHFracBarrel_BXm1Filled  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracBarrel_BXm1Filled    = map_of_MEs[DirName+"/"+"CHFracBarrel_BXm1Filled"];     if (  meCHFracBarrel_BXm1Filled  && meCHFracBarrel_BXm1Filled ->getRootObject())  meCHFracBarrel_BXm1Filled  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtBarrel_BXm1Filled    = map_of_MEs[DirName+"/"+"PtBarrel_BXm1Filled"];     if (  mePtBarrel_BXm1Filled  && mePtBarrel_BXm1Filled ->getRootObject())  mePtBarrel_BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() > -3.0 && correctedJet.eta() <= -1.3) {
	    mePhFracEndCapMinus_BXm1Filled    = map_of_MEs[DirName+"/"+"PhFracEndCapMinus_BXm1Filled"];     if (  mePhFracEndCapMinus_BXm1Filled  && mePhFracEndCapMinus_BXm1Filled ->getRootObject())  mePhFracEndCapMinus_BXm1Filled  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracEndCapMinus_BXm1Filled    = map_of_MEs[DirName+"/"+"NHFracEndCapMinus_BXm1Filled"];     if (  meNHFracEndCapMinus_BXm1Filled  && meNHFracEndCapMinus_BXm1Filled ->getRootObject())  meNHFracEndCapMinus_BXm1Filled  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracEndCapMinus_BXm1Filled    = map_of_MEs[DirName+"/"+"CHFracEndCapMinus_BXm1Filled"];     if (  meCHFracEndCapMinus_BXm1Filled  && meCHFracEndCapMinus_BXm1Filled ->getRootObject())  meCHFracEndCapMinus_BXm1Filled  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtEndCapMinus_BXm1Filled    = map_of_MEs[DirName+"/"+"PtEndCapMinus_BXm1Filled"];     if (  mePtEndCapMinus_BXm1Filled  && mePtEndCapMinus_BXm1Filled ->getRootObject())  mePtEndCapMinus_BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() >= 1.3 && correctedJet.eta() < 3.0) {
	    mePhFracEndCapPlus_BXm1Filled    = map_of_MEs[DirName+"/"+"PhFracEndCapPlus_BXm1Filled"];     if (  mePhFracEndCapPlus_BXm1Filled  && mePhFracEndCapPlus_BXm1Filled ->getRootObject())  mePhFracEndCapPlus_BXm1Filled  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracEndCapPlus_BXm1Filled    = map_of_MEs[DirName+"/"+"NHFracEndCapPlus_BXm1Filled"];     if (  meNHFracEndCapPlus_BXm1Filled  && meNHFracEndCapPlus_BXm1Filled ->getRootObject())  meNHFracEndCapPlus_BXm1Filled  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracEndCapPlus_BXm1Filled    = map_of_MEs[DirName+"/"+"CHFracEndCapPlus_BXm1Filled"];     if (  meCHFracEndCapPlus_BXm1Filled  && meCHFracEndCapPlus_BXm1Filled ->getRootObject())  meCHFracEndCapPlus_BXm1Filled  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtEndCapPlus_BXm1Filled    = map_of_MEs[DirName+"/"+"PtEndCapPlus_BXm1Filled"];     if (  mePtEndCapPlus_BXm1Filled  && mePtEndCapPlus_BXm1Filled ->getRootObject())  mePtEndCapPlus_BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() > -5.0 && correctedJet.eta() <= -3.0) {
	    mePtForwardMinus_BXm1Filled    = map_of_MEs[DirName+"/"+"PtForwardMinus_BXm1Filled"];     if (  mePtForwardMinus_BXm1Filled  && mePtForwardMinus_BXm1Filled ->getRootObject())  mePtForwardMinus_BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	    meHFHFracMinus_BXm1Filled    = map_of_MEs[DirName+"/"+"HFHFracMinus_BXm1Filled"];     if (  meHFHFracMinus_BXm1Filled  && meHFHFracMinus_BXm1Filled ->getRootObject())  meHFHFracMinus_BXm1Filled  ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    meHFEMFracMinus_BXm1Filled    = map_of_MEs[DirName+"/"+"HFEMFracMinus_BXm1Filled"];     if (  meHFEMFracMinus_BXm1Filled  && meHFEMFracMinus_BXm1Filled ->getRootObject())  meHFEMFracMinus_BXm1Filled  ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	  }else if (correctedJet.eta() >= 3.0 && correctedJet.eta() < 5.0) {
	    mePtForwardPlus_BXm1Filled    = map_of_MEs[DirName+"/"+"PtForwardPlus_BXm1Filled"];     if (  mePtForwardPlus_BXm1Filled  && mePtForwardPlus_BXm1Filled ->getRootObject())  mePtForwardPlus_BXm1Filled  ->Fill((*pfJets)[ijet].pt());
	    meHFHFracPlus_BXm1Filled    = map_of_MEs[DirName+"/"+"HFHFracPlus_BXm1Filled"];     if (  meHFHFracPlus_BXm1Filled  && meHFHFracPlus_BXm1Filled ->getRootObject())  meHFHFracPlus_BXm1Filled  ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    meHFEMFracPlus_BXm1Filled    = map_of_MEs[DirName+"/"+"HFEMFracPlus_BXm1Filled"];     if (  meHFEMFracPlus_BXm1Filled  && meHFEMFracPlus_BXm1Filled ->getRootObject())  meHFEMFracPlus_BXm1Filled  ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	  }
	}
	if(techTriggerResultBx0 && !techTriggerResultBxE && !techTriggerResultBxF){
	  meEta_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"Eta_BXm2BXm1Empty"];     if (  meEta_BXm2BXm1Empty  && meEta_BXm2BXm1Empty ->getRootObject())  meEta_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].eta());
	  if(fabs(correctedJet.eta()) <= 1.3) {
	    mePhFracBarrel_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PhFracBarrel_BXm2BXm1Empty"];     if (  mePhFracBarrel_BXm2BXm1Empty  && mePhFracBarrel_BXm2BXm1Empty ->getRootObject())  mePhFracBarrel_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracBarrel_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"NHFracBarrel_BXm2BXm1Empty"];     if (  meNHFracBarrel_BXm2BXm1Empty  && meNHFracBarrel_BXm2BXm1Empty ->getRootObject())  meNHFracBarrel_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracBarrel_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"CHFracBarrel_BXm2BXm1Empty"];     if (  meCHFracBarrel_BXm2BXm1Empty  && meCHFracBarrel_BXm2BXm1Empty ->getRootObject())  meCHFracBarrel_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtBarrel_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PtBarrel_BXm2BXm1Empty"];     if (  mePtBarrel_BXm2BXm1Empty  && mePtBarrel_BXm2BXm1Empty ->getRootObject())  mePtBarrel_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() > -3.0 && correctedJet.eta() <= -1.3) {
	    mePhFracEndCapMinus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PhFracEndCapMinus_BXm2BXm1Empty"];     if (  mePhFracEndCapMinus_BXm2BXm1Empty  && mePhFracEndCapMinus_BXm2BXm1Empty ->getRootObject())  mePhFracEndCapMinus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracEndCapMinus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"NHFracEndCapMinus_BXm2BXm1Empty"];     if (  meNHFracEndCapMinus_BXm2BXm1Empty  && meNHFracEndCapMinus_BXm2BXm1Empty ->getRootObject())  meNHFracEndCapMinus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracEndCapMinus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"CHFracEndCapMinus_BXm2BXm1Empty"];     if (  meCHFracEndCapMinus_BXm2BXm1Empty  && meCHFracEndCapMinus_BXm2BXm1Empty ->getRootObject())  meCHFracEndCapMinus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtEndCapMinus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PtEndCapMinus_BXm2BXm1Empty"];     if (  mePtEndCapMinus_BXm2BXm1Empty  && mePtEndCapMinus_BXm2BXm1Empty ->getRootObject())  mePtEndCapMinus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() >= 1.3 && correctedJet.eta() < 3.0) {
	    mePhFracEndCapPlus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PhFracEndCapPlus_BXm2BXm1Empty"];     if (  mePhFracEndCapPlus_BXm2BXm1Empty  && mePhFracEndCapPlus_BXm2BXm1Empty ->getRootObject())  mePhFracEndCapPlus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracEndCapPlus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"NHFracEndCapPlus_BXm2BXm1Empty"];     if (  meNHFracEndCapPlus_BXm2BXm1Empty  && meNHFracEndCapPlus_BXm2BXm1Empty ->getRootObject())  meNHFracEndCapPlus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracEndCapPlus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"CHFracEndCapPlus_BXm2BXm1Empty"];     if (  meCHFracEndCapPlus_BXm2BXm1Empty  && meCHFracEndCapPlus_BXm2BXm1Empty ->getRootObject())  meCHFracEndCapPlus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtEndCapPlus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PtEndCapPlus_BXm2BXm1Empty"];     if (  mePtEndCapPlus_BXm2BXm1Empty  && mePtEndCapPlus_BXm2BXm1Empty ->getRootObject())  mePtEndCapPlus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() > -5.0 && correctedJet.eta() <= -3.0) {
	    mePtForwardMinus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PtForwardMinus_BXm2BXm1Empty"];     if (  mePtForwardMinus_BXm2BXm1Empty  && mePtForwardMinus_BXm2BXm1Empty ->getRootObject())  mePtForwardMinus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	    meHFHFracMinus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"HFHFracMinus_BXm2BXm1Empty"];     if (  meHFHFracMinus_BXm2BXm1Empty  && meHFHFracMinus_BXm2BXm1Empty ->getRootObject())  meHFHFracMinus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    meHFEMFracMinus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"HFEMFracMinus_BXm2BXm1Empty"];     if (  meHFEMFracMinus_BXm2BXm1Empty  && meHFEMFracMinus_BXm2BXm1Empty ->getRootObject()) meHFEMFracMinus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	  }else if (correctedJet.eta() >= 3.0 && correctedJet.eta() < 5.0) {
	    mePtForwardPlus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"PtForwardPlus_BXm2BXm1Empty"];     if (  mePtForwardPlus_BXm2BXm1Empty  && mePtForwardPlus_BXm2BXm1Empty ->getRootObject())  mePtForwardPlus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	    meHFHFracPlus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"HFHFracPlus_BXm2BXm1Empty"];     if (  meHFHFracPlus_BXm2BXm1Empty  && meHFHFracPlus_BXm2BXm1Empty ->getRootObject())  meHFHFracPlus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    meHFEMFracPlus_BXm2BXm1Empty    = map_of_MEs[DirName+"/"+"HFEMFracPlus_BXm2BXm1Empty"];     if (  meHFEMFracPlus_BXm2BXm1Empty  && meHFEMFracPlus_BXm2BXm1Empty ->getRootObject())  meHFEMFracPlus_BXm2BXm1Empty  ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	  }
	}
	if(techTriggerResultBx0 && !techTriggerResultBxF){
	  meEta_BXm1Empty    = map_of_MEs[DirName+"/"+"Eta_BXm1Empty"];     if (  meEta_BXm1Empty  && meEta_BXm1Empty ->getRootObject())  meEta_BXm1Empty  ->Fill((*pfJets)[ijet].eta());
	  if(fabs(correctedJet.eta()) <= 1.3) {
	    mePhFracBarrel_BXm1Empty    = map_of_MEs[DirName+"/"+"PhFracBarrel_BXm1Empty"];     if (  mePhFracBarrel_BXm1Empty  && mePhFracBarrel_BXm1Empty ->getRootObject())  mePhFracBarrel_BXm1Empty  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracBarrel_BXm1Empty    = map_of_MEs[DirName+"/"+"NHFracBarrel_BXm1Empty"];     if (  meNHFracBarrel_BXm1Empty  && meNHFracBarrel_BXm1Empty ->getRootObject())  meNHFracBarrel_BXm1Empty  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracBarrel_BXm1Empty    = map_of_MEs[DirName+"/"+"CHFracBarrel_BXm1Empty"];     if (  meCHFracBarrel_BXm1Empty  && meCHFracBarrel_BXm1Empty ->getRootObject())  meCHFracBarrel_BXm1Empty  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtBarrel_BXm1Empty    = map_of_MEs[DirName+"/"+"PtBarrel_BXm1Empty"];     if (  mePtBarrel_BXm1Empty  && mePtBarrel_BXm1Empty ->getRootObject())  mePtBarrel_BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() > -3.0 && correctedJet.eta() <= -1.3) {
	    mePhFracEndCapMinus_BXm1Empty    = map_of_MEs[DirName+"/"+"PhFracEndCapMinus_BXm1Empty"];     if (  mePhFracEndCapMinus_BXm1Empty  && mePhFracEndCapMinus_BXm1Empty ->getRootObject())  mePhFracEndCapMinus_BXm1Empty  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracEndCapMinus_BXm1Empty    = map_of_MEs[DirName+"/"+"NHFracEndCapMinus_BXm1Empty"];     if (  meNHFracEndCapMinus_BXm1Empty  && meNHFracEndCapMinus_BXm1Empty ->getRootObject())  meNHFracEndCapMinus_BXm1Empty  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracEndCapMinus_BXm1Empty    = map_of_MEs[DirName+"/"+"CHFracEndCapMinus_BXm1Empty"];     if (  meCHFracEndCapMinus_BXm1Empty  && meCHFracEndCapMinus_BXm1Empty ->getRootObject())  meCHFracEndCapMinus_BXm1Empty  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtEndCapMinus_BXm1Empty    = map_of_MEs[DirName+"/"+"PtEndCapMinus_BXm1Empty"];     if (  mePtEndCapMinus_BXm1Empty  && mePtEndCapMinus_BXm1Empty ->getRootObject())  mePtEndCapMinus_BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() >= 1.3 && correctedJet.eta() < 3.0) {
	    mePhFracEndCapPlus_BXm1Empty    = map_of_MEs[DirName+"/"+"PhFracEndCapPlus_BXm1Empty"];     if (  mePhFracEndCapPlus_BXm1Empty  && mePhFracEndCapPlus_BXm1Empty ->getRootObject())  mePhFracEndCapPlus_BXm1Empty  ->Fill((*pfJets)[ijet].photonEnergyFraction());
	    meNHFracEndCapPlus_BXm1Empty    = map_of_MEs[DirName+"/"+"NHFracEndCapPlus_BXm1Empty"];     if (  meNHFracEndCapPlus_BXm1Empty  && meNHFracEndCapPlus_BXm1Empty ->getRootObject())  meNHFracEndCapPlus_BXm1Empty  ->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	    meCHFracEndCapPlus_BXm1Empty    = map_of_MEs[DirName+"/"+"CHFracEndCapPlus_BXm1Empty"];     if (  meCHFracEndCapPlus_BXm1Empty  && meCHFracEndCapPlus_BXm1Empty ->getRootObject())  meCHFracEndCapPlus_BXm1Empty  ->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	    mePtEndCapPlus_BXm1Empty    = map_of_MEs[DirName+"/"+"PtEndCapPlus_BXm1Empty"];     if (  mePtEndCapPlus_BXm1Empty  && mePtEndCapPlus_BXm1Empty ->getRootObject())  mePtEndCapPlus_BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	  }else if (correctedJet.eta() > -5.0 && correctedJet.eta() <= -3.0) {
	    mePtForwardMinus_BXm1Empty    = map_of_MEs[DirName+"/"+"PtForwardMinus_BXm1Empty"];     if (  mePtForwardMinus_BXm1Empty  && mePtForwardMinus_BXm1Empty ->getRootObject())  mePtForwardMinus_BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	    meHFHFracMinus_BXm1Empty    = map_of_MEs[DirName+"/"+"HFHFracMinus_BXm1Empty"];     if (  meHFHFracMinus_BXm1Empty  && meHFHFracMinus_BXm1Empty ->getRootObject())  meHFHFracMinus_BXm1Empty  ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    meHFEMFracMinus_BXm1Empty    = map_of_MEs[DirName+"/"+"HFEMFracMinus_BXm1Empty"];     if (  meHFEMFracMinus_BXm1Empty  && meHFEMFracMinus_BXm1Empty ->getRootObject())  meHFEMFracMinus_BXm1Empty  ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	  }else if (correctedJet.eta() >= 3.0 && correctedJet.eta() < 5.0) {
	    mePtForwardPlus_BXm1Empty    = map_of_MEs[DirName+"/"+"PtForwardPlus_BXm1Empty"];     if (  mePtForwardPlus_BXm1Empty  && mePtForwardPlus_BXm1Empty ->getRootObject())  mePtForwardPlus_BXm1Empty  ->Fill((*pfJets)[ijet].pt());
	    meHFHFracPlus_BXm1Empty    = map_of_MEs[DirName+"/"+"HFHFracPlus_BXm1Empty"];     if (  meHFHFracPlus_BXm1Empty  && meHFHFracPlus_BXm1Empty ->getRootObject())  meHFHFracPlus_BXm1Empty  ->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	    meHFEMFracPlus_BXm1Empty    = map_of_MEs[DirName+"/"+"HFEMFracPlus_BXm1Empty"];     if (  meHFEMFracPlus_BXm1Empty  && meHFEMFracPlus_BXm1Empty ->getRootObject())  meHFEMFracPlus_BXm1Empty  ->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	  }
	}
	mChargedHadronEnergy = map_of_MEs[DirName+"/"+"ChargedHadronEnergy"]; if (mChargedHadronEnergy && mChargedHadronEnergy->getRootObject())  mChargedHadronEnergy->Fill ((*pfJets)[ijet].chargedHadronEnergy());
	mNeutralHadronEnergy = map_of_MEs[DirName+"/"+"NeutralHadronEnergy"]; if (mNeutralHadronEnergy && mNeutralHadronEnergy->getRootObject())  mNeutralHadronEnergy->Fill ((*pfJets)[ijet].neutralHadronEnergy());
	mChargedEmEnergy = map_of_MEs[DirName+"/"+"ChargedEmEnergy"]; if (mChargedEmEnergy && mChargedEmEnergy->getRootObject()) mChargedEmEnergy->Fill((*pfJets)[ijet].chargedEmEnergy());
	mChargedMuEnergy = map_of_MEs[DirName+"/"+"ChargedMuEnergy"]; if (mChargedMuEnergy && mChargedMuEnergy->getRootObject()) mChargedMuEnergy->Fill ((*pfJets)[ijet].chargedMuEnergy ());
	mNeutralEmEnergy = map_of_MEs[DirName+"/"+"NeutralEmEnergy"]; if (mNeutralEmEnergy && mNeutralEmEnergy->getRootObject()) mNeutralEmEnergy->Fill((*pfJets)[ijet].neutralEmEnergy());
	mChargedMultiplicity = map_of_MEs[DirName+"/"+"ChargedMultiplicity"]; if (mChargedMultiplicity && mChargedMultiplicity->getRootObject()) mChargedMultiplicity->Fill((*pfJets)[ijet].chargedMultiplicity());
	mNeutralMultiplicity = map_of_MEs[DirName+"/"+"NeutralMultiplicity"]; if (mNeutralMultiplicity && mNeutralMultiplicity->getRootObject()) mNeutralMultiplicity->Fill((*pfJets)[ijet].neutralMultiplicity());
	mMuonMultiplicity = map_of_MEs[DirName+"/"+"MuonMultiplicity"]; if (mMuonMultiplicity && mMuonMultiplicity->getRootObject()) mMuonMultiplicity->Fill ((*pfJets)[ijet]. muonMultiplicity());
	//_______________________________________________________
	mNeutralFraction = map_of_MEs[DirName+"/"+"NeutralConstituentsFraction"];if (mNeutralFraction && mNeutralFraction->getRootObject()) mNeutralFraction->Fill ((double)(*pfJets)[ijet].neutralMultiplicity()/(double)(*pfJets)[ijet].nConstituents());
	mChargedHadronEnergy_profile = map_of_MEs[DirName+"/"+"ChargedHadronEnergy_profile"]; if (mChargedHadronEnergy_profile && mChargedHadronEnergy_profile->getRootObject()) mChargedHadronEnergy_profile->Fill(numPV, (*pfJets)[ijet].chargedHadronEnergy());
	mNeutralHadronEnergy_profile = map_of_MEs[DirName+"/"+"NeutralHadronEnergy_profile"];if (mNeutralHadronEnergy_profile && mNeutralHadronEnergy_profile->getRootObject()) mNeutralHadronEnergy_profile->Fill(numPV, (*pfJets)[ijet].neutralHadronEnergy());
	mChargedEmEnergy_profile = map_of_MEs[DirName+"/"+"ChargedEmEnergy_profile"]; if (mChargedEmEnergy_profile && mChargedEmEnergy_profile->getRootObject())     mChargedEmEnergy_profile    ->Fill(numPV, (*pfJets)[ijet].chargedEmEnergy());
	mChargedMuEnergy_profile = map_of_MEs[DirName+"/"+"ChargedMuEnergy_profile"];if (mChargedMuEnergy_profile && mChargedMuEnergy_profile->getRootObject())     mChargedMuEnergy_profile    ->Fill(numPV, (*pfJets)[ijet].chargedMuEnergy ());
	mNeutralEmEnergy_profile = map_of_MEs[DirName+"/"+"NeutralEmEnergy_profile"];if (mNeutralEmEnergy_profile && mNeutralEmEnergy_profile->getRootObject())     mNeutralEmEnergy_profile    ->Fill(numPV, (*pfJets)[ijet].neutralEmEnergy());
	mChargedMultiplicity_profile = map_of_MEs[DirName+"/"+"ChargedMultiplicity_profile"]; if (mChargedMultiplicity_profile && mChargedMultiplicity_profile->getRootObject()) mChargedMultiplicity_profile->Fill(numPV, (*pfJets)[ijet].chargedMultiplicity());
	mNeutralMultiplicity_profile = map_of_MEs[DirName+"/"+"NeutralMultiplicity_profile"];if (mNeutralMultiplicity_profile && mNeutralMultiplicity_profile->getRootObject()) mNeutralMultiplicity_profile->Fill(numPV, (*pfJets)[ijet].neutralMultiplicity());
	mMuonMultiplicity_profile = map_of_MEs[DirName+"/"+"MuonMultiplicity_profile"]; if (mMuonMultiplicity_profile && mMuonMultiplicity_profile->getRootObject())    mMuonMultiplicity_profile   ->Fill(numPV, (*pfJets)[ijet].muonMultiplicity());	
      }//cleaned PFJets
    }//PFJet specific loop
    //IDs have been defined by now
    //check here already for ordering of jets -> if we choose later to soften pt-thresholds for physics selections
    //compared to the default jet histograms
    if(pass_Z_selection){//if Z selection not passed, don't need to find out of muons and Jets are overlapping
      if(deltaR((*Muons)[mu_index0].eta(),(*Muons)[mu_index0].phi(),correctedJet.eta(),correctedJet.phi())>0.2 && deltaR((*Muons)[mu_index1].eta(),(*Muons)[mu_index1].phi(),correctedJet.eta(),correctedJet.phi())>0.2 ){
	if(correctedJet.pt()>pt1_mu_vetoed){
	  pt2_mu_vetoed=pt1_mu_vetoed;
	  ind2_mu_vetoed=ind1_mu_vetoed;
	  cleaned_second_jet_mu_vetoed=cleaned_first_jet_mu_vetoed;
	  pt1_mu_vetoed=correctedJet.pt();
	  ind1_mu_vetoed=ijet;
	  cleaned_first_jet_mu_vetoed=JetIDWPU;
	} else if(correctedJet.pt()>pt2_mu_vetoed){
	  pt2_mu_vetoed=correctedJet.pt();
	  ind2_mu_vetoed=ijet;
	  cleaned_second_jet_mu_vetoed=JetIDWPU;
	}
      } 
    }
    
    if(correctedJet.pt()>pt1){
      pt3=pt2;
      ind3=ind2;
      //cleaned_third_jet=cleaned_second_jet;
      pt2=pt1;
      ind2=ind1;
      cleaned_second_jet=cleaned_first_jet;
      pt1=correctedJet.pt();
      ind1=ijet;
      cleaned_first_jet=JetIDWPU;
    } else if(correctedJet.pt()>pt2){
      pt3=pt2;
      ind3=ind2;
      //cleaned_third_jet=cleaned_second_jet;
      pt2=correctedJet.pt();
      ind2=ijet;
      cleaned_second_jet=JetIDWPU;
    } else if(correctedJet.pt()>pt3){
      pt3=correctedJet.pt();
      ind3=ijet;
      //cleaned_third_jet=JetIDWPU;
    }
    if(!pass_corrected){
      continue;
    }
    //after jettype specific variables are filled -> perform histograms for all jets
    //fill JetID efficiencies if uncleaned selection is chosen
    if(!runcosmics_ && pass_corrected){
      if(jetpassid) {
	mLooseJIDPassFractionVSeta = map_of_MEs[DirName+"/"+"JetIDPassFractionVSeta"]; if (mLooseJIDPassFractionVSeta && mLooseJIDPassFractionVSeta->getRootObject())  mLooseJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	mLooseJIDPassFractionVSpt = map_of_MEs[DirName+"/"+"JetIDPassFractionVSpt"]; if (mLooseJIDPassFractionVSpt && mLooseJIDPassFractionVSpt->getRootObject()) mLooseJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	if(fabs(correctedJet.eta())<3.0){
	  mLooseJIDPassFractionVSptNoHF= map_of_MEs[DirName+"/"+"JetIDPassFractionVSptNoHF"]; if (mLooseJIDPassFractionVSptNoHF && mLooseJIDPassFractionVSptNoHF->getRootObject()) mLooseJIDPassFractionVSptNoHF->Fill(correctedJet.pt(),1.);
	}
      } else {
	mLooseJIDPassFractionVSeta = map_of_MEs[DirName+"/"+"JetIDPassFractionVSeta"]; if (mLooseJIDPassFractionVSeta && mLooseJIDPassFractionVSeta->getRootObject()) mLooseJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	mLooseJIDPassFractionVSpt = map_of_MEs[DirName+"/"+"JetIDPassFractionVSpt"]; if (mLooseJIDPassFractionVSpt && mLooseJIDPassFractionVSpt->getRootObject()) mLooseJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	if(fabs(correctedJet.eta())<3.0){
	  mLooseJIDPassFractionVSptNoHF= map_of_MEs[DirName+"/"+"JetIDPassFractionVSptNoHF"]; if (mLooseJIDPassFractionVSptNoHF && mLooseJIDPassFractionVSptNoHF->getRootObject()) mLooseJIDPassFractionVSptNoHF->Fill(correctedJet.pt(),0.);
	}
      }
    }
    //here we so far consider calojets ->check for PFJets and JPT jets again
    if(Thiscleaned && pass_corrected){//might be softer than loose jet ID 
      numofjets++;
      if(isCaloJet_){
	jetME = map_of_MEs[DirName+"/"+"jetReco"]; if(jetME && jetME->getRootObject()) jetME->Fill(1);
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(correctedJet.pt()/(*caloJets)[ijet].pt());
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(correctedJet.eta(),correctedJet.pt()/(*caloJets)[ijet].pt());
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(correctedJet.pt(),correctedJet.pt()/(*caloJets)[ijet].pt());
      }
      if(isPFJet_){
	jetME = map_of_MEs[DirName+"/"+"jetReco"]; if(jetME && jetME->getRootObject()) jetME->Fill(2);
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(correctedJet.pt()/(*pfJets)[ijet].pt());
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject())mJetEnergyCorrVSEta->Fill(correctedJet.eta(),correctedJet.pt()/(*pfJets)[ijet].pt());
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(correctedJet.pt(),correctedJet.pt()/(*pfJets)[ijet].pt());
      }
      if(isMiniAODJet_){
	jetME = map_of_MEs[DirName+"/"+"jetReco"]; if(jetME && jetME->getRootObject()) jetME->Fill(4);
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(1./(*patJets)[ijet].jecFactor("Uncorrected"));
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject())mJetEnergyCorrVSEta->Fill(correctedJet.eta(),1./(*patJets)[ijet].jecFactor("Uncorrected"));
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(correctedJet.pt(),1./(*patJets)[ijet].jecFactor("Uncorrected"));
      }
      // --- Event passed the low pt jet trigger
      if (jetLoPass_ == 1) {	  
	mPhi_Lo = map_of_MEs[DirName+"/"+"Phi_Lo"]; if (mPhi_Lo && mPhi_Lo->getRootObject()) mPhi_Lo->Fill (correctedJet.phi());
	mPt_Lo = map_of_MEs[DirName+"/"+"Pt_Lo"]; if (mPt_Lo && mPt_Lo->getRootObject())  mPt_Lo->Fill (correctedJet.pt());
	
      }
      // --- Event passed the high pt jet trigger
      if (jetHiPass_ == 1&& correctedJet.pt()>100. ) {
	mEta_Hi = map_of_MEs[DirName+"/"+"Eta_Hi"]; if (mEta_Hi && mEta_Hi->getRootObject()) mEta_Hi->Fill (correctedJet.eta());
	mPhi_Hi = map_of_MEs[DirName+"/"+"Phi_Hi"]; if (mPhi_Hi && mPhi_Hi->getRootObject()) mPhi_Hi->Fill (correctedJet.phi());
	mPt_Hi = map_of_MEs[DirName+"/"+"Pt_Hi"]; if (mPt_Hi && mPt_Hi->getRootObject()) mPt_Hi->Fill (correctedJet.pt());		     		    
      }
      mPt = map_of_MEs[DirName+"/"+"Pt"]; if (mPt && mPt->getRootObject()) mPt->Fill (correctedJet.pt());
      mPt_1 = map_of_MEs[DirName+"/"+"Pt_1"]; if (mPt_1 && mPt_1->getRootObject())  mPt_1->Fill (correctedJet.pt());
      mPt_2 = map_of_MEs[DirName+"/"+"Pt_2"]; if (mPt_2 && mPt_2->getRootObject()) mPt_2->Fill (correctedJet.pt());
      mPt_3 = map_of_MEs[DirName+"/"+"Pt_3"]; if (mPt_3 && mPt_3->getRootObject()) mPt_3->Fill (correctedJet.pt());
      mEta = map_of_MEs[DirName+"/"+"Eta"]; if (mEta && mEta->getRootObject()) mEta->Fill (correctedJet.eta());
      mPhi = map_of_MEs[DirName+"/"+"Phi"]; if (mPhi && mPhi->getRootObject())  mPhi->Fill (correctedJet.phi());	 
      
      mPhiVSEta = map_of_MEs[DirName+"/"+"PhiVSEta"]; if (mPhiVSEta && mPhiVSEta->getRootObject()) mPhiVSEta->Fill(correctedJet.eta(),correctedJet.phi());
      //if(!isJPTJet_){
      mConstituents = map_of_MEs[DirName+"/"+"Constituents"]; if (mConstituents && mConstituents->getRootObject()) mConstituents->Fill (correctedJet.nConstituents());
      //}
      // Fill NPV profiles
      //--------------------------------------------------------------------
      mPt_profile = map_of_MEs[DirName+"/"+"Pt_profile"]; if (mPt_profile && mPt_profile->getRootObject())  mPt_profile ->Fill(numPV, correctedJet.pt());
      mEta_profile = map_of_MEs[DirName+"/"+"Eta_profile"]; if (mEta_profile && mEta_profile->getRootObject())  mEta_profile         ->Fill(numPV, correctedJet.eta());
      mPhi_profile = map_of_MEs[DirName+"/"+"Phi_profile"]; if (mPhi_profile && mPhi_profile->getRootObject())  mPhi_profile         ->Fill(numPV, correctedJet.phi());
      //if(!isJPTJet_){
      mConstituents_profile = map_of_MEs[DirName+"/"+"Constituents_profile"]; if (mConstituents_profile && mConstituents_profile->getRootObject())  mConstituents_profile->Fill(numPV, correctedJet.nConstituents());
      //}
      if (fabs(correctedJet.eta()) <= 1.3) {
	mPt_Barrel = map_of_MEs[DirName+"/"+"Pt_Barrel"]; if (mPt_Barrel && mPt_Barrel->getRootObject()) mPt_Barrel->Fill (correctedJet.pt());
	mPhi_Barrel = map_of_MEs[DirName+"/"+"Phi_Barrel"]; if (mPhi_Barrel && mPhi_Barrel->getRootObject()) mPhi_Barrel->Fill (correctedJet.phi());
	//if (mE_Barrel)    mE_Barrel->Fill (correctedJet.energy());
	//if(!isJPTJet_){
	mConstituents_Barrel = map_of_MEs[DirName+"/"+"Constituents_Barrel"]; if (mConstituents_Barrel && mConstituents_Barrel->getRootObject()) mConstituents_Barrel->Fill(correctedJet.nConstituents());
	//}
      }else if (fabs(correctedJet.eta()) <= 3) {
	mPt_EndCap = map_of_MEs[DirName+"/"+"Pt_EndCap"]; if (mPt_EndCap && mPt_EndCap->getRootObject()) mPt_EndCap->Fill (correctedJet.pt());
	mPhi_EndCap = map_of_MEs[DirName+"/"+"Phi_EndCap"]; if (mPhi_EndCap && mPhi_EndCap->getRootObject())  mPhi_EndCap->Fill (correctedJet.phi());
	//if (mE_EndCap)    mE_EndCap->Fill (correctedJet.energy());
	//if(!isJPTJet_){
	mConstituents_EndCap = map_of_MEs[DirName+"/"+"Constituents_EndCap"]; if (mConstituents_EndCap && mConstituents_EndCap->getRootObject())  mConstituents_EndCap->Fill(correctedJet.nConstituents());
	//}
      }else{
	mPt_Forward = map_of_MEs[DirName+"/"+"Pt_Forward"]; if (mPt_Forward && mPt_Forward->getRootObject()) mPt_Forward->Fill (correctedJet.pt());
	mPhi_Forward = map_of_MEs[DirName+"/"+"Phi_Forward"]; if (mPhi_Forward && mPhi_Forward->getRootObject()) mPhi_Forward->Fill (correctedJet.phi());
	//if (mE_Forward)    mE_Forward->Fill (correctedJet.energy());
	//if(!isJPTJet_){
	mConstituents_Forward = map_of_MEs[DirName+"/"+"Constituents_Forward"]; if (mConstituents_Forward && mConstituents_Forward->getRootObject())   mConstituents_Forward->Fill(correctedJet.nConstituents());
	//}
      }
    }// pass ID for corrected jets --> inclusive selection
  }//loop over uncorrected jets 
  
  
  mNJets = map_of_MEs[DirName+"/"+"NJets"]; if (mNJets && mNJets->getRootObject())  mNJets->Fill (numofjets);
  mNJets_profile = map_of_MEs[DirName+"/"+"NJets_profile"]; if (mNJets_profile && mNJets_profile->getRootObject())  mNJets_profile->Fill(numPV, numofjets);
  
  sort(recoJets.begin(),recoJets.end(),jetSortingRule);

  //for non dijet selection, otherwise numofjets==0
  if(numofjets>0){//keep threshold for dijet counting at the original one
    //check ID of the leading jet

    if(cleaned_first_jet){
      mEtaFirst = map_of_MEs[DirName+"/"+"EtaFirst"]; if (mEtaFirst && mEtaFirst->getRootObject())  mEtaFirst->Fill ((recoJets)[0].eta());
      mPhiFirst = map_of_MEs[DirName+"/"+"PhiFirst"]; if (mPhiFirst && mPhiFirst->getRootObject())  mPhiFirst->Fill ((recoJets)[0].phi());
      mPtFirst = map_of_MEs[DirName+"/"+"PtFirst"]; if (mPtFirst && mPtFirst->getRootObject())  mPtFirst->Fill ((recoJets)[0].pt());
      //check ID of second check for DPhi plots
      if(numofjets>1 && cleaned_second_jet) {
	double dphi=fabs((recoJets)[0].phi()-(recoJets)[1].phi());
	if(dphi>acos(-1.)){
	  dphi=2*acos(-1.)-dphi;
	}
	mDPhi = map_of_MEs[DirName+"/"+"DPhi"]; if (mDPhi && mDPhi->getRootObject())  mDPhi->Fill (dphi);
      }
    }
    //if(cleaned_second_jet && isJPTJet_){
    //mPtSecond = map_of_MEs[DirName+"/"+"PtSecond"]; if (mPtSecond && mPtSecond->getRootObject())  mPtSecond->Fill(recoJets[1].pt());
    //}
    //if(cleaned_third_jet && isJPTJet_){
    //mPtThird = map_of_MEs[DirName+"/"+"PtThird"]; if (mPtThird && mPtThird->getRootObject())  mPtThird->Fill(recoJets[2].pt());
    //}
  }
  //dijet selection -> recoJets are on corrected level, both jets cleaned, fill folder only for cleaned jet selection
  if(jetCleaningFlag_ && recoJets.size()>1 && cleaned_first_jet && cleaned_second_jet ){
    //pt threshold checked before filling
    if(jetCleaningFlag_){
      DirName = "JetMET/Jet/Cleaned"+mInputCollection_.label()+"/DiJet";
    }
    //if(fabs(recoJets[0].eta())<3. && fabs(recoJets[1].eta())<3. ){
    //calculate dphi
    double dphi=fabs((recoJets)[0].phi()-(recoJets)[1].phi());
    if(dphi>acos(-1.)){
      dphi=2*acos(-1.)-dphi;
    } 
    mDPhi = map_of_MEs[DirName+"/"+"DPhi"]; if (mDPhi && mDPhi->getRootObject()) if (mDPhi) mDPhi->Fill (dphi);
    //dphi cut
    if(fabs(dphi)>2.1){
      if(isCaloJet_){
	if(!runcosmics_){
	  reco::CaloJetRef calojetref1(caloJets, ind1);
	  reco::JetID jetID1 = (*jetID_ValueMap_Handle)[calojetref1];
	  reco::CaloJetRef calojetref2(caloJets, ind2);
	  reco::JetID jetID2 = (*jetID_ValueMap_Handle)[calojetref2];	
	  mN90Hits = map_of_MEs[DirName+"/"+"N90Hits"]; if (mN90Hits && mN90Hits->getRootObject()) mN90Hits->Fill (jetID1.n90Hits);
	  mfHPD = map_of_MEs[DirName+"/"+"fHPD"]; if (mfHPD && mfHPD->getRootObject())             mfHPD->Fill (jetID1.fHPD);
	  mresEMF = map_of_MEs[DirName+"/"+"resEMF"]; if (mresEMF && mresEMF->getRootObject())     mresEMF->Fill (jetID1.restrictedEMF);
	  mfRBX = map_of_MEs[DirName+"/"+"fRBX"]; if (mfRBX && mfRBX->getRootObject())             mfRBX->Fill (jetID1.fRBX);	  
	  mN90Hits = map_of_MEs[DirName+"/"+"N90Hits"]; if (mN90Hits && mN90Hits->getRootObject()) mN90Hits->Fill (jetID2.n90Hits);
	  mfHPD = map_of_MEs[DirName+"/"+"fHPD"]; if (mfHPD && mfHPD->getRootObject())             mfHPD->Fill (jetID2.fHPD);
	  mresEMF = map_of_MEs[DirName+"/"+"resEMF"]; if (mresEMF && mresEMF->getRootObject())     mresEMF->Fill (jetID2.restrictedEMF);
	  mfRBX = map_of_MEs[DirName+"/"+"fRBX"]; if (mfRBX && mfRBX->getRootObject())             mfRBX->Fill (jetID2.fRBX);
	}
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject()) mHFrac->Fill ((*caloJets)[ind1].energyFractionHadronic());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject()) mEFrac->Fill ((*caloJets)[ind1].emEnergyFraction());
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject()) mHFrac->Fill ((*caloJets)[ind2].energyFractionHadronic());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject()) mEFrac->Fill ((*caloJets)[ind2].emEnergyFraction());
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject())   mHFrac_profile       ->Fill(numPV, (*caloJets)[ind1].energyFractionHadronic());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject())   mEFrac_profile       ->Fill(numPV, (*caloJets)[ind1].emEnergyFraction());
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject())   mHFrac_profile       ->Fill(numPV, (*caloJets)[ind2].energyFractionHadronic());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject())   mEFrac_profile       ->Fill(numPV, (*caloJets)[ind2].emEnergyFraction());
	
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(recoJets[0].pt()/(*caloJets)[ind1].pt());
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[0].eta(),recoJets[0].pt()/(*caloJets)[ind1].pt());
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[0].pt(),recoJets[0].pt()/(*caloJets)[ind1].pt());
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(recoJets[1].pt()/(*caloJets)[ind2].pt());
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[1].eta(),recoJets[1].pt()/(*caloJets)[ind2].pt());
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[1].pt(),recoJets[1].pt()/(*caloJets)[ind2].pt());
      }
      //if(isJPTJet_){
      //mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject())     mHFrac->Fill ((*jptJets)[ind1].chargedHadronEnergyFraction()+(*jptJets)[ind1].neutralHadronEnergyFraction());
      //mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject())     mEFrac->Fill (1.-(*jptJets)[ind1].chargedHadronEnergyFraction()-(*jptJets)[ind1].neutralHadronEnergyFraction());
      //mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"];      mHFrac_profile       ->Fill(numPV, (*jptJets)[ind1].chargedHadronEnergyFraction()+(*jptJets)[ind1].neutralHadronEnergyFraction());
      //mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"];      mEFrac_profile       ->Fill(numPV, 1.-(*jptJets)[ind1].chargedHadronEnergyFraction()-(*jptJets)[ind1].neutralHadronEnergyFraction());
      //mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject())     mHFrac->Fill ((*jptJets)[ind2].chargedHadronEnergyFraction()+(*jptJets)[ind2].neutralHadronEnergyFraction());
      //mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject())     mEFrac->Fill (1.-(*jptJets)[ind2].chargedHadronEnergyFraction()-(*jptJets)[ind2].neutralHadronEnergyFraction());
      //mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"];      mHFrac_profile       ->Fill(numPV, (*jptJets)[ind2].chargedHadronEnergyFraction()+(*jptJets)[ind2].neutralHadronEnergyFraction());
      //mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"];      mEFrac_profile       ->Fill(numPV, 1.-(*jptJets)[ind2].chargedHadronEnergyFraction()-(*jptJets)[ind2].neutralHadronEnergyFraction());
      //
      //mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(recoJets[0].pt()/(*jptJets)[ind1].pt());
      //mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[0].eta(),recoJets[0].pt()/(*jptJets)[ind1].pt());
      //mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[0].pt(),recoJets[0].pt()/(*jptJets)[ind1].pt());
      //mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(recoJets[1].pt()/(*jptJets)[ind2].pt());
      //mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[1].eta(),recoJets[1].pt()/(*jptJets)[ind2].pt());
      //mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[1].pt(),recoJets[1].pt()/(*jptJets)[ind2].pt());
      //}
      if(isPFJet_){
	mCHFrac = map_of_MEs[DirName+"/"+"CHFrac"]; if (mCHFrac && mCHFrac->getRootObject())         mCHFrac ->Fill((*pfJets)[ind1].chargedHadronEnergyFraction());
	mNHFrac = map_of_MEs[DirName+"/"+"NHFrac"]; if (mNHFrac && mNHFrac->getRootObject())         mNHFrac ->Fill((*pfJets)[ind1].neutralHadronEnergyFraction());
	mPhFrac = map_of_MEs[DirName+"/"+"PhFrac"]; if (mPhFrac && mPhFrac->getRootObject())         mPhFrac ->Fill((*pfJets)[ind1].neutralEmEnergyFraction());
	mHFEMFrac = map_of_MEs[DirName+"/"+"HFEMFrac"]; if (mHFEMFrac && mHFEMFrac->getRootObject()) mHFEMFrac ->Fill((*pfJets)[ind1].HFEMEnergyFraction());
	mHFHFrac = map_of_MEs[DirName+"/"+"HFHFrac"]; if (mHFHFrac && mHFHFrac->getRootObject())     mHFHFrac ->Fill((*pfJets)[ind1].HFHadronEnergyFraction());
	
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(recoJets[0].pt()/(*pfJets)[ind1].pt());
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[0].eta(),recoJets[0].pt()/(*pfJets)[ind1].pt());
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[0].pt(),recoJets[0].pt()/(*pfJets)[ind1].pt());
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(recoJets[1].pt()/(*pfJets)[ind2].pt());
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[1].eta(),recoJets[1].pt()/(*pfJets)[ind2].pt());
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[1].pt(),recoJets[1].pt()/(*pfJets)[ind2].pt());
	
	mChargedMultiplicity = map_of_MEs[DirName+"/"+"ChargedMultiplicity"]; if(mChargedMultiplicity && mChargedMultiplicity->getRootObject()) mChargedMultiplicity->Fill((*pfJets)[ind1].chargedMultiplicity());
	mNeutralMultiplicity = map_of_MEs[DirName+"/"+"NeutralMultiplicity"]; if(mNeutralMultiplicity && mNeutralMultiplicity->getRootObject()) mNeutralMultiplicity->Fill((*pfJets)[ind1].neutralMultiplicity());
	mMuonMultiplicity = map_of_MEs[DirName+"/"+"MuonMultiplicity"]; if(mMuonMultiplicity && mMuonMultiplicity->getRootObject())             mMuonMultiplicity->Fill((*pfJets)[ind1].muonMultiplicity());
	//Filling variables for second jet	
	mCHFrac = map_of_MEs[DirName+"/"+"CHFrac"]; if (mCHFrac && mCHFrac->getRootObject())         mCHFrac ->Fill((*pfJets)[ind2].chargedHadronEnergyFraction());
	mNHFrac = map_of_MEs[DirName+"/"+"NHFrac"]; if (mNHFrac && mNHFrac->getRootObject())         mNHFrac ->Fill((*pfJets)[ind2].neutralHadronEnergyFraction());
	mPhFrac = map_of_MEs[DirName+"/"+"PhFrac"]; if (mPhFrac && mPhFrac->getRootObject())         mPhFrac ->Fill((*pfJets)[ind2].neutralEmEnergyFraction());
	mHFEMFrac = map_of_MEs[DirName+"/"+"HFEMFrac"]; if (mHFEMFrac && mHFEMFrac->getRootObject()) mHFEMFrac ->Fill((*pfJets)[ind2].HFEMEnergyFraction());
	mHFHFrac = map_of_MEs[DirName+"/"+"HFHFrac"]; if (mHFHFrac && mHFHFrac->getRootObject())     mHFHFrac ->Fill((*pfJets)[ind2].HFHadronEnergyFraction());
	
	mNeutralFraction = map_of_MEs[DirName+"/"+"NeutralConstituentsFraction"];if (mNeutralFraction && mNeutralFraction->getRootObject()) mNeutralFraction->Fill ((double)(*pfJets)[ind1].neutralMultiplicity()/(double)(*pfJets)[ind1].nConstituents());
	
	mChargedMultiplicity = map_of_MEs[DirName+"/"+"ChargedMultiplicity"]; if(mChargedMultiplicity && mChargedMultiplicity->getRootObject()) mChargedMultiplicity->Fill((*pfJets)[ind2].chargedMultiplicity());
	mNeutralMultiplicity = map_of_MEs[DirName+"/"+"NeutralMultiplicity"]; if(mNeutralMultiplicity && mNeutralMultiplicity->getRootObject()) mNeutralMultiplicity->Fill((*pfJets)[ind2].neutralMultiplicity());
	mMuonMultiplicity = map_of_MEs[DirName+"/"+"MuonMultiplicity"]; if(mMuonMultiplicity && mMuonMultiplicity->getRootObject())             mMuonMultiplicity->Fill((*pfJets)[ind2].muonMultiplicity());
	
	//now fill PFJet profiles for leading jet
	mCHFrac_profile = map_of_MEs[DirName+"/"+"CHFrac_profile"]; if (mCHFrac_profile && mCHFrac_profile->getRootObject())         mCHFrac_profile ->Fill(numPV, (*pfJets)[ind1].chargedHadronEnergyFraction());
	mNHFrac_profile = map_of_MEs[DirName+"/"+"NHFrac_profile"]; if (mNHFrac_profile && mNHFrac_profile->getRootObject())         mNHFrac_profile ->Fill(numPV, (*pfJets)[ind1].neutralHadronEnergyFraction());
	mPhFrac_profile = map_of_MEs[DirName+"/"+"PhFrac_profile"]; if (mPhFrac_profile && mPhFrac_profile->getRootObject())         mPhFrac_profile ->Fill(numPV, (*pfJets)[ind1].neutralEmEnergyFraction());
	mHFEMFrac_profile = map_of_MEs[DirName+"/"+"HFEMFrac_profile"]; if (mHFEMFrac_profile && mHFEMFrac_profile->getRootObject()) mHFEMFrac_profile ->Fill(numPV, (*pfJets)[ind1].HFEMEnergyFraction());
	mHFHFrac_profile = map_of_MEs[DirName+"/"+"HFHFrac_profile"]; if (mHFHFrac_profile && mHFHFrac_profile->getRootObject())     mHFHFrac_profile ->Fill(numPV, (*pfJets)[ind1].HFHadronEnergyFraction());
	
	mNeutralFraction = map_of_MEs[DirName+"/"+"NeutralConstituentsFraction"];if (mNeutralFraction && mNeutralFraction->getRootObject()) mNeutralFraction->Fill ((double)(*pfJets)[ind2].neutralMultiplicity()/(double)(*pfJets)[ind2].nConstituents());
	
	mChargedMultiplicity_profile = map_of_MEs[DirName+"/"+"ChargedMultiplicity_profile"]; if(mChargedMultiplicity_profile && mChargedMultiplicity_profile->getRootObject()) mChargedMultiplicity_profile->Fill(numPV, (*pfJets)[ind1].chargedMultiplicity());
	mNeutralMultiplicity_profile = map_of_MEs[DirName+"/"+"NeutralMultiplicity_profile"]; if(mNeutralMultiplicity_profile && mNeutralMultiplicity_profile->getRootObject()) mNeutralMultiplicity_profile->Fill(numPV, (*pfJets)[ind1].neutralMultiplicity());
	mMuonMultiplicity_profile = map_of_MEs[DirName+"/"+"MuonMultiplicity_profile"]; if(mMuonMultiplicity_profile && mMuonMultiplicity_profile->getRootObject())             mMuonMultiplicity->Fill(numPV, (*pfJets)[ind1].muonMultiplicity());
	//now fill PFJet profiles for second leading jet
	mCHFrac_profile = map_of_MEs[DirName+"/"+"CHFrac_profile"]; if (mCHFrac_profile && mCHFrac_profile->getRootObject())         mCHFrac_profile ->Fill(numPV, (*pfJets)[ind2].chargedHadronEnergyFraction());
	mNHFrac_profile = map_of_MEs[DirName+"/"+"NHFrac_profile"]; if (mNHFrac_profile && mNHFrac_profile->getRootObject())         mNHFrac_profile ->Fill(numPV, (*pfJets)[ind2].neutralHadronEnergyFraction());
	mPhFrac_profile = map_of_MEs[DirName+"/"+"PhFrac_profile"]; if (mPhFrac_profile && mPhFrac_profile->getRootObject())         mPhFrac_profile ->Fill(numPV, (*pfJets)[ind2].neutralEmEnergyFraction());
	mHFEMFrac_profile = map_of_MEs[DirName+"/"+"HFEMFrac_profile"]; if (mHFEMFrac_profile && mHFEMFrac_profile->getRootObject()) mHFEMFrac_profile ->Fill(numPV, (*pfJets)[ind2].HFEMEnergyFraction());
	mHFHFrac_profile = map_of_MEs[DirName+"/"+"HFHFrac_profile"]; if (mHFHFrac_profile && mHFHFrac_profile->getRootObject())     mHFHFrac_profile ->Fill(numPV, (*pfJets)[ind2].HFHadronEnergyFraction());
	
	mChargedMultiplicity_profile = map_of_MEs[DirName+"/"+"ChargedMultiplicity_profile"]; if(mChargedMultiplicity_profile && mChargedMultiplicity_profile->getRootObject()) mChargedMultiplicity_profile->Fill(numPV, (*pfJets)[ind2].chargedMultiplicity());
	mNeutralMultiplicity_profile = map_of_MEs[DirName+"/"+"NeutralMultiplicity_profile"]; if(mNeutralMultiplicity_profile && mNeutralMultiplicity_profile->getRootObject()) mNeutralMultiplicity_profile->Fill(numPV, (*pfJets)[ind2].neutralMultiplicity());
	mMuonMultiplicity_profile = map_of_MEs[DirName+"/"+"MuonMultiplicity_profile"]; if(mMuonMultiplicity_profile && mMuonMultiplicity_profile->getRootObject())             mMuonMultiplicity_profile->Fill(numPV, (*pfJets)[ind2].muonMultiplicity());
 

	int QGmulti=-1;
	float QGLikelihood=-10;
	float QGptD=-10;
	float QGaxis2=-10;
	if(fill_CHS_histos){
	  reco::PFJetRef pfjetref(pfJets,ind1);
	  QGmulti=(*qgMultiplicity)[pfjetref];
	  QGLikelihood=(*qgLikelihood)[pfjetref];
	  QGptD=(*qgptD)[pfjetref];
	  QGaxis2=(*qgaxis2)[pfjetref];
	  if(fabs(recoJets[0].eta())<1.3){//barrel jets
	    //fractions for barrel
	    if (recoJets[0].pt()>=20. && recoJets[0].pt()<=50.) {
	      mAxis2_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_Barrel"];if(mAxis2_lowPt_Barrel && mAxis2_lowPt_Barrel->getRootObject()) mAxis2_lowPt_Barrel->Fill(QGaxis2);
	      mpTD_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_Barrel"]; if(mpTD_lowPt_Barrel && mpTD_lowPt_Barrel->getRootObject()) mpTD_lowPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_Barrel"]; if(mMultiplicityQG_lowPt_Barrel && mMultiplicityQG_lowPt_Barrel->getRootObject()) mMultiplicityQG_lowPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_Barrel"]; if(mqgLikelihood_lowPt_Barrel && mqgLikelihood_lowPt_Barrel->getRootObject()) mqgLikelihood_lowPt_Barrel->Fill(QGLikelihood);
	    }
	    if (recoJets[0].pt()>50. && recoJets[0].pt()<=140.) {
	      mAxis2_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_Barrel"];if(mAxis2_mediumPt_Barrel && mAxis2_mediumPt_Barrel->getRootObject()) mAxis2_mediumPt_Barrel->Fill(QGaxis2);
	      mpTD_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_Barrel"]; if(mpTD_mediumPt_Barrel && mpTD_mediumPt_Barrel->getRootObject()) mpTD_mediumPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_Barrel"]; if(mMultiplicityQG_mediumPt_Barrel && mMultiplicityQG_mediumPt_Barrel->getRootObject()) mMultiplicityQG_mediumPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_Barrel"]; if(mqgLikelihood_mediumPt_Barrel && mqgLikelihood_mediumPt_Barrel->getRootObject()) mqgLikelihood_mediumPt_Barrel->Fill(QGLikelihood);
	    }
	    if (recoJets[0].pt()>140.) {
	      mAxis2_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_Barrel"];if(mAxis2_highPt_Barrel && mAxis2_highPt_Barrel->getRootObject()) mAxis2_highPt_Barrel->Fill(QGaxis2);
	      mpTD_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_highPt_Barrel"]; if(mpTD_highPt_Barrel && mpTD_highPt_Barrel->getRootObject()) mpTD_highPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_Barrel"]; if(mMultiplicityQG_highPt_Barrel && mMultiplicityQG_highPt_Barrel->getRootObject()) mMultiplicityQG_highPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_Barrel"]; if(mqgLikelihood_highPt_Barrel && mqgLikelihood_highPt_Barrel->getRootObject()) mqgLikelihood_highPt_Barrel->Fill(QGLikelihood);
	    }
	  }else if(fabs(recoJets[0].eta())<3.0){//endcap jets 
	    if (recoJets[0].pt()>20. && recoJets[0].pt()<=50.) {
	      mAxis2_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_EndCap"];if(mAxis2_lowPt_EndCap && mAxis2_lowPt_EndCap->getRootObject()) mAxis2_lowPt_EndCap->Fill(QGaxis2);
	      mpTD_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_EndCap"]; if(mpTD_lowPt_EndCap && mpTD_lowPt_EndCap->getRootObject()) mpTD_lowPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_EndCap"]; if(mMultiplicityQG_lowPt_EndCap && mMultiplicityQG_lowPt_EndCap->getRootObject()) mMultiplicityQG_lowPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_EndCap"]; if(mqgLikelihood_lowPt_EndCap && mqgLikelihood_lowPt_EndCap->getRootObject()) mqgLikelihood_lowPt_EndCap->Fill(QGLikelihood);
	    }
	    if (recoJets[0].pt()>50. && recoJets[0].pt()<=140.) {
	      mAxis2_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_EndCap"];if(mAxis2_mediumPt_EndCap && mAxis2_mediumPt_EndCap->getRootObject()) mAxis2_mediumPt_EndCap->Fill(QGaxis2);
	      mpTD_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_EndCap"]; if(mpTD_mediumPt_EndCap && mpTD_mediumPt_EndCap->getRootObject()) mpTD_mediumPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_EndCap"]; if(mMultiplicityQG_mediumPt_EndCap && mMultiplicityQG_mediumPt_EndCap->getRootObject()) mMultiplicityQG_mediumPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_EndCap"]; if(mqgLikelihood_mediumPt_EndCap && mqgLikelihood_mediumPt_EndCap->getRootObject()) mqgLikelihood_mediumPt_EndCap->Fill(QGLikelihood);
	    }
	    if (recoJets[0].pt()>140.) {
	      mAxis2_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_EndCap"];if(mAxis2_highPt_EndCap && mAxis2_highPt_EndCap->getRootObject()) mAxis2_highPt_EndCap->Fill(QGaxis2);
	      mpTD_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_highPt_EndCap"]; if(mpTD_highPt_EndCap && mpTD_highPt_EndCap->getRootObject()) mpTD_highPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_EndCap"]; if(mMultiplicityQG_highPt_EndCap && mMultiplicityQG_highPt_EndCap->getRootObject()) mMultiplicityQG_highPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_EndCap"]; if(mqgLikelihood_highPt_EndCap && mqgLikelihood_highPt_EndCap->getRootObject()) mqgLikelihood_highPt_EndCap->Fill(QGLikelihood);
	    }
	  }else{
	    if (recoJets[0].pt()>20. && recoJets[0].pt()<=50.) {
	      mAxis2_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_Forward"];if(mAxis2_lowPt_Forward && mAxis2_lowPt_Forward->getRootObject()) mAxis2_lowPt_Forward->Fill(QGaxis2);
	      mpTD_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_Forward"]; if(mpTD_lowPt_Forward && mpTD_lowPt_Forward->getRootObject()) mpTD_lowPt_Forward->Fill(QGptD);
	      mMultiplicityQG_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_Forward"]; if(mMultiplicityQG_lowPt_Forward && mMultiplicityQG_lowPt_Forward->getRootObject()) mMultiplicityQG_lowPt_Forward->Fill(QGmulti);
	      mqgLikelihood_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_Forward"]; if(mqgLikelihood_lowPt_Forward && mqgLikelihood_lowPt_Forward->getRootObject()) mqgLikelihood_lowPt_Forward->Fill(QGLikelihood);
	    }
	    if (recoJets[0].pt()>50. && recoJets[0].pt()<=140.) {
	      mAxis2_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_Forward"];if(mAxis2_mediumPt_Forward && mAxis2_mediumPt_Forward->getRootObject()) mAxis2_mediumPt_Forward->Fill(QGaxis2);
	      mpTD_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_Forward"]; if(mpTD_mediumPt_Forward && mpTD_mediumPt_Forward->getRootObject()) mpTD_mediumPt_Forward->Fill(QGptD);
	      mMultiplicityQG_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_Forward"]; if(mMultiplicityQG_mediumPt_Forward && mMultiplicityQG_mediumPt_Forward->getRootObject()) mMultiplicityQG_mediumPt_Forward->Fill(QGmulti);
	      mqgLikelihood_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_Forward"]; if(mqgLikelihood_mediumPt_Forward && mqgLikelihood_mediumPt_Forward->getRootObject()) mqgLikelihood_mediumPt_Forward->Fill(QGLikelihood);
	    }
	    if (recoJets[0].pt()>140.) {
	      mAxis2_highPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_Forward"];if(mAxis2_highPt_Forward && mAxis2_highPt_Forward->getRootObject()) mAxis2_highPt_Forward->Fill(QGaxis2);
	      mpTD_highPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_highPt_Forward"]; if(mpTD_highPt_Forward && mpTD_highPt_Forward->getRootObject()) mpTD_highPt_Forward->Fill(QGptD);
	      mMultiplicityQG_highPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_Forward"]; if(mMultiplicityQG_highPt_Forward && mMultiplicityQG_highPt_Forward->getRootObject()) mMultiplicityQG_highPt_Forward->Fill(QGmulti);
	      mqgLikelihood_highPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_Forward"]; if(mqgLikelihood_highPt_Forward && mqgLikelihood_highPt_Forward->getRootObject()) mqgLikelihood_highPt_Forward->Fill(QGLikelihood);
	    }
	  }//done for first jet
	  reco::PFJetRef pfjetref1(pfJets,ind2);
	  QGmulti=(*qgMultiplicity)[pfjetref1];
	  QGLikelihood=(*qgLikelihood)[pfjetref1];
	  QGptD=(*qgptD)[pfjetref1];
	  QGaxis2=(*qgaxis2)[pfjetref1];
	  if(fabs(recoJets[1].eta())<1.3){//barrel jets
	    //fractions for barrel
	    if (recoJets[1].pt()>=20. && recoJets[1].pt()<=50.) {
	      mAxis2_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_Barrel"];if(mAxis2_lowPt_Barrel && mAxis2_lowPt_Barrel->getRootObject()) mAxis2_lowPt_Barrel->Fill(QGaxis2);
	      mpTD_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_Barrel"]; if(mpTD_lowPt_Barrel && mpTD_lowPt_Barrel->getRootObject()) mpTD_lowPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_Barrel"]; if(mMultiplicityQG_lowPt_Barrel && mMultiplicityQG_lowPt_Barrel->getRootObject()) mMultiplicityQG_lowPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_Barrel"]; if(mqgLikelihood_lowPt_Barrel && mqgLikelihood_lowPt_Barrel->getRootObject()) mqgLikelihood_lowPt_Barrel->Fill(QGLikelihood);
	    }
	    if (recoJets[1].pt()>50. && recoJets[1].pt()<=140.) {
	      mAxis2_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_Barrel"];if(mAxis2_mediumPt_Barrel && mAxis2_mediumPt_Barrel->getRootObject()) mAxis2_mediumPt_Barrel->Fill(QGaxis2);
	      mpTD_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_Barrel"]; if(mpTD_mediumPt_Barrel && mpTD_mediumPt_Barrel->getRootObject()) mpTD_mediumPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_Barrel"]; if(mMultiplicityQG_mediumPt_Barrel && mMultiplicityQG_mediumPt_Barrel->getRootObject()) mMultiplicityQG_mediumPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_Barrel"]; if(mqgLikelihood_mediumPt_Barrel && mqgLikelihood_mediumPt_Barrel->getRootObject()) mqgLikelihood_mediumPt_Barrel->Fill(QGLikelihood);
	    }
	    if (recoJets[1].pt()>140.) {
	      mAxis2_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_Barrel"];if(mAxis2_highPt_Barrel && mAxis2_highPt_Barrel->getRootObject()) mAxis2_highPt_Barrel->Fill(QGaxis2);
	      mpTD_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_highPt_Barrel"]; if(mpTD_highPt_Barrel && mpTD_highPt_Barrel->getRootObject()) mpTD_highPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_Barrel"]; if(mMultiplicityQG_highPt_Barrel && mMultiplicityQG_highPt_Barrel->getRootObject()) mMultiplicityQG_highPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_Barrel"]; if(mqgLikelihood_highPt_Barrel && mqgLikelihood_highPt_Barrel->getRootObject()) mqgLikelihood_highPt_Barrel->Fill(QGLikelihood);
	    }
	  }else if(fabs(recoJets[1].eta())<3.0){//endcap jets 
	    if (recoJets[1].pt()>20. && recoJets[1].pt()<=50.) {
	      mAxis2_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_EndCap"];if(mAxis2_lowPt_EndCap && mAxis2_lowPt_EndCap->getRootObject()) mAxis2_lowPt_EndCap->Fill(QGaxis2);
	      mpTD_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_EndCap"]; if(mpTD_lowPt_EndCap && mpTD_lowPt_EndCap->getRootObject()) mpTD_lowPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_EndCap"]; if(mMultiplicityQG_lowPt_EndCap && mMultiplicityQG_lowPt_EndCap->getRootObject()) mMultiplicityQG_lowPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_EndCap"]; if(mqgLikelihood_lowPt_EndCap && mqgLikelihood_lowPt_EndCap->getRootObject()) mqgLikelihood_lowPt_EndCap->Fill(QGLikelihood);
	    }
	    if (recoJets[1].pt()>50. && recoJets[1].pt()<=140.) {
	      mAxis2_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_EndCap"];if(mAxis2_mediumPt_EndCap && mAxis2_mediumPt_EndCap->getRootObject()) mAxis2_mediumPt_EndCap->Fill(QGaxis2);
	      mpTD_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_EndCap"]; if(mpTD_mediumPt_EndCap && mpTD_mediumPt_EndCap->getRootObject()) mpTD_mediumPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_EndCap"]; if(mMultiplicityQG_mediumPt_EndCap && mMultiplicityQG_mediumPt_EndCap->getRootObject()) mMultiplicityQG_mediumPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_EndCap"]; if(mqgLikelihood_mediumPt_EndCap && mqgLikelihood_mediumPt_EndCap->getRootObject()) mqgLikelihood_mediumPt_EndCap->Fill(QGLikelihood);
	    }
	    if (recoJets[1].pt()>140.) {
	      mAxis2_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_EndCap"];if(mAxis2_highPt_EndCap && mAxis2_highPt_EndCap->getRootObject()) mAxis2_highPt_EndCap->Fill(QGaxis2);
	      mpTD_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_highPt_EndCap"]; if(mpTD_highPt_EndCap && mpTD_highPt_EndCap->getRootObject()) mpTD_highPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_EndCap"]; if(mMultiplicityQG_highPt_EndCap && mMultiplicityQG_highPt_EndCap->getRootObject()) mMultiplicityQG_highPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_EndCap"]; if(mqgLikelihood_highPt_EndCap && mqgLikelihood_highPt_EndCap->getRootObject()) mqgLikelihood_highPt_EndCap->Fill(QGLikelihood);
	    }
	  }else{
	    if (recoJets[1].pt()>20. && recoJets[1].pt()<=50.) {
	      mAxis2_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_Forward"];if(mAxis2_lowPt_Forward && mAxis2_lowPt_Forward->getRootObject()) mAxis2_lowPt_Forward->Fill(QGaxis2);
	      mpTD_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_Forward"]; if(mpTD_lowPt_Forward && mpTD_lowPt_Forward->getRootObject()) mpTD_lowPt_Forward->Fill(QGptD);
	      mMultiplicityQG_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_Forward"]; if(mMultiplicityQG_lowPt_Forward && mMultiplicityQG_lowPt_Forward->getRootObject()) mMultiplicityQG_lowPt_Forward->Fill(QGmulti);
	      mqgLikelihood_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_Forward"]; if(mqgLikelihood_lowPt_Forward && mqgLikelihood_lowPt_Forward->getRootObject()) mqgLikelihood_lowPt_Forward->Fill(QGLikelihood);
	    }
	    if (recoJets[1].pt()>50. && recoJets[1].pt()<=140.) {
	      mAxis2_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_Forward"];if(mAxis2_mediumPt_Forward && mAxis2_mediumPt_Forward->getRootObject()) mAxis2_mediumPt_Forward->Fill(QGaxis2);
	      mpTD_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_Forward"]; if(mpTD_mediumPt_Forward && mpTD_mediumPt_Forward->getRootObject()) mpTD_mediumPt_Forward->Fill(QGptD);
	      mMultiplicityQG_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_Forward"]; if(mMultiplicityQG_mediumPt_Forward && mMultiplicityQG_mediumPt_Forward->getRootObject()) mMultiplicityQG_mediumPt_Forward->Fill(QGmulti);
	      mqgLikelihood_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_Forward"]; if(mqgLikelihood_mediumPt_Forward && mqgLikelihood_mediumPt_Forward->getRootObject()) mqgLikelihood_mediumPt_Forward->Fill(QGLikelihood);
	    }
	    if (recoJets[1].pt()>140.) {
	      mAxis2_highPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_Forward"];if(mAxis2_highPt_Forward && mAxis2_highPt_Forward->getRootObject()) mAxis2_highPt_Forward->Fill(QGaxis2);
	      mpTD_highPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_highPt_Forward"]; if(mpTD_highPt_Forward && mpTD_highPt_Forward->getRootObject()) mpTD_highPt_Forward->Fill(QGptD);
	      mMultiplicityQG_highPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_Forward"]; if(mMultiplicityQG_highPt_Forward && mMultiplicityQG_highPt_Forward->getRootObject()) mMultiplicityQG_highPt_Forward->Fill(QGmulti);
	      mqgLikelihood_highPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_Forward"]; if(mqgLikelihood_highPt_Forward && mqgLikelihood_highPt_Forward->getRootObject()) mqgLikelihood_highPt_Forward->Fill(QGLikelihood);
	    }
	  }//deal with second jet
	}//fill quark gluon tagged variables
      }//pfjet 	  
      if(isMiniAODJet_){
	mCHFrac = map_of_MEs[DirName+"/"+"CHFrac"]; if (mCHFrac && mCHFrac->getRootObject())         mCHFrac ->Fill((*patJets)[ind1].chargedHadronEnergyFraction());
	mNHFrac = map_of_MEs[DirName+"/"+"NHFrac"]; if (mNHFrac && mNHFrac->getRootObject())         mNHFrac ->Fill((*patJets)[ind1].neutralHadronEnergyFraction());
	mPhFrac = map_of_MEs[DirName+"/"+"PhFrac"]; if (mPhFrac && mPhFrac->getRootObject())         mPhFrac ->Fill((*patJets)[ind1].neutralEmEnergyFraction());
	mHFEMFrac = map_of_MEs[DirName+"/"+"HFEMFrac"]; if (mHFEMFrac && mHFEMFrac->getRootObject()) mHFEMFrac ->Fill((*patJets)[ind1].HFEMEnergyFraction());
	mHFHFrac = map_of_MEs[DirName+"/"+"HFHFrac"]; if (mHFHFrac && mHFHFrac->getRootObject())     mHFHFrac ->Fill((*patJets)[ind1].HFHadronEnergyFraction());
	
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(1./(*patJets)[ind1].jecFactor("Uncorrected"));
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[0].eta(),1./(*patJets)[ind1].jecFactor("Uncorrected"));
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[0].pt(),1./(*patJets)[ind1].jecFactor("Uncorrected"));
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(1./(*patJets)[ind2].jecFactor("Uncorrected"));
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[0].eta(),1./(*patJets)[ind2].jecFactor("Uncorrected"));
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[0].pt(),1./(*patJets)[ind2].jecFactor("Uncorrected"));
	
	mChargedMultiplicity = map_of_MEs[DirName+"/"+"ChargedMultiplicity"]; if(mChargedMultiplicity && mChargedMultiplicity->getRootObject()) mChargedMultiplicity->Fill((*patJets)[ind1].chargedMultiplicity());
	mNeutralMultiplicity = map_of_MEs[DirName+"/"+"NeutralMultiplicity"]; if(mNeutralMultiplicity && mNeutralMultiplicity->getRootObject()) mNeutralMultiplicity->Fill((*patJets)[ind1].neutralMultiplicity());
	mMuonMultiplicity = map_of_MEs[DirName+"/"+"MuonMultiplicity"]; if(mMuonMultiplicity && mMuonMultiplicity->getRootObject())             mMuonMultiplicity->Fill((*patJets)[ind1].muonMultiplicity());
	//Filling variables for second jet	
	mCHFrac = map_of_MEs[DirName+"/"+"CHFrac"]; if (mCHFrac && mCHFrac->getRootObject())         mCHFrac ->Fill((*patJets)[ind2].chargedHadronEnergyFraction());
	mNHFrac = map_of_MEs[DirName+"/"+"NHFrac"]; if (mNHFrac && mNHFrac->getRootObject())         mNHFrac ->Fill((*patJets)[ind2].neutralHadronEnergyFraction());
	mPhFrac = map_of_MEs[DirName+"/"+"PhFrac"]; if (mPhFrac && mPhFrac->getRootObject())         mPhFrac ->Fill((*patJets)[ind2].neutralEmEnergyFraction());
	mHFEMFrac = map_of_MEs[DirName+"/"+"HFEMFrac"]; if (mHFEMFrac && mHFEMFrac->getRootObject()) mHFEMFrac ->Fill((*patJets)[ind2].HFEMEnergyFraction());
	mHFHFrac = map_of_MEs[DirName+"/"+"HFHFrac"]; if (mHFHFrac && mHFHFrac->getRootObject())     mHFHFrac ->Fill((*patJets)[ind2].HFHadronEnergyFraction());
	
	mNeutralFraction = map_of_MEs[DirName+"/"+"NeutralConstituentsFraction"];if (mNeutralFraction && mNeutralFraction->getRootObject()) mNeutralFraction->Fill ((double)(*patJets)[ind1].neutralMultiplicity()/(double)(*patJets)[ind1].nConstituents());
	
	mChargedMultiplicity = map_of_MEs[DirName+"/"+"ChargedMultiplicity"]; if(mChargedMultiplicity && mChargedMultiplicity->getRootObject()) mChargedMultiplicity->Fill((*patJets)[ind2].chargedMultiplicity());
	mNeutralMultiplicity = map_of_MEs[DirName+"/"+"NeutralMultiplicity"]; if(mNeutralMultiplicity && mNeutralMultiplicity->getRootObject()) mNeutralMultiplicity->Fill((*patJets)[ind2].neutralMultiplicity());
	mMuonMultiplicity = map_of_MEs[DirName+"/"+"MuonMultiplicity"]; if(mMuonMultiplicity && mMuonMultiplicity->getRootObject())             mMuonMultiplicity->Fill((*patJets)[ind2].muonMultiplicity());
	
	//now fill PATJet profiles for leading jet
	mCHFrac_profile = map_of_MEs[DirName+"/"+"CHFrac_profile"]; if (mCHFrac_profile && mCHFrac_profile->getRootObject())         mCHFrac_profile ->Fill(numPV, (*patJets)[ind1].chargedHadronEnergyFraction());
	mNHFrac_profile = map_of_MEs[DirName+"/"+"NHFrac_profile"]; if (mNHFrac_profile && mNHFrac_profile->getRootObject())         mNHFrac_profile ->Fill(numPV, (*patJets)[ind1].neutralHadronEnergyFraction());
	mPhFrac_profile = map_of_MEs[DirName+"/"+"PhFrac_profile"]; if (mPhFrac_profile && mPhFrac_profile->getRootObject())         mPhFrac_profile ->Fill(numPV, (*patJets)[ind1].neutralEmEnergyFraction());
	mHFEMFrac_profile = map_of_MEs[DirName+"/"+"HFEMFrac_profile"]; if (mHFEMFrac_profile && mHFEMFrac_profile->getRootObject()) mHFEMFrac_profile ->Fill(numPV, (*patJets)[ind1].HFEMEnergyFraction());
	mHFHFrac_profile = map_of_MEs[DirName+"/"+"HFHFrac_profile"]; if (mHFHFrac_profile && mHFHFrac_profile->getRootObject())     mHFHFrac_profile ->Fill(numPV, (*patJets)[ind1].HFHadronEnergyFraction());
	
	mNeutralFraction = map_of_MEs[DirName+"/"+"NeutralConstituentsFraction"];if (mNeutralFraction && mNeutralFraction->getRootObject()) mNeutralFraction->Fill ((double)(*patJets)[ind2].neutralMultiplicity()/(double)(*patJets)[ind2].nConstituents());
	
	mChargedMultiplicity_profile = map_of_MEs[DirName+"/"+"ChargedMultiplicity_profile"]; if(mChargedMultiplicity_profile && mChargedMultiplicity_profile->getRootObject()) mChargedMultiplicity_profile->Fill(numPV, (*patJets)[ind1].chargedMultiplicity());
	mNeutralMultiplicity_profile = map_of_MEs[DirName+"/"+"NeutralMultiplicity_profile"]; if(mNeutralMultiplicity_profile && mNeutralMultiplicity_profile->getRootObject()) mNeutralMultiplicity_profile->Fill(numPV, (*patJets)[ind1].neutralMultiplicity());
	mMuonMultiplicity_profile = map_of_MEs[DirName+"/"+"MuonMultiplicity_profile"]; if(mMuonMultiplicity_profile && mMuonMultiplicity_profile->getRootObject())             mMuonMultiplicity->Fill(numPV, (*patJets)[ind1].muonMultiplicity());
	//now fill PATJet profiles for second leading jet
	mCHFrac_profile = map_of_MEs[DirName+"/"+"CHFrac_profile"]; if (mCHFrac_profile && mCHFrac_profile->getRootObject())         mCHFrac_profile ->Fill(numPV, (*patJets)[ind2].chargedHadronEnergyFraction());
	mNHFrac_profile = map_of_MEs[DirName+"/"+"NHFrac_profile"]; if (mNHFrac_profile && mNHFrac_profile->getRootObject())         mNHFrac_profile ->Fill(numPV, (*patJets)[ind2].neutralHadronEnergyFraction());
	mPhFrac_profile = map_of_MEs[DirName+"/"+"PhFrac_profile"]; if (mPhFrac_profile && mPhFrac_profile->getRootObject())         mPhFrac_profile ->Fill(numPV, (*patJets)[ind2].neutralEmEnergyFraction());
	mHFEMFrac_profile = map_of_MEs[DirName+"/"+"HFEMFrac_profile"]; if (mHFEMFrac_profile && mHFEMFrac_profile->getRootObject()) mHFEMFrac_profile ->Fill(numPV, (*patJets)[ind2].HFEMEnergyFraction());
	mHFHFrac_profile = map_of_MEs[DirName+"/"+"HFHFrac_profile"]; if (mHFHFrac_profile && mHFHFrac_profile->getRootObject())     mHFHFrac_profile ->Fill(numPV, (*patJets)[ind2].HFHadronEnergyFraction());
	
	mChargedMultiplicity_profile = map_of_MEs[DirName+"/"+"ChargedMultiplicity_profile"]; if(mChargedMultiplicity_profile && mChargedMultiplicity_profile->getRootObject()) mChargedMultiplicity_profile->Fill(numPV, (*patJets)[ind2].chargedMultiplicity());
	mNeutralMultiplicity_profile = map_of_MEs[DirName+"/"+"NeutralMultiplicity_profile"]; if(mNeutralMultiplicity_profile && mNeutralMultiplicity_profile->getRootObject()) mNeutralMultiplicity_profile->Fill(numPV, (*patJets)[ind2].neutralMultiplicity());
	mMuonMultiplicity_profile = map_of_MEs[DirName+"/"+"MuonMultiplicity_profile"]; if(mMuonMultiplicity_profile && mMuonMultiplicity_profile->getRootObject())             mMuonMultiplicity_profile->Fill(numPV, (*patJets)[ind2].muonMultiplicity());
      }	  


      //fill histos for first jet
      mPt = map_of_MEs[DirName+"/"+"Pt"]; if (mPt && mPt->getRootObject())      mPt->Fill (recoJets[0].pt());
      mEta = map_of_MEs[DirName+"/"+"Eta"]; if (mEta && mEta->getRootObject())  mEta->Fill (recoJets[0].eta());
      mPhi = map_of_MEs[DirName+"/"+"Phi"]; if (mPhi && mPhi->getRootObject())  mPhi->Fill (recoJets[0].phi());
      mPhiVSEta = map_of_MEs[DirName+"/"+"PhiVSEta"]; if (mPhiVSEta && mPhiVSEta->getRootObject()) mPhiVSEta->Fill(recoJets[0].eta(),recoJets[0].phi());
      //if(!isJPTJet_){
      mConstituents = map_of_MEs[DirName+"/"+"Constituents"]; if (mConstituents && mConstituents->getRootObject()) mConstituents->Fill (recoJets[0].nConstituents());
      //}
      mPt = map_of_MEs[DirName+"/"+"Pt"]; if (mPt && mPt->getRootObject())      mPt->Fill (recoJets[1].pt());
      mEta = map_of_MEs[DirName+"/"+"Eta"]; if (mEta && mEta->getRootObject())  mEta->Fill (recoJets[1].eta());
      mPhi = map_of_MEs[DirName+"/"+"Phi"]; if (mPhi && mPhi->getRootObject())  mPhi->Fill (recoJets[1].phi());
      mPhiVSEta = map_of_MEs[DirName+"/"+"PhiVSEta"]; if (mPhiVSEta && mPhiVSEta->getRootObject()) mPhiVSEta->Fill(recoJets[1].eta(),recoJets[1].phi());
      //if(!isJPTJet_){
      mConstituents = map_of_MEs[DirName+"/"+"Constituents"]; if (mConstituents && mConstituents->getRootObject()) mConstituents->Fill (recoJets[1].nConstituents());
      //}
      //PV profiles 
      mPt_profile = map_of_MEs[DirName+"/"+"Pt_profile"]; if (mPt_profile && mPt_profile->getRootObject())        mPt_profile          ->Fill(numPV, recoJets[0].pt());
      mEta_profile = map_of_MEs[DirName+"/"+"Eta_profile"]; if (mEta_profile && mEta_profile->getRootObject())    mEta_profile         ->Fill(numPV, recoJets[0].eta());
      mPhi_profile = map_of_MEs[DirName+"/"+"Phi_profile"]; if (mPhi_profile && mPhi_profile->getRootObject())    mPhi_profile         ->Fill(numPV, recoJets[0].phi());
      //if(!isJPTJet_){
      mConstituents_profile = map_of_MEs[DirName+"/"+"Constituents_profile"]; if (mConstituents_profile && mConstituents_profile->getRootObject()) mConstituents_profile->Fill(numPV, recoJets[0].nConstituents());
      //}
      mPt_profile = map_of_MEs[DirName+"/"+"Pt_profile"]; if (mPt_profile && mPt_profile->getRootObject())         mPt_profile          ->Fill(numPV, recoJets[1].pt());
      mEta_profile = map_of_MEs[DirName+"/"+"Eta_profile"]; if (mEta_profile && mEta_profile->getRootObject())     mEta_profile         ->Fill(numPV, recoJets[1].eta());
      mPhi_profile = map_of_MEs[DirName+"/"+"Phi_profile"]; if (mPhi_profile && mPhi_profile->getRootObject())     mPhi_profile         ->Fill(numPV, recoJets[1].phi());
      //if(!isJPTJet_){
      mConstituents_profile = map_of_MEs[DirName+"/"+"Constituents_profile"]; if (mConstituents_profile && mConstituents_profile->getRootObject()) mConstituents_profile->Fill(numPV, recoJets[1].nConstituents());
      //}
      if (fabs(recoJets[0].eta() < 1.4)) {
	double pt_dijet = (recoJets[0].pt() + recoJets[1].pt())/2;      
	if (dphi > 2.7) {//cut even toughter on dijet balance
	  double pt_probe;
	  double pt_barrel;
	  int jet1, jet2;
	  //int randJet = rand() % 2;
	  int randJet =iEvent.id().event()%2;
	  if (fabs(recoJets[1].eta() < 1.4)) {
	    if (randJet) {
	      jet1 = 0;
	      jet2 = 1;
	    }
	    else {
	      jet1 = 1;
	      jet2 = 0;
	    }	  
	    // ***Di-Jet Asymmetry****
	    // * leading jets eta < 1.4
	    // * leading jets dphi > 2.7
	    // * pt_third jet < threshold
	    // * A = (pt_1 - pt_2)/(pt_1 + pt_2)
	    // * jets 1 and two are randomly ordered
	    // **
	    bool thirdJetCut = true;
	    //that doesn't make any sense -> imagine you have 5 jets,
	    //jet 3 is quite hard (almost as hard as the second jet, i.e. 200/80/79/20/15, cutoff is 30
	    //the 4th and 5th jet are soft enough -> then you would fill the asymmetry twice, 
	    //although jet 2 and 3 are basically identical
	    //do third jet relative to second jet
	    //JME-10-014 suggests pt3/pt_dijet<0.15
	    if(ind3>0){
	      if (pt3 > asymmetryThirdJetCut_){
		thirdJetCut=false;
	      } 
	    }
	    if(thirdJetCut){
	      double dijetAsymmetry =(recoJets[jet1].pt() - recoJets[jet2].pt()) / (recoJets[jet1].pt() + recoJets[jet2].pt());
	      mDijetAsymmetry = map_of_MEs[DirName+"/"+"DijetAsymmetry"]; if (mDijetAsymmetry && mDijetAsymmetry->getRootObject()) mDijetAsymmetry->Fill(dijetAsymmetry);
	    }// end restriction on third jet pt in asymmetry calculation
	    
	  }
	  else {
	    jet1 = 0;
	    jet2 = 1;
	  }
	  
	  pt_barrel = recoJets[jet1].pt();
	  pt_probe  = recoJets[jet2].pt();
	  
	  //dijet balance cuts
	  // ***Di-Jet Balance****
	  // * pt_dijet = (pt_probe+pt_barrel)/2
	  // * leading jets dphi > 2.7
	  // * reject evnets where pt_third/pt_dijet > 0.2
	  // * pv selection
	  // * B = (pt_probe - pt_barrel)/pt_dijet
	  // * select probe randomly from 2 jets if both leading jets are in the barrel
	  bool thirdJetCut = true;
	  if(ind3>0){
	    if (pt3/pt_dijet > balanceThirdJetCut_){ 
	      thirdJetCut = false;
	    }
	  }
	  if (thirdJetCut) {
	    double dijetBalance = (pt_probe - pt_barrel) / pt_dijet;
	    mDijetBalance = map_of_MEs[DirName+"/"+"DijetBalance"]; if (mDijetBalance && mDijetBalance->getRootObject()) mDijetBalance->Fill(dijetBalance);
	  }// end restriction on third jet pt ratio in balance calculation
	  
	}// dPhi > 2.7 for dijetbalance and asymmetrie
      }//leading jet in barrel
    }//DPhi cut of 2.1
  }//dijet selection, check if both leading jets are IDed
  //now do the ZJets selection -> pass_Z_selection cuts already on the Z-pt>30 GeV
  if(pass_Z_selection && ind1_mu_vetoed>=0 && pt1_mu_vetoed>12 && cleaned_first_jet_mu_vetoed && isPFJet_){
    bool pass_second_jet_mu_vetoed=false;
    if(cleaned_second_jet_mu_vetoed){
      if(ind2_mu_vetoed>=0 && pt2_mu_vetoed/zCand.Pt()<0.2){
	pass_second_jet_mu_vetoed=true;
      }
    }
    if(pass_second_jet_mu_vetoed){
      Jet recoJet1; 
      if(isPFJet_){
	recoJet1=(*pfJets)[ind1_mu_vetoed];
      }
      if (pass_correction_flag && !isMiniAODJet_) {
	double scale=1;
	if (isCaloJet_){
	  scale = jetCorr->correction((*caloJets)[ind1_mu_vetoed]);
	}
	if (isPFJet_){ 
	  scale = jetCorr->correction((*pfJets)[ind1_mu_vetoed]);
	}
	recoJet1.scaleEnergy(scale);	    
      }
      double dphi=fabs(recoJet1.phi()-zCand.Phi());
      if(dphi>acos(-1.)){
	dphi=2*acos(-1.)-dphi;
      }
      if(jetCleaningFlag_){
	DirName = "JetMET/Jet/Cleaned"+mInputCollection_.label()+"/ZJets";
      }                 
      mDPhiZJet  = map_of_MEs[DirName+"/"+"DPhiZJ"]; if (mDPhiZJet && mDPhiZJet->getRootObject()) mDPhiZJet ->Fill(dphi);  
      if(fabs(dphi-acos(-1.))<0.34){    
	//get now MET collections for MPF studies
	edm::Handle<reco::CaloMETCollection> calometcoll;
	edm::Handle<reco::PFMETCollection> pfmetcoll;
	//edm::Handle<pat::METCollection> patmetcoll;
	const MET *met=NULL;
	if(isCaloJet_){
	  iEvent.getByToken(caloMetToken_, calometcoll);
	  if(!calometcoll.isValid()) return;
	  met=&(calometcoll->front());
	}
	if(isPFJet_){
	  iEvent.getByToken(pfMetToken_, pfmetcoll);
	  if(!pfmetcoll.isValid()) return;
	  met=&(pfmetcoll->front());
	}
	//if(isMiniAODJet_){
	//iEvent.getByToken(patMetToken_, patmetcoll);
	//if(!patmetcoll.isValid()) return;
	//met=&(patmetcoll->front());
	//}
	mZMass   = map_of_MEs[DirName+"/"+"DiMuonMass"]; if(mZMass && mZMass->getRootObject()) mZMass->Fill(zCand.M());
	mZJetAsymmetry     = map_of_MEs[DirName+"/"+"ZJetAsymmetry"]; if(mZJetAsymmetry && mZJetAsymmetry->getRootObject()) mZJetAsymmetry->Fill((zCand.Pt()-recoJet1.pt())/(zCand.Pt()+recoJet1.pt()));
	if(recoJet1.pt()>20){
	  mPt = map_of_MEs[DirName+"/"+"Pt"]; if (mPt && mPt->getRootObject())      mPt->Fill (recoJet1.pt());
	  mEta = map_of_MEs[DirName+"/"+"Eta"]; if (mEta && mEta->getRootObject())  mEta->Fill (recoJet1.eta());
	  mPhi = map_of_MEs[DirName+"/"+"Phi"]; if (mPhi && mPhi->getRootObject())  mPhi->Fill (recoJet1.phi());
	  //PV profiles 
	  mPt_profile = map_of_MEs[DirName+"/"+"Pt_profile"]; if (mPt_profile && mPt_profile->getRootObject())        mPt_profile          ->Fill(numPV, recoJet1.pt());
	  mEta_profile = map_of_MEs[DirName+"/"+"Eta_profile"]; if (mEta_profile && mEta_profile->getRootObject())    mEta_profile         ->Fill(numPV, recoJet1.eta());
	  mPhi_profile = map_of_MEs[DirName+"/"+"Phi_profile"]; if (mPhi_profile && mPhi_profile->getRootObject())    mPhi_profile         ->Fill(numPV, recoJet1.phi());
	  mConstituents = map_of_MEs[DirName+"/"+"Constituents"]; if (mConstituents && mConstituents->getRootObject()) mConstituents->Fill(recoJet1.nConstituents());
	  mConstituents_profile = map_of_MEs[DirName+"/"+"Constituents_profile"]; if (mConstituents_profile && mConstituents_profile->getRootObject()) mConstituents_profile->Fill(numPV, recoJet1.nConstituents());
	  mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(recoJet1.pt()/(*pfJets)[ind1_mu_vetoed].pt());
	  mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJet1.eta(),recoJet1.pt()/(*pfJets)[ind1_mu_vetoed].pt());
	  mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJet1.pt(),recoJet1.pt()/(*pfJets)[ind1_mu_vetoed].pt());
	  mCHFrac = map_of_MEs[DirName+"/"+"CHFrac"]; if (mCHFrac && mCHFrac->getRootObject())         mCHFrac ->Fill((*pfJets)[ind1_mu_vetoed].chargedHadronEnergyFraction());
	  mNHFrac = map_of_MEs[DirName+"/"+"NHFrac"]; if (mNHFrac && mNHFrac->getRootObject())         mNHFrac ->Fill((*pfJets)[ind1_mu_vetoed].neutralHadronEnergyFraction());
	  mPhFrac = map_of_MEs[DirName+"/"+"PhFrac"]; if (mPhFrac && mPhFrac->getRootObject())         mPhFrac ->Fill((*pfJets)[ind1_mu_vetoed].neutralEmEnergyFraction());
	  mHFEMFrac = map_of_MEs[DirName+"/"+"HFEMFrac"]; if (mHFEMFrac && mHFEMFrac->getRootObject()) mHFEMFrac ->Fill((*pfJets)[ind1_mu_vetoed].HFEMEnergyFraction());
	  mHFHFrac = map_of_MEs[DirName+"/"+"HFHFrac"]; if (mHFHFrac && mHFHFrac->getRootObject())     mHFHFrac ->Fill((*pfJets)[ind1_mu_vetoed].HFHadronEnergyFraction());
	  //now fill PFJet profiles for second leading jet
	  mCHFrac_profile = map_of_MEs[DirName+"/"+"CHFrac_profile"]; if (mCHFrac_profile && mCHFrac_profile->getRootObject())         mCHFrac_profile ->Fill(numPV, (*pfJets)[ind1_mu_vetoed].chargedHadronEnergyFraction());
	  mNHFrac_profile = map_of_MEs[DirName+"/"+"NHFrac_profile"]; if (mNHFrac_profile && mNHFrac_profile->getRootObject())         mNHFrac_profile ->Fill(numPV, (*pfJets)[ind1_mu_vetoed].neutralHadronEnergyFraction());
	  mPhFrac_profile = map_of_MEs[DirName+"/"+"PhFrac_profile"]; if (mPhFrac_profile && mPhFrac_profile->getRootObject())         mPhFrac_profile ->Fill(numPV, (*pfJets)[ind1_mu_vetoed].neutralEmEnergyFraction());
	  mHFEMFrac_profile = map_of_MEs[DirName+"/"+"HFEMFrac_profile"]; if (mHFEMFrac_profile && mHFEMFrac_profile->getRootObject()) mHFEMFrac_profile ->Fill(numPV, (*pfJets)[ind1_mu_vetoed].HFEMEnergyFraction());
	  mHFHFrac_profile = map_of_MEs[DirName+"/"+"HFHFrac_profile"]; if (mHFHFrac_profile && mHFHFrac_profile->getRootObject())     mHFHFrac_profile ->Fill(numPV, (*pfJets)[ind1_mu_vetoed].HFHadronEnergyFraction());
	}
	double MPF=1.+(met->px()*zCand.Px()+met->py()*zCand.Py())/(zCand.Pt()*zCand.Pt());
	if(fabs(recoJet1.eta())<1.3){//barrel jets
	  mJ1Pt_over_ZPt_J_Barrel = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_J_Barrel"]; if(mJ1Pt_over_ZPt_J_Barrel && mJ1Pt_over_ZPt_J_Barrel->getRootObject())mJ1Pt_over_ZPt_J_Barrel->Fill(recoJet1.pt()/zCand.Pt());
	  mMPF_J_Barrel = map_of_MEs[DirName+"/"+"MPF_J_Barrel"]; if(mMPF_J_Barrel && mMPF_J_Barrel->getRootObject())mMPF_J_Barrel->Fill(MPF);
	  if(zCand.Pt()<90){//lower cut on 30 already right from the start
	    mJetZBalance_lowZPt_J_Barrel = map_of_MEs[DirName+"/"+"JZB_lowZPt_J_Barrel"]; if(mJetZBalance_lowZPt_J_Barrel && mJetZBalance_lowZPt_J_Barrel->getRootObject())mJetZBalance_lowZPt_J_Barrel->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_lowZPt_J_Barrel = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_lowZPt_J_Barrel"]; if(mJ1Pt_over_ZPt_lowZPt_J_Barrel && mJ1Pt_over_ZPt_lowZPt_J_Barrel->getRootObject())mJ1Pt_over_ZPt_lowZPt_J_Barrel->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_lowZPt_J_Barrel = map_of_MEs[DirName+"/"+"MPF_lowZPt_J_Barrel"]; if(mMPF_lowZPt_J_Barrel && mMPF_lowZPt_J_Barrel->getRootObject())mMPF_lowZPt_J_Barrel->Fill(MPF);
	    //mMPF_J_Barrel = map_of_MEs[DirName+"/"+"MPF_J_Barrel"]; if(mMPF_J_Barrel && mMPF_J_Barrel->getRootObject())mMPF_J_Barrel->Fill(MPF);
	  }else if (zCand.Pt()<140){
	    mJetZBalance_mediumZPt_J_Barrel = map_of_MEs[DirName+"/"+"JZB_mediumZPt_J_Barrel"]; if(mJetZBalance_mediumZPt_J_Barrel && mJetZBalance_mediumZPt_J_Barrel->getRootObject())mJetZBalance_mediumZPt_J_Barrel->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_mediumZPt_J_Barrel = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_mediumZPt_J_Barrel"]; if(mJ1Pt_over_ZPt_mediumZPt_J_Barrel && mJ1Pt_over_ZPt_mediumZPt_J_Barrel->getRootObject())mJ1Pt_over_ZPt_mediumZPt_J_Barrel->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_mediumZPt_J_Barrel = map_of_MEs[DirName+"/"+"MPF_mediumZPt_J_Barrel"]; if(mMPF_mediumZPt_J_Barrel && mMPF_mediumZPt_J_Barrel->getRootObject())mMPF_mediumZPt_J_Barrel->Fill(MPF);
	  }else{
	    mJetZBalance_highZPt_J_Barrel = map_of_MEs[DirName+"/"+"JZB_highZPt_J_Barrel"]; if(mJetZBalance_highZPt_J_Barrel && mJetZBalance_highZPt_J_Barrel->getRootObject())mJetZBalance_highZPt_J_Barrel->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_highZPt_J_Barrel = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_highZPt_J_Barrel"]; if(mJ1Pt_over_ZPt_highZPt_J_Barrel && mJ1Pt_over_ZPt_highZPt_J_Barrel->getRootObject())mJ1Pt_over_ZPt_highZPt_J_Barrel->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_highZPt_J_Barrel = map_of_MEs[DirName+"/"+"MPF_highZPt_J_Barrel"]; if(mMPF_highZPt_J_Barrel && mMPF_highZPt_J_Barrel->getRootObject())mMPF_highZPt_J_Barrel->Fill(MPF);
	  }
	  if(zCand.Pt()>30){
	    if(zCand.Pt()<55){
	      mDeltaPt_Z_j1_over_ZPt_30_55_J_Barrel = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_30_55_J_Barrel"];if(mDeltaPt_Z_j1_over_ZPt_30_55_J_Barrel && mDeltaPt_Z_j1_over_ZPt_30_55_J_Barrel->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_30_55_J_Barrel->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else if (zCand.Pt()<75){
	      mDeltaPt_Z_j1_over_ZPt_55_75_J_Barrel = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_55_75_J_Barrel"];if(mDeltaPt_Z_j1_over_ZPt_55_75_J_Barrel && mDeltaPt_Z_j1_over_ZPt_55_75_J_Barrel->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_55_75_J_Barrel->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else if (zCand.Pt()<150){
	      mDeltaPt_Z_j1_over_ZPt_75_150_J_Barrel = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_75_150_J_Barrel"];if(mDeltaPt_Z_j1_over_ZPt_75_150_J_Barrel && mDeltaPt_Z_j1_over_ZPt_75_150_J_Barrel->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_75_150_J_Barrel->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else if (zCand.Pt()<290){
	      mDeltaPt_Z_j1_over_ZPt_150_290_J_Barrel = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_150_290_J_Barrel"];if(mDeltaPt_Z_j1_over_ZPt_150_290_J_Barrel && mDeltaPt_Z_j1_over_ZPt_150_290_J_Barrel->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_150_290_J_Barrel->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else{
	      mDeltaPt_Z_j1_over_ZPt_290_J_Barrel = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_290_J_Barrel"];if(mDeltaPt_Z_j1_over_ZPt_290_J_Barrel && mDeltaPt_Z_j1_over_ZPt_290_J_Barrel->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_290_J_Barrel->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }
	  }
	}else if(fabs(recoJet1.eta())<3.0){//endcap jets
	  mJ1Pt_over_ZPt_J_EndCap = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_J_EndCap"]; if(mJ1Pt_over_ZPt_J_EndCap && mJ1Pt_over_ZPt_J_EndCap->getRootObject())mJ1Pt_over_ZPt_J_EndCap->Fill(recoJet1.pt()/zCand.Pt());
	  mMPF_J_EndCap = map_of_MEs[DirName+"/"+"MPF_J_EndCap"]; if(mMPF_J_EndCap && mMPF_J_EndCap->getRootObject())mMPF_J_EndCap->Fill(MPF);
	  if(zCand.Pt()<90){//lower cut on 30 already right from the start
	    mJetZBalance_lowZPt_J_EndCap = map_of_MEs[DirName+"/"+"JZB_lowZPt_J_EndCap"]; if(mJetZBalance_lowZPt_J_EndCap && mJetZBalance_lowZPt_J_EndCap->getRootObject())mJetZBalance_lowZPt_J_EndCap->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_lowZPt_J_EndCap = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_lowZPt_J_EndCap"]; if(mJ1Pt_over_ZPt_lowZPt_J_EndCap && mJ1Pt_over_ZPt_lowZPt_J_EndCap->getRootObject())mJ1Pt_over_ZPt_lowZPt_J_EndCap->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_lowZPt_J_EndCap = map_of_MEs[DirName+"/"+"MPF_lowZPt_J_EndCap"]; if(mMPF_lowZPt_J_EndCap && mMPF_lowZPt_J_EndCap->getRootObject())mMPF_lowZPt_J_EndCap->Fill(MPF);
	  }else if (zCand.Pt()<140){
	    mJetZBalance_mediumZPt_J_EndCap = map_of_MEs[DirName+"/"+"JZB_mediumZPt_J_EndCap"]; if(mJetZBalance_mediumZPt_J_EndCap && mJetZBalance_mediumZPt_J_EndCap->getRootObject())mJetZBalance_mediumZPt_J_EndCap->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_mediumZPt_J_EndCap = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_mediumZPt_J_EndCap"]; if(mJ1Pt_over_ZPt_mediumZPt_J_EndCap && mJ1Pt_over_ZPt_mediumZPt_J_EndCap->getRootObject())mJ1Pt_over_ZPt_mediumZPt_J_EndCap->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_mediumZPt_J_EndCap = map_of_MEs[DirName+"/"+"MPF_mediumZPt_J_EndCap"]; if(mMPF_mediumZPt_J_EndCap && mMPF_mediumZPt_J_EndCap->getRootObject())mMPF_mediumZPt_J_EndCap->Fill(MPF);
	  }else{
	    mJetZBalance_highZPt_J_EndCap = map_of_MEs[DirName+"/"+"JZB_highZPt_J_EndCap"]; if(mJetZBalance_highZPt_J_EndCap && mJetZBalance_highZPt_J_EndCap->getRootObject())mJetZBalance_highZPt_J_EndCap->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_highZPt_J_EndCap = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_highZPt_J_EndCap"]; if(mJ1Pt_over_ZPt_highZPt_J_EndCap && mJ1Pt_over_ZPt_highZPt_J_EndCap->getRootObject())mJ1Pt_over_ZPt_highZPt_J_EndCap->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_highZPt_J_EndCap = map_of_MEs[DirName+"/"+"MPF_highZPt_J_EndCap"]; if(mMPF_highZPt_J_EndCap && mMPF_highZPt_J_EndCap->getRootObject())mMPF_highZPt_J_EndCap->Fill(MPF);
	  }
	  if(zCand.Pt()>30){
	    if(zCand.Pt()<55){
	      mDeltaPt_Z_j1_over_ZPt_30_55_J_EndCap = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_30_55_J_EndCap"];if(mDeltaPt_Z_j1_over_ZPt_30_55_J_EndCap && mDeltaPt_Z_j1_over_ZPt_30_55_J_EndCap->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_30_55_J_EndCap->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else if (zCand.Pt()<75){
	      mDeltaPt_Z_j1_over_ZPt_55_75_J_EndCap = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_55_75_J_EndCap"];if(mDeltaPt_Z_j1_over_ZPt_55_75_J_EndCap && mDeltaPt_Z_j1_over_ZPt_55_75_J_EndCap->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_55_75_J_EndCap->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else if (zCand.Pt()<150){
	      mDeltaPt_Z_j1_over_ZPt_75_150_J_EndCap = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_75_150_J_EndCap"];if(mDeltaPt_Z_j1_over_ZPt_75_150_J_EndCap && mDeltaPt_Z_j1_over_ZPt_75_150_J_EndCap->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_75_150_J_EndCap->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else if (zCand.Pt()<290){
	      mDeltaPt_Z_j1_over_ZPt_150_290_J_EndCap = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_150_290_J_EndCap"];if(mDeltaPt_Z_j1_over_ZPt_150_290_J_EndCap && mDeltaPt_Z_j1_over_ZPt_150_290_J_EndCap->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_150_290_J_EndCap->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else{
	      mDeltaPt_Z_j1_over_ZPt_290_J_EndCap = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_290_J_EndCap"];if(mDeltaPt_Z_j1_over_ZPt_290_J_EndCap && mDeltaPt_Z_j1_over_ZPt_290_J_EndCap->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_290_J_EndCap->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }
	  }
	}else{//forward jets
	  mJ1Pt_over_ZPt_J_Forward = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_J_Forward"]; if(mJ1Pt_over_ZPt_J_Forward && mJ1Pt_over_ZPt_J_Forward->getRootObject())mJ1Pt_over_ZPt_J_Forward->Fill(recoJet1.pt()/zCand.Pt());
	  mMPF_J_Forward = map_of_MEs[DirName+"/"+"MPF_J_Forward"]; if(mMPF_J_Forward && mMPF_J_Forward->getRootObject())mMPF_J_Forward->Fill(MPF);
	  if(zCand.Pt()<90){//lower cut on 30 already right from the start
	    mJetZBalance_lowZPt_J_Forward = map_of_MEs[DirName+"/"+"JZB_lowZPt_J_Forward"]; if(mJetZBalance_lowZPt_J_Forward && mJetZBalance_lowZPt_J_Forward->getRootObject())mJetZBalance_lowZPt_J_Forward->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_lowZPt_J_Forward = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_lowZPt_J_Forward"]; if(mJ1Pt_over_ZPt_lowZPt_J_Forward && mJ1Pt_over_ZPt_lowZPt_J_Forward->getRootObject())mJ1Pt_over_ZPt_lowZPt_J_Forward->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_lowZPt_J_Forward = map_of_MEs[DirName+"/"+"MPF_lowZPt_J_Forward"]; if(mMPF_lowZPt_J_Forward && mMPF_lowZPt_J_Forward->getRootObject())mMPF_lowZPt_J_Forward->Fill(MPF);
	  }else if (zCand.Pt()<140){
	    mJetZBalance_mediumZPt_J_Forward = map_of_MEs[DirName+"/"+"JZB_mediumZPt_J_Forward"]; if(mJetZBalance_mediumZPt_J_Forward && mJetZBalance_mediumZPt_J_Forward->getRootObject())mJetZBalance_mediumZPt_J_Forward->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_mediumZPt_J_Forward = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_mediumZPt_J_Forward"]; if(mJ1Pt_over_ZPt_mediumZPt_J_Forward && mJ1Pt_over_ZPt_mediumZPt_J_Forward->getRootObject())mJ1Pt_over_ZPt_mediumZPt_J_Forward->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_mediumZPt_J_Forward = map_of_MEs[DirName+"/"+"MPF_mediumZPt_J_Forward"]; if(mMPF_mediumZPt_J_Forward && mMPF_mediumZPt_J_Forward->getRootObject())mMPF_mediumZPt_J_Forward->Fill(MPF);
	  }else{
	    mJetZBalance_highZPt_J_Forward = map_of_MEs[DirName+"/"+"JZB_highZPt_J_Forward"]; if(mJetZBalance_highZPt_J_Forward && mJetZBalance_highZPt_J_Forward->getRootObject())mJetZBalance_highZPt_J_Forward->Fill(recoJet1.pt()-zCand.Pt());
	    mJ1Pt_over_ZPt_highZPt_J_Forward = map_of_MEs[DirName+"/"+"J1Pt_over_ZPt_highZPt_J_Forward"]; if(mJ1Pt_over_ZPt_highZPt_J_Forward && mJ1Pt_over_ZPt_highZPt_J_Forward->getRootObject())mJ1Pt_over_ZPt_highZPt_J_Forward->Fill(recoJet1.pt()/zCand.Pt());
	    mMPF_highZPt_J_Forward = map_of_MEs[DirName+"/"+"MPF_highZPt_J_Forward"]; if(mMPF_highZPt_J_Forward && mMPF_highZPt_J_Forward->getRootObject())mMPF_highZPt_J_Forward->Fill(MPF);
	  }
	  if(zCand.Pt()>30){
	    if(zCand.Pt()<55){
	      mDeltaPt_Z_j1_over_ZPt_30_55_J_Forward = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_30_55_J_Forward"];if(mDeltaPt_Z_j1_over_ZPt_30_55_J_Forward && mDeltaPt_Z_j1_over_ZPt_30_55_J_Forward->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_30_55_J_Forward->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else if (zCand.Pt()<100){
	      mDeltaPt_Z_j1_over_ZPt_55_100_J_Forward = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_55_100_J_Forward"];if(mDeltaPt_Z_j1_over_ZPt_55_100_J_Forward && mDeltaPt_Z_j1_over_ZPt_55_100_J_Forward->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_55_100_J_Forward->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }else{
	      mDeltaPt_Z_j1_over_ZPt_100_J_Forward = map_of_MEs[DirName+"/"+"DeltaPt_Z_j1_over_ZPt_100_J_Forward"];if(mDeltaPt_Z_j1_over_ZPt_100_J_Forward && mDeltaPt_Z_j1_over_ZPt_100_J_Forward->getRootObject() ) mDeltaPt_Z_j1_over_ZPt_100_J_Forward->Fill((zCand.Pt()-recoJet1.pt())/zCand.Pt());
	    }
	  }
	}
	int QGmulti=-1;
	float QGLikelihood=-10;
	float QGptD=-10;
	float QGaxis2=-10;
	if(fill_CHS_histos){
	  reco::PFJetRef pfjetref(pfJets, ind1_mu_vetoed);
	  QGmulti=(*qgMultiplicity)[pfjetref];
	  QGLikelihood=(*qgLikelihood)[pfjetref];
	  QGptD=(*qgptD)[pfjetref];
	  QGaxis2=(*qgaxis2)[pfjetref];
	  if(fabs(recoJet1.eta())<1.3){//barrel jets
	    //fractions for barrel
	    if (recoJet1.pt()>=20. && recoJet1.pt()<=50.) {
	      mAxis2_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_Barrel"];if(mAxis2_lowPt_Barrel && mAxis2_lowPt_Barrel->getRootObject()) mAxis2_lowPt_Barrel->Fill(QGaxis2);
	      mpTD_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_Barrel"]; if(mpTD_lowPt_Barrel && mpTD_lowPt_Barrel->getRootObject()) mpTD_lowPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_Barrel"]; if(mMultiplicityQG_lowPt_Barrel && mMultiplicityQG_lowPt_Barrel->getRootObject()) mMultiplicityQG_lowPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_lowPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_Barrel"]; if(mqgLikelihood_lowPt_Barrel && mqgLikelihood_lowPt_Barrel->getRootObject()) mqgLikelihood_lowPt_Barrel->Fill(QGLikelihood);
	    }
	    if (recoJet1.pt()>50. && recoJet1.pt()<=140.) {
	      mAxis2_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_Barrel"];if(mAxis2_mediumPt_Barrel && mAxis2_mediumPt_Barrel->getRootObject()) mAxis2_mediumPt_Barrel->Fill(QGaxis2);
	      mpTD_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_Barrel"]; if(mpTD_mediumPt_Barrel && mpTD_mediumPt_Barrel->getRootObject()) mpTD_mediumPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_Barrel"]; if(mMultiplicityQG_mediumPt_Barrel && mMultiplicityQG_mediumPt_Barrel->getRootObject()) mMultiplicityQG_mediumPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_mediumPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_Barrel"]; if(mqgLikelihood_mediumPt_Barrel && mqgLikelihood_mediumPt_Barrel->getRootObject()) mqgLikelihood_mediumPt_Barrel->Fill(QGLikelihood);
	    }
	    if (recoJet1.pt()>140.) {
	      mAxis2_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_Barrel"];if(mAxis2_highPt_Barrel && mAxis2_highPt_Barrel->getRootObject()) mAxis2_highPt_Barrel->Fill(QGaxis2);
	      mpTD_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_pTD_highPt_Barrel"]; if(mpTD_highPt_Barrel && mpTD_highPt_Barrel->getRootObject()) mpTD_highPt_Barrel->Fill(QGptD);
	      mMultiplicityQG_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_Barrel"]; if(mMultiplicityQG_highPt_Barrel && mMultiplicityQG_highPt_Barrel->getRootObject()) mMultiplicityQG_highPt_Barrel->Fill(QGmulti);
	      mqgLikelihood_highPt_Barrel = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_Barrel"]; if(mqgLikelihood_highPt_Barrel && mqgLikelihood_highPt_Barrel->getRootObject()) mqgLikelihood_highPt_Barrel->Fill(QGLikelihood);
	    }
	  }else if(fabs(recoJet1.eta())<3.0){//endcap jets 
	    if (recoJet1.pt()>20. && recoJet1.pt()<=50.) {
	      mAxis2_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_EndCap"];if(mAxis2_lowPt_EndCap && mAxis2_lowPt_EndCap->getRootObject()) mAxis2_lowPt_EndCap->Fill(QGaxis2);
	      mpTD_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_EndCap"]; if(mpTD_lowPt_EndCap && mpTD_lowPt_EndCap->getRootObject()) mpTD_lowPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_EndCap"]; if(mMultiplicityQG_lowPt_EndCap && mMultiplicityQG_lowPt_EndCap->getRootObject()) mMultiplicityQG_lowPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_lowPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_EndCap"]; if(mqgLikelihood_lowPt_EndCap && mqgLikelihood_lowPt_EndCap->getRootObject()) mqgLikelihood_lowPt_EndCap->Fill(QGLikelihood);
	    }
	    if (recoJet1.pt()>50. && recoJet1.pt()<=140.) {
	      mAxis2_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_EndCap"];if(mAxis2_mediumPt_EndCap && mAxis2_mediumPt_EndCap->getRootObject()) mAxis2_mediumPt_EndCap->Fill(QGaxis2);
	      mpTD_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_EndCap"]; if(mpTD_mediumPt_EndCap && mpTD_mediumPt_EndCap->getRootObject()) mpTD_mediumPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_EndCap"]; if(mMultiplicityQG_mediumPt_EndCap && mMultiplicityQG_mediumPt_EndCap->getRootObject()) mMultiplicityQG_mediumPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_mediumPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_EndCap"]; if(mqgLikelihood_mediumPt_EndCap && mqgLikelihood_mediumPt_EndCap->getRootObject()) mqgLikelihood_mediumPt_EndCap->Fill(QGLikelihood);
	    }
	    if (recoJet1.pt()>140.) {
	      mAxis2_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_EndCap"];if(mAxis2_highPt_EndCap && mAxis2_highPt_EndCap->getRootObject()) mAxis2_highPt_EndCap->Fill(QGaxis2);
	      mpTD_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_pTD_highPt_EndCap"]; if(mpTD_highPt_EndCap && mpTD_highPt_EndCap->getRootObject()) mpTD_highPt_EndCap->Fill(QGptD);
	      mMultiplicityQG_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_EndCap"]; if(mMultiplicityQG_highPt_EndCap && mMultiplicityQG_highPt_EndCap->getRootObject()) mMultiplicityQG_highPt_EndCap->Fill(QGmulti);
	      mqgLikelihood_highPt_EndCap = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_EndCap"]; if(mqgLikelihood_highPt_EndCap && mqgLikelihood_highPt_EndCap->getRootObject()) mqgLikelihood_highPt_EndCap->Fill(QGLikelihood);
	    }
	  }else{
	    if (recoJet1.pt()>20. && recoJet1.pt()<=50.) {
	      mAxis2_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_lowPt_Forward"];if(mAxis2_lowPt_Forward && mAxis2_lowPt_Forward->getRootObject()) mAxis2_lowPt_Forward->Fill(QGaxis2);
	      mpTD_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_lowPt_Forward"]; if(mpTD_lowPt_Forward && mpTD_lowPt_Forward->getRootObject()) mpTD_lowPt_Forward->Fill(QGptD);
	      mMultiplicityQG_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_lowPt_Forward"]; if(mMultiplicityQG_lowPt_Forward && mMultiplicityQG_lowPt_Forward->getRootObject()) mMultiplicityQG_lowPt_Forward->Fill(QGmulti);
	      mqgLikelihood_lowPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_lowPt_Forward"]; if(mqgLikelihood_lowPt_Forward && mqgLikelihood_lowPt_Forward->getRootObject()) mqgLikelihood_lowPt_Forward->Fill(QGLikelihood);
	    }
	    if (recoJet1.pt()>50. && recoJet1.pt()<=140.) {
	      mAxis2_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_mediumPt_Forward"];if(mAxis2_mediumPt_Forward && mAxis2_mediumPt_Forward->getRootObject()) mAxis2_mediumPt_Forward->Fill(QGaxis2);
	      mpTD_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_mediumPt_Forward"]; if(mpTD_mediumPt_Forward && mpTD_mediumPt_Forward->getRootObject()) mpTD_mediumPt_Forward->Fill(QGptD);
	      mMultiplicityQG_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_mediumPt_Forward"]; if(mMultiplicityQG_mediumPt_Forward && mMultiplicityQG_mediumPt_Forward->getRootObject()) mMultiplicityQG_mediumPt_Forward->Fill(QGmulti);
	      mqgLikelihood_mediumPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_mediumPt_Forward"]; if(mqgLikelihood_mediumPt_Forward && mqgLikelihood_mediumPt_Forward->getRootObject()) mqgLikelihood_mediumPt_Forward->Fill(QGLikelihood);
	    }
	    if (recoJet1.pt()>140.) {
	      mAxis2_highPt_Forward = map_of_MEs[DirName+"/"+"qg_Axis2_highPt_Forward"];if(mAxis2_highPt_Forward && mAxis2_highPt_Forward->getRootObject()) mAxis2_highPt_Forward->Fill(QGaxis2);
	      mpTD_highPt_Forward = map_of_MEs[DirName+"/"+"qg_pTD_highPt_Forward"]; if(mpTD_highPt_Forward && mpTD_highPt_Forward->getRootObject()) mpTD_highPt_Forward->Fill(QGptD);
	      mMultiplicityQG_highPt_Forward = map_of_MEs[DirName+"/"+"qg_multiplicity_highPt_Forward"]; if(mMultiplicityQG_highPt_Forward && mMultiplicityQG_highPt_Forward->getRootObject()) mMultiplicityQG_highPt_Forward->Fill(QGmulti);
	      mqgLikelihood_highPt_Forward = map_of_MEs[DirName+"/"+"qg_Likelihood_highPt_Forward"]; if(mqgLikelihood_highPt_Forward && mqgLikelihood_highPt_Forward->getRootObject()) mqgLikelihood_highPt_Forward->Fill(QGLikelihood);
	    }
	  }
	}//fill quark gluon tagged variables
      }//jet back to back to Z      
    }//2nd jet veto
  }//Z selection + hard leading jet
}
