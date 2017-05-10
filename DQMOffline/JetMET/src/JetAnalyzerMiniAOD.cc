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

#include "DQMOffline/JetMET/interface/JetAnalyzerMiniAOD.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
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

using namespace edm;
using namespace reco;
using namespace std;


// ***********************************************************
JetAnalyzerMiniAOD::JetAnalyzerMiniAOD(const edm::ParameterSet& pSet)
{
  parameters_ = pSet.getParameter<edm::ParameterSet>("jetAnalysis");
  mInputCollection_           =    pSet.getParameter<edm::InputTag>       ("jetsrc");

  jetType_ = pSet.getParameter<std::string>("JetType");

  fill_jet_high_level_histo=pSet.getParameter<bool>("filljetHighLevel"),
  
  isCaloJet_ = (std::string("calo")==jetType_);
  isPFJet_   = (std::string("pf") ==jetType_);

  patJetsToken_ = consumes< edm::View<pat::Jet> >(mInputCollection_);
  
  JetIDQuality_  = pSet.getParameter<string>("JetIDQuality");
  JetIDVersion_  = pSet.getParameter<string>("JetIDVersion");

  // JetID definitions for Calo and JPT Jets
  if(/*isJPTJet_ || */isCaloJet_){
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
  if(isPFJet_){
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
  
  highPtJetEventFlag_ = new GenericTriggerEventFlag( highptjetparms, consumesCollector() );
  lowPtJetEventFlag_  = new GenericTriggerEventFlag( lowptjetparms , consumesCollector() );
  
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
JetAnalyzerMiniAOD::~JetAnalyzerMiniAOD() {
  
  delete highPtJetEventFlag_;
  delete lowPtJetEventFlag_;

  delete DCSFilterForDCSMonitoring_;
  delete DCSFilterForJetMonitoring_;
  LogTrace(metname)<<"[JetAnalyzerMiniAOD] Saving the histos";
}

// ***********************************************************
void JetAnalyzerMiniAOD::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				     edm::EventSetup const & ) {
  
  if(jetCleaningFlag_){
    if(isCaloJet_){
      ibooker.setCurrentFolder("JetMET/Jet/CleanedCalo"+mInputCollection_.label());
      DirName = "JetMET/Jet/CleanedCalo"+mInputCollection_.label();
    }else if(isPFJet_){
      ibooker.setCurrentFolder("JetMET/Jet/CleanedPF"+mInputCollection_.label());
      DirName = "JetMET/Jet/CleanedPF"+mInputCollection_.label();
    }
  }else{
    if(isCaloJet_){
      ibooker.setCurrentFolder("JetMET/Jet/UncleanedCalo"+mInputCollection_.label());
      DirName = "JetMET/Jet/UncleanedCalo"+mInputCollection_.label();
    }else if(isPFJet_){
      ibooker.setCurrentFolder("JetMET/Jet/UncleanedPF"+mInputCollection_.label());
      DirName = "JetMET/Jet/UncleanedPF"+mInputCollection_.label();
    }
  }

  jetME = ibooker.book1D("jetReco", "jetReco", 3, 1, 4);
  jetME->setBinLabel(1,"CaloJets",1);
  jetME->setBinLabel(2,"PFJets",1);
  jetME->setBinLabel(3,"JPTJets",1);

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
  mHFrac        = ibooker.book1D("HFrac",        "HFrac",                140,   -0.2,    1.2);
  mEFrac        = ibooker.book1D("EFrac",        "EFrac",           52,   -0.02,    1.02);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorr" ,mJetEnergyCorr));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorrVSEta" ,mJetEnergyCorrVSEta));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorrVSPt" ,mJetEnergyCorrVSPt));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac" ,mHFrac));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac" ,mEFrac));
  
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
  mHFrac_profile        = ibooker.bookProfile("HFrac_profile",        "HFrac",             nbinsPV_, nPVlow_, nPVhigh_,     140,   -0.2,    1.2);
  mEFrac_profile        = ibooker.bookProfile("EFrac_profile",        "EFrac",             nbinsPV_, nPVlow_, nPVhigh_,     52,   -0.02,    1.02);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_profile" ,mPt_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_profile",mEta_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_profile",mPhi_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_profile",mHFrac_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_profile",mEFrac_profile));


  if(!jetCleaningFlag_){//JIDPassFrac_ defines a collection of cleaned jets, for which we will want to fill the cleaning passing fraction
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
  mHFrac_profile       ->setAxisTitle("nvtx",1);
  mEFrac_profile       ->setAxisTitle("nvtx",1);

  mNJets_profile->setAxisTitle("nvtx",1);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_profile" ,mPt_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_profile",mEta_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_profile",mPhi_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_profile" ,mHFrac_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_profile" ,mEFrac_profile));
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
  mHFrac_Barrel            = ibooker.book1D("HFrac_Barrel", "HFrac Barrel", 100, 0, 1);
  mEFrac_Barrel            = ibooker.book1D("EFrac_Barrel", "EFrac Barrel", 52, -0.02, 1.02);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_Barrel" ,mHFrac_Barrel));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_Barrel" ,mEFrac_Barrel));
  
  //mPt_EndCap_Lo            = ibooker.book1D("Pt_EndCap_Lo", "Pt EndCap (Pass Low Pt Jet Trigger)", 20, 0, 100);   
  //mPhi_EndCap_Lo           = ibooker.book1D("Phi_EndCap_Lo", "Phi EndCap (Pass Low Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  //if(!isJPTJet_){
  mConstituents_EndCap     = ibooker.book1D("Constituents_EndCap", "Constituents EndCap", 50, 0, 100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_EndCap",mConstituents_EndCap));
  //}
  mHFrac_EndCap            = ibooker.book1D("HFrac_EndCap", "HFrac EndCap", 100, 0, 1);
  mEFrac_EndCap            = ibooker.book1D("EFrac_EndCap", "EFrac EndCap", 52, -0.02, 1.02);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_EndCap" ,mHFrac_EndCap));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_EndCap" ,mEFrac_EndCap));

  //mPt_Forward_Lo           = ibooker.book1D("Pt_Forward_Lo", "Pt Forward (Pass Low Pt Jet Trigger)", 20, 0, 100);  
  //mPhi_Forward_Lo          = ibooker.book1D("Phi_Forward_Lo", "Phi Forward (Pass Low Pt Jet Trigger)", phiBin_, phiMin_, phiMax_);
  //if(!isJPTJet_){
  mConstituents_Forward    = ibooker.book1D("Constituents_Forward", "Constituents Forward", 50, 0, 100);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_Forward",mConstituents_Forward));
  //}
  mHFrac_Forward           = ibooker.book1D("HFrac_Forward", "HFrac Forward", 140, -0.2, 1.2);
  mEFrac_Forward           = ibooker.book1D("EFrac_Forward", "EFrac Forward", 52, -0.02, 1.02);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_Forward" ,mHFrac_Forward));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_Forward" ,mEFrac_Forward));


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
  mEtaFirst                = ibooker.book1D("EtaFirst", "EtaFirst", 100, -5, 5);
  mPhiFirst                = ibooker.book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5);
  mPtFirst                 = ibooker.book1D("PtFirst", "PtFirst", ptBin_, ptMin_, ptMax_);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EtaFirst" ,mEtaFirst));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PtFirst"  ,mPtFirst));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhiFirst" ,mPhiFirst));
  
  //--- Calo jet selection only
  if(isCaloJet_) {

    // CaloJet specific
    mMaxEInEmTowers         = ibooker.book1D("MaxEInEmTowers", "MaxEInEmTowers", 150, 0, 150);
    mMaxEInHadTowers        = ibooker.book1D("MaxEInHadTowers", "MaxEInHadTowers", 150, 0, 150);
    
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MaxEInEmTowers"  ,mMaxEInEmTowers));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MaxEInHadTowers" ,mMaxEInHadTowers));

    mHadEnergyInHO          = ibooker.book1D("HadEnergyInHO", "HadEnergyInHO", 100, 0, 20);
    mHadEnergyInHB          = ibooker.book1D("HadEnergyInHB", "HadEnergyInHB", 100, 0, 100);
    mHadEnergyInHF          = ibooker.book1D("HadEnergyInHF", "HadEnergyInHF", 100, 0, 100);
    mHadEnergyInHE          = ibooker.book1D("HadEnergyInHE", "HadEnergyInHE", 100, 0, 200);
    mEmEnergyInEB           = ibooker.book1D("EmEnergyInEB", "EmEnergyInEB", 100, 0, 100);
    mEmEnergyInEE           = ibooker.book1D("EmEnergyInEE", "EmEnergyInEE", 100, 0, 100);
    mEmEnergyInHF           = ibooker.book1D("EmEnergyInHF", "EmEnergyInHF", 120, -20, 200);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HadEnergyInHO"  ,mHadEnergyInHO));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HadEnergyInHB"  ,mHadEnergyInHB));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HadEnergyInHF"  ,mHadEnergyInHF));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HadEnergyInHE"  ,mHadEnergyInHE));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EmEnergyInEB" ,mEmEnergyInEB));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EmEnergyInEE" ,mEmEnergyInEE));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EmEnergyInHF" ,mEmEnergyInHF));
    
    //JetID variables
    mresEMF                 = ibooker.book1D("resEMF", "resEMF", 50, 0., 1.);
    mN90Hits                = ibooker.book1D("N90Hits", "N90Hits", 100, 0., 100);
    mfHPD                   = ibooker.book1D("fHPD", "fHPD", 50, 0., 1.);
    mfRBX                   = ibooker.book1D("fRBX", "fRBX", 50, 0., 1.);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"resEMF" ,mresEMF));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"N90Hits" ,mN90Hits));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"fHPD" ,mfHPD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"fRBX" ,mfRBX));
  }

  if(isPFJet_) {
    //PFJet specific histograms
    mCHFracVSeta_lowPt= ibooker.bookProfile("CHFracVSeta_lowPt","CHFracVSeta_lowPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mNHFracVSeta_lowPt= ibooker.bookProfile("NHFracVSeta_lowPt","NHFracVSeta_lowPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mPhFracVSeta_lowPt= ibooker.bookProfile("PhFracVSeta_lowPt","PhFracVSeta_lowPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mElFracVSeta_lowPt= ibooker.bookProfile("ElFracVSeta_lowPt","ElFracVSeta_lowPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mMuFracVSeta_lowPt= ibooker.bookProfile("MuFracVSeta_lowPt","MuFracVSeta_lowPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mCHFracVSeta_mediumPt= ibooker.bookProfile("CHFracVSeta_mediumPt","CHFracVSeta_mediumPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mNHFracVSeta_mediumPt= ibooker.bookProfile("NHFracVSeta_mediumPt","NHFracVSeta_mediumPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mPhFracVSeta_mediumPt= ibooker.bookProfile("PhFracVSeta_mediumPt","PhFracVSeta_mediumPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mElFracVSeta_mediumPt= ibooker.bookProfile("ElFracVSeta_mediumPt","ElFracVSeta_mediumPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mMuFracVSeta_mediumPt= ibooker.bookProfile("MuFracVSeta_mediumPt","MuFracVSeta_mediumPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mCHFracVSeta_highPt= ibooker.bookProfile("CHFracVSeta_highPt","CHFracVSeta_highPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mNHFracVSeta_highPt= ibooker.bookProfile("NHFracVSeta_highPt","NHFracVSeta_highPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mPhFracVSeta_highPt= ibooker.bookProfile("PhFracVSeta_highPt","PhFracVSeta_highPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mElFracVSeta_highPt= ibooker.bookProfile("ElFracVSeta_highPt","ElFracVSeta_highPt",etaBin_, etaMin_, etaMax_,0.,1.2);
    mMuFracVSeta_highPt= ibooker.bookProfile("MuFracVSeta_highPt","MuFracVSeta_highPt",etaBin_, etaMin_, etaMax_,0.,1.2);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSeta_lowPt" ,mCHFracVSeta_lowPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSeta_lowPt" ,mNHFracVSeta_lowPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSeta_lowPt" ,mPhFracVSeta_lowPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFracVSeta_lowPt" ,mElFracVSeta_lowPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFracVSeta_lowPt" ,mMuFracVSeta_lowPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSeta_mediumPt" ,mCHFracVSeta_mediumPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSeta_mediumPt" ,mNHFracVSeta_mediumPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSeta_mediumPt" ,mPhFracVSeta_mediumPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFracVSeta_mediumPt" ,mElFracVSeta_mediumPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFracVSeta_mediumPt" ,mMuFracVSeta_mediumPt)); 
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSeta_highPt" ,mCHFracVSeta_highPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSeta_highPt" ,mNHFracVSeta_highPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSeta_highPt" ,mPhFracVSeta_highPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFracVSeta_highPt" ,mElFracVSeta_highPt));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFracVSeta_highPt" ,mMuFracVSeta_highPt)); 

    //barrel histograms for PFJets
    // energy fractions
    mCHFrac_lowPt_Barrel     = ibooker.book1D("CHFrac_lowPt_Barrel", "CHFrac_lowPt_Barrel", 120, -0.1, 1.1);
    mNHFrac_lowPt_Barrel     = ibooker.book1D("NHFrac_lowPt_Barrel", "NHFrac_lowPt_Barrel", 120, -0.1, 1.1);
    mPhFrac_lowPt_Barrel     = ibooker.book1D("PhFrac_lowPt_Barrel", "PhFrac_lowPt_Barrel", 120, -0.1, 1.1);
    mElFrac_lowPt_Barrel     = ibooker.book1D("ElFrac_lowPt_Barrel", "ElFrac_lowPt_Barrel", 120, -0.1, 1.1);
    mMuFrac_lowPt_Barrel     = ibooker.book1D("MuFrac_lowPt_Barrel", "MuFrac_lowPt_Barrel", 120, -0.1, 1.1);
    mCHFrac_mediumPt_Barrel  = ibooker.book1D("CHFrac_mediumPt_Barrel", "CHFrac_mediumPt_Barrel", 120, -0.1, 1.1);
    mNHFrac_mediumPt_Barrel  = ibooker.book1D("NHFrac_mediumPt_Barrel", "NHFrac_mediumPt_Barrel", 120, -0.1, 1.1);
    mPhFrac_mediumPt_Barrel  = ibooker.book1D("PhFrac_mediumPt_Barrel", "PhFrac_mediumPt_Barrel", 120, -0.1, 1.1);
    mElFrac_mediumPt_Barrel  = ibooker.book1D("ElFrac_mediumPt_Barrel", "ElFrac_mediumPt_Barrel", 120, -0.1, 1.1);
    mMuFrac_mediumPt_Barrel  = ibooker.book1D("MuFrac_mediumPt_Barrel", "MuFrac_mediumPt_Barrel", 120, -0.1, 1.1);
    mCHFrac_highPt_Barrel    = ibooker.book1D("CHFrac_highPt_Barrel", "CHFrac_highPt_Barrel", 120, -0.1, 1.1);
    mNHFrac_highPt_Barrel    = ibooker.book1D("NHFrac_highPt_Barrel", "NHFrac_highPt_Barrel", 120, -0.1, 1.1);
    mPhFrac_highPt_Barrel    = ibooker.book1D("PhFrac_highPt_Barrel", "PhFrac_highPt_Barrel", 120, -0.1, 1.1);
    mElFrac_highPt_Barrel    = ibooker.book1D("ElFrac_highPt_Barrel", "ElFrac_highPt_Barrel", 120, -0.1, 1.1);
    mMuFrac_highPt_Barrel    = ibooker.book1D("MuFrac_highPt_Barrel", "MuFrac_highPt_Barrel", 120, -0.1, 1.1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_lowPt_Barrel" ,mCHFrac_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_lowPt_Barrel" ,mNHFrac_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_lowPt_Barrel" ,mPhFrac_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFrac_lowPt_Barrel" ,mElFrac_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFrac_lowPt_Barrel" ,mMuFrac_lowPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_mediumPt_Barrel" ,mCHFrac_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_mediumPt_Barrel" ,mNHFrac_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_mediumPt_Barrel" ,mPhFrac_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFrac_mediumPt_Barrel" ,mElFrac_mediumPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFrac_mediumPt_Barrel" ,mMuFrac_mediumPt_Barrel)); 
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_highPt_Barrel" ,mCHFrac_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_highPt_Barrel" ,mNHFrac_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_highPt_Barrel" ,mPhFrac_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFrac_highPt_Barrel" ,mElFrac_highPt_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFrac_highPt_Barrel" ,mMuFrac_highPt_Barrel)); 
    
    //energies
    mCHEn_lowPt_Barrel     = ibooker.book1D("CHEn_lowPt_Barrel", "CHEn_lowPt_Barrel", ptBin_, 0., ptMax_);
    mNHEn_lowPt_Barrel     = ibooker.book1D("NHEn_lowPt_Barrel", "NHEn_lowPt_Barrel", ptBin_, 0., ptMax_);
    mPhEn_lowPt_Barrel     = ibooker.book1D("PhEn_lowPt_Barrel", "PhEn_lowPt_Barrel", ptBin_, 0., ptMax_);
    mElEn_lowPt_Barrel     = ibooker.book1D("ElEn_lowPt_Barrel", "ElEn_lowPt_Barrel", ptBin_, 0., ptMax_);
    mMuEn_lowPt_Barrel     = ibooker.book1D("MuEn_lowPt_Barrel", "MuEn_lowPt_Barrel", ptBin_, 0., ptMax_);
    mCHEn_mediumPt_Barrel  = ibooker.book1D("CHEn_mediumPt_Barrel", "CHEn_mediumPt_Barrel", ptBin_, 0., ptMax_);
    mNHEn_mediumPt_Barrel  = ibooker.book1D("NHEn_mediumPt_Barrel", "NHEn_mediumPt_Barrel", ptBin_, 0., ptMax_);
    mPhEn_mediumPt_Barrel  = ibooker.book1D("PhEn_mediumPt_Barrel", "PhEn_mediumPt_Barrel", ptBin_, 0., ptMax_);
    mElEn_mediumPt_Barrel  = ibooker.book1D("ElEn_mediumPt_Barrel", "ElEn_mediumPt_Barrel", ptBin_, 0., ptMax_);
    mMuEn_mediumPt_Barrel  = ibooker.book1D("MuEn_mediumPt_Barrel", "MuEn_mediumPt_Barrel", ptBin_, 0., ptMax_);
    mCHEn_highPt_Barrel    = ibooker.book1D("CHEn_highPt_Barrel", "CHEn_highPt_Barrel", ptBin_, 0., 1.1*ptMax_);
    mNHEn_highPt_Barrel    = ibooker.book1D("NHEn_highPt_Barrel", "NHEn_highPt_Barrel", ptBin_, 0., ptMax_);
    mPhEn_highPt_Barrel    = ibooker.book1D("PhEn_highPt_Barrel", "PhEn_highPt_Barrel", ptBin_, 0., ptMax_);
    mElEn_highPt_Barrel    = ibooker.book1D("ElEn_highPt_Barrel", "ElEn_highPt_Barrel", ptBin_, 0., ptMax_);
    mMuEn_highPt_Barrel    = ibooker.book1D("MuEn_highPt_Barrel", "MuEn_highPt_Barrel", ptBin_, 0., ptMax_);

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

    //
    mCHFracVSpT_Barrel= ibooker.bookProfile("CHFracVSpT_Barrel","CHFracVSpT_Barrel",ptBin_, ptMin_, ptMax_,0.,1.2);
    mNHFracVSpT_Barrel= ibooker.bookProfile("NHFracVSpT_Barrel","NHFracVSpT_Barrel",ptBin_, ptMin_, ptMax_,0.,1.2);
    mPhFracVSpT_Barrel= ibooker.bookProfile("PhFracVSpT_Barrel","PhFracVSpT_Barrel",ptBin_, ptMin_, ptMax_,0.,1.2);
    mElFracVSpT_Barrel= ibooker.bookProfile("ElFracVSpT_Barrel","ElFracVSpT_Barrel",ptBin_, ptMin_, ptMax_,0.,1.2);
    mMuFracVSpT_Barrel= ibooker.bookProfile("MuFracVSpT_Barrel","MuFracVSpT_Barrel",ptBin_, ptMin_, ptMax_,0.,1.2);
    mCHFracVSpT_EndCap= ibooker.bookProfile("CHFracVSpT_EndCap","CHFracVSpT_EndCap",ptBin_, ptMin_, ptMax_,0.,1.2);
    mNHFracVSpT_EndCap= ibooker.bookProfile("NHFracVSpT_EndCap","NHFracVSpT_EndCap",ptBin_, ptMin_, ptMax_,0.,1.2);
    mPhFracVSpT_EndCap= ibooker.bookProfile("PhFracVSpT_EndCap","PhFracVSpT_EndCap",ptBin_, ptMin_, ptMax_,0.,1.2);
    mElFracVSpT_EndCap= ibooker.bookProfile("ElFracVSpT_EndCap","ElFracVSpT_EndCap",ptBin_, ptMin_, ptMax_,0.,1.2);
    mMuFracVSpT_EndCap= ibooker.bookProfile("MuFracVSpT_EndCap","MuFracVSpT_EndCap",ptBin_, ptMin_, ptMax_,0.,1.2);
    mHFHFracVSpT_Forward= ibooker.bookProfile("HFHFracVSpT_Forward","HFHFracVSpT_Forward",ptBin_, ptMin_, ptMax_,-0.2,1.2);
    mHFEFracVSpT_Forward= ibooker.bookProfile("HFEFracVSpT_Forward","HFEFracVSpT_Forward",ptBin_, ptMin_, ptMax_,-0.2,1.2);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSpT_Barrel" ,mCHFracVSpT_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSpT_Barrel" ,mNHFracVSpT_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSpT_Barrel" ,mPhFracVSpT_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFracVSpT_Barrel" ,mElFracVSpT_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFracVSpT_Barrel" ,mMuFracVSpT_Barrel));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFracVSpT_EndCap" ,mCHFracVSpT_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFracVSpT_EndCap" ,mNHFracVSpT_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFracVSpT_EndCap" ,mPhFracVSpT_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFracVSpT_EndCap" ,mElFracVSpT_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFracVSpT_Forward" ,mHFHFracVSpT_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEFracVSpT_Forward" ,mHFEFracVSpT_Forward));

    //endcap monitoring
    //energy fractions
    mCHFrac_lowPt_EndCap     = ibooker.book1D("CHFrac_lowPt_EndCap", "CHFrac_lowPt_EndCap", 120, -0.1, 1.1);
    mNHFrac_lowPt_EndCap     = ibooker.book1D("NHFrac_lowPt_EndCap", "NHFrac_lowPt_EndCap", 120, -0.1, 1.1);
    mPhFrac_lowPt_EndCap     = ibooker.book1D("PhFrac_lowPt_EndCap", "PhFrac_lowPt_EndCap", 120, -0.1, 1.1);
    mElFrac_lowPt_EndCap     = ibooker.book1D("ElFrac_lowPt_EndCap", "ElFrac_lowPt_EndCap", 120, -0.1, 1.1);
    mMuFrac_lowPt_EndCap     = ibooker.book1D("MuFrac_lowPt_EndCap", "MuFrac_lowPt_EndCap", 120, -0.1, 1.1);
    mCHFrac_mediumPt_EndCap  = ibooker.book1D("CHFrac_mediumPt_EndCap", "CHFrac_mediumPt_EndCap", 120, -0.1, 1.1);
    mNHFrac_mediumPt_EndCap  = ibooker.book1D("NHFrac_mediumPt_EndCap", "NHFrac_mediumPt_EndCap", 120, -0.1, 1.1);
    mPhFrac_mediumPt_EndCap  = ibooker.book1D("PhFrac_mediumPt_EndCap", "PhFrac_mediumPt_EndCap", 120, -0.1, 1.1);
    mElFrac_mediumPt_EndCap  = ibooker.book1D("ElFrac_mediumPt_EndCap", "ElFrac_mediumPt_EndCap", 120, -0.1, 1.1);
    mMuFrac_mediumPt_EndCap  = ibooker.book1D("MuFrac_mediumPt_EndCap", "MuFrac_mediumPt_EndCap", 120, -0.1, 1.1);
    mCHFrac_highPt_EndCap    = ibooker.book1D("CHFrac_highPt_EndCap", "CHFrac_highPt_EndCap", 120, -0.1, 1.1);
    mNHFrac_highPt_EndCap    = ibooker.book1D("NHFrac_highPt_EndCap", "NHFrac_highPt_EndCap", 120, -0.1, 1.1);
    mPhFrac_highPt_EndCap    = ibooker.book1D("PhFrac_highPt_EndCap", "PhFrac_highPt_EndCap", 120, -0.1, 1.1);
    mElFrac_highPt_EndCap    = ibooker.book1D("ElFrac_highPt_EndCap", "ElFrac_highPt_EndCap", 120, -0.1, 1.1);
    mMuFrac_highPt_EndCap    = ibooker.book1D("MuFrac_highPt_EndCap", "MuFrac_highPt_EndCap", 120, -0.1, 1.1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_lowPt_EndCap" ,mCHFrac_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_lowPt_EndCap" ,mNHFrac_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_lowPt_EndCap" ,mPhFrac_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFrac_lowPt_EndCap" ,mElFrac_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFrac_lowPt_EndCap" ,mMuFrac_lowPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_mediumPt_EndCap" ,mCHFrac_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_mediumPt_EndCap" ,mNHFrac_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_mediumPt_EndCap" ,mPhFrac_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFrac_mediumPt_EndCap" ,mElFrac_mediumPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFrac_mediumPt_EndCap" ,mMuFrac_mediumPt_EndCap)); 
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_highPt_EndCap" ,mCHFrac_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_highPt_EndCap" ,mNHFrac_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_highPt_EndCap" ,mPhFrac_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFrac_highPt_EndCap" ,mElFrac_highPt_EndCap));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFrac_highPt_EndCap" ,mMuFrac_highPt_EndCap)); 

    //energies
    mCHEn_lowPt_EndCap     = ibooker.book1D("CHEn_lowPt_EndCap", "CHEn_lowPt_EndCap", ptBin_, 0., ptMax_);
    mNHEn_lowPt_EndCap     = ibooker.book1D("NHEn_lowPt_EndCap", "NHEn_lowPt_EndCap", ptBin_, 0., ptMax_);
    mPhEn_lowPt_EndCap     = ibooker.book1D("PhEn_lowPt_EndCap", "PhEn_lowPt_EndCap", ptBin_, 0., ptMax_);
    mElEn_lowPt_EndCap     = ibooker.book1D("ElEn_lowPt_EndCap", "ElEn_lowPt_EndCap", ptBin_, 0., ptMax_);
    mMuEn_lowPt_EndCap     = ibooker.book1D("MuEn_lowPt_EndCap", "MuEn_lowPt_EndCap", ptBin_, 0., ptMax_);
    mCHEn_mediumPt_EndCap  = ibooker.book1D("CHEn_mediumPt_EndCap", "CHEn_mediumPt_EndCap", ptBin_, 0., ptMax_);
    mNHEn_mediumPt_EndCap  = ibooker.book1D("NHEn_mediumPt_EndCap", "NHEn_mediumPt_EndCap", ptBin_, 0., ptMax_);
    mPhEn_mediumPt_EndCap  = ibooker.book1D("PhEn_mediumPt_EndCap", "PhEn_mediumPt_EndCap", ptBin_, 0., ptMax_);
    mElEn_mediumPt_EndCap  = ibooker.book1D("ElEn_mediumPt_EndCap", "ElEn_mediumPt_EndCap", ptBin_, 0., ptMax_);
    mMuEn_mediumPt_EndCap  = ibooker.book1D("MuEn_mediumPt_EndCap", "MuEn_mediumPt_EndCap", ptBin_, 0., ptMax_);
    mCHEn_highPt_EndCap    = ibooker.book1D("CHEn_highPt_EndCap", "CHEn_highPt_EndCap", ptBin_, 0., 1.5*ptMax_);
    mNHEn_highPt_EndCap    = ibooker.book1D("NHEn_highPt_EndCap", "NHEn_highPt_EndCap", ptBin_, 0., 1.5*ptMax_);
    mPhEn_highPt_EndCap    = ibooker.book1D("PhEn_highPt_EndCap", "PhEn_highPt_EndCap", ptBin_, 0., 1.5*ptMax_);
    mElEn_highPt_EndCap    = ibooker.book1D("ElEn_highPt_EndCap", "ElEn_highPt_EndCap", ptBin_, 0., ptMax_);
    mMuEn_highPt_EndCap    = ibooker.book1D("MuEn_highPt_EndCap", "MuEn_highPt_EndCap", ptBin_, 0., ptMax_);

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
    mHFEFrac_lowPt_Forward    = ibooker.book1D("HFEFrac_lowPt_Forward", "HFEFrac_lowPt_Forward", 140, -0.2, 1.2);
    mHFHFrac_lowPt_Forward    = ibooker.book1D("HFHFrac_lowPt_Forward", "HFHFrac_lowPt_Forward", 140, -0.2, 1.2);
    mHFEFrac_mediumPt_Forward = ibooker.book1D("HFEFrac_mediumPt_Forward", "HFEFrac_mediumPt_Forward", 140, -0.2, 1.2);
    mHFHFrac_mediumPt_Forward = ibooker.book1D("HFHFrac_mediumPt_Forward", "HFHFrac_mediumPt_Forward", 140, -0.2, 1.2);
    mHFEFrac_highPt_Forward   = ibooker.book1D("HFEFrac_highPt_Forward", "HFEFrac_highPt_Forward", 140, -0.2, 1.2);
    mHFHFrac_highPt_Forward   = ibooker.book1D("HFHFrac_highPt_Forward", "HFHFrac_highPt_Forward", 140, -0.2, 1.2);
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
    mChMultiplicity_lowPt_Forward     = ibooker.book1D("ChMultiplicity_lowPt_Forward", "ChMultiplicity_lowPt_Forward", 60,0,60);
    mNeutMultiplicity_lowPt_Forward    = ibooker.book1D("NeutMultiplicity_lowPt_Forward", "NeutMultiplicity_lowPt_Forward", 60,0,60);
    mChMultiplicity_mediumPt_Forward  = ibooker.book1D("ChMultiplicity_mediumPt_Forward", "ChMultiplicity_mediumPt_Forward", 60,0,60);
    mNeutMultiplicity_mediumPt_Forward = ibooker.book1D("NeutMultiplicity_mediumPt_Forward", "NeutMultiplicity_mediumPt_Forward", 60,0,60);
    mChMultiplicity_highPt_Forward    = ibooker.book1D("ChMultiplicity_highPt_Forward", "ChMultiplicity_highPt_Forward", 60,0,60);
    mNeutMultiplicity_highPt_Forward   = ibooker.book1D("NeutMultiplicity_highPt_Forward", "NeutMultiplicity_highPt_Forward", 60,0,60);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_lowPt_Forward" ,mChMultiplicity_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_lowPt_Forward" ,mNeutMultiplicity_lowPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_mediumPt_Forward" ,mChMultiplicity_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_mediumPt_Forward" ,mNeutMultiplicity_mediumPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChMultiplicity_highPt_Forward" ,mChMultiplicity_highPt_Forward));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutMultiplicity_highPt_Forward" ,mNeutMultiplicity_highPt_Forward));
    
    mChargedHadronEnergy = ibooker.book1D("ChargedHadronEnergy", "charged HAD energy",    100, 0, 100);
    mNeutralHadronEnergy = ibooker.book1D("NeutralHadronEnergy", "neutral HAD energy",    100, 0, 100);
    mChargedEmEnergy     = ibooker.book1D("ChargedEmEnergy",    "charged EM energy ",    100, 0, 100);
    mChargedMuEnergy     = ibooker.book1D("ChargedMuEnergy",     "charged Mu energy",     100, 0, 100);
    mNeutralEmEnergy     = ibooker.book1D("NeutralEmEnergy",     "neutral EM energy",     100, 0, 100);
    mChargedMultiplicity = ibooker.book1D("ChargedMultiplicity", "charged multiplicity ", 100, 0, 100);
    mNeutralMultiplicity = ibooker.book1D("NeutralMultiplicity", "neutral multiplicity",  100, 0, 100);
    mMuonMultiplicity    = ibooker.book1D("MuonMultiplicity",    "muon multiplicity",     100, 0, 100);
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
    mChargedHadronEnergy_profile = ibooker.bookProfile("ChargedHadronEnergy_profile", "charged HAD energy",   nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mNeutralHadronEnergy_profile = ibooker.bookProfile("NeutralHadronEnergy_profile", "neutral HAD energy",   nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mChargedEmEnergy_profile     = ibooker.bookProfile("ChargedEmEnergy_profile",     "charged EM energy",    nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mChargedMuEnergy_profile     = ibooker.bookProfile("ChargedMuEnergy_profile",     "charged Mu energy",    nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mNeutralEmEnergy_profile     = ibooker.bookProfile("NeutralEmEnergy_profile",     "neutral EM energy",    nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mChargedMultiplicity_profile = ibooker.bookProfile("ChargedMultiplicity_profile", "charged multiplicity", nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mNeutralMultiplicity_profile = ibooker.bookProfile("NeutralMultiplicity_profile", "neutral multiplicity", nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mMuonMultiplicity_profile    = ibooker.bookProfile("MuonMultiplicity_profile",    "muon multiplicity",    nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    
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

void JetAnalyzerMiniAOD::bookMESetSelection(std::string DirName, DQMStore::IBooker & ibooker)
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
  mHFrac        = ibooker.book1D("HFrac",        "HFrac",                140,   -0.2,    1.2);
  mEFrac        = ibooker.book1D("EFrac",        "EFrac",                140,   -0.2,    1.2);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorr" ,mJetEnergyCorr));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorrVSEta" ,mJetEnergyCorrVSEta));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"JetEnergyCorrVSPt" ,mJetEnergyCorrVSPt));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac" ,mHFrac));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac" ,mEFrac));

  mDPhi                   = ibooker.book1D("DPhi", "dPhi btw the two leading jets", 100, 0., acos(-1.));
  mDijetAsymmetry                   = ibooker.book1D("DijetAsymmetry", "DijetAsymmetry", 100, -1., 1.);
  mDijetBalance                     = ibooker.book1D("DijetBalance",   "DijetBalance",   100, -2., 2.);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DPhi" ,mDPhi));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DijetAsymmetry" ,mDijetAsymmetry));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"DijetBalance"   ,mDijetBalance));

  // Book NPV profiles
  //----------------------------------------------------------------------------
  mPt_profile           = ibooker.bookProfile("Pt_profile",           "pt",                nbinsPV_, nPVlow_, nPVhigh_,   ptBin_,  ptMin_,  ptMax_);
  mEta_profile          = ibooker.bookProfile("Eta_profile",          "eta",               nbinsPV_, nPVlow_, nPVhigh_,  etaBin_, etaMin_, etaMax_);
  mPhi_profile          = ibooker.bookProfile("Phi_profile",          "phi",               nbinsPV_, nPVlow_, nPVhigh_,  phiBin_, phiMin_, phiMax_);
  //if(!isJPTJet_){
  mConstituents_profile = ibooker.bookProfile("Constituents_profile", "# of constituents", nbinsPV_, nPVlow_, nPVhigh_,      50,      0,    100);
  //}
  mHFrac_profile        = ibooker.bookProfile("HFrac_profile",        "HFrac",             nbinsPV_, nPVlow_, nPVhigh_,     140,   -0.2,    1.2);
  mEFrac_profile        = ibooker.bookProfile("EFrac_profile",        "EFrac",             nbinsPV_, nPVlow_, nPVhigh_,     140,   -0.2,    1.2);
  // met NPV profiles x-axis title
  //----------------------------------------------------------------------------
  mPt_profile          ->setAxisTitle("nvtx",1);
  mEta_profile         ->setAxisTitle("nvtx",1);
  mPhi_profile         ->setAxisTitle("nvtx",1);
  //if(!isJPTJet_){
  mConstituents_profile->setAxisTitle("nvtx",1);
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Constituents_profile",mConstituents_profile));
  //}
  mHFrac_profile       ->setAxisTitle("nvtx",1);
  mEFrac_profile       ->setAxisTitle("nvtx",1);

  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Pt_profile" ,mPt_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Eta_profile",mEta_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"Phi_profile",mPhi_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFrac_profile",mHFrac_profile));
  map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"EFrac_profile",mEFrac_profile));

  //
  //--- Calo jet melection only
  if(isCaloJet_) {
    // CaloJet mpecific
    mMaxEInEmTowers         = ibooker.book1D("MaxEInEmTowers", "MaxEInEmTowers", 150, 0, 150);
    mMaxEInHadTowers        = ibooker.book1D("MaxEInHadTowers", "MaxEInHadTowers", 150, 0, 150);
    //JetID variables
    mresEMF                 = ibooker.book1D("resEMF", "resEMF", 50, 0., 1.);
    mN90Hits                = ibooker.book1D("N90Hits", "N90Hits", 100, 0., 100);
    mfHPD                   = ibooker.book1D("fHPD", "fHPD", 50, 0., 1.);
    mfRBX                   = ibooker.book1D("fRBX", "fRBX", 50, 0., 1.);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MaxEInEmTowers"  ,mMaxEInEmTowers));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MaxEInHadTowers" ,mMaxEInHadTowers));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"resEMF" ,mresEMF));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"N90Hits",mN90Hits));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"fHPD" ,mfHPD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"fRBX" ,mfRBX));

  }

  if(isPFJet_){ 
    //barrel histograms for PFJets
    // energy fractions
    mCHFrac     = ibooker.book1D("CHFrac", "CHFrac", 120, -0.1, 1.1);
    mNHFrac     = ibooker.book1D("NHFrac", "NHFrac", 120, -0.1, 1.1);
    mPhFrac     = ibooker.book1D("PhFrac", "PhFrac", 120, -0.1, 1.1);
    mElFrac     = ibooker.book1D("ElFrac", "ElFrac", 120, -0.1, 1.1);
    mMuFrac     = ibooker.book1D("MuFrac", "MuFrac", 120, -0.1, 1.1);
    mHFEMFrac   = ibooker.book1D("HFEMFrac","HFEMFrac", 120, -0.1, 1.1);
    mHFHFrac   = ibooker.book1D("HFHFrac", "HFHFrac", 120, -0.1, 1.1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac"  ,mCHFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac"  ,mNHFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac"  ,mPhFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFrac"  ,mElFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFrac"  ,mMuFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFrac",mHFEMFrac));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFrac" ,mHFHFrac));

    mChargedMultiplicity = ibooker.book1D("ChargedMultiplicity", "charged multiplicity ", 100, 0, 100);
    mNeutralMultiplicity = ibooker.book1D("NeutralMultiplicity", "neutral multiplicity",  100, 0, 100);
    mMuonMultiplicity    = ibooker.book1D("MuonMultiplicity",    "muon multiplicity",     100, 0, 100);
   
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedMultiplicity" ,mChargedMultiplicity));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralMultiplicity" ,mNeutralMultiplicity));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuonMultiplicity"    ,mMuonMultiplicity));
    
    // Book NPV profiles
    //----------------------------------------------------------------------------
    mCHFrac_profile = ibooker.bookProfile("CHFrac_profile", "charged HAD fraction profile",   nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 1);
    mNHFrac_profile = ibooker.bookProfile("NHFrac_profile", "neutral HAD fraction profile",   nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 1);
    mElFrac_profile    = ibooker.bookProfile("ElFrac_profile",     "Electron Fraction Profile",    nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 1);
    mMuFrac_profile    = ibooker.bookProfile("MuFrac_profile",     "Muon Fraction Profile",    nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 1);
    mPhFrac_profile    = ibooker.bookProfile("PhFrac_profile",     "Photon Fraction Profile",    nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 1);
    mHFEMFrac_profile  = ibooker.bookProfile("HFEMFrac_profile","HF electomagnetic fraction Profile", nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 1);
    mHFHFrac_profile   = ibooker.bookProfile("HFHFrac_profile", "HF hadronic fraction profile", nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 1);
    mChargedMultiplicity_profile = ibooker.bookProfile("ChargedMultiplicity_profile", "charged multiplicity", nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mNeutralMultiplicity_profile = ibooker.bookProfile("NeutralMultiplicity_profile", "neutral multiplicity", nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    mMuonMultiplicity_profile    = ibooker.bookProfile("MuonMultiplicity_profile",    "muon multiplicity",    nbinsPV_, nPVlow_, nPVhigh_, 100, 0, 100);
    
    // met NPV profiles x-axis title
    //----------------------------------------------------------------------------
    mCHFrac_profile    ->setAxisTitle("nvtx",1);
    mNHFrac_profile    ->setAxisTitle("nvtx",1);
    mElFrac_profile    ->setAxisTitle("nvtx",1);
    mMuFrac_profile    ->setAxisTitle("nvtx",1);
    mPhFrac_profile    ->setAxisTitle("nvtx",1);
    mHFEMFrac_profile  ->setAxisTitle("nvtx",1);
    mHFHFrac_profile   ->setAxisTitle("nvtx",1);
    mChargedMultiplicity_profile->setAxisTitle("nvtx",1);
    mNeutralMultiplicity_profile->setAxisTitle("nvtx",1);
    mMuonMultiplicity_profile   ->setAxisTitle("nvtx",1);

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"CHFrac_profile"  ,mCHFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NHFrac_profile"  ,mNHFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"PhFrac_profile"  ,mPhFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ElFrac_profile"  ,mElFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuFrac_profile"  ,mMuFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFEMFrac_profile",mHFEMFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"HFHFrac_profile" ,mHFHFrac_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"ChargedMultiplicity_profile" ,mChargedMultiplicity_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralMultiplicity_profile" ,mNeutralMultiplicity_profile));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"MuonMultiplicity_profile"    ,mMuonMultiplicity_profile));
   
    mNeutralFraction     = ibooker.book1D("NeutralConstituentsFraction","Neutral Constituents Fraction",100,0,1);
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(DirName+"/"+"NeutralConstituentsFraction" ,mNeutralFraction));
  }
 
}

// ***********************************************************
void JetAnalyzerMiniAOD::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  //LogDebug("JetAnalyzerMiniAOD") << "beginRun, run " << run.id();
  //

  if ( highPtJetEventFlag_->on() ) highPtJetEventFlag_->initRun( iRun, iSetup );
  if ( lowPtJetEventFlag_ ->on() ) lowPtJetEventFlag_ ->initRun( iRun, iSetup );

  if (highPtJetEventFlag_->on() && highPtJetEventFlag_->expressionsFromDB(highPtJetEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highPtJetExpr_ = highPtJetEventFlag_->expressionsFromDB(highPtJetEventFlag_->hltDBKey(), iSetup);
  if (lowPtJetEventFlag_->on() && lowPtJetEventFlag_->expressionsFromDB(lowPtJetEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    lowPtJetExpr_  = lowPtJetEventFlag_->expressionsFromDB(lowPtJetEventFlag_->hltDBKey(),   iSetup);
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
      LogDebug("JetAnalyzerMiniAOD") << "HLTConfigProvider failed to initialize.";
    }
  }
 
}

// ***********************************************************
void JetAnalyzerMiniAOD::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
}

// ***********************************************************
void JetAnalyzerMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {


  //set general folders first --> change later on for different folders
  if(jetCleaningFlag_){
    if(isCaloJet_){
      DirName = "JetMET/Jet/CleanedCalo"+mInputCollection_.label();
    }else if(isPFJet_){
      DirName = "JetMET/Jet/CleanedPF"+mInputCollection_.label();
    }
  }else{
    if(isCaloJet_){
      DirName = "JetMET/Jet/UncleanedCalo"+mInputCollection_.label();
    }else if(isPFJet_){
      DirName = "JetMET/Jet/UncleanedPF"+mInputCollection_.label();
    }
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
  }else{
    std::cout<<"vertexhandle is not valid"<<std::endl;
  }
  bool bPrimaryVertex = (bypassAllPVChecks_ || (numPV>0));
  if(fill_jet_high_level_histo){//should be filled for all events, no selection up to this point
    verticesME = map_of_MEs["JetMET/vertices"]; if(verticesME && verticesME->getRootObject())verticesME->Fill(numPV);
  }
  // ==========================================================

  edm::Handle<L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  iEvent.getByToken(gtToken_, gtReadoutRecord);


  if (!gtReadoutRecord.isValid()) {
    LogInfo("JetAnalyzerMiniAOD") << "JetAnalyzerMiniAOD: Could not find GT readout record" << std::endl;
    if (verbose_) std::cout << "JetAnalyzerMiniAOD: Could not find GT readout record product" << std::endl;
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
  edm::Handle<edm::View<pat::Jet> >jets;

  iEvent.getByToken(patJetsToken_,jets);

  edm::View<pat::Jet> patJets = *jets;
  
  bool jetCollectionIsValid = jets.isValid();


  //check for collections AND DCS filters
  bool dcsDecision = DCSFilterForJetMonitoring_->filter(iEvent, iSetup);

  if (jetCleaningFlag_ && (!jetCollectionIsValid || !bPrimaryVertex || !dcsDecision)) return;

  //now start changes for jets
  std::vector<Jet> corJets;
  corJets.clear();

  //maybe not most elegant solution, but works for sure
  unsigned int ind1=-1;
  double pt1=-1;
  unsigned int ind2=-1;
  double pt2=-1;
  unsigned int ind3=-1;
  double pt3=-1;

  bool cleaned_first_jet=false;
  bool cleaned_second_jet=false;

  //now start changes for jets
  std::vector<Jet> recoJets;
  recoJets.clear();

  int numofjets=0;
  int ijet=0;
  for(edm::View<pat::Jet>::const_iterator i_jet = patJets.begin();i_jet != patJets.end(); ++i_jet) {
    bool Thiscleaned=true;
    bool jetpassid=false;
    if(i_jet->pt()> ptThreshold_){
      recoJets.push_back(*i_jet);
    }

    if(isCaloJet_&& i_jet->isCaloJet() ){
      reco::JetID jetID= i_jet->jetID (); 
      jetpassid = jetIDFunctor((*i_jet));
      if(jetCleaningFlag_){
	Thiscleaned=jetpassid;
      }
      if(i_jet->pt()> ptThreshold_ && Thiscleaned){//if cleaning requested->jet passes a loose ID
	mN90Hits = map_of_MEs[DirName+"/"+"N90Hits"]; if (mN90Hits && mN90Hits->getRootObject()) mN90Hits->Fill (jetID.n90Hits);
	mfHPD = map_of_MEs[DirName+"/"+"fHPD"]; if (mfHPD && mfHPD->getRootObject())             mfHPD->Fill (jetID.fHPD);
	mresEMF = map_of_MEs[DirName+"/"+"resEMF"]; if (mresEMF && mresEMF->getRootObject())     mresEMF->Fill (jetID.restrictedEMF);
	mfRBX = map_of_MEs[DirName+"/"+"fRBX"]; if (mfRBX && mfRBX->getRootObject())             mfRBX->Fill (jetID.fRBX);
	
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject())        mHFrac->Fill (i_jet->energyFractionHadronic());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject())        mEFrac->Fill (i_jet->emEnergyFraction());
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject())        mHFrac_profile       ->Fill(numPV, i_jet->energyFractionHadronic());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject())        mEFrac_profile       ->Fill(numPV, i_jet->emEnergyFraction());
	if (fabs(i_jet->eta()) <= 1.3) {	
	  mHFrac_Barrel = map_of_MEs[DirName+"/"+"HFrac_Barrel"]; if (mHFrac_Barrel && mHFrac_Barrel->getRootObject())           mHFrac_Barrel->Fill(i_jet->energyFractionHadronic());	
	  mEFrac_Barrel = map_of_MEs[DirName+"/"+"EFrac_Barrel"]; if (mEFrac_Barrel && mEFrac_Barrel->getRootObject())           mEFrac_Barrel->Fill(i_jet->emEnergyFraction());	
	}else if(fabs(i_jet->eta()) <3.0){
	  mHFrac_EndCap = map_of_MEs[DirName+"/"+"HFrac_EndCap"]; if (mHFrac_EndCap && mHFrac_EndCap->getRootObject())           mHFrac_EndCap->Fill(i_jet->energyFractionHadronic());	
	  mEFrac_EndCap = map_of_MEs[DirName+"/"+"EFrac_EndCap"]; if (mEFrac_EndCap && mEFrac_EndCap->getRootObject())           mEFrac_EndCap->Fill(i_jet->emEnergyFraction());
	}else{
	  mHFrac_Forward = map_of_MEs[DirName+"/"+"HFrac_Forward"]; if (mHFrac_Forward && mHFrac_Forward->getRootObject())           mHFrac_Forward->Fill(i_jet->energyFractionHadronic());	
	  mEFrac_Forward = map_of_MEs[DirName+"/"+"EFrac_Forward"]; if (mEFrac_Forward && mEFrac_Forward->getRootObject())           mEFrac_Forward->Fill(i_jet->emEnergyFraction());
	}
	mMaxEInEmTowers = map_of_MEs[DirName+"/"+"MaxEInEmTowers"]; if (mMaxEInEmTowers && mMaxEInEmTowers->getRootObject())  mMaxEInEmTowers->Fill (i_jet->maxEInEmTowers());
	mMaxEInHadTowers = map_of_MEs[DirName+"/"+"MaxEInHadTowers"]; if (mMaxEInHadTowers && mMaxEInHadTowers->getRootObject()) mMaxEInHadTowers->Fill (i_jet->maxEInHadTowers());
	
	mHadEnergyInHO = map_of_MEs[DirName+"/"+"HadEnergyInHO"]; if (mHadEnergyInHO && mHadEnergyInHO->getRootObject())   mHadEnergyInHO->Fill (i_jet->hadEnergyInHO());
	mHadEnergyInHB = map_of_MEs[DirName+"/"+"HadEnergyInHB"]; if (mHadEnergyInHB && mHadEnergyInHB->getRootObject())   mHadEnergyInHB->Fill (i_jet->hadEnergyInHB());
	mHadEnergyInHF = map_of_MEs[DirName+"/"+"HadEnergyInHF"]; if (mHadEnergyInHF && mHadEnergyInHF->getRootObject())   mHadEnergyInHF->Fill (i_jet->hadEnergyInHF());
	mHadEnergyInHE = map_of_MEs[DirName+"/"+"HadEnergyInHE"]; if (mHadEnergyInHE && mHadEnergyInHE->getRootObject())   mHadEnergyInHE->Fill (i_jet->hadEnergyInHE());
	mEmEnergyInEB = map_of_MEs[DirName+"/"+"EmEnergyInEB"]; if (mEmEnergyInEB && mEmEnergyInEB->getRootObject())    mEmEnergyInEB->Fill (i_jet->emEnergyInEB());
	mEmEnergyInEE = map_of_MEs[DirName+"/"+"EmEnergyInEE"]; if (mEmEnergyInEE && mEmEnergyInEE->getRootObject())    mEmEnergyInEE->Fill (i_jet->emEnergyInEE());
	mEmEnergyInHF = map_of_MEs[DirName+"/"+"EmEnergyInHF"]; if (mEmEnergyInHF && mEmEnergyInHF->getRootObject())    mEmEnergyInHF->Fill (i_jet->emEnergyInHF());
	
      }	
    }
    if(isPFJet_&& i_jet->isPFJet()){
      jetpassid = pfjetIDFunctor(*i_jet);
      if(jetCleaningFlag_){
	Thiscleaned = jetpassid;
      }
      if(i_jet->pt()> ptThreshold_ && Thiscleaned){//if cleaning requested->jet passes a loose ID
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject()) mHFrac->Fill (i_jet->chargedHadronEnergyFraction()+i_jet->neutralHadronEnergyFraction()+i_jet->HFHadronEnergyFraction ());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject()) mEFrac->Fill (i_jet->chargedEmEnergyFraction() +i_jet->neutralEmEnergyFraction()+i_jet->HFEMEnergyFraction ());
	if (i_jet->pt()<= 50) {
	  mCHFracVSeta_lowPt = map_of_MEs[DirName+"/"+"CHFracVSeta_lowPt"]; if (mCHFracVSeta_lowPt &&  mCHFracVSeta_lowPt->getRootObject()) mCHFracVSeta_lowPt->Fill(i_jet->eta(),i_jet->chargedHadronEnergyFraction());
	  mNHFracVSeta_lowPt = map_of_MEs[DirName+"/"+"NHFracVSeta_lowPt"]; if (mNHFracVSeta_lowPt &&  mNHFracVSeta_lowPt->getRootObject()) mNHFracVSeta_lowPt->Fill(i_jet->eta(),i_jet->neutralHadronEnergyFraction());
	  mPhFracVSeta_lowPt = map_of_MEs[DirName+"/"+"PhFracVSeta_lowPt"]; if (mPhFracVSeta_lowPt &&  mPhFracVSeta_lowPt->getRootObject()) mPhFracVSeta_lowPt->Fill(i_jet->eta(),i_jet->neutralEmEnergyFraction());
	  mElFracVSeta_lowPt = map_of_MEs[DirName+"/"+"ElFracVSeta_lowPt"]; if (mElFracVSeta_lowPt &&  mElFracVSeta_lowPt->getRootObject()) mElFracVSeta_lowPt->Fill(i_jet->eta(),i_jet->chargedEmEnergyFraction());
	  mMuFracVSeta_lowPt = map_of_MEs[DirName+"/"+"MuFracVSeta_lowPt"]; if (mMuFracVSeta_lowPt &&  mMuFracVSeta_lowPt->getRootObject()) mMuFracVSeta_lowPt->Fill(i_jet->eta(),i_jet->chargedMuEnergyFraction());
	}
	if (i_jet->pt()>50. && i_jet->pt()<=140.) {
	  mCHFracVSeta_mediumPt = map_of_MEs[DirName+"/"+"CHFracVSeta_mediumPt"]; if (mCHFracVSeta_mediumPt &&  mCHFracVSeta_mediumPt->getRootObject()) mCHFracVSeta_mediumPt->Fill(i_jet->eta(),i_jet->chargedHadronEnergyFraction());
	  mNHFracVSeta_mediumPt = map_of_MEs[DirName+"/"+"NHFracVSeta_mediumPt"]; if (mNHFracVSeta_mediumPt &&  mNHFracVSeta_mediumPt->getRootObject()) mNHFracVSeta_mediumPt->Fill(i_jet->eta(),i_jet->neutralHadronEnergyFraction());
	  mPhFracVSeta_mediumPt = map_of_MEs[DirName+"/"+"PhFracVSeta_mediumPt"]; if (mPhFracVSeta_mediumPt &&  mPhFracVSeta_mediumPt->getRootObject()) mPhFracVSeta_mediumPt->Fill(i_jet->eta(),i_jet->neutralEmEnergyFraction());
	  mElFracVSeta_mediumPt = map_of_MEs[DirName+"/"+"ElFracVSeta_mediumPt"]; if (mElFracVSeta_mediumPt &&  mElFracVSeta_mediumPt->getRootObject()) mElFracVSeta_mediumPt->Fill(i_jet->eta(),i_jet->chargedEmEnergyFraction());
	  mMuFracVSeta_mediumPt = map_of_MEs[DirName+"/"+"MuFracVSeta_mediumPt"]; if (mMuFracVSeta_mediumPt &&  mMuFracVSeta_mediumPt->getRootObject()) mMuFracVSeta_mediumPt->Fill(i_jet->eta(),i_jet->chargedMuEnergyFraction());
	}
	if (i_jet->pt()>140.) {
	  mCHFracVSeta_highPt = map_of_MEs[DirName+"/"+"CHFracVSeta_highPt"]; if (mCHFracVSeta_highPt &&  mCHFracVSeta_highPt->getRootObject()) mCHFracVSeta_highPt->Fill(i_jet->eta(),i_jet->chargedHadronEnergyFraction());
	  mNHFracVSeta_highPt = map_of_MEs[DirName+"/"+"NHFracVSeta_highPt"]; if (mNHFracVSeta_highPt &&  mNHFracVSeta_highPt->getRootObject()) mNHFracVSeta_highPt->Fill(i_jet->eta(),i_jet->neutralHadronEnergyFraction());
	  mPhFracVSeta_highPt = map_of_MEs[DirName+"/"+"PhFracVSeta_highPt"]; if (mPhFracVSeta_highPt &&  mPhFracVSeta_highPt->getRootObject()) mPhFracVSeta_highPt->Fill(i_jet->eta(),i_jet->neutralEmEnergyFraction());
	  mElFracVSeta_highPt = map_of_MEs[DirName+"/"+"ElFracVSeta_highPt"]; if (mElFracVSeta_highPt &&  mElFracVSeta_highPt->getRootObject()) mElFracVSeta_highPt->Fill(i_jet->eta(),i_jet->chargedEmEnergyFraction());
	  mMuFracVSeta_highPt = map_of_MEs[DirName+"/"+"MuFracVSeta_highPt"]; if (mMuFracVSeta_highPt &&  mMuFracVSeta_highPt->getRootObject()) mMuFracVSeta_highPt->Fill(i_jet->eta(),i_jet->chargedMuEnergyFraction());
	}
	if (fabs(i_jet->eta()) <= 1.3) {
	  mHFrac_Barrel = map_of_MEs[DirName+"/"+"HFrac_Barrel"]; if (mHFrac_Barrel && mHFrac_Barrel->getRootObject())   mHFrac_Barrel->Fill(i_jet->chargedHadronEnergyFraction() + i_jet->neutralHadronEnergyFraction() );
	  mEFrac_Barrel = map_of_MEs[DirName+"/"+"EFrac_Barrel"]; if (mEFrac_Barrel && mEFrac_Barrel->getRootObject())   mEFrac_Barrel->Fill (i_jet->chargedEmEnergyFraction() + i_jet->neutralEmEnergyFraction());
	  //fractions for barrel
	  if (i_jet->pt()<=50.) {
	    mCHFrac_lowPt_Barrel = map_of_MEs[DirName+"/"+"CHFrac_lowPt_Barrel"]; if (mCHFrac_lowPt_Barrel &&  mCHFrac_lowPt_Barrel->getRootObject()) mCHFrac_lowPt_Barrel->Fill(i_jet->chargedHadronEnergyFraction());
	    mNHFrac_lowPt_Barrel = map_of_MEs[DirName+"/"+"NHFrac_lowPt_Barrel"]; if (mNHFrac_lowPt_Barrel &&  mNHFrac_lowPt_Barrel->getRootObject()) mNHFrac_lowPt_Barrel->Fill(i_jet->neutralHadronEnergyFraction());
	    mPhFrac_lowPt_Barrel = map_of_MEs[DirName+"/"+"PhFrac_lowPt_Barrel"]; if (mPhFrac_lowPt_Barrel &&  mPhFrac_lowPt_Barrel->getRootObject()) mPhFrac_lowPt_Barrel->Fill(i_jet->neutralEmEnergyFraction());
	    mElFrac_lowPt_Barrel = map_of_MEs[DirName+"/"+"ElFrac_lowPt_Barrel"]; if (mElFrac_lowPt_Barrel &&  mElFrac_lowPt_Barrel->getRootObject()) mElFrac_lowPt_Barrel->Fill(i_jet->chargedEmEnergyFraction());
	    mMuFrac_lowPt_Barrel = map_of_MEs[DirName+"/"+"MuFrac_lowPt_Barrel"]; if (mMuFrac_lowPt_Barrel &&  mMuFrac_lowPt_Barrel->getRootObject()) mMuFrac_lowPt_Barrel->Fill(i_jet->chargedMuEnergyFraction());
	    mCHEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"CHEn_lowPt_Barrel"]; if (mCHEn_lowPt_Barrel &&  mCHEn_lowPt_Barrel->getRootObject()) mCHEn_lowPt_Barrel->Fill(i_jet->chargedHadronEnergy());
	    mNHEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"NHEn_lowPt_Barrel"]; if (mNHEn_lowPt_Barrel &&  mNHEn_lowPt_Barrel->getRootObject()) mNHEn_lowPt_Barrel->Fill(i_jet->neutralHadronEnergy());
	    mPhEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"PhEn_lowPt_Barrel"]; if (mPhEn_lowPt_Barrel &&  mPhEn_lowPt_Barrel->getRootObject()) mPhEn_lowPt_Barrel->Fill(i_jet->neutralEmEnergy());
	    mElEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"ElEn_lowPt_Barrel"]; if (mElEn_lowPt_Barrel &&  mElEn_lowPt_Barrel->getRootObject()) mElEn_lowPt_Barrel->Fill(i_jet->chargedEmEnergy());
	    mMuEn_lowPt_Barrel = map_of_MEs[DirName+"/"+"MuEn_lowPt_Barrel"]; if (mMuEn_lowPt_Barrel &&  mMuEn_lowPt_Barrel->getRootObject()) mMuEn_lowPt_Barrel->Fill(i_jet->chargedMuEnergy());
	    mChMultiplicity_lowPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_lowPt_Barrel"]; if(mChMultiplicity_lowPt_Barrel && mChMultiplicity_lowPt_Barrel->getRootObject())  mChMultiplicity_lowPt_Barrel->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_lowPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_lowPt_Barrel"]; if(mNeutMultiplicity_lowPt_Barrel && mNeutMultiplicity_lowPt_Barrel->getRootObject())  mNeutMultiplicity_lowPt_Barrel->Fill(i_jet->neutralMultiplicity());
	    mMuMultiplicity_lowPt_Barrel = map_of_MEs[DirName+"/"+"MuMultiplicity_lowPt_Barrel"]; if(mMuMultiplicity_lowPt_Barrel && mMuMultiplicity_lowPt_Barrel->getRootObject())  mMuMultiplicity_lowPt_Barrel->Fill(i_jet->muonMultiplicity());
	  }
	  if (i_jet->pt()>50. && i_jet->pt()<=140.) {
	    mCHFrac_mediumPt_Barrel = map_of_MEs[DirName+"/"+"CHFrac_mediumPt_Barrel"]; if (mCHFrac_mediumPt_Barrel &&  mCHFrac_mediumPt_Barrel->getRootObject()) mCHFrac_mediumPt_Barrel->Fill(i_jet->chargedHadronEnergyFraction());
	    mNHFrac_mediumPt_Barrel = map_of_MEs[DirName+"/"+"NHFrac_mediumPt_Barrel"]; if (mNHFrac_mediumPt_Barrel &&  mNHFrac_mediumPt_Barrel->getRootObject()) mNHFrac_mediumPt_Barrel->Fill(i_jet->neutralHadronEnergyFraction());
	    mPhFrac_mediumPt_Barrel = map_of_MEs[DirName+"/"+"PhFrac_mediumPt_Barrel"]; if (mPhFrac_mediumPt_Barrel &&  mPhFrac_mediumPt_Barrel->getRootObject()) mPhFrac_mediumPt_Barrel->Fill(i_jet->neutralEmEnergyFraction());
	    mElFrac_mediumPt_Barrel = map_of_MEs[DirName+"/"+"ElFrac_mediumPt_Barrel"]; if (mElFrac_mediumPt_Barrel &&  mElFrac_mediumPt_Barrel->getRootObject()) mElFrac_mediumPt_Barrel->Fill(i_jet->chargedEmEnergyFraction());
	    mMuFrac_mediumPt_Barrel = map_of_MEs[DirName+"/"+"MuFrac_mediumPt_Barrel"]; if (mMuFrac_mediumPt_Barrel &&  mMuFrac_mediumPt_Barrel->getRootObject()) mMuFrac_mediumPt_Barrel->Fill(i_jet->chargedMuEnergyFraction());
	    mCHEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"CHEn_mediumPt_Barrel"]; if (mCHEn_mediumPt_Barrel &&  mCHEn_mediumPt_Barrel->getRootObject()) mCHEn_mediumPt_Barrel->Fill(i_jet->chargedHadronEnergy());
	    mNHEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"NHEn_mediumPt_Barrel"]; if (mNHEn_mediumPt_Barrel &&  mNHEn_mediumPt_Barrel->getRootObject()) mNHEn_mediumPt_Barrel->Fill(i_jet->neutralHadronEnergy());
	    mPhEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"PhEn_mediumPt_Barrel"]; if (mPhEn_mediumPt_Barrel &&  mPhEn_mediumPt_Barrel->getRootObject()) mPhEn_mediumPt_Barrel->Fill(i_jet->neutralEmEnergy());
	    mElEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"ElEn_mediumPt_Barrel"]; if (mElEn_mediumPt_Barrel &&  mElEn_mediumPt_Barrel->getRootObject()) mElEn_mediumPt_Barrel->Fill(i_jet->chargedEmEnergy());
	    mMuEn_mediumPt_Barrel = map_of_MEs[DirName+"/"+"MuEn_mediumPt_Barrel"]; if (mMuEn_mediumPt_Barrel &&  mMuEn_mediumPt_Barrel->getRootObject()) mMuEn_mediumPt_Barrel->Fill(i_jet->chargedMuEnergy());
	    mChMultiplicity_mediumPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_mediumPt_Barrel"]; if(mChMultiplicity_mediumPt_Barrel && mChMultiplicity_mediumPt_Barrel->getRootObject())  mChMultiplicity_mediumPt_Barrel->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_mediumPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_mediumPt_Barrel"]; if(mNeutMultiplicity_mediumPt_Barrel && mNeutMultiplicity_mediumPt_Barrel->getRootObject())  mNeutMultiplicity_mediumPt_Barrel->Fill(i_jet->neutralMultiplicity());
	    mMuMultiplicity_mediumPt_Barrel = map_of_MEs[DirName+"/"+"MuMultiplicity_mediumPt_Barrel"]; if(mMuMultiplicity_mediumPt_Barrel && mMuMultiplicity_mediumPt_Barrel->getRootObject())  mMuMultiplicity_mediumPt_Barrel->Fill(i_jet->muonMultiplicity());
	  }
	  if (i_jet->pt()>140.) {
	    mCHFrac_highPt_Barrel = map_of_MEs[DirName+"/"+"CHFrac_highPt_Barrel"]; if (mCHFrac_highPt_Barrel &&  mCHFrac_highPt_Barrel->getRootObject()) mCHFrac_highPt_Barrel->Fill(i_jet->chargedHadronEnergyFraction());
	    mNHFrac_highPt_Barrel = map_of_MEs[DirName+"/"+"NHFrac_highPt_Barrel"]; if (mNHFrac_highPt_Barrel &&  mNHFrac_highPt_Barrel->getRootObject()) mNHFrac_highPt_Barrel->Fill(i_jet->neutralHadronEnergyFraction());
	    mPhFrac_highPt_Barrel = map_of_MEs[DirName+"/"+"PhFrac_highPt_Barrel"]; if (mPhFrac_highPt_Barrel &&  mPhFrac_highPt_Barrel->getRootObject()) mPhFrac_highPt_Barrel->Fill(i_jet->neutralEmEnergyFraction());
	    mElFrac_highPt_Barrel = map_of_MEs[DirName+"/"+"ElFrac_highPt_Barrel"]; if (mElFrac_highPt_Barrel &&  mElFrac_highPt_Barrel->getRootObject()) mElFrac_highPt_Barrel->Fill(i_jet->chargedEmEnergyFraction());
	    mMuFrac_highPt_Barrel = map_of_MEs[DirName+"/"+"MuFrac_highPt_Barrel"]; if (mMuFrac_highPt_Barrel &&  mMuFrac_highPt_Barrel->getRootObject()) mMuFrac_highPt_Barrel->Fill(i_jet->chargedMuEnergyFraction());
	    mCHEn_highPt_Barrel = map_of_MEs[DirName+"/"+"CHEn_highPt_Barrel"]; if (mCHEn_highPt_Barrel &&  mCHEn_highPt_Barrel->getRootObject()) mCHEn_highPt_Barrel->Fill(i_jet->chargedHadronEnergy());
	    mNHEn_highPt_Barrel = map_of_MEs[DirName+"/"+"NHEn_highPt_Barrel"]; if (mNHEn_highPt_Barrel &&  mNHEn_highPt_Barrel->getRootObject()) mNHEn_highPt_Barrel->Fill(i_jet->neutralHadronEnergy());
	    mPhEn_highPt_Barrel = map_of_MEs[DirName+"/"+"PhEn_highPt_Barrel"]; if (mPhEn_highPt_Barrel &&  mPhEn_highPt_Barrel->getRootObject()) mPhEn_highPt_Barrel->Fill(i_jet->neutralEmEnergy());
	    mElEn_highPt_Barrel = map_of_MEs[DirName+"/"+"ElEn_highPt_Barrel"]; if (mElEn_highPt_Barrel &&  mElEn_highPt_Barrel->getRootObject()) mElEn_highPt_Barrel->Fill(i_jet->chargedEmEnergy());
	    mMuEn_highPt_Barrel = map_of_MEs[DirName+"/"+"MuEn_highPt_Barrel"]; if (mMuEn_highPt_Barrel &&  mMuEn_highPt_Barrel->getRootObject()) mMuEn_highPt_Barrel->Fill(i_jet->chargedMuEnergy());
	    mChMultiplicity_highPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_highPt_Barrel"]; if(mChMultiplicity_highPt_Barrel && mChMultiplicity_highPt_Barrel->getRootObject())  mChMultiplicity_highPt_Barrel->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_highPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_highPt_Barrel"]; if(mNeutMultiplicity_highPt_Barrel && mNeutMultiplicity_highPt_Barrel->getRootObject())  mNeutMultiplicity_highPt_Barrel->Fill(i_jet->neutralMultiplicity());
	    mMuMultiplicity_highPt_Barrel = map_of_MEs[DirName+"/"+"MuMultiplicity_highPt_Barrel"]; if(mMuMultiplicity_highPt_Barrel && mMuMultiplicity_highPt_Barrel->getRootObject())  mMuMultiplicity_highPt_Barrel->Fill(i_jet->muonMultiplicity());
	  }
	  mCHFracVSpT_Barrel = map_of_MEs[DirName+"/"+"CHFracVSpT_Barrel"]; if(mCHFracVSpT_Barrel && mCHFracVSpT_Barrel->getRootObject()) mCHFracVSpT_Barrel->Fill(i_jet->pt(),i_jet->chargedHadronEnergyFraction());
	  mNHFracVSpT_Barrel = map_of_MEs[DirName+"/"+"NHFracVSpT_Barrel"];if (mNHFracVSpT_Barrel && mNHFracVSpT_Barrel->getRootObject()) mNHFracVSpT_Barrel->Fill(i_jet->pt(),i_jet->neutralHadronEnergyFraction());
	  mPhFracVSpT_Barrel = map_of_MEs[DirName+"/"+"PhFracVSpT_Barrel"];if (mPhFracVSpT_Barrel && mPhFracVSpT_Barrel->getRootObject()) mPhFracVSpT_Barrel->Fill(i_jet->pt(),i_jet->neutralEmEnergyFraction());
	  mElFracVSpT_Barrel = map_of_MEs[DirName+"/"+"ElFracVSpT_Barrel"];if (mElFracVSpT_Barrel && mElFracVSpT_Barrel->getRootObject()) mElFracVSpT_Barrel->Fill(i_jet->pt(),i_jet->chargedEmEnergyFraction());
	  mMuFracVSpT_Barrel = map_of_MEs[DirName+"/"+"MuFracVSpT_Barrel"];if (mMuFracVSpT_Barrel && mMuFracVSpT_Barrel->getRootObject()) mMuFracVSpT_Barrel->Fill(i_jet->pt(),i_jet->chargedMuEnergyFraction());
	}else if(fabs(i_jet->eta()) <= 3) {
	  mHFrac_EndCap = map_of_MEs[DirName+"/"+"HFrac_EndCap"]; if (mHFrac_EndCap && mHFrac_EndCap->getRootObject())   mHFrac_EndCap->Fill(i_jet->chargedHadronEnergyFraction() + i_jet->neutralHadronEnergyFraction()+i_jet->HFHadronEnergyFraction ());
	  mEFrac_EndCap = map_of_MEs[DirName+"/"+"EFrac_EndCap"]; if (mEFrac_EndCap && mEFrac_EndCap->getRootObject())    mEFrac_EndCap->Fill (i_jet->chargedEmEnergyFraction() + i_jet->neutralEmEnergyFraction()+i_jet->HFEMEnergyFraction ());
	  //fractions for endcap
	  if (i_jet->pt()<=50.) {
	    mCHFrac_lowPt_EndCap = map_of_MEs[DirName+"/"+"CHFrac_lowPt_EndCap"]; if (mCHFrac_lowPt_EndCap &&  mCHFrac_lowPt_EndCap->getRootObject()) mCHFrac_lowPt_EndCap->Fill(i_jet->chargedHadronEnergyFraction());
	    mNHFrac_lowPt_EndCap = map_of_MEs[DirName+"/"+"NHFrac_lowPt_EndCap"]; if (mNHFrac_lowPt_EndCap &&  mNHFrac_lowPt_EndCap->getRootObject()) mNHFrac_lowPt_EndCap->Fill(i_jet->neutralHadronEnergyFraction());
	    mPhFrac_lowPt_EndCap = map_of_MEs[DirName+"/"+"PhFrac_lowPt_EndCap"]; if (mPhFrac_lowPt_EndCap &&  mPhFrac_lowPt_EndCap->getRootObject()) mPhFrac_lowPt_EndCap->Fill(i_jet->neutralEmEnergyFraction());
	    mElFrac_lowPt_EndCap = map_of_MEs[DirName+"/"+"ElFrac_lowPt_EndCap"]; if (mElFrac_lowPt_EndCap &&  mElFrac_lowPt_EndCap->getRootObject()) mElFrac_lowPt_EndCap->Fill(i_jet->chargedEmEnergyFraction());
	    mMuFrac_lowPt_EndCap = map_of_MEs[DirName+"/"+"MuFrac_lowPt_EndCap"]; if (mMuFrac_lowPt_EndCap &&  mMuFrac_lowPt_EndCap->getRootObject()) mMuFrac_lowPt_EndCap->Fill(i_jet->chargedMuEnergyFraction());
	    mCHEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"CHEn_lowPt_EndCap"]; if (mCHEn_lowPt_EndCap &&  mCHEn_lowPt_EndCap->getRootObject()) mCHEn_lowPt_EndCap->Fill(i_jet->chargedHadronEnergy());
	    mNHEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"NHEn_lowPt_EndCap"]; if (mNHEn_lowPt_EndCap &&  mNHEn_lowPt_EndCap->getRootObject()) mNHEn_lowPt_EndCap->Fill(i_jet->neutralHadronEnergy());
	    mPhEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"PhEn_lowPt_EndCap"]; if (mPhEn_lowPt_EndCap &&  mPhEn_lowPt_EndCap->getRootObject()) mPhEn_lowPt_EndCap->Fill(i_jet->neutralEmEnergy());
	    mElEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"ElEn_lowPt_EndCap"]; if (mElEn_lowPt_EndCap &&  mElEn_lowPt_EndCap->getRootObject()) mElEn_lowPt_EndCap->Fill(i_jet->chargedEmEnergy());
	    mMuEn_lowPt_EndCap = map_of_MEs[DirName+"/"+"MuEn_lowPt_EndCap"]; if (mMuEn_lowPt_EndCap &&  mMuEn_lowPt_EndCap->getRootObject()) mMuEn_lowPt_EndCap->Fill(i_jet->chargedMuEnergy());
	    mChMultiplicity_lowPt_EndCap = map_of_MEs[DirName+"/"+"ChMultiplicity_lowPt_EndCap"]; if(mChMultiplicity_lowPt_EndCap && mChMultiplicity_lowPt_EndCap->getRootObject())  mChMultiplicity_lowPt_EndCap->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_lowPt_EndCap = map_of_MEs[DirName+"/"+"NeutMultiplicity_lowPt_EndCap"]; if(mNeutMultiplicity_lowPt_EndCap && mNeutMultiplicity_lowPt_EndCap->getRootObject())  mNeutMultiplicity_lowPt_EndCap->Fill(i_jet->neutralMultiplicity());
	    mMuMultiplicity_lowPt_EndCap = map_of_MEs[DirName+"/"+"MuMultiplicity_lowPt_EndCap"]; if(mMuMultiplicity_lowPt_EndCap && mMuMultiplicity_lowPt_EndCap->getRootObject())  mMuMultiplicity_lowPt_EndCap->Fill(i_jet->muonMultiplicity());
	  }
	  if (i_jet->pt()>50. && i_jet->pt()<=140.) {
	    mCHFrac_mediumPt_EndCap = map_of_MEs[DirName+"/"+"CHFrac_mediumPt_EndCap"]; if (mCHFrac_mediumPt_EndCap &&  mCHFrac_mediumPt_EndCap->getRootObject()) mCHFrac_mediumPt_EndCap->Fill(i_jet->chargedHadronEnergyFraction());
	    mNHFrac_mediumPt_EndCap = map_of_MEs[DirName+"/"+"NHFrac_mediumPt_EndCap"]; if (mNHFrac_mediumPt_EndCap &&  mNHFrac_mediumPt_EndCap->getRootObject()) mNHFrac_mediumPt_EndCap->Fill(i_jet->neutralHadronEnergyFraction());
	    mPhFrac_mediumPt_EndCap = map_of_MEs[DirName+"/"+"PhFrac_mediumPt_EndCap"]; if (mPhFrac_mediumPt_EndCap &&  mPhFrac_mediumPt_EndCap->getRootObject()) mPhFrac_mediumPt_EndCap->Fill(i_jet->neutralEmEnergyFraction());
	    mElFrac_mediumPt_EndCap = map_of_MEs[DirName+"/"+"ElFrac_mediumPt_EndCap"]; if (mElFrac_mediumPt_EndCap &&  mElFrac_mediumPt_EndCap->getRootObject()) mElFrac_mediumPt_EndCap->Fill(i_jet->chargedEmEnergyFraction());
	    mMuFrac_mediumPt_EndCap = map_of_MEs[DirName+"/"+"MuFrac_mediumPt_EndCap"]; if (mMuFrac_mediumPt_EndCap &&  mMuFrac_mediumPt_EndCap->getRootObject()) mMuFrac_mediumPt_EndCap->Fill(i_jet->chargedMuEnergyFraction());
	    mCHEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"CHEn_mediumPt_EndCap"]; if (mCHEn_mediumPt_EndCap &&  mCHEn_mediumPt_EndCap->getRootObject()) mCHEn_mediumPt_EndCap->Fill(i_jet->chargedHadronEnergy());
	    mNHEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"NHEn_mediumPt_EndCap"]; if (mNHEn_mediumPt_EndCap &&  mNHEn_mediumPt_EndCap->getRootObject()) mNHEn_mediumPt_EndCap->Fill(i_jet->neutralHadronEnergy());
	    mPhEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"PhEn_mediumPt_EndCap"]; if (mPhEn_mediumPt_EndCap &&  mPhEn_mediumPt_EndCap->getRootObject()) mPhEn_mediumPt_EndCap->Fill(i_jet->neutralEmEnergy());
	    mElEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"ElEn_mediumPt_EndCap"]; if (mElEn_mediumPt_EndCap &&  mElEn_mediumPt_EndCap->getRootObject()) mElEn_mediumPt_EndCap->Fill(i_jet->chargedEmEnergy());
	    mMuEn_mediumPt_EndCap = map_of_MEs[DirName+"/"+"MuEn_mediumPt_EndCap"]; if (mMuEn_mediumPt_EndCap &&  mMuEn_mediumPt_EndCap->getRootObject()) mMuEn_mediumPt_EndCap->Fill(i_jet->chargedMuEnergy());
	    mChMultiplicity_mediumPt_EndCap = map_of_MEs[DirName+"/"+"ChMultiplicity_mediumPt_EndCap"]; if(mChMultiplicity_mediumPt_EndCap && mChMultiplicity_mediumPt_EndCap->getRootObject())  mChMultiplicity_mediumPt_EndCap->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_mediumPt_EndCap = map_of_MEs[DirName+"/"+"NeutMultiplicity_mediumPt_EndCap"]; if(mNeutMultiplicity_mediumPt_EndCap && mNeutMultiplicity_mediumPt_EndCap->getRootObject())  mNeutMultiplicity_mediumPt_EndCap->Fill(i_jet->neutralMultiplicity());
	    mMuMultiplicity_mediumPt_EndCap = map_of_MEs[DirName+"/"+"MuMultiplicity_mediumPt_EndCap"]; if(mMuMultiplicity_mediumPt_EndCap && mMuMultiplicity_mediumPt_EndCap->getRootObject())  mMuMultiplicity_mediumPt_EndCap->Fill(i_jet->muonMultiplicity());
	  }
	  if (i_jet->pt()>140.) {
	    mCHFrac_highPt_EndCap = map_of_MEs[DirName+"/"+"CHFrac_highPt_EndCap"]; if (mCHFrac_highPt_EndCap &&  mCHFrac_highPt_EndCap->getRootObject()) mCHFrac_highPt_EndCap->Fill(i_jet->chargedHadronEnergyFraction());
	    mNHFrac_highPt_EndCap = map_of_MEs[DirName+"/"+"NHFrac_highPt_EndCap"]; if (mNHFrac_highPt_EndCap &&  mNHFrac_highPt_EndCap->getRootObject()) mNHFrac_highPt_EndCap->Fill(i_jet->neutralHadronEnergyFraction());
	    mPhFrac_highPt_EndCap = map_of_MEs[DirName+"/"+"PhFrac_highPt_EndCap"]; if (mPhFrac_highPt_EndCap &&  mPhFrac_highPt_EndCap->getRootObject()) mPhFrac_highPt_EndCap->Fill(i_jet->neutralEmEnergyFraction());
	    mElFrac_highPt_EndCap = map_of_MEs[DirName+"/"+"ElFrac_highPt_EndCap"]; if (mElFrac_highPt_EndCap &&  mElFrac_highPt_EndCap->getRootObject()) mElFrac_highPt_EndCap->Fill(i_jet->chargedEmEnergyFraction());
	    mMuFrac_highPt_EndCap = map_of_MEs[DirName+"/"+"MuFrac_highPt_EndCap"]; if (mMuFrac_highPt_EndCap &&  mMuFrac_highPt_EndCap->getRootObject()) mMuFrac_highPt_EndCap->Fill(i_jet->chargedMuEnergyFraction());
	    mCHEn_highPt_EndCap = map_of_MEs[DirName+"/"+"CHEn_highPt_EndCap"]; if (mCHEn_highPt_EndCap &&  mCHEn_highPt_EndCap->getRootObject()) mCHEn_highPt_EndCap->Fill(i_jet->chargedHadronEnergy());
	    mNHEn_highPt_EndCap = map_of_MEs[DirName+"/"+"NHEn_highPt_EndCap"]; if (mNHEn_highPt_EndCap &&  mNHEn_highPt_EndCap->getRootObject()) mNHEn_highPt_EndCap->Fill(i_jet->neutralHadronEnergy());
	    mPhEn_highPt_EndCap = map_of_MEs[DirName+"/"+"PhEn_highPt_EndCap"]; if (mPhEn_highPt_EndCap &&  mPhEn_highPt_EndCap->getRootObject()) mPhEn_highPt_EndCap->Fill(i_jet->neutralEmEnergy());
	    mElEn_highPt_EndCap = map_of_MEs[DirName+"/"+"ElEn_highPt_EndCap"]; if (mElEn_highPt_EndCap &&  mElEn_highPt_EndCap->getRootObject()) mElEn_highPt_EndCap->Fill(i_jet->chargedEmEnergy());
	    mMuEn_highPt_EndCap = map_of_MEs[DirName+"/"+"MuEn_highPt_EndCap"]; if (mMuEn_highPt_EndCap &&  mMuEn_highPt_EndCap->getRootObject()) mMuEn_highPt_EndCap->Fill(i_jet->chargedMuEnergy());
	    mChMultiplicity_highPt_EndCap = map_of_MEs[DirName+"/"+"ChMultiplicity_highPt_EndCap"]; if(mChMultiplicity_highPt_EndCap && mChMultiplicity_highPt_EndCap->getRootObject())  mChMultiplicity_highPt_EndCap->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_highPt_EndCap = map_of_MEs[DirName+"/"+"NeutMultiplicity_highPt_EndCap"]; if(mNeutMultiplicity_highPt_EndCap && mNeutMultiplicity_highPt_EndCap->getRootObject())  mNeutMultiplicity_highPt_EndCap->Fill(i_jet->neutralMultiplicity());
	    mMuMultiplicity_highPt_EndCap = map_of_MEs[DirName+"/"+"MuMultiplicity_highPt_EndCap"]; if(mMuMultiplicity_highPt_EndCap && mMuMultiplicity_highPt_EndCap->getRootObject())  mMuMultiplicity_highPt_EndCap->Fill(i_jet->muonMultiplicity());
	  }
	  mCHFracVSpT_EndCap = map_of_MEs[DirName+"/"+"CHFracVSpT_EndCap"]; if(mCHFracVSpT_EndCap && mCHFracVSpT_EndCap->getRootObject()) mCHFracVSpT_EndCap->Fill(i_jet->pt(),i_jet->chargedHadronEnergyFraction());
	  mNHFracVSpT_EndCap = map_of_MEs[DirName+"/"+"NHFracVSpT_EndCap"];if (mNHFracVSpT_EndCap && mNHFracVSpT_EndCap->getRootObject()) mNHFracVSpT_EndCap->Fill(i_jet->pt(),i_jet->neutralHadronEnergyFraction());
	  mPhFracVSpT_EndCap = map_of_MEs[DirName+"/"+"PhFracVSpT_EndCap"];if (mPhFracVSpT_EndCap && mPhFracVSpT_EndCap->getRootObject()) mPhFracVSpT_EndCap->Fill(i_jet->pt(),i_jet->neutralEmEnergyFraction());
	  mElFracVSpT_EndCap = map_of_MEs[DirName+"/"+"ElFracVSpT_EndCap"];if (mElFracVSpT_EndCap && mElFracVSpT_EndCap->getRootObject()) mElFracVSpT_EndCap->Fill(i_jet->pt(),i_jet->chargedEmEnergyFraction());
	  mMuFracVSpT_EndCap = map_of_MEs[DirName+"/"+"MuFracVSpT_EndCap"];if (mMuFracVSpT_EndCap && mMuFracVSpT_EndCap->getRootObject()) mMuFracVSpT_EndCap->Fill(i_jet->pt(),i_jet->chargedMuEnergyFraction());
	}else{
	  mHFrac_Forward = map_of_MEs[DirName+"/"+"HFrac_Forward"]; if (mHFrac_Forward && mHFrac_Forward->getRootObject())    mHFrac_Forward->Fill(i_jet->chargedHadronEnergyFraction() + i_jet->neutralHadronEnergyFraction()+i_jet->HFHadronEnergyFraction ());	
	  mEFrac_Forward = map_of_MEs[DirName+"/"+"EFrac_Forward"]; if (mEFrac_Forward && mEFrac_Forward->getRootObject()) mEFrac_Forward->Fill (i_jet->chargedEmEnergyFraction() + i_jet->neutralEmEnergyFraction()+i_jet->HFEMEnergyFraction ());
	  mHFHFracVSpT_Forward = map_of_MEs[DirName+"/"+"HFHFracVSpT_Forward"]; if (mHFHFracVSpT_Forward && mHFHFracVSpT_Forward->getRootObject())    mHFHFracVSpT_Forward->Fill(i_jet->pt(),i_jet->chargedHadronEnergyFraction() + i_jet->neutralHadronEnergyFraction()+i_jet->HFHadronEnergyFraction ());	
	  mHFEFracVSpT_Forward = map_of_MEs[DirName+"/"+"HFEFracVSpT_Forward"]; if (mHFEFracVSpT_Forward && mHFEFracVSpT_Forward->getRootObject())    mHFEFracVSpT_Forward->Fill (i_jet->pt(),i_jet->chargedEmEnergyFraction() + i_jet->neutralEmEnergyFraction()+i_jet->HFEMEnergyFraction ());
	  //fractions
	  if (i_jet->pt()<=50.) {
	    mHFEFrac_lowPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_lowPt_Forward"]; if(mHFEFrac_lowPt_Forward && mHFEFrac_lowPt_Forward->getRootObject()) mHFEFrac_lowPt_Forward->Fill(i_jet->HFEMEnergyFraction());
	    mHFEFrac_lowPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_lowPt_Forward"]; if(mHFHFrac_lowPt_Forward && mHFHFrac_lowPt_Forward->getRootObject()) mHFHFrac_lowPt_Forward->Fill(i_jet->HFHadronEnergyFraction());
	    mHFEEn_lowPt_Forward = map_of_MEs[DirName+"/"+"HFEEn_lowPt_Forward"];     if(mHFEEn_lowPt_Forward && mHFEEn_lowPt_Forward->getRootObject())     mHFEEn_lowPt_Forward->Fill(i_jet->HFEMEnergy());
	    mHFHEn_lowPt_Forward = map_of_MEs[DirName+"/"+"HFHEn_lowPt_Forward"];    if(mHFHEn_lowPt_Forward && mHFHEn_lowPt_Forward->getRootObject())     mHFHEn_lowPt_Forward->Fill(i_jet->HFHadronEnergy());
	    mChMultiplicity_lowPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_lowPt_Barrel"]; if(mChMultiplicity_lowPt_Forward && mChMultiplicity_lowPt_Forward->getRootObject())  mChMultiplicity_lowPt_Forward->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_lowPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_lowPt_Barrel"]; if(mNeutMultiplicity_lowPt_Forward && mNeutMultiplicity_lowPt_Forward->getRootObject())  mNeutMultiplicity_lowPt_Forward->Fill(i_jet->neutralMultiplicity());
	  }
	  if (i_jet->pt()>50. && i_jet->pt()<=140.) {
	    mHFEFrac_mediumPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_mediumPt_Forward"]; if(mHFEFrac_mediumPt_Forward && mHFEFrac_mediumPt_Forward->getRootObject()) mHFEFrac_mediumPt_Forward->Fill(i_jet->HFEMEnergyFraction());
	    mHFEFrac_mediumPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_mediumPt_Forward"]; if(mHFHFrac_mediumPt_Forward && mHFHFrac_mediumPt_Forward->getRootObject()) mHFHFrac_mediumPt_Forward->Fill(i_jet->HFHadronEnergyFraction());
	    mHFEEn_mediumPt_Forward = map_of_MEs[DirName+"/"+"HFEEn_mediumPt_Forward"];     if(mHFEEn_mediumPt_Forward && mHFEEn_mediumPt_Forward->getRootObject())     mHFEEn_mediumPt_Forward->Fill(i_jet->HFEMEnergy());
	    mHFHEn_mediumPt_Forward = map_of_MEs[DirName+"/"+"HFHEn_mediumPt_Forward"];    if(mHFHEn_mediumPt_Forward && mHFHEn_mediumPt_Forward->getRootObject())     mHFHEn_mediumPt_Forward->Fill(i_jet->HFHadronEnergy());
	    mChMultiplicity_mediumPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_mediumPt_Barrel"]; if(mChMultiplicity_mediumPt_Forward && mChMultiplicity_mediumPt_Forward->getRootObject())  mChMultiplicity_mediumPt_Forward->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_mediumPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_mediumPt_Barrel"]; if(mNeutMultiplicity_mediumPt_Forward && mNeutMultiplicity_mediumPt_Forward->getRootObject())  mNeutMultiplicity_mediumPt_Forward->Fill(i_jet->neutralMultiplicity());
	  }
	  if (i_jet->pt()>140.) {
	    mHFEFrac_highPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_highPt_Forward"]; if(mHFEFrac_highPt_Forward && mHFEFrac_highPt_Forward->getRootObject()) mHFEFrac_highPt_Forward->Fill(i_jet->HFEMEnergyFraction());
	    mHFEFrac_highPt_Forward = map_of_MEs[DirName+"/"+"HFEFrac_highPt_Forward"]; if(mHFHFrac_highPt_Forward && mHFHFrac_highPt_Forward->getRootObject()) mHFHFrac_highPt_Forward->Fill(i_jet->HFHadronEnergyFraction());
	    mHFEEn_highPt_Forward = map_of_MEs[DirName+"/"+"HFEEn_highPt_Forward"];     if(mHFEEn_highPt_Forward && mHFEEn_highPt_Forward->getRootObject())     mHFEEn_highPt_Forward->Fill(i_jet->HFEMEnergy());
	    mHFHEn_highPt_Forward = map_of_MEs[DirName+"/"+"HFHEn_highPt_Forward"];    if(mHFHEn_highPt_Forward && mHFHEn_highPt_Forward->getRootObject())     mHFHEn_highPt_Forward->Fill(i_jet->HFHadronEnergy());
	    mChMultiplicity_highPt_Barrel = map_of_MEs[DirName+"/"+"ChMultiplicity_highPt_Barrel"]; if(mChMultiplicity_highPt_Forward && mChMultiplicity_highPt_Forward->getRootObject())  mChMultiplicity_highPt_Forward->Fill(i_jet->chargedMultiplicity());
	    mNeutMultiplicity_highPt_Barrel = map_of_MEs[DirName+"/"+"NeutMultiplicity_highPt_Barrel"]; if(mNeutMultiplicity_highPt_Forward && mNeutMultiplicity_highPt_Forward->getRootObject())  mNeutMultiplicity_highPt_Forward->Fill(i_jet->neutralMultiplicity());
	  }
	}
	mChargedHadronEnergy = map_of_MEs[DirName+"/"+"ChargedHadronEnergy"]; if (mChargedHadronEnergy && mChargedHadronEnergy->getRootObject())  mChargedHadronEnergy->Fill (i_jet->chargedHadronEnergy());
	mNeutralHadronEnergy = map_of_MEs[DirName+"/"+"NeutralHadronEnergy"]; if (mNeutralHadronEnergy && mNeutralHadronEnergy->getRootObject())  mNeutralHadronEnergy->Fill (i_jet->neutralHadronEnergy());
	mChargedEmEnergy = map_of_MEs[DirName+"/"+"ChargedEmEnergy"]; if (mChargedEmEnergy && mChargedEmEnergy->getRootObject()) mChargedEmEnergy->Fill(i_jet->chargedEmEnergy());
	mChargedMuEnergy = map_of_MEs[DirName+"/"+"ChargedMuEnergy"]; if (mChargedMuEnergy && mChargedMuEnergy->getRootObject()) mChargedMuEnergy->Fill (i_jet->chargedMuEnergy ());
	mNeutralEmEnergy = map_of_MEs[DirName+"/"+"NeutralEmEnergy"]; if (mNeutralEmEnergy && mNeutralEmEnergy->getRootObject()) mNeutralEmEnergy->Fill(i_jet->neutralEmEnergy());
	mChargedMultiplicity = map_of_MEs[DirName+"/"+"ChargedMultiplicity"]; if (mChargedMultiplicity && mChargedMultiplicity->getRootObject()) mChargedMultiplicity->Fill(i_jet->chargedMultiplicity());
	mNeutralMultiplicity = map_of_MEs[DirName+"/"+"NeutralMultiplicity"]; if (mNeutralMultiplicity && mNeutralMultiplicity->getRootObject()) mNeutralMultiplicity->Fill(i_jet->neutralMultiplicity());
	mMuonMultiplicity = map_of_MEs[DirName+"/"+"MuonMultiplicity"]; if (mMuonMultiplicity && mMuonMultiplicity->getRootObject()) mMuonMultiplicity->Fill (i_jet-> muonMultiplicity());
	//_______________________________________________________
	mNeutralFraction = map_of_MEs[DirName+"/"+"NeutralConstituentsFraction"];if (mNeutralFraction && mNeutralFraction->getRootObject()) mNeutralFraction->Fill ((double)i_jet->neutralMultiplicity()/(double)i_jet->nConstituents());
	
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject()) mHFrac_profile       ->Fill(numPV, i_jet->chargedHadronEnergyFraction() + i_jet->neutralHadronEnergyFraction());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject()) mEFrac_profile       ->Fill(numPV, i_jet->chargedEmEnergyFraction()     + i_jet->neutralEmEnergyFraction());
	
	mChargedHadronEnergy_profile = map_of_MEs[DirName+"/"+"ChargedHadronEnergy_profile"]; if (mChargedHadronEnergy_profile && mChargedHadronEnergy_profile->getRootObject()) mChargedHadronEnergy_profile->Fill(numPV, i_jet->chargedHadronEnergy());
	mNeutralHadronEnergy_profile = map_of_MEs[DirName+"/"+"NeutralHadronEnergy_profile"];if (mNeutralHadronEnergy_profile && mNeutralHadronEnergy_profile->getRootObject()) mNeutralHadronEnergy_profile->Fill(numPV, i_jet->neutralHadronEnergy());
	mChargedEmEnergy_profile = map_of_MEs[DirName+"/"+"ChargedEmEnergy_profile"]; if (mChargedEmEnergy_profile && mChargedEmEnergy_profile->getRootObject())     mChargedEmEnergy_profile    ->Fill(numPV, i_jet->chargedEmEnergy());
	mChargedMuEnergy_profile = map_of_MEs[DirName+"/"+"ChargedMuEnergy_profile"];if (mChargedMuEnergy_profile && mChargedMuEnergy_profile->getRootObject())     mChargedMuEnergy_profile    ->Fill(numPV, i_jet->chargedMuEnergy ());
	mNeutralEmEnergy_profile = map_of_MEs[DirName+"/"+"NeutralEmEnergy_profile"];if (mNeutralEmEnergy_profile && mNeutralEmEnergy_profile->getRootObject())     mNeutralEmEnergy_profile    ->Fill(numPV, i_jet->neutralEmEnergy());
	mChargedMultiplicity_profile = map_of_MEs[DirName+"/"+"ChargedMultiplicity_profile"]; if (mChargedMultiplicity_profile && mChargedMultiplicity_profile->getRootObject()) mChargedMultiplicity_profile->Fill(numPV, i_jet->chargedMultiplicity());
	mNeutralMultiplicity_profile = map_of_MEs[DirName+"/"+"NeutralMultiplicity_profile"];if (mNeutralMultiplicity_profile && mNeutralMultiplicity_profile->getRootObject()) mNeutralMultiplicity_profile->Fill(numPV, i_jet->neutralMultiplicity());
	mMuonMultiplicity_profile = map_of_MEs[DirName+"/"+"MuonMultiplicity_profile"]; if (mMuonMultiplicity_profile && mMuonMultiplicity_profile->getRootObject())    mMuonMultiplicity_profile   ->Fill(numPV, i_jet->muonMultiplicity());
	
      }
    }
    //now for all jets, check if we have each type separated
    if( (isCaloJet_&& i_jet->isCaloJet()) || (isPFJet_&& i_jet->isPFJet())){
      if(i_jet->pt()*i_jet->jecFactor(0)> ptThresholdUnc_ && Thiscleaned){
	mPt_uncor = map_of_MEs[DirName+"/"+"Pt_uncor"]; if (mPt_uncor && mPt_uncor->getRootObject())   mPt_uncor->Fill (i_jet->pt());
	mEta_uncor = map_of_MEs[DirName+"/"+"Eta_uncor"]; if (mEta_uncor && mEta_uncor->getRootObject()) mEta_uncor->Fill (i_jet->eta());
	mPhi_uncor = map_of_MEs[DirName+"/"+"Phi_uncor"]; if (mPhi_uncor && mPhi_uncor->getRootObject()) mPhi_uncor->Fill (i_jet->phi());
	 //if(!isJPTJet_){
	mConstituents_uncor = map_of_MEs[DirName+"/"+"Constituents_uncor"]; if (mConstituents_uncor && mConstituents_uncor->getRootObject()) mConstituents_uncor->Fill (i_jet->nConstituents());
      }
      if(i_jet->pt()> ptThreshold_ && Thiscleaned){
	if(i_jet->pt()>pt1){
	  pt2=pt1;
	  ind2=ind1;
	  cleaned_second_jet=cleaned_first_jet;
	  pt1=i_jet->pt();
	  ind1=ijet;
	  cleaned_first_jet=jetpassid;
	} else if(i_jet->pt()>pt2){
	  pt2=i_jet->pt();
	  ind2=ijet;
	  cleaned_second_jet=jetpassid;
	} 
	//after jettype specific variables are filled -> perform histograms for all jets
	//fill JetID efficiencies if uncleaned selection is chosen
	if(!jetCleaningFlag_){
	  if(jetpassid) {
	    mLooseJIDPassFractionVSeta = map_of_MEs[DirName+"/"+"JetIDPassFractionVSeta"]; if (mLooseJIDPassFractionVSeta && mLooseJIDPassFractionVSeta->getRootObject())  mLooseJIDPassFractionVSeta->Fill(i_jet->eta(),1.);
	    mLooseJIDPassFractionVSpt = map_of_MEs[DirName+"/"+"JetIDPassFractionVSpt"]; if (mLooseJIDPassFractionVSpt && mLooseJIDPassFractionVSpt->getRootObject()) mLooseJIDPassFractionVSpt->Fill(i_jet->pt(),1.);
	    if(i_jet->eta()<3.0){
	      mLooseJIDPassFractionVSptNoHF= map_of_MEs[DirName+"/"+"JetIDPassFractionVSptNoHF"]; if (mLooseJIDPassFractionVSptNoHF && mLooseJIDPassFractionVSptNoHF->getRootObject()) mLooseJIDPassFractionVSptNoHF->Fill(i_jet->pt(),1.);
	    }
	  } else {
	    mLooseJIDPassFractionVSeta = map_of_MEs[DirName+"/"+"JetIDPassFractionVSeta"]; if (mLooseJIDPassFractionVSeta && mLooseJIDPassFractionVSeta->getRootObject()) mLooseJIDPassFractionVSeta->Fill(i_jet->eta(),0.);
	    mLooseJIDPassFractionVSpt = map_of_MEs[DirName+"/"+"JetIDPassFractionVSpt"]; if (mLooseJIDPassFractionVSpt && mLooseJIDPassFractionVSpt->getRootObject()) mLooseJIDPassFractionVSpt->Fill(i_jet->pt(),0.);
	    if(i_jet->eta()<3.0){
	      mLooseJIDPassFractionVSptNoHF= map_of_MEs[DirName+"/"+"JetIDPassFractionVSptNoHF"]; if (mLooseJIDPassFractionVSptNoHF && mLooseJIDPassFractionVSptNoHF->getRootObject()) mLooseJIDPassFractionVSptNoHF->Fill(i_jet->pt(),0.);
	    }
	  }
	}
	//here we so far consider calojets ->check for PFJets and JPT jets again
	if(Thiscleaned){//might be softer than loose jet ID 
	  numofjets++;
	  
	  if(isCaloJet_){
	    jetME = map_of_MEs[DirName+"/"+"jetReco"]; if(jetME && jetME->getRootObject()) jetME->Fill(1);
	    mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(1./i_jet->jecFactor(0));
	    mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(i_jet->eta(),1./i_jet->jecFactor(0));
	    mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(i_jet->pt(),1./i_jet->jecFactor(0));
	  }
	  if(isPFJet_){
	    jetME = map_of_MEs[DirName+"/"+"jetReco"]; if(jetME && jetME->getRootObject()) jetME->Fill(2);
	    mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(1./i_jet->jecFactor(0));
	    mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject())mJetEnergyCorrVSEta->Fill(i_jet->eta(),1./i_jet->jecFactor(0));
	    mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(i_jet->pt(),1./i_jet->jecFactor(0));
	  }
	  // --- Event passed the low pt jet trigger
	  if (jetLoPass_ == 1) {	  
	    mPhi_Lo = map_of_MEs[DirName+"/"+"Phi_Lo"]; if (mPhi_Lo && mPhi_Lo->getRootObject()) mPhi_Lo->Fill (i_jet->phi());
	    mPt_Lo = map_of_MEs[DirName+"/"+"Pt_Lo"]; if (mPt_Lo && mPt_Lo->getRootObject())  mPt_Lo->Fill (i_jet->pt());
	    
	  }
	  // --- Event passed the high pt jet trigger
	  if (jetHiPass_ == 1&& i_jet->pt()>100. ) {
	    mEta_Hi = map_of_MEs[DirName+"/"+"Eta_Hi"]; if (mEta_Hi && mEta_Hi->getRootObject()) mEta_Hi->Fill (i_jet->eta());
	    mPhi_Hi = map_of_MEs[DirName+"/"+"Phi_Hi"]; if (mPhi_Hi && mPhi_Hi->getRootObject()) mPhi_Hi->Fill (i_jet->phi());
	    mPt_Hi = map_of_MEs[DirName+"/"+"Pt_Hi"]; if (mPt_Hi && mPt_Hi->getRootObject()) mPt_Hi->Fill (i_jet->pt());		     		    
	  }
	  mPt = map_of_MEs[DirName+"/"+"Pt"]; if (mPt && mPt->getRootObject()) mPt->Fill (i_jet->pt());
	  mPt_1 = map_of_MEs[DirName+"/"+"Pt_1"]; if (mPt_1 && mPt_1->getRootObject())  mPt_1->Fill (i_jet->pt());
	  mPt_2 = map_of_MEs[DirName+"/"+"Pt_2"]; if (mPt_2 && mPt_2->getRootObject()) mPt_2->Fill (i_jet->pt());
	  mPt_3 = map_of_MEs[DirName+"/"+"Pt_3"]; if (mPt_3 && mPt_3->getRootObject()) mPt_3->Fill (i_jet->pt());
	  mEta = map_of_MEs[DirName+"/"+"Eta"]; if (mEta && mEta->getRootObject()) mEta->Fill (i_jet->eta());
	  mPhi = map_of_MEs[DirName+"/"+"Phi"]; if (mPhi && mPhi->getRootObject())  mPhi->Fill (i_jet->phi());	 
	  
	  mPhiVSEta = map_of_MEs[DirName+"/"+"PhiVSEta"]; if (mPhiVSEta && mPhiVSEta->getRootObject()) mPhiVSEta->Fill(i_jet->eta(),i_jet->phi());
	  //if(!isJPTJet_){
	  mConstituents = map_of_MEs[DirName+"/"+"Constituents"]; if (mConstituents && mConstituents->getRootObject()) mConstituents->Fill (i_jet->nConstituents());
	  //}
	  // Fill NPV profiles
	  //--------------------------------------------------------------------
	  mPt_profile = map_of_MEs[DirName+"/"+"Pt_profile"]; if (mPt_profile && mPt_profile->getRootObject())  mPt_profile ->Fill(numPV, i_jet->pt());
	  mEta_profile = map_of_MEs[DirName+"/"+"Eta_profile"]; if (mEta_profile && mEta_profile->getRootObject())  mEta_profile         ->Fill(numPV, i_jet->eta());
	  mPhi_profile = map_of_MEs[DirName+"/"+"Phi_profile"]; if (mPhi_profile && mPhi_profile->getRootObject())  mPhi_profile         ->Fill(numPV, i_jet->phi());
	  //if(!isJPTJet_){
	  mConstituents_profile = map_of_MEs[DirName+"/"+"Constituents_profile"]; if (mConstituents_profile && mConstituents_profile->getRootObject())  mConstituents_profile->Fill(numPV, i_jet->nConstituents());
	  //}
	  if (fabs(i_jet->eta()) <= 1.3) {
	    mPt_Barrel = map_of_MEs[DirName+"/"+"Pt_Barrel"]; if (mPt_Barrel && mPt_Barrel->getRootObject()) mPt_Barrel->Fill (i_jet->pt());
	    mPhi_Barrel = map_of_MEs[DirName+"/"+"Phi_Barrel"]; if (mPhi_Barrel && mPhi_Barrel->getRootObject()) mPhi_Barrel->Fill (i_jet->phi());
	    //if (mE_Barrel)    mE_Barrel->Fill (i_jet->energy());
	    //if(!isJPTJet_){
	    mConstituents_Barrel = map_of_MEs[DirName+"/"+"Constituents_Barrel"]; if (mConstituents_Barrel && mConstituents_Barrel->getRootObject()) mConstituents_Barrel->Fill(i_jet->nConstituents());
	    //}
	  }else if (fabs(i_jet->eta()) <= 3) {
	    mPt_EndCap = map_of_MEs[DirName+"/"+"Pt_EndCap"]; if (mPt_EndCap && mPt_EndCap->getRootObject()) mPt_EndCap->Fill (i_jet->pt());
	    mPhi_EndCap = map_of_MEs[DirName+"/"+"Phi_EndCap"]; if (mPhi_EndCap && mPhi_EndCap->getRootObject())  mPhi_EndCap->Fill (i_jet->phi());
	    //if (mE_EndCap)    mE_EndCap->Fill (i_jet->energy());
	    //if(!isJPTJet_){
	    mConstituents_EndCap = map_of_MEs[DirName+"/"+"Constituents_EndCap"]; if (mConstituents_EndCap && mConstituents_EndCap->getRootObject())  mConstituents_EndCap->Fill(i_jet->nConstituents());
	    //}
	  }else{
	    mPt_Forward = map_of_MEs[DirName+"/"+"Pt_Forward"]; if (mPt_Forward && mPt_Forward->getRootObject()) mPt_Forward->Fill (i_jet->pt());
	    mPhi_Forward = map_of_MEs[DirName+"/"+"Phi_Forward"]; if (mPhi_Forward && mPhi_Forward->getRootObject()) mPhi_Forward->Fill (i_jet->phi());
	    //if (mE_Forward)    mE_Forward->Fill (i_jet->energy());
	    //if(!isJPTJet_){
	    mConstituents_Forward = map_of_MEs[DirName+"/"+"Constituents_Forward"]; if (mConstituents_Forward && mConstituents_Forward->getRootObject())   mConstituents_Forward->Fill(i_jet->nConstituents());
	    //}
	  }
	}
      }
    }
    //since we work now with iterators, need to change the index count here at end of loop
    ijet+=1;
  }

  mNJets = map_of_MEs[DirName+"/"+"NJets"]; if (mNJets && mNJets->getRootObject())  mNJets->Fill (numofjets);
  mNJets_profile = map_of_MEs[DirName+"/"+"NJets_profile"]; if (mNJets_profile && mNJets_profile->getRootObject())  mNJets_profile->Fill(numPV, numofjets);


  sort(recoJets.begin(),recoJets.end(),jetSortingRule);

  //for non dijet selection, otherwise numofjets==0
  if(numofjets>0){
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
  }
  //dijet selection -> recoJets are on corrected level, both jets cleaned, fill folder only for cleaned jet selection
  if(jetCleaningFlag_ && recoJets.size()>1 && cleaned_first_jet && cleaned_second_jet ){
    edm::View<pat::Jet>::const_iterator i_jet1 = patJets.begin()+ind1;
    edm::View<pat::Jet>::const_iterator i_jet2 = patJets.begin()+ind2;
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
	  reco::JetID jetID1 = i_jet1->jetID();
	  reco::JetID jetID2 = i_jet2->jetID();	
	  mN90Hits = map_of_MEs[DirName+"/"+"N90Hits"]; if (mN90Hits && mN90Hits->getRootObject()) mN90Hits->Fill (jetID1.n90Hits);
	  mfHPD = map_of_MEs[DirName+"/"+"fHPD"]; if (mfHPD && mfHPD->getRootObject())             mfHPD->Fill (jetID1.fHPD);
	  mresEMF = map_of_MEs[DirName+"/"+"resEMF"]; if (mresEMF && mresEMF->getRootObject())     mresEMF->Fill (jetID1.restrictedEMF);
	  mfRBX = map_of_MEs[DirName+"/"+"fRBX"]; if (mfRBX && mfRBX->getRootObject())             mfRBX->Fill (jetID1.fRBX);	  
	  mN90Hits = map_of_MEs[DirName+"/"+"N90Hits"]; if (mN90Hits && mN90Hits->getRootObject()) mN90Hits->Fill (jetID2.n90Hits);
	  mfHPD = map_of_MEs[DirName+"/"+"fHPD"]; if (mfHPD && mfHPD->getRootObject())             mfHPD->Fill (jetID2.fHPD);
	  mresEMF = map_of_MEs[DirName+"/"+"resEMF"]; if (mresEMF && mresEMF->getRootObject())     mresEMF->Fill (jetID2.restrictedEMF);
	  mfRBX = map_of_MEs[DirName+"/"+"fRBX"]; if (mfRBX && mfRBX->getRootObject())             mfRBX->Fill (jetID2.fRBX);
	}
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject()) mHFrac->Fill (i_jet1->energyFractionHadronic());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject()) mEFrac->Fill (i_jet1->emEnergyFraction());
	mMaxEInEmTowers = map_of_MEs[DirName+"/"+"MaxEInEmTowers"]; if (mMaxEInEmTowers && mMaxEInEmTowers->getRootObject())     mMaxEInEmTowers->Fill (i_jet1->maxEInEmTowers());
	mMaxEInHadTowers = map_of_MEs[DirName+"/"+"MaxEInHadTowers"]; if (mMaxEInHadTowers && mMaxEInHadTowers->getRootObject()) mMaxEInHadTowers->Fill (i_jet1->maxEInHadTowers());
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject()) mHFrac->Fill (i_jet2->energyFractionHadronic());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject()) mEFrac->Fill (i_jet2->emEnergyFraction());
	mMaxEInEmTowers = map_of_MEs[DirName+"/"+"MaxEInEmTowers"]; if (mMaxEInEmTowers && mMaxEInEmTowers->getRootObject())     mMaxEInEmTowers->Fill (i_jet2->maxEInEmTowers());
	mMaxEInHadTowers = map_of_MEs[DirName+"/"+"MaxEInHadTowers"]; if (mMaxEInHadTowers && mMaxEInHadTowers->getRootObject()) mMaxEInHadTowers->Fill (i_jet2->maxEInHadTowers()); 
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject())   mHFrac_profile       ->Fill(numPV, i_jet1->energyFractionHadronic());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject())   mEFrac_profile       ->Fill(numPV, i_jet1->emEnergyFraction());
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject())   mHFrac_profile       ->Fill(numPV, i_jet2->energyFractionHadronic());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject())   mEFrac_profile       ->Fill(numPV, i_jet2->emEnergyFraction());
	
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(1./i_jet1->jecFactor(0));
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[0].eta(),1./i_jet1->jecFactor(0));
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[0].pt(),1./i_jet1->jecFactor(0));
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(1./i_jet2->jecFactor(0));
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[1].eta(),1./i_jet2->jecFactor(0));
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[1].pt(),1./i_jet2->jecFactor(0));
      }
      if(isPFJet_){
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject())      mHFrac->Fill (i_jet1->chargedHadronEnergyFraction()+i_jet1->neutralHadronEnergyFraction()+i_jet1->HFHadronEnergyFraction());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject())      mEFrac->Fill (i_jet1->chargedEmEnergyFraction() +i_jet1->neutralEmEnergyFraction()+i_jet1->HFEMEnergyFraction());
	
	mCHFrac = map_of_MEs[DirName+"/"+"CHFrac"]; if (mCHFrac && mCHFrac->getRootObject())         mCHFrac ->Fill(i_jet1->chargedHadronEnergyFraction());
	mNHFrac = map_of_MEs[DirName+"/"+"NHFrac"]; if (mNHFrac && mNHFrac->getRootObject())         mNHFrac ->Fill(i_jet1->neutralHadronEnergyFraction());
	mPhFrac = map_of_MEs[DirName+"/"+"PhFrac"]; if (mPhFrac && mPhFrac->getRootObject())         mPhFrac ->Fill(i_jet1->neutralEmEnergyFraction());
	mElFrac = map_of_MEs[DirName+"/"+"ElFrac"]; if (mElFrac && mElFrac->getRootObject())         mElFrac ->Fill(i_jet1->chargedEmEnergyFraction());
	mMuFrac = map_of_MEs[DirName+"/"+"MuFrac"]; if (mMuFrac && mMuFrac->getRootObject())         mMuFrac ->Fill(i_jet1->chargedMuEnergyFraction());
	mHFEMFrac = map_of_MEs[DirName+"/"+"HFEMFrac"]; if (mHFEMFrac && mHFEMFrac->getRootObject()) mHFEMFrac ->Fill(i_jet1->HFEMEnergyFraction());
	mHFHFrac = map_of_MEs[DirName+"/"+"HFHFrac"]; if (mHFHFrac && mHFHFrac->getRootObject())     mHFHFrac ->Fill(i_jet1->HFHadronEnergyFraction());
	
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(1./i_jet1->jecFactor(0));
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[0].eta(),1./i_jet1->jecFactor(0));
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[0].pt(),1./i_jet1->jecFactor(0));
	mJetEnergyCorr = map_of_MEs[DirName+"/"+"JetEnergyCorr"]; if(mJetEnergyCorr && mJetEnergyCorr->getRootObject()) mJetEnergyCorr->Fill(i_jet2->jecFactor(0));
	mJetEnergyCorrVSEta = map_of_MEs[DirName+"/"+"JetEnergyCorrVSEta"]; if(mJetEnergyCorrVSEta && mJetEnergyCorrVSEta->getRootObject()) mJetEnergyCorrVSEta->Fill(recoJets[1].eta(),1./i_jet2->jecFactor(0));
	mJetEnergyCorrVSPt = map_of_MEs[DirName+"/"+"JetEnergyCorrVSPt"]; if(mJetEnergyCorrVSPt && mJetEnergyCorrVSPt->getRootObject()) mJetEnergyCorrVSPt->Fill(recoJets[1].pt(),1./i_jet2->jecFactor(0));
	
	mChargedMultiplicity = map_of_MEs[DirName+"/"+"ChargedMultiplicity"]; if(mChargedMultiplicity && mChargedMultiplicity->getRootObject()) mChargedMultiplicity->Fill(i_jet1->chargedMultiplicity());
	mNeutralMultiplicity = map_of_MEs[DirName+"/"+"NeutralMultiplicity"]; if(mNeutralMultiplicity && mNeutralMultiplicity->getRootObject()) mNeutralMultiplicity->Fill(i_jet1->neutralMultiplicity());
	mMuonMultiplicity = map_of_MEs[DirName+"/"+"MuonMultiplicity"]; if(mMuonMultiplicity && mMuonMultiplicity->getRootObject())             mMuonMultiplicity->Fill(i_jet1->muonMultiplicity());
	//Filling variables for second jet
	mHFrac = map_of_MEs[DirName+"/"+"HFrac"]; if (mHFrac && mHFrac->getRootObject())      mHFrac->Fill (i_jet2->chargedHadronEnergyFraction()+i_jet2->neutralHadronEnergyFraction()+i_jet2->HFHadronEnergyFraction());
	mEFrac = map_of_MEs[DirName+"/"+"EFrac"]; if (mEFrac && mHFrac->getRootObject())      mEFrac->Fill (i_jet2->chargedEmEnergyFraction() +i_jet2->neutralEmEnergyFraction()+i_jet2->HFEMEnergyFraction());
	
	mCHFrac = map_of_MEs[DirName+"/"+"CHFrac"]; if (mCHFrac && mCHFrac->getRootObject())         mCHFrac ->Fill(i_jet2->chargedHadronEnergyFraction());
	mNHFrac = map_of_MEs[DirName+"/"+"NHFrac"]; if (mNHFrac && mNHFrac->getRootObject())         mNHFrac ->Fill(i_jet2->neutralHadronEnergyFraction());
	mPhFrac = map_of_MEs[DirName+"/"+"PhFrac"]; if (mPhFrac && mPhFrac->getRootObject())         mPhFrac ->Fill(i_jet2->neutralEmEnergyFraction());
	mElFrac = map_of_MEs[DirName+"/"+"ElFrac"]; if (mElFrac && mElFrac->getRootObject())         mElFrac ->Fill(i_jet2->chargedEmEnergyFraction());
	mMuFrac = map_of_MEs[DirName+"/"+"MuFrac"]; if (mMuFrac && mMuFrac->getRootObject())         mMuFrac ->Fill(i_jet2->chargedMuEnergyFraction());
	mHFEMFrac = map_of_MEs[DirName+"/"+"HFEMFrac"]; if (mHFEMFrac && mHFEMFrac->getRootObject()) mHFEMFrac ->Fill(i_jet2->HFEMEnergyFraction());
	mHFHFrac = map_of_MEs[DirName+"/"+"HFHFrac"]; if (mHFHFrac && mHFHFrac->getRootObject())     mHFHFrac ->Fill(i_jet2->HFHadronEnergyFraction());
	
	mNeutralFraction = map_of_MEs[DirName+"/"+"NeutralConstituentsFraction"];if (mNeutralFraction && mNeutralFraction->getRootObject()) mNeutralFraction->Fill ((double)i_jet1->neutralMultiplicity()/(double)i_jet1->nConstituents());
	
	mChargedMultiplicity = map_of_MEs[DirName+"/"+"ChargedMultiplicity"]; if(mChargedMultiplicity && mChargedMultiplicity->getRootObject()) mChargedMultiplicity->Fill(i_jet2->chargedMultiplicity());
	mNeutralMultiplicity = map_of_MEs[DirName+"/"+"NeutralMultiplicity"]; if(mNeutralMultiplicity && mNeutralMultiplicity->getRootObject()) mNeutralMultiplicity->Fill(i_jet2->neutralMultiplicity());
	mMuonMultiplicity = map_of_MEs[DirName+"/"+"MuonMultiplicity"]; if(mMuonMultiplicity && mMuonMultiplicity->getRootObject())             mMuonMultiplicity->Fill(i_jet2->muonMultiplicity());
	
	//now fill PFJet profiles for leading jet
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject())     mHFrac_profile       ->Fill(numPV, i_jet1->chargedHadronEnergyFraction() + i_jet1->neutralHadronEnergyFraction());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject())     mEFrac_profile       ->Fill(numPV, i_jet1->chargedEmEnergyFraction()     + i_jet1->neutralEmEnergyFraction());
	mCHFrac_profile = map_of_MEs[DirName+"/"+"CHFrac_profile"]; if (mCHFrac_profile && mCHFrac_profile->getRootObject())         mCHFrac_profile ->Fill(numPV, i_jet1->chargedHadronEnergyFraction());
	mNHFrac_profile = map_of_MEs[DirName+"/"+"NHFrac_profile"]; if (mNHFrac_profile && mNHFrac_profile->getRootObject())         mNHFrac_profile ->Fill(numPV, i_jet1->neutralHadronEnergyFraction());
	mPhFrac_profile = map_of_MEs[DirName+"/"+"PhFrac_profile"]; if (mPhFrac_profile && mPhFrac_profile->getRootObject())         mPhFrac_profile ->Fill(numPV, i_jet1->neutralEmEnergyFraction());
	mElFrac_profile = map_of_MEs[DirName+"/"+"ElFrac_profile"]; if (mElFrac_profile && mElFrac_profile->getRootObject())         mElFrac_profile ->Fill(numPV, i_jet1->chargedEmEnergyFraction());
	mMuFrac_profile = map_of_MEs[DirName+"/"+"MuFrac_profile"]; if (mMuFrac_profile && mMuFrac_profile->getRootObject())         mMuFrac_profile ->Fill(numPV, i_jet1->chargedMuEnergyFraction());
	mHFEMFrac_profile = map_of_MEs[DirName+"/"+"HFEMFrac_profile"]; if (mHFEMFrac_profile && mHFEMFrac_profile->getRootObject()) mHFEMFrac_profile ->Fill(numPV, i_jet1->HFEMEnergyFraction());
	mHFHFrac_profile = map_of_MEs[DirName+"/"+"HFHFrac_profile"]; if (mHFHFrac_profile && mHFHFrac_profile->getRootObject())     mHFHFrac_profile ->Fill(numPV, i_jet1->HFHadronEnergyFraction());
	
	mNeutralFraction = map_of_MEs[DirName+"/"+"NeutralConstituentsFraction"];if (mNeutralFraction && mNeutralFraction->getRootObject()) mNeutralFraction->Fill ((double)i_jet2->neutralMultiplicity()/(double)i_jet2->nConstituents());
	
	mChargedMultiplicity_profile = map_of_MEs[DirName+"/"+"ChargedMultiplicity_profile"]; if(mChargedMultiplicity_profile && mChargedMultiplicity_profile->getRootObject()) mChargedMultiplicity_profile->Fill(numPV, i_jet1->chargedMultiplicity());
	mNeutralMultiplicity_profile = map_of_MEs[DirName+"/"+"NeutralMultiplicity_profile"]; if(mNeutralMultiplicity_profile && mNeutralMultiplicity_profile->getRootObject()) mNeutralMultiplicity_profile->Fill(numPV, i_jet1->neutralMultiplicity());
	mMuonMultiplicity_profile = map_of_MEs[DirName+"/"+"MuonMultiplicity_profile"]; if(mMuonMultiplicity_profile && mMuonMultiplicity_profile->getRootObject())             mMuonMultiplicity->Fill(numPV, i_jet1->muonMultiplicity());
	//now fill PFJet profiles for second leading jet
	mHFrac_profile = map_of_MEs[DirName+"/"+"HFrac_profile"]; if (mHFrac_profile && mHFrac_profile->getRootObject())     mHFrac_profile       ->Fill(numPV, i_jet2->chargedHadronEnergyFraction() + i_jet1->neutralHadronEnergyFraction());
	mEFrac_profile = map_of_MEs[DirName+"/"+"EFrac_profile"]; if (mEFrac_profile && mEFrac_profile->getRootObject())     mEFrac_profile       ->Fill(numPV, i_jet2->chargedEmEnergyFraction()     + i_jet1->neutralEmEnergyFraction());
	mCHFrac_profile = map_of_MEs[DirName+"/"+"CHFrac_profile"]; if (mCHFrac_profile && mCHFrac_profile->getRootObject())         mCHFrac_profile ->Fill(numPV, i_jet2->chargedHadronEnergyFraction());
	mNHFrac_profile = map_of_MEs[DirName+"/"+"NHFrac_profile"]; if (mNHFrac_profile && mNHFrac_profile->getRootObject())         mNHFrac_profile ->Fill(numPV, i_jet2->neutralHadronEnergyFraction());
	mPhFrac_profile = map_of_MEs[DirName+"/"+"PhFrac_profile"]; if (mPhFrac_profile && mPhFrac_profile->getRootObject())         mPhFrac_profile ->Fill(numPV, i_jet2->neutralEmEnergyFraction());
	mElFrac_profile = map_of_MEs[DirName+"/"+"ElFrac_profile"]; if (mElFrac_profile && mElFrac_profile->getRootObject())         mElFrac_profile ->Fill(numPV, i_jet2->chargedEmEnergyFraction());
	mMuFrac_profile = map_of_MEs[DirName+"/"+"MuFrac_profile"]; if (mMuFrac_profile && mMuFrac_profile->getRootObject())         mMuFrac_profile ->Fill(numPV, i_jet2->chargedMuEnergyFraction());
	mHFEMFrac_profile = map_of_MEs[DirName+"/"+"HFEMFrac_profile"]; if (mHFEMFrac_profile && mHFEMFrac_profile->getRootObject()) mHFEMFrac_profile ->Fill(numPV, i_jet2->HFEMEnergyFraction());
	mHFHFrac_profile = map_of_MEs[DirName+"/"+"HFHFrac_profile"]; if (mHFHFrac_profile && mHFHFrac_profile->getRootObject())     mHFHFrac_profile ->Fill(numPV, i_jet2->HFHadronEnergyFraction());
	
	mChargedMultiplicity_profile = map_of_MEs[DirName+"/"+"ChargedMultiplicity_profile"]; if(mChargedMultiplicity_profile && mChargedMultiplicity_profile->getRootObject()) mChargedMultiplicity_profile->Fill(numPV, i_jet2->chargedMultiplicity());
	mNeutralMultiplicity_profile = map_of_MEs[DirName+"/"+"NeutralMultiplicity_profile"]; if(mNeutralMultiplicity_profile && mNeutralMultiplicity_profile->getRootObject()) mNeutralMultiplicity_profile->Fill(numPV, i_jet2->neutralMultiplicity());
	mMuonMultiplicity_profile = map_of_MEs[DirName+"/"+"MuonMultiplicity_profile"]; if(mMuonMultiplicity_profile && mMuonMultiplicity_profile->getRootObject())             mMuonMultiplicity_profile->Fill(numPV, i_jet2->muonMultiplicity());
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
  
}
