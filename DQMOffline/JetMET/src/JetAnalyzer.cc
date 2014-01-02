/*
 *  See header file for a description of this class.
 *
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
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

#define DEBUG 0

using namespace edm;
using namespace reco;
using namespace std;

namespace jetAnalysis {
  
  // Helpper class to propagate tracks to the calo surface using the same implementation as the JetTrackAssociator
  class TrackPropagatorToCalo
  {
   public:
    TrackPropagatorToCalo();
    void update(const edm::EventSetup& eventSetup);
    math::XYZPoint impactPoint(const reco::Track& track) const;
   private:
    const MagneticField* magneticField_;
    const Propagator* propagator_;
    uint32_t magneticFieldCacheId_;
    uint32_t propagatorCacheId_;
  };
  /*
  // Helper class to calculate strip signal to noise and manage the necessary ES objects
  class StripSignalOverNoiseCalculator
  {
   public:
   StripSignalOverNoiseCalculator(const std::string& theQualityLabel = std::string(""));
    void update(const edm::EventSetup& eventSetup);
    double signalOverNoise(const SiStripCluster& cluster,
			   const uint32_t& id) const;
    double operator () (const SiStripCluster& cluster,
			const uint32_t& id) const
    { return signalOverNoise(cluster,id); }
   private:
    const std::string qualityLabel_;
    const SiStripQuality* quality_;
    const SiStripNoises* noise_;
    const SiStripGain* gain_;
    uint32_t qualityCacheId_;
    uint32_t noiseCacheId_;
    uint32_t gainCacheId_;
  };
  */
}

// ***********************************************************
JetAnalyzer::JetAnalyzer(const edm::ParameterSet& pSet)
  : trackPropagator_(new jetAnalysis::TrackPropagatorToCalo)//,
    //sOverNCalculator_(new jetAnalysis::StripSignalOverNoiseCalculator)
{
  
  parameters = pSet.getParameter<edm::ParameterSet>("jetAnalysis");
  mInputCollection           =    pSet.getParameter<edm::InputTag>       ("jetsrc");
  
  mOutputFile   = pSet.getParameter<std::string>("OutputFile");
  JetType = pSet.getParameter<std::string>("JetType");
  JetCorrectionService = pSet.getParameter<std::string> ("JetCorrections");
  
  
  
  
  isCaloJet = (std::string("calo")==JetType);
  isJPTJet = (std::string("jpt") ==JetType);
  isPFJet = (std::string("pf") ==JetType);
  
  if (isCaloJet) caloJetsToken_ = consumes<reco::CaloJetCollection>(mInputCollection);
  if (isJPTJet) jptJetsToken_ = consumes<reco::JPTJetCollection>(mInputCollection);
  if (isPFJet) pfJetsToken_ = consumes<reco::PFJetCollection>(mInputCollection);
  
  // Used for Jet DQM 
  if(isJPTJet || isCaloJet){
    jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));
  }

  //check later if some of those are also needed for PFJets
  _leadJetFlag = 0;
  _JetLoPass   = 0;
  _JetHiPass   = 0;
  _ptThreshold = 20.;
  _ptThresholdUnc = 20.;
  _asymmetryThirdJetCut = 5.;
  _balanceThirdJetCut   = 0.2; 
  
  _n90HitsMin =0;
  _fHPDMax=1.;
  _resEMFMin=0.;
  _n90HitsMinLoose =0;
  _fHPDMaxLoose=1.;
  _resEMFMinLoose=0.;
  _n90HitsMinTight =0;
  _fHPDMaxTight=1.;
  _resEMFMinTight=0.;
  _sigmaEtaMinTight=-999.;
  _sigmaPhiMinTight=-999.;
  
  _LooseCHFMin = -999.;
  _LooseNHFMax = -999.;
  _LooseCEFMax = -999.;
  _LooseNEFMax = -999.;
  _TightCHFMin = -999.;
  _TightNHFMax = -999.;
  _TightCEFMax = -999.;
  _TightNEFMax = -999.;
  _ThisCHFMin = -999.;
  _ThisNHFMax = -999.;
  _ThisCEFMax = -999.;
  _ThisNEFMax = -999.;
  
  //energycorrected=false;
  
  theTriggerResultsLabel        = pSet.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_= consumes<edm::TriggerResults>(edm::InputTag(theTriggerResultsLabel));
  //
  theDiJetSelectionFlag         = pSet.getUntrackedParameter<bool>("DoDiJetSelection", true);
  theJetCleaningFlag            = pSet.getUntrackedParameter<bool>("JetCleaningFlag", true);
  if(theDiJetSelectionFlag){
    makedijetselection=1;
  }
  //makedijetselection         = pSet.getUntrackedParameter<bool>("DoDiJetSelection", true);
  //
  // ==========================================================
  //DCS information
  // ==========================================================
  DCSFilterCalo = new JetMETDQMDCSFilter(pSet.getParameter<ParameterSet>("DCSFilterCalo"));
  DCSFilterPF   = new JetMETDQMDCSFilter(pSet.getParameter<ParameterSet>("DCSFilterPF"));
  DCSFilterJPT  = new JetMETDQMDCSFilter(pSet.getParameter<ParameterSet>("DCSFilterJPT"));
  DCSFilterAll  = new JetMETDQMDCSFilter(pSet.getParameter<ParameterSet>("DCSFilterAll"));
  
  //
  
  fillJIDPassFrac = pSet.getParameter<int>("fillJIDPassFrac");
  //makedijetselection = pSet.getParameter<int>("makedijetselection");
  
  //Trigger selectoin
  edm::ParameterSet highptjetparms = pSet.getParameter<edm::ParameterSet>("highPtJetTrigger");
  edm::ParameterSet lowptjetparms  = pSet.getParameter<edm::ParameterSet>("lowPtJetTrigger" );
  
  _HighPtJetEventFlag = new GenericTriggerEventFlag( highptjetparms, consumesCollector() );
  _LowPtJetEventFlag  = new GenericTriggerEventFlag( lowptjetparms , consumesCollector() );
  
  highPtJetExpr_ = highptjetparms.getParameter<std::vector<std::string> >("hltPaths");
  lowPtJetExpr_  = lowptjetparms .getParameter<std::vector<std::string> >("hltPaths");
  
  _LSBegin     = pSet.getParameter<int>("LSBegin");
  _LSEnd       = pSet.getParameter<int>("LSEnd");
  
  processname_ = pSet.getParameter<std::string>("processname");
  
  //jet cleanup parameters
  theCleaningParameters = pSet.getParameter<ParameterSet>("CleaningParameters");

  
  _doPVCheck          = theCleaningParameters.getParameter<bool>("doPrimaryVertexCheck");
  //_doHLTPhysicsOn     = theCleaningParameters.getParameter<bool>("doHLTPhysicsOn");
  
  //_tightBHFiltering   = theCleaningParameters.getParameter<bool>("tightBHFiltering");
  
  _theVertexLabel     = theCleaningParameters.getParameter<edm::InputTag>("vertexLabel");
  _theGTLabel         = theCleaningParameters.getParameter<edm::InputTag>("gtLabel");
  gtToken_= consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(_theGTLabel));
  vertexToken_= consumes<std::vector<reco::Vertex> >(edm::InputTag(_theVertexLabel));
  
  //Vertex requirements
  if (_doPVCheck) {
    _nvtx_min        = theCleaningParameters.getParameter<int>("nvtx_min");
    _nvtxtrks_min    = theCleaningParameters.getParameter<int>("nvtxtrks_min");
    _vtxndof_min     = theCleaningParameters.getParameter<int>("vtxndof_min");
    _vtxchi2_max     = theCleaningParameters.getParameter<double>("vtxchi2_max");
    _vtxz_max        = theCleaningParameters.getParameter<double>("vtxz_max");
  }
  
  std::string inputCollectionLabel(mInputCollection.label());
  
}  
  

// ***********************************************************
JetAnalyzer::~JetAnalyzer() {
  
  delete _HighPtJetEventFlag;
  delete _LowPtJetEventFlag;

  delete DCSFilterCalo;
  delete DCSFilterPF;
  delete DCSFilterJPT;
  delete DCSFilterAll;
  

}



// ***********************************************************
void JetAnalyzer::beginJob(void) {
  
  dbe = edm::Service<DQMStore>().operator->();
  if(theDiJetSelectionFlag){
    dbe->setCurrentFolder("JetMET/Jet/DiJets"+mInputCollection.label());
  }else{
    if(theJetCleaningFlag){
      dbe->setCurrentFolder("JetMET/Jet/Cleaned"+mInputCollection.label());
    }else{
      dbe->setCurrentFolder("JetMET/Jet/Uncleaned"+mInputCollection.label());
    }
  }

  fillJIDPassFrac = parameters.getParameter<int>("fillJIDPassFrac");
  //makedijetselection = parameters.getParameter<int>("makedijetselection");
  //makedijetselection=theDiJetSelectionFlag;

  // monitoring of eta parameter
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");

  // monitoring of phi paramater
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");

  // monitoring of the transverse momentum
  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");

  // 
  eBin = parameters.getParameter<int>("eBin");
  eMin = parameters.getParameter<double>("eMin");
  eMax = parameters.getParameter<double>("eMax");

  // 
  pBin = parameters.getParameter<int>("pBin");
  pMin = parameters.getParameter<double>("pMin");
  pMax = parameters.getParameter<double>("pMax");

  // 
  nbinsPV = parameters.getParameter<int>("pVBin");
  PVlow   = parameters.getParameter<double>("pVMin");
  PVhigh  = parameters.getParameter<double>("pVMax");

  //
  _ptThreshold = parameters.getParameter<double>("ptThreshold");
  _ptThresholdUnc=parameters.getParameter<double>("ptThresholdUnc");
  _asymmetryThirdJetCut = parameters.getParameter<double>("asymmetryThirdJetCut");
  _balanceThirdJetCut   = parameters.getParameter<double>("balanceThirdJetCut");

  // Generic jet parameters
  mPt           = dbe->book1D("Pt",           "pt",                 ptBin,  ptMin,  ptMax);
  mEta          = dbe->book1D("Eta",          "eta",               etaBin, etaMin, etaMax);
  mPhi          = dbe->book1D("Phi",          "phi",               phiBin, phiMin, phiMax);
  if(!isJPTJet){
    mConstituents = dbe->book1D("Constituents", "# of constituents",     50,      0,    100);
  }
  mJetEnergyCorr= dbe->book1D("JetEnergyCorr", "jet energy correction factor", 50, 0.0,3.0);
  mJetEnergyCorrVsEta= dbe->bookProfile("JetEnergyCorrVsEta", "jet energy correction factor Vs eta", etaBin, etaMin,etaMax, 0.0,3.0);
  mHFrac        = dbe->book1D("HFrac",        "HFrac",                120,   -0.1,    1.1);
  mEFrac        = dbe->book1D("EFrac",        "EFrac",                120,   -0.1,    1.1);

  mPt_uncor           = dbe->book1D("Pt_uncor",           "pt for uncorrected jets",                 ptBin,  _ptThresholdUnc,  ptMax);
  mEta_uncor          = dbe->book1D("Eta_uncor",          "eta for uncorrected jets",               etaBin, etaMin, etaMax);
  mPhi_uncor          = dbe->book1D("Phi_uncor",          "phi for uncorrected jets",               phiBin, phiMin, phiMax);
  if(!isJPTJet){
    mConstituents_uncor = dbe->book1D("Constituents_uncor", "# of constituents for uncorrected jets",     50,      0,    100);
  }

  mDPhi                   = dbe->book1D("DPhi", "dPhi btw the two leading jets", 100, 0., acos(-1.));
  
  mDijetAsymmetry                   = dbe->book1D("DijetAsymmetry", "DijetAsymmetry", 100, -1., 1.);
  mDijetBalance                     = dbe->book1D("DijetBalance",   "DijetBalance",   100, -2., 2.);

  // Book NPV profiles
  //----------------------------------------------------------------------------
  mPt_profile           = dbe->bookProfile("Pt_profile",           "pt",                nbinsPV, PVlow, PVhigh,   ptBin,  ptMin,  ptMax);
  mEta_profile          = dbe->bookProfile("Eta_profile",          "eta",               nbinsPV, PVlow, PVhigh,  etaBin, etaMin, etaMax);
  mPhi_profile          = dbe->bookProfile("Phi_profile",          "phi",               nbinsPV, PVlow, PVhigh,  phiBin, phiMin, phiMax);
  if(!isJPTJet){
    mConstituents_profile = dbe->bookProfile("Constituents_profile", "# of constituents", nbinsPV, PVlow, PVhigh,      50,      0,    100);
  }
  mHFrac_profile        = dbe->bookProfile("HFrac_profile",        "HFrac",             nbinsPV, PVlow, PVhigh,     120,   -0.1,    1.1);
  mEFrac_profile        = dbe->bookProfile("EFrac_profile",        "EFrac",             nbinsPV, PVlow, PVhigh,     120,   -0.1,    1.1);

  if(fillJIDPassFrac==1){//fillJIDPassFrac defines a collection of cleaned jets, for which we will want to fill the cleaning passing fraction
    mLooseJIDPassFractionVSeta      = dbe->bookProfile("LooseJIDPassFractionVSeta","LooseJIDPassFractionVSeta",etaBin, etaMin, etaMax,0.,1.2);
    mLooseJIDPassFractionVSpt       = dbe->bookProfile("LooseJIDPassFractionVSpt","LooseJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
    mTightJIDPassFractionVSeta      = dbe->bookProfile("TightJIDPassFractionVSeta","TightJIDPassFractionVSeta",etaBin, etaMin, etaMax,0.,1.2);
    mTightJIDPassFractionVSpt       = dbe->bookProfile("TightJIDPassFractionVSpt","TightJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
  }

  if (makedijetselection != 1){
    mNJets_profile = dbe->bookProfile("NJets_profile", "number of jets", nbinsPV, PVlow, PVhigh, 100, 0, 100);
  }

  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  mPt_profile          ->setAxisTitle("nvtx",1);
  mEta_profile         ->setAxisTitle("nvtx",1);
  mPhi_profile         ->setAxisTitle("nvtx",1);
  if(!isJPTJet){
    mConstituents_profile->setAxisTitle("nvtx",1);
  }
  mHFrac_profile       ->setAxisTitle("nvtx",1);
  mEFrac_profile       ->setAxisTitle("nvtx",1);

  if (makedijetselection != 1) {
    mNJets_profile->setAxisTitle("nvtx",1);
  }


  //mE                       = dbe->book1D("E", "E", eBin, eMin, eMax);
  //mP                       = dbe->book1D("P", "P", pBin, pMin, pMax);
  //  mMass                    = dbe->book1D("Mass", "Mass", 100, 0, 25);
  //
  mPhiVSEta                     = dbe->book2D("PhiVSEta", "PhiVSEta", 50, etaMin, etaMax, 24, phiMin, phiMax);
  mPhiVSEta->getTH2F()->SetOption("colz");
  mPhiVSEta->setAxisTitle("#eta",1);
  mPhiVSEta->setAxisTitle("#phi",2);
  if(makedijetselection!=1){
    mPt_1                    = dbe->book1D("Pt_1", "Pt spectrum of jets - range 1", 20, 0, 100);   
    mPt_2                    = dbe->book1D("Pt_2", "Pt spectrum of jets - range 2", 60, 0, 300);   
    mPt_3                    = dbe->book1D("Pt_3", "Pt spectrum of jets - range 3", 100, 0, 5000);
    // Low and high pt trigger paths
    mPt_Lo                  = dbe->book1D("Pt_Lo", "Pt (Pass Low Pt Jet Trigger)", 20, 0, 100);   
    //mEta_Lo                 = dbe->book1D("Eta_Lo", "Eta (Pass Low Pt Jet Trigger)", etaBin, etaMin, etaMax);
    mPhi_Lo                 = dbe->book1D("Phi_Lo", "Phi (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
    
    mPt_Hi                  = dbe->book1D("Pt_Hi", "Pt (Pass Hi Pt Jet Trigger)", 60, 0, 300);   
    mEta_Hi                 = dbe->book1D("Eta_Hi", "Eta (Pass Hi Pt Jet Trigger)", etaBin, etaMin, etaMax);
    mPhi_Hi                 = dbe->book1D("Phi_Hi", "Phi (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);
    mNJets                   = dbe->book1D("NJets", "number of jets", 100, 0, 100);

    //mPt_Barrel_Lo            = dbe->book1D("Pt_Barrel_Lo", "Pt Barrel (Pass Low Pt Jet Trigger)", 20, 0, 100);   
    //mPhi_Barrel_Lo           = dbe->book1D("Phi_Barrel_Lo", "Phi Barrel (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
    if(!isJPTJet){
      mConstituents_Barrel     = dbe->book1D("Constituents_Barrel", "Constituents Barrel", 50, 0, 100);
    }
    mHFrac_Barrel            = dbe->book1D("HFrac_Barrel", "HFrac Barrel", 100, 0, 1);
    mEFrac_Barrel            = dbe->book1D("EFrac_Barrel", "EFrac Barrel", 110, -0.05, 1.05);
    
    //mPt_EndCap_Lo            = dbe->book1D("Pt_EndCap_Lo", "Pt EndCap (Pass Low Pt Jet Trigger)", 20, 0, 100);   
    //mPhi_EndCap_Lo           = dbe->book1D("Phi_EndCap_Lo", "Phi EndCap (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
    if(!isJPTJet){
      mConstituents_EndCap     = dbe->book1D("Constituents_EndCap", "Constituents EndCap", 50, 0, 100);
    }
    mHFrac_EndCap            = dbe->book1D("HFrac_Endcap", "HFrac EndCap", 100, 0, 1);
    mEFrac_EndCap            = dbe->book1D("EFrac_Endcap", "EFrac EndCap", 110, -0.05, 1.05);
    
    //mPt_Forward_Lo           = dbe->book1D("Pt_Forward_Lo", "Pt Forward (Pass Low Pt Jet Trigger)", 20, 0, 100);  
    //mPhi_Forward_Lo          = dbe->book1D("Phi_Forward_Lo", "Phi Forward (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
    if(!isJPTJet){
      mConstituents_Forward    = dbe->book1D("Constituents_Forward", "Constituents Forward", 50, 0, 100);
    }
    mHFrac_Forward           = dbe->book1D("HFrac_Forward", "HFrac Forward", 100, 0, 1);
    mEFrac_Forward           = dbe->book1D("EFrac_Forward", "EFrac Forward", 110, -0.05, 1.05);
    
    mPt_Barrel_Hi            = dbe->book1D("Pt_Barrel_Hi", "Pt Barrel (Pass Hi Pt Jet Trigger)", 60, 0, 300);   
    mPhi_Barrel_Hi           = dbe->book1D("Phi_Barrel_Hi", "Phi Barrel (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);
    //mConstituents_Barrel_Hi  = dbe->book1D("Constituents_Barrel_Hi", "Constituents Barrel (Pass Hi Pt Jet Trigger)", 50, 0, 100);
    //mHFrac_Barrel_Hi         = dbe->book1D("HFrac_Barrel_Hi", "HFrac Barrel (Pass Hi Pt Jet Trigger)", 100, 0, 1);
    
    mPt_EndCap_Hi            = dbe->book1D("Pt_EndCap_Hi", "Pt EndCap (Pass Hi Pt Jet Trigger)", 60, 0, 300);  
    mPhi_EndCap_Hi           = dbe->book1D("Phi_EndCap_Hi", "Phi EndCap (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);
    //mConstituents_EndCap_Hi  = dbe->book1D("Constituents_EndCap_Hi", "Constituents EndCap (Pass Hi Pt Jet Trigger)", 50, 0, 100);
    //mHFrac_EndCap_Hi         = dbe->book1D("HFrac_EndCap_Hi", "HFrac EndCap (Pass Hi Pt Jet Trigger)", 100, 0, 1);
    
    mPt_Forward_Hi           = dbe->book1D("Pt_Forward_Hi", "Pt Forward (Pass Hi Pt Jet Trigger)", 60, 0, 300);  
    mPhi_Forward_Hi          = dbe->book1D("Phi_Forward_Hi", "Phi Forward (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);
    //mConstituents_Forward_Hi = dbe->book1D("Constituents_Forward_Hi", "Constituents Forward (Pass Hi Pt Jet Trigger)", 50, 0, 100);
    //mHFrac_Forward_Hi        = dbe->book1D("HFrac_Forward_Hi", "HFrac Forward (Pass Hi Pt Jet Trigger)", 100, 0, 1);
    
    mPhi_Barrel              = dbe->book1D("Phi_Barrel", "Phi_Barrel", phiBin, phiMin, phiMax);
    //mE_Barrel                = dbe->book1D("E_Barrel", "E_Barrel", eBin, eMin, eMax);
    mPt_Barrel               = dbe->book1D("Pt_Barrel", "Pt_Barrel", ptBin, ptMin, ptMax);
    
    mPhi_EndCap              = dbe->book1D("Phi_EndCap", "Phi_EndCap", phiBin, phiMin, phiMax);
    //mE_EndCap                = dbe->book1D("E_EndCap", "E_EndCap", eBin, eMin, 2*eMax);
    mPt_EndCap               = dbe->book1D("Pt_EndCap", "Pt_EndCap", ptBin, ptMin, ptMax);
    
    mPhi_Forward             = dbe->book1D("Phi_Forward", "Phi_Forward", phiBin, phiMin, phiMax);
    //mE_Forward               = dbe->book1D("E_Forward", "E_Forward", eBin, eMin, 4*eMax);
    mPt_Forward              = dbe->book1D("Pt_Forward", "Pt_Forward", ptBin, ptMin, ptMax);
    
    // Leading Jet Parameters
    mEtaFirst                = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5);
    mPhiFirst                = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5);
    //mEFirst                  = dbe->book1D("EFirst", "EFirst", 100, 0, 1000);
    mPtFirst                 = dbe->book1D("PtFirst", "PtFirst", ptBin, ptMin, ptMax);

    //if (fillJIDPassFrac==1) {
    //mLooseJIDPassFractionVSeta  = dbe->bookProfile("LooseJIDPassFractionVSeta","LooseJIDPassFractionVSeta",50, -3., 3.,0.,1.2);
    //mLooseJIDPassFractionVSpt   = dbe->bookProfile("LooseJIDPassFractionVSpt","LooseJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
    //mTightJIDPassFractionVSeta  = dbe->bookProfile("TightJIDPassFractionVSeta","TightJIDPassFractionVSeta",50, -3., 3.,0.,1.2);
    //mTightJIDPassFractionVSpt   = dbe->bookProfile("TightJIDPassFractionVSpt","TightJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
    //}
  }
  //
  //--- Calo jet selection only
  if(isCaloJet) {
    _n90HitsMin = parameters.getParameter<int>("n90HitsMin");
    _fHPDMax = parameters.getParameter<double>("fHPDMax");
    _resEMFMin = parameters.getParameter<double>("resEMFMin");
    _sigmaEtaMinTight = parameters.getParameter<double>("sigmaEtaMinTight");
    _sigmaPhiMinTight = parameters.getParameter<double>("sigmaPhiMinTight");
    
    _n90HitsMinLoose = parameters.getParameter<int>("n90HitsMinLoose");
    _fHPDMaxLoose = parameters.getParameter<double>("fHPDMaxLoose");
    _resEMFMinLoose = parameters.getParameter<double>("resEMFMinLoose");
    _n90HitsMinTight = parameters.getParameter<int>("n90HitsMinTight");
    _fHPDMaxTight = parameters.getParameter<double>("fHPDMaxTight");
    _resEMFMinTight = parameters.getParameter<double>("resEMFMinTight");

    // CaloJet specific
    mMaxEInEmTowers         = dbe->book1D("MaxEInEmTowers", "MaxEInEmTowers", 100, 0, 100);
    mMaxEInHadTowers        = dbe->book1D("MaxEInHadTowers", "MaxEInHadTowers", 100, 0, 100);
    if(makedijetselection!=1) {
      mHadEnergyInHO          = dbe->book1D("HadEnergyInHO", "HadEnergyInHO", 100, 0, 10);
      mHadEnergyInHB          = dbe->book1D("HadEnergyInHB", "HadEnergyInHB", 100, 0, 50);
      mHadEnergyInHF          = dbe->book1D("HadEnergyInHF", "HadEnergyInHF", 100, 0, 50);
      mHadEnergyInHE          = dbe->book1D("HadEnergyInHE", "HadEnergyInHE", 100, 0, 100);
      mEmEnergyInEB           = dbe->book1D("EmEnergyInEB", "EmEnergyInEB", 100, 0, 50);
      mEmEnergyInEE           = dbe->book1D("EmEnergyInEE", "EmEnergyInEE", 100, 0, 50);
      mEmEnergyInHF           = dbe->book1D("EmEnergyInHF", "EmEnergyInHF", 120, -20, 100);
    }

  
    //JetID variables
  
    mresEMF                 = dbe->book1D("resEMF", "resEMF", 50, 0., 1.);
    mN90Hits                = dbe->book1D("N90Hits", "N90Hits", 50, 0., 50);
    mfHPD                   = dbe->book1D("fHPD", "fHPD", 50, 0., 1.);
    mfRBX                   = dbe->book1D("fRBX", "fRBX", 50, 0., 1.);
    
  }

  //
  //--- Calo jet selection only
  if(isCaloJet) {
    jetME = dbe->book1D("jetReco", "jetReco", 3, 1, 4);
    jetME->setBinLabel(1,"CaloJets",1);
    // CaloJet specific
    mMaxEInEmTowers         = dbe->book1D("MaxEInEmTowers", "MaxEInEmTowers", 100, 0, 100);
    mMaxEInHadTowers        = dbe->book1D("MaxEInHadTowers", "MaxEInHadTowers", 100, 0, 100);
    if(makedijetselection!=1) {
      mHadEnergyInHO          = dbe->book1D("HadEnergyInHO", "HadEnergyInHO", 100, 0, 10);
      mHadEnergyInHB          = dbe->book1D("HadEnergyInHB", "HadEnergyInHB", 100, 0, 50);
      mHadEnergyInHF          = dbe->book1D("HadEnergyInHF", "HadEnergyInHF", 100, 0, 50);
      mHadEnergyInHE          = dbe->book1D("HadEnergyInHE", "HadEnergyInHE", 100, 0, 100);
      mEmEnergyInEB           = dbe->book1D("EmEnergyInEB", "EmEnergyInEB", 100, 0, 50);
      mEmEnergyInEE           = dbe->book1D("EmEnergyInEE", "EmEnergyInEE", 100, 0, 50);
      mEmEnergyInHF           = dbe->book1D("EmEnergyInHF", "EmEnergyInHF", 120, -20, 100);
    }

  
    //JetID variables
  
    mresEMF                 = dbe->book1D("resEMF", "resEMF", 50, 0., 1.);
    mN90Hits                = dbe->book1D("N90Hits", "N90Hits", 50, 0., 50);
    mfHPD                   = dbe->book1D("fHPD", "fHPD", 50, 0., 1.);
    mfRBX                   = dbe->book1D("fRBX", "fRBX", 50, 0., 1.);
    
  }
  if(isJPTJet) {
   jetME = dbe->book1D("jetReco", "jetReco", 3, 1, 4);
   jetME->setBinLabel(3,"JPTJets",1);
   _n90HitsMin = parameters.getParameter<int>("n90HitsMin");
   _fHPDMax = parameters.getParameter<double>("fHPDMax");
   _resEMFMin = parameters.getParameter<double>("resEMFMin");
   _sigmaEtaMinTight = parameters.getParameter<double>("sigmaEtaMinTight");
   _sigmaPhiMinTight = parameters.getParameter<double>("sigmaPhiMinTight");
   
   _n90HitsMinLoose = parameters.getParameter<int>("n90HitsMinLoose");
   _fHPDMaxLoose = parameters.getParameter<double>("fHPDMaxLoose");
   _resEMFMinLoose = parameters.getParameter<double>("resEMFMinLoose");
   _n90HitsMinTight = parameters.getParameter<int>("n90HitsMinTight");
   _fHPDMaxTight = parameters.getParameter<double>("fHPDMaxTight");
   _resEMFMinTight = parameters.getParameter<double>("resEMFMinTight");
   if(makedijetselection!=1){
     //jpt histograms
     mE   = dbe->book1D("E", "E", eBin, eMin, eMax);
     mEt  = dbe->book1D("Et", "Et", ptBin, ptMin, ptMax);
     mP   = dbe->book1D("P", "P", ptBin, ptMin, ptMax);
     mPtSecond = dbe->book1D("PtSecond", "PtSecond", ptBin, ptMin, ptMax);
     mPtThird = dbe->book1D("PtThird", "PtThird", ptBin, ptMin, ptMax);
     mPx  = dbe->book1D("Px", "Px", ptBin, ptMin, ptMax);
     mPy  = dbe->book1D("Py", "Py", ptBin, ptMin, ptMax);
     mPz  = dbe->book1D("Pz", "Pz", ptBin, ptMin, ptMax);
     mnTracks  = dbe->book1D("nTracks", "number of tracks for correction per jet", 100, 0, 100);
     mnTracksVsJetPt= dbe->bookProfile("nTracksVSJetPt","number of tracks for correction per jet vs raw jet p_{T}",ptBin, ptMin, ptMax,100,0,100);
     mnTracksVsJetEta= dbe->bookProfile("nTracksVsJetEta","number of tracks for correction per jet vs jet #eta",etaBin, etaMin, etaMax,100,0,100);
     mnTracksVsJetPt ->setAxisTitle("raw JetPt",1);
     mnTracksVsJetEta ->setAxisTitle("raw JetEta",1);
     //define meaningful limits

     //ntrackbins,0,ntrackbins
     mnallPionsTracksPerJet=dbe->book1D("nallPionTracks", "number of pion tracks for correction per jet", 100, 0, 100);
     //trackptbins,0,trackptmax
     //introduce etamax?
     mallPionsTracksPt=dbe->book1D("allPionTrackPt", "pion track p_{T}", 100, 0., 50.);
     mallPionsTracksEta=dbe->book1D("allPionTrackEta", "pion track #eta", 50, -2.5, 2.5);
     //phibins,phimax,phimin
     mallPionsTracksPhi=dbe->book1D("allPionTracksPhi", "pion track #phi", phiBin,phiMin, phiMax);
     //etabins/etamax/etamin
     mallPionsTracksPtVsEta=dbe->bookProfile("allPionTracksPtVsEta", "pion track p_{T} vs track #eta", 50, -2.5, 2.5,100,0.,50.);

     mnInVertexInCaloPionsTracksPerJet=dbe->book1D("nInVertexInCaloPionTracks", "number of pion in cone at calo and vertexs for correction per jet", 100, 0, 100);
     mInVertexInCaloPionsTracksPt=dbe->book1D("InVertexInCaloPionTrackPt", "pion in cone at calo and vertex p_{T}", 100, 0., 50.);
     mInVertexInCaloPionsTracksEta=dbe->book1D("InVertexInCaloPionTrackEta", "pion in cone at calo and vertex #eta", 50, -2.5, 2.5);
     mInVertexInCaloPionsTracksPhi=dbe->book1D("InVertexInCaloPionTracksPhi", "pion in cone at calo and vertex #phi", phiBin,phiMin, phiMax);
     mInVertexInCaloPionsTracksPtVsEta=dbe->bookProfile("InVertexInCaloPionTracksPtVsEta", "pion in cone at calo and vertex p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);
     //monitor element trackDirectionJetDRHisto
     //monitor element trackImpactPointJetDRHisto
     mnOutVertexInCaloPionsTracksPerJet=dbe->book1D("nOutVertexInCaloPionTracks", "number of pion in cone at calo and out at vertex for correction per jet", 100, 0, 100);
     mOutVertexInCaloPionsTracksPt=dbe->book1D("OutVertexInCaloPionTrackPt", "pion in cone at calo and out at vertex p_{T}", 100, 0., 50.);
     mOutVertexInCaloPionsTracksEta=dbe->book1D("OutVertexInCaloPionTrackEta", "pion in cone at calo and out at vertex #eta", 50, -2.5, 2.5);
     mOutVertexInCaloPionsTracksPhi=dbe->book1D("OutVertexInCaloPionTracksPhi", "pion in cone at calo and out at vertex #phi", phiBin,phiMin, phiMax);
     mOutVertexInCaloPionsTracksPtVsEta=dbe->bookProfile("OutVertexInCaloPionTracksPtVsEta", "pion in cone at calo and out at vertex p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);

     mnInVertexOutCaloPionsTracksPerJet=dbe->book1D("nInVertexOutCaloPionTracks", "number of pions out cone at calo and and in cone at vertex for correction per jet", 100, 0, 100);
     mInVertexOutCaloPionsTracksPt=dbe->book1D("InVertexOutCaloPionTrackPt", "pion out cone at calo and in cone at vertex p_{T}", 100, 0., 50.);
     mInVertexOutCaloPionsTracksEta=dbe->book1D("InVertexOutCaloPionTrackEta", "pion out cone at calo and in cone at vertex #eta", 50, -2.5, 2.5);
     mInVertexOutCaloPionsTracksPhi=dbe->book1D("InVertexOutCaloPionTracksPhi", "pion out cone at calo and in cone at vertex #phi", phiBin,phiMin, phiMax);
     mInVertexOutCaloPionsTracksPtVsEta=dbe->bookProfile("InVertexOutCaloPionTracksPtVsEta", "pion out cone at calo and in cone at vertex p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);

     mnallMuonsTracksPerJet=dbe->book1D("nallMuonTracks", "number of muon tracks for correction per jet", 10, 0, 10);
     mallMuonsTracksPt=dbe->book1D("allMuonTrackPt", "muon track p_{T}", 100, 0., 50.);
     mallMuonsTracksEta=dbe->book1D("allMuonTrackEta", "muon track #eta", 50, -2.5, 2.5);
     mallMuonsTracksPhi=dbe->book1D("allMuonTracksPhi", "muon track #phi", phiBin,phiMin, phiMax);
     mallMuonsTracksPtVsEta=dbe->bookProfile("allMuonTracksPtVsEta", "muon track p_{T} vs track #eta", 50, -2.5, 2.5,100,0.,50.);
 
     mnInVertexInCaloMuonsTracksPerJet=dbe->book1D("nInVertexInCaloMuonTracks", "number of muons in cone at calo and vertex for correction per jet", 10, 0, 10);
     mInVertexInCaloMuonsTracksPt=dbe->book1D("InVertexInCaloMuonTrackPt", "muon in cone at calo and vertex p_{T}", 100, 0., 50.);
     mInVertexInCaloMuonsTracksEta=dbe->book1D("InVertexInCaloMuonTrackEta", "muon in cone at calo and vertex #eta", 50, -2.5, 2.5);
     mInVertexInCaloMuonsTracksPhi=dbe->book1D("InVertexInCaloMuonTracksPhi", "muon in cone at calo and vertex #phi", phiBin,phiMin, phiMax);
     mInVertexInCaloMuonsTracksPtVsEta=dbe->bookProfile("InVertexInCaloMuonTracksPtVsEta", "muon in cone at calo and vertex p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);

     mnOutVertexInCaloMuonsTracksPerJet=dbe->book1D("nOutVertexInCaloMuonTracks", "number of muons in cone at calo and out cone at vertex for correction per jet", 10, 0, 10);
     mOutVertexInCaloMuonsTracksPt=dbe->book1D("OutVertexInCaloMuonTrackPt", "muon in cone at calo and out cone at vertex p_{T}", 100, 0., 50.);
     mOutVertexInCaloMuonsTracksEta=dbe->book1D("OutVertexInCaloMuonTrackEta", "muon in cone at calo and out cone at vertex #eta", 50, -2.5, 2.5);
     mOutVertexInCaloMuonsTracksPhi=dbe->book1D("OutVertexInCaloMuonTracksPhi", "muon in cone at calo and out cone at vertex #phi", phiBin,phiMin, phiMax);
     mOutVertexInCaloMuonsTracksPtVsEta=dbe->bookProfile("OutVertexInCaloMuonTracksPtVsEta", "muon oin cone at calo and out cone at vertex p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);

     mnInVertexOutCaloMuonsTracksPerJet=dbe->book1D("nInVertexOutCaloMuonTracks", "number of muons out cone at calo and in cone at vertex for correction per jet", 10, 0, 10);
     mInVertexOutCaloMuonsTracksPt=dbe->book1D("InVertexOutCaloMuonTrackPt", "muon out cone at calo and in cone at vertex p_{T}", 100, 0., 50.);
     mInVertexOutCaloMuonsTracksEta=dbe->book1D("InVertexOutCaloMuonTrackEta", "muon out cone at calo and in cone at vertex #eta", 50, -2.5, 2.5);
     mInVertexOutCaloMuonsTracksPhi=dbe->book1D("InVertexOutCaloMuonTracksPhi", "muon out cone at calo and in cone at vertex #phi", phiBin,phiMin, phiMax);
     mInVertexOutCaloMuonsTracksPtVsEta=dbe->bookProfile("InVertexOutCaloMuonTracksPtVsEta", "muon out cone at calo and in cone at vertex p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);

     mnallElectronsTracksPerJet=dbe->book1D("nallElectronTracks", "number of electron tracks for correction per jet", 10, 0, 10);
     mallElectronsTracksPt=dbe->book1D("allElectronTrackPt", "electron track p_{T}", 100, 0., 50.);
     mallElectronsTracksEta=dbe->book1D("allElectronTrackEta", "electron track #eta", 50, -2.5, 2.5);
     mallElectronsTracksPhi=dbe->book1D("allElectronTracksPhi", "electron track #phi", phiBin,phiMin, phiMax);
     mallElectronsTracksPtVsEta=dbe->bookProfile("allElectronTracksPtVsEta", "electron track p_{T} vs track #eta", 50, -2.5, 2.5,100,0.,50.);

     mnInVertexInCaloElectronsTracksPerJet=dbe->book1D("nInVertexInCaloElectronTracks", "number of electrons in cone at calo and vertex for correction per jet", 10, 0, 10);
     mInVertexInCaloElectronsTracksPt=dbe->book1D("InVertexInCaloElectronTrackPt", "electron in cone at calo and vertex p_{T}", 100, 0., 50.);
     mInVertexInCaloElectronsTracksEta=dbe->book1D("InVertexInCaloElectronTrackEta", "electron in cone at calo and vertex #eta", 50, -2.5, 2.5);
     mInVertexInCaloElectronsTracksPhi=dbe->book1D("InVertexInCaloElectronTracksPhi", "electron in cone at calo and vertex #phi", phiBin,phiMin, phiMax);
     mInVertexInCaloElectronsTracksPtVsEta=dbe->bookProfile("InVertexInCaloElectronTracksPtVsEta", "electron in cone at calo and vertex  p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);

     mnOutVertexInCaloElectronsTracksPerJet=dbe->book1D("nOutVertexInCaloElectronTracks", "number of electrons in cone at calo and out cone at vertex for correction per jet", 10, 0, 10);
     mOutVertexInCaloElectronsTracksPt=dbe->book1D("OutVertexInCaloElectronTrackPt", "electron in cone at calo and out cone at vertex p_{T}", 100, 0., 50.);
     mOutVertexInCaloElectronsTracksEta=dbe->book1D("OutVertexInCaloElectronTrackEta", "electron in cone at calo and out cone at vertex #eta", 50, -2.5, 2.5);
     mOutVertexInCaloElectronsTracksPhi=dbe->book1D("OutVertexInCaloElectronTracksPhi", "electron in cone at calo and out cone at vertex #phi", phiBin,phiMin, phiMax);
     mOutVertexInCaloElectronsTracksPtVsEta=dbe->bookProfile("OutVertexInCaloElectronTracksPtVsEta", "electron in cone at calo and out cone at vertex p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);

     mnInVertexOutCaloElectronsTracksPerJet=dbe->book1D("nInVertexOutCaloElectronTracks", "number of electrons out cone at calo and in cone at vertex for correction per jet", 10, 0, 10);
     mInVertexOutCaloElectronsTracksPt=dbe->book1D("InVertexOutCaloElectronTrackPt", "electron out cone at calo and in cone at vertex p_{T}", 100, 0., 50.);
     mInVertexOutCaloElectronsTracksEta=dbe->book1D("InVertexOutCaloElectronTrackEta", "electron out cone at calo and in cone at vertex #eta", 50, -2.5, 2.5);
     mInVertexOutCaloElectronsTracksPhi=dbe->book1D("InVertexOutCaloElectronTracksPhi", "electron out cone at calo and in cone at vertex #phi", phiBin,phiMin, phiMax);
     mInVertexOutCaloElectronsTracksPtVsEta=dbe->bookProfile("InVertexOutCaloElectronTracksPtVsEta", "electron out cone at calo and in cone at vertex p_{T} vs #eta", 50, -2.5, 2.5,100,0.,50.);

     mInCaloTrackDirectionJetDRHisto_      = dbe->book1D("InCaloTrackDirectionJetDR",
							"#Delta R between track direction at vertex and jet axis (track in cone at calo)",50,0.,2.0);
     mOutCaloTrackDirectionJetDRHisto_      = dbe->book1D("OutCaloTrackDirectionJetDR",
							"#Delta R between track direction at vertex and jet axis (track out cone at calo)",50,0.,2.0);
     mInVertexTrackImpactPointJetDRHisto_  = dbe->book1D("InVertexTrackImpactPointJetDR",
							 "#Delta R between track impact point on calo and jet axis (track in cone at vertex)",50,0.,2.0);
     mOutVertexTrackImpactPointJetDRHisto_ = dbe->book1D("OutVertexTrackImpactPointJetDR",
							 "#Delta R between track impact point on calo and jet axis (track out of cone at vertex)",50,0.,2.0);
   }
  }

  if(isPFJet) {
    jetME = dbe->book1D("jetReco", "jetReco", 3, 1, 4);
    jetME->setBinLabel(2,"PFJets",1);

    _TightCHFMin = parameters.getParameter<double>("TightCHFMin");
    _TightNHFMax = parameters.getParameter<double>("TightNHFMax");
    _TightCEFMax = parameters.getParameter<double>("TightCEFMax");
    _TightNEFMax = parameters.getParameter<double>("TightNEFMax");
    _LooseCHFMin = parameters.getParameter<double>("LooseCHFMin");
    _LooseNHFMax = parameters.getParameter<double>("LooseNHFMax");
    _LooseCEFMax = parameters.getParameter<double>("LooseCEFMax");
    _LooseNEFMax = parameters.getParameter<double>("LooseNEFMax");

    _ThisCHFMin = parameters.getParameter<double>("ThisCHFMin");
    _ThisNHFMax = parameters.getParameter<double>("ThisNHFMax");
    _ThisCEFMax = parameters.getParameter<double>("ThisCEFMax");
    _ThisNEFMax = parameters.getParameter<double>("ThisNEFMax");

    if(makedijetselection!=1){
      //PFJet specific
      mCHFracVSeta_lowPt= dbe->bookProfile("CHFracVSeta_lowPt","CHFracVSeta_lowPt",etaBin, etaMin, etaMax,0.,1.2);
      mNHFracVSeta_lowPt= dbe->bookProfile("NHFracVSeta_lowPt","NHFracVSeta_lowPt",etaBin, etaMin, etaMax,0.,1.2);
      mPhFracVSeta_lowPt= dbe->bookProfile("PhFracVSeta_lowPt","PhFracVSeta_lowPt",etaBin, etaMin, etaMax,0.,1.2);
      mElFracVSeta_lowPt= dbe->bookProfile("ElFracVSeta_lowPt","ElFracVSeta_lowPt",etaBin, etaMin, etaMax,0.,1.2);
      mMuFracVSeta_lowPt= dbe->bookProfile("MuFracVSeta_lowPt","MuFracVSeta_lowPt",etaBin, etaMin, etaMax,0.,1.2);
      mCHFracVSeta_mediumPt= dbe->bookProfile("CHFracVSeta_mediumPt","CHFracVSeta_mediumPt",etaBin, etaMin, etaMax,0.,1.2);
      mNHFracVSeta_mediumPt= dbe->bookProfile("NHFracVSeta_mediumPt","NHFracVSeta_mediumPt",etaBin, etaMin, etaMax,0.,1.2);
      mPhFracVSeta_mediumPt= dbe->bookProfile("PhFracVSeta_mediumPt","PhFracVSeta_mediumPt",etaBin, etaMin, etaMax,0.,1.2);
      mElFracVSeta_mediumPt= dbe->bookProfile("ElFracVSeta_mediumPt","ElFracVSeta_mediumPt",etaBin, etaMin, etaMax,0.,1.2);
      mMuFracVSeta_mediumPt= dbe->bookProfile("MuFracVSeta_mediumPt","MuFracVSeta_mediumPt",etaBin, etaMin, etaMax,0.,1.2);
      mCHFracVSeta_highPt= dbe->bookProfile("CHFracVSeta_highPt","CHFracVSeta_highPt",etaBin, etaMin, etaMax,0.,1.2);
      mNHFracVSeta_highPt= dbe->bookProfile("NHFracVSeta_highPt","NHFracVSeta_highPt",etaBin, etaMin, etaMax,0.,1.2);
      mPhFracVSeta_highPt= dbe->bookProfile("PhFracVSeta_highPt","PhFracVSeta_highPt",etaBin, etaMin, etaMax,0.,1.2);
      mElFracVSeta_highPt= dbe->bookProfile("ElFracVSeta_highPt","ElFracVSeta_highPt",etaBin, etaMin, etaMax,0.,1.2);
      mMuFracVSeta_highPt= dbe->bookProfile("MuFracVSeta_highPt","MuFracVSeta_highPt",etaBin, etaMin, etaMax,0.,1.2);
      
      //barrel histograms for PFJets
      // energy fractions
      mCHFrac_lowPt_Barrel     = dbe->book1D("CHFrac_lowPt_Barrel", "CHFrac_lowPt_Barrel", 120, -0.1, 1.1);
      mNHFrac_lowPt_Barrel     = dbe->book1D("NHFrac_lowPt_Barrel", "NHFrac_lowPt_Barrel", 120, -0.1, 1.1);
      mPhFrac_lowPt_Barrel     = dbe->book1D("PhFrac_lowPt_Barrel", "PhFrac_lowPt_Barrel", 120, -0.1, 1.1);
      mElFrac_lowPt_Barrel     = dbe->book1D("ElFrac_lowPt_Barrel", "ElFrac_lowPt_Barrel", 120, -0.1, 1.1);
      mMuFrac_lowPt_Barrel     = dbe->book1D("MuFrac_lowPt_Barrel", "MuFrac_lowPt_Barrel", 120, -0.1, 1.1);
      mCHFrac_mediumPt_Barrel  = dbe->book1D("CHFrac_mediumPt_Barrel", "CHFrac_mediumPt_Barrel", 120, -0.1, 1.1);
      mNHFrac_mediumPt_Barrel  = dbe->book1D("NHFrac_mediumPt_Barrel", "NHFrac_mediumPt_Barrel", 120, -0.1, 1.1);
      mPhFrac_mediumPt_Barrel  = dbe->book1D("PhFrac_mediumPt_Barrel", "PhFrac_mediumPt_Barrel", 120, -0.1, 1.1);
      mElFrac_mediumPt_Barrel  = dbe->book1D("ElFrac_mediumPt_Barrel", "ElFrac_mediumPt_Barrel", 120, -0.1, 1.1);
      mMuFrac_mediumPt_Barrel  = dbe->book1D("MuFrac_mediumPt_Barrel", "MuFrac_mediumPt_Barrel", 120, -0.1, 1.1);
      mCHFrac_highPt_Barrel    = dbe->book1D("CHFrac_highPt_Barrel", "CHFrac_highPt_Barrel", 120, -0.1, 1.1);
      mNHFrac_highPt_Barrel    = dbe->book1D("NHFrac_highPt_Barrel", "NHFrac_highPt_Barrel", 120, -0.1, 1.1);
      mPhFrac_highPt_Barrel    = dbe->book1D("PhFrac_highPt_Barrel", "PhFrac_highPt_Barrel", 120, -0.1, 1.1);
      mElFrac_highPt_Barrel    = dbe->book1D("ElFrac_highPt_Barrel", "ElFrac_highPt_Barrel", 120, -0.1, 1.1);
      mMuFrac_highPt_Barrel    = dbe->book1D("MuFrac_highPt_Barrel", "MuFrac_highPt_Barrel", 120, -0.1, 1.1);
      
      //energies
      mCHEn_lowPt_Barrel     = dbe->book1D("CHEn_lowPt_Barrel", "CHEn_lowPt_Barrel", ptBin, ptMin, ptMax);
      mNHEn_lowPt_Barrel     = dbe->book1D("NHEn_lowPt_Barrel", "NHEn_lowPt_Barrel", ptBin, ptMin, ptMax);
      mPhEn_lowPt_Barrel     = dbe->book1D("PhEn_lowPt_Barrel", "PhEn_lowPt_Barrel", ptBin, ptMin, ptMax);
      mElEn_lowPt_Barrel     = dbe->book1D("ElEn_lowPt_Barrel", "ElEn_lowPt_Barrel", ptBin, ptMin, ptMax);
      mMuEn_lowPt_Barrel     = dbe->book1D("MuEn_lowPt_Barrel", "MuEn_lowPt_Barrel", ptBin, ptMin, ptMax);
      mCHEn_mediumPt_Barrel  = dbe->book1D("CHEn_mediumPt_Barrel", "CHEn_mediumPt_Barrel", ptBin, ptMin, ptMax);
      mNHEn_mediumPt_Barrel  = dbe->book1D("NHEn_mediumPt_Barrel", "NHEn_mediumPt_Barrel", ptBin, ptMin, ptMax);
      mPhEn_mediumPt_Barrel  = dbe->book1D("PhEn_mediumPt_Barrel", "PhEn_mediumPt_Barrel", ptBin, ptMin, ptMax);
      mElEn_mediumPt_Barrel  = dbe->book1D("ElEn_mediumPt_Barrel", "ElEn_mediumPt_Barrel", ptBin, ptMin, ptMax);
      mMuEn_mediumPt_Barrel  = dbe->book1D("MuEn_mediumPt_Barrel", "MuEn_mediumPt_Barrel", ptBin, ptMin, ptMax);
      mCHEn_highPt_Barrel    = dbe->book1D("CHEn_highPt_Barrel", "CHEn_highPt_Barrel", ptBin, ptMin, ptMax);
      mNHEn_highPt_Barrel    = dbe->book1D("NHEn_highPt_Barrel", "NHEn_highPt_Barrel", ptBin, ptMin, ptMax);
      mPhEn_highPt_Barrel    = dbe->book1D("PhEn_highPt_Barrel", "PhEn_highPt_Barrel", ptBin, ptMin, ptMax);
      mElEn_highPt_Barrel    = dbe->book1D("ElEn_highPt_Barrel", "ElEn_highPt_Barrel", ptBin, ptMin, ptMax);
      mMuEn_highPt_Barrel    = dbe->book1D("MuEn_highPt_Barrel", "MuEn_highPt_Barrel", ptBin, ptMin, ptMax);
      //multiplicities
      mChMultiplicity_lowPt_Barrel    = dbe->book1D("ChMultiplicity_lowPt_Barrel", "ChMultiplicity_lowPt_Barrel", 30,0,30);
      mNeuMultiplicity_lowPt_Barrel   = dbe->book1D("NeuMultiplicity_lowPt_Barrel", "NeuMultiplicity_lowPt_Barrel", 30,0,30);
      mMuMultiplicity_lowPt_Barrel    = dbe->book1D("MuMultiplicity_lowPt_Barrel", "MuMultiplicity_lowPt_Barrel", 30,0,30);
      mChMultiplicity_mediumPt_Barrel    = dbe->book1D("ChMultiplicity_mediumPt_Barrel", "ChMultiplicity_mediumPt_Barrel", 30,0,30);
      mNeuMultiplicity_mediumPt_Barrel   = dbe->book1D("NeuMultiplicity_mediumPt_Barrel", "NeuMultiplicity_mediumPt_Barrel", 30,0,30);
      mMuMultiplicity_mediumPt_Barrel    = dbe->book1D("MuMultiplicity_mediumPt_Barrel", "MuMultiplicity_mediumPt_Barrel", 30,0,30);
      mChMultiplicity_highPt_Barrel    = dbe->book1D("ChMultiplicity_highPt_Barrel", "ChMultiplicity_highPt_Barrel", 30,0,30);
      mNeuMultiplicity_highPt_Barrel   = dbe->book1D("NeuMultiplicity_highPt_Barrel", "NeuMultiplicity_highPt_Barrel", 30,0,30);
      mMuMultiplicity_highPt_Barrel    = dbe->book1D("MuMultiplicity_highPt_Barrel", "MuMultiplicity_highPt_Barrel", 30,0,30);
      //
      mCHFracVSpT_Barrel= dbe->bookProfile("CHFracVSpT_Barrel","CHFracVSpT_Barrel",ptBin, ptMin, ptMax,0.,1.2);
      mNHFracVSpT_Barrel= dbe->bookProfile("NHFracVSpT_Barrel","NHFracVSpT_Barrel",ptBin, ptMin, ptMax,0.,1.2);
      mPhFracVSpT_Barrel= dbe->bookProfile("PhFracVSpT_Barrel","PhFracVSpT_Barrel",ptBin, ptMin, ptMax,0.,1.2);
      mElFracVSpT_Barrel= dbe->bookProfile("ElFracVSpT_Barrel","ElFracVSpT_Barrel",ptBin, ptMin, ptMax,0.,1.2);
      mMuFracVSpT_Barrel= dbe->bookProfile("MuFracVSpT_Barrel","MuFracVSpT_Barrel",ptBin, ptMin, ptMax,0.,1.2);
      mCHFracVSpT_EndCap= dbe->bookProfile("CHFracVSpT_EndCap","CHFracVSpT_EndCap",ptBin, ptMin, ptMax,0.,1.2);
      mNHFracVSpT_EndCap= dbe->bookProfile("NHFracVSpT_EndCap","NHFracVSpT_EndCap",ptBin, ptMin, ptMax,0.,1.2);
      mPhFracVSpT_EndCap= dbe->bookProfile("PhFracVSpT_EndCap","PhFracVSpT_EndCap",ptBin, ptMin, ptMax,0.,1.2);
      mElFracVSpT_EndCap= dbe->bookProfile("ElFracVSpT_EndCap","ElFracVSpT_EndCap",ptBin, ptMin, ptMax,0.,1.2);
      mMuFracVSpT_EndCap= dbe->bookProfile("MuFracVSpT_EndCap","MuFracVSpT_EndCap",ptBin, ptMin, ptMax,0.,1.2);
      mHFHFracVSpT_Forward= dbe->bookProfile("HFHFracVSpT_Forward","HFHFracVSpT_Forward",ptBin, ptMin, ptMax,0.,1.2);
      mHFEFracVSpT_Forward= dbe->bookProfile("HFEFracVSpT_Forward","HFEFracVSpT_Forward",ptBin, ptMin, ptMax,0.,1.2);
      //endcap monitoring
      //energy fractions
      mCHFrac_lowPt_EndCap     = dbe->book1D("CHFrac_lowPt_EndCap", "CHFrac_lowPt_EndCap", 120, -0.1, 1.1);
      mNHFrac_lowPt_EndCap     = dbe->book1D("NHFrac_lowPt_EndCap", "NHFrac_lowPt_EndCap", 120, -0.1, 1.1);
      mPhFrac_lowPt_EndCap     = dbe->book1D("PhFrac_lowPt_EndCap", "PhFrac_lowPt_EndCap", 120, -0.1, 1.1);
      mElFrac_lowPt_EndCap     = dbe->book1D("ElFrac_lowPt_EndCap", "ElFrac_lowPt_EndCap", 120, -0.1, 1.1);
      mMuFrac_lowPt_EndCap     = dbe->book1D("MuFrac_lowPt_EndCap", "MuFrac_lowPt_EndCap", 120, -0.1, 1.1);
      mCHFrac_mediumPt_EndCap  = dbe->book1D("CHFrac_mediumPt_EndCap", "CHFrac_mediumPt_EndCap", 120, -0.1, 1.1);
      mNHFrac_mediumPt_EndCap  = dbe->book1D("NHFrac_mediumPt_EndCap", "NHFrac_mediumPt_EndCap", 120, -0.1, 1.1);
      mPhFrac_mediumPt_EndCap  = dbe->book1D("PhFrac_mediumPt_EndCap", "PhFrac_mediumPt_EndCap", 120, -0.1, 1.1);
      mElFrac_mediumPt_EndCap  = dbe->book1D("ElFrac_mediumPt_EndCap", "ElFrac_mediumPt_EndCap", 120, -0.1, 1.1);
      mMuFrac_mediumPt_EndCap  = dbe->book1D("MuFrac_mediumPt_EndCap", "MuFrac_mediumPt_EndCap", 120, -0.1, 1.1);
      mCHFrac_highPt_EndCap    = dbe->book1D("CHFrac_highPt_EndCap", "CHFrac_highPt_EndCap", 120, -0.1, 1.1);
      mNHFrac_highPt_EndCap    = dbe->book1D("NHFrac_highPt_EndCap", "NHFrac_highPt_EndCap", 120, -0.1, 1.1);
      mPhFrac_highPt_EndCap    = dbe->book1D("PhFrac_highPt_EndCap", "PhFrac_highPt_EndCap", 120, -0.1, 1.1);
      mElFrac_highPt_EndCap    = dbe->book1D("ElFrac_highPt_EndCap", "ElFrac_highPt_EndCap", 120, -0.1, 1.1);
      mMuFrac_highPt_EndCap    = dbe->book1D("MuFrac_highPt_EndCap", "MuFrac_highPt_EndCap", 120, -0.1, 1.1);
      //energies
      mCHEn_lowPt_EndCap     = dbe->book1D("CHEn_lowPt_EndCap", "CHEn_lowPt_EndCap", ptBin, ptMin, ptMax);
      mNHEn_lowPt_EndCap     = dbe->book1D("NHEn_lowPt_EndCap", "NHEn_lowPt_EndCap", ptBin, ptMin, ptMax);
      mPhEn_lowPt_EndCap     = dbe->book1D("PhEn_lowPt_EndCap", "PhEn_lowPt_EndCap", ptBin, ptMin, ptMax);
      mElEn_lowPt_EndCap     = dbe->book1D("ElEn_lowPt_EndCap", "ElEn_lowPt_EndCap", ptBin, ptMin, ptMax);
      mMuEn_lowPt_EndCap     = dbe->book1D("MuEn_lowPt_EndCap", "MuEn_lowPt_EndCap", ptBin, ptMin, ptMax);
      mCHEn_mediumPt_EndCap  = dbe->book1D("CHEn_mediumPt_EndCap", "CHEn_mediumPt_EndCap", ptBin, ptMin, ptMax);
      mNHEn_mediumPt_EndCap  = dbe->book1D("NHEn_mediumPt_EndCap", "NHEn_mediumPt_EndCap", ptBin, ptMin, ptMax);
      mPhEn_mediumPt_EndCap  = dbe->book1D("PhEn_mediumPt_EndCap", "PhEn_mediumPt_EndCap", ptBin, ptMin, ptMax);
      mElEn_mediumPt_EndCap  = dbe->book1D("ElEn_mediumPt_EndCap", "ElEn_mediumPt_EndCap", ptBin, ptMin, ptMax);
      mMuEn_mediumPt_EndCap  = dbe->book1D("MuEn_mediumPt_EndCap", "MuEn_mediumPt_EndCap", ptBin, ptMin, ptMax);
      mCHEn_highPt_EndCap    = dbe->book1D("CHEn_highPt_EndCap", "CHEn_highPt_EndCap", ptBin, ptMin, ptMax);
      mNHEn_highPt_EndCap    = dbe->book1D("NHEn_highPt_EndCap", "NHEn_highPt_EndCap", ptBin, ptMin, ptMax);
      mPhEn_highPt_EndCap    = dbe->book1D("PhEn_highPt_EndCap", "PhEn_highPt_EndCap", ptBin, ptMin, ptMax);
      mElEn_highPt_EndCap    = dbe->book1D("ElEn_highPt_EndCap", "ElEn_highPt_EndCap", ptBin, ptMin, ptMax);
      mMuEn_highPt_EndCap    = dbe->book1D("MuEn_highPt_EndCap", "MuEn_highPt_EndCap", ptBin, ptMin, ptMax);
      //multiplicities
      mChMultiplicity_lowPt_EndCap    = dbe->book1D("ChMultiplicity_lowPt_EndCap", "ChMultiplicity_lowPt_EndCap", 30,0,30);
      mNeuMultiplicity_lowPt_EndCap   = dbe->book1D("NeuMultiplicity_lowPt_EndCap", "NeuMultiplicity_lowPt_EndCap", 30,0,30);
      mMuMultiplicity_lowPt_EndCap    = dbe->book1D("MuMultiplicity_lowPt_EndCap", "MuMultiplicity_lowPt_EndCap", 30,0,30);
      mChMultiplicity_mediumPt_EndCap    = dbe->book1D("ChMultiplicity_mediumPt_EndCap", "ChMultiplicity_mediumPt_EndCap", 30,0,30);
      mNeuMultiplicity_mediumPt_EndCap   = dbe->book1D("NeuMultiplicity_mediumPt_EndCap", "NeuMultiplicity_mediumPt_EndCap", 30,0,30);
      mMuMultiplicity_mediumPt_EndCap    = dbe->book1D("MuMultiplicity_mediumPt_EndCap", "MuMultiplicity_mediumPt_EndCap", 30,0,30);
      mChMultiplicity_highPt_EndCap    = dbe->book1D("ChMultiplicity_highPt_EndCap", "ChMultiplicity_highPt_EndCap", 30,0,30);
      mNeuMultiplicity_highPt_EndCap   = dbe->book1D("NeuMultiplicity_highPt_EndCap", "NeuMultiplicity_highPt_EndCap", 30,0,30);
      mMuMultiplicity_highPt_EndCap    = dbe->book1D("MuMultiplicity_highPt_EndCap", "MuMultiplicity_highPt_EndCap", 30,0,30);
      //forward monitoring
      //energy fraction
      mHFEFrac_lowPt_Forward    = dbe->book1D("HFEFrac_lowPt_Forward", "HFEFrac_lowPt_Forward", 120, -0.1, 1.1);
      mHFHFrac_lowPt_Forward    = dbe->book1D("HFHFrac_lowPt_Forward", "HFHFrac_lowPt_Forward", 120, -0.1, 1.1);
      mHFEFrac_mediumPt_Forward = dbe->book1D("HFEFrac_mediumPt_Forward", "HFEFrac_mediumPt_Forward", 120, -0.1, 1.1);
      mHFHFrac_mediumPt_Forward = dbe->book1D("HFHFrac_mediumPt_Forward", "HFHFrac_mediumPt_Forward", 120, -0.1, 1.1);
      mHFEFrac_highPt_Forward   = dbe->book1D("HFEFrac_highPt_Forward", "HFEFrac_highPt_Forward", 120, -0.1, 1.1);
      mHFHFrac_highPt_Forward   = dbe->book1D("HFHFrac_highPt_Forward", "HFHFrac_highPt_Forward", 120, -0.1, 1.1);
      //energies
      mHFEEn_lowPt_Forward    = dbe->book1D("HFEEn_lowPt_Forward", "HFEEn_lowPt_Forward", ptBin, ptMin, ptMax);
      mHFHEn_lowPt_Forward    = dbe->book1D("HFHEn_lowPt_Forward", "HFHEn_lowPt_Forward", ptBin, ptMin, ptMax);
      mHFEEn_mediumPt_Forward = dbe->book1D("HFEEn_mediumPt_Forward", "HFEEn_mediumPt_Forward", ptBin, ptMin, ptMax);
      mHFHEn_mediumPt_Forward = dbe->book1D("HFHEn_mediumPt_Forward", "HFHEn_mediumPt_Forward", ptBin, ptMin, ptMax);
      mHFEEn_highPt_Forward   = dbe->book1D("HFEEn_highPt_Forward", "HFEEn_highPt_Forward", ptBin, ptMin, ptMax);
      mHFHEn_highPt_Forward   = dbe->book1D("HFHEn_highPt_Forward", "HFHEn_highPt_Forward", ptBin, ptMin, ptMax);
      //multiplicities
      mChMultiplicity_lowPt_Forward     = dbe->book1D("ChMultiplicity_lowPt_Forward", "ChMultiplicity_lowPt_Forward", 30,0,30);
      mNeuMultiplicity_lowPt_Forward    = dbe->book1D("NeuMultiplicity_lowPt_Forward", "NeuMultiplicity_lowPt_Forward", 30,0,30);
      mMuMultiplicity_lowPt_Forward     = dbe->book1D("MuMultiplicity_lowPt_Forward", "MuMultiplicity_lowPt_Forward", 30,0,30);
      mChMultiplicity_mediumPt_Forward  = dbe->book1D("ChMultiplicity_mediumPt_Forward", "ChMultiplicity_mediumPt_Forward", 30,0,30);
      mNeuMultiplicity_mediumPt_Forward = dbe->book1D("NeuMultiplicity_mediumPt_Forward", "NeuMultiplicity_mediumPt_Forward", 30,0,30);
      mMuMultiplicity_mediumPt_Forward  = dbe->book1D("MuMultiplicity_mediumPt_Forward", "MuMultiplicity_mediumPt_Forward", 30,0,30);
      mChMultiplicity_highPt_Forward    = dbe->book1D("ChMultiplicity_highPt_Forward", "ChMultiplicity_highPt_Forward", 30,0,30);
      mNeuMultiplicity_highPt_Forward   = dbe->book1D("NeuMultiplicity_highPt_Forward", "NeuMultiplicity_highPt_Forward", 30,0,30);
      mMuMultiplicity_highPt_Forward    = dbe->book1D("MuMultiplicity_highPt_Forward", "MuMultiplicity_highPt_Forward", 30,0,30);
    }
    
    mChargedHadronEnergy = dbe->book1D("mChargedHadronEnergy", "charged HAD energy",    100, 0, 100);
    mNeutralHadronEnergy = dbe->book1D("mNeutralHadronEnergy", "neutral HAD energy",    100, 0, 100);
    mChargedEmEnergy     = dbe->book1D("mChargedEmEnergy ",    "charged EM energy ",    100, 0, 100);
    mChargedMuEnergy     = dbe->book1D("mChargedMuEnergy",     "charged Mu energy",     100, 0, 100);
    mNeutralEmEnergy     = dbe->book1D("mNeutralEmEnergy",     "neutral EM energy",     100, 0, 100);
    mChargedMultiplicity = dbe->book1D("mChargedMultiplicity", "charged multiplicity ", 100, 0, 100);
    mNeutralMultiplicity = dbe->book1D("mNeutralMultiplicity", "neutral multiplicity",  100, 0, 100);
    mMuonMultiplicity    = dbe->book1D("mMuonMultiplicity",    "muon multiplicity",     100, 0, 100);
    
    
    // Book NPV profiles
    //----------------------------------------------------------------------------
    mChargedHadronEnergy_profile = dbe->bookProfile("mChargedHadronEnergy_profile", "charged HAD energy",   nbinsPV, PVlow, PVhigh, 100, 0, 100);
    mNeutralHadronEnergy_profile = dbe->bookProfile("mNeutralHadronEnergy_profile", "neutral HAD energy",   nbinsPV, PVlow, PVhigh, 100, 0, 100);
    mChargedEmEnergy_profile     = dbe->bookProfile("mChargedEmEnergy_profile",     "charged EM energy",    nbinsPV, PVlow, PVhigh, 100, 0, 100);
    mChargedMuEnergy_profile     = dbe->bookProfile("mChargedMuEnergy_profile",     "charged Mu energy",    nbinsPV, PVlow, PVhigh, 100, 0, 100);
    mNeutralEmEnergy_profile     = dbe->bookProfile("mNeutralEmEnergy_profile",     "neutral EM energy",    nbinsPV, PVlow, PVhigh, 100, 0, 100);
    mChargedMultiplicity_profile = dbe->bookProfile("mChargedMultiplicity_profile", "charged multiplicity", nbinsPV, PVlow, PVhigh, 100, 0, 100);
    mNeutralMultiplicity_profile = dbe->bookProfile("mNeutralMultiplicity_profile", "neutral multiplicity", nbinsPV, PVlow, PVhigh, 100, 0, 100);
    mMuonMultiplicity_profile    = dbe->bookProfile("mMuonMultiplicity_profile",    "muon multiplicity",    nbinsPV, PVlow, PVhigh, 100, 0, 100);
    
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
    
    mNeutralFraction     = dbe->book1D("NeutralConstituentsFraction","Neutral Constituens Fraction",100,0,1);

  }

  dbe->setCurrentFolder("JetMET");
  lumisecME = dbe->book1D("lumisec", "lumisec", 500, 0., 500.);
  cleanupME = dbe->book1D("cleanup", "cleanup", 10, 0., 10.);
  cleanupME->setBinLabel(1,"Primary Vertex");
  cleanupME->setBinLabel(2,"DCS::Pixel");
  cleanupME->setBinLabel(3,"DCS::SiStrip");
  cleanupME->setBinLabel(4,"DCS::ECAL");
  cleanupME->setBinLabel(5,"DCS::ES");
  cleanupME->setBinLabel(6,"DCS::HBHE");
  cleanupME->setBinLabel(7,"DCS::HF");
  cleanupME->setBinLabel(8,"DCS::HO");
  cleanupME->setBinLabel(9,"DCS::Muon");

  verticesME = dbe->book1D("vertices", "vertices", 50, 0, 50);


}

// ***********************************************************
void JetAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  //LogDebug("JetAnalyzer") << "beginRun, run " << run.id();
  //

  if ( _HighPtJetEventFlag->on() ) _HighPtJetEventFlag->initRun( iRun, iSetup );
  if ( _LowPtJetEventFlag ->on() ) _LowPtJetEventFlag ->initRun( iRun, iSetup );

  if (_HighPtJetEventFlag->on() && _HighPtJetEventFlag->expressionsFromDB(_HighPtJetEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highPtJetExpr_ = _HighPtJetEventFlag->expressionsFromDB(_HighPtJetEventFlag->hltDBKey(), iSetup);
  if (_LowPtJetEventFlag->on() && _LowPtJetEventFlag->expressionsFromDB(_LowPtJetEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    lowPtJetExpr_  = _LowPtJetEventFlag->expressionsFromDB(_LowPtJetEventFlag->hltDBKey(),   iSetup);
  if (!JetCorrectionService.empty()){
    energycorrected=true;
  }
  //--- htlConfig_
  //processname_="HLT";
  bool changed(true);
  _hlt_initialized = hltConfig_.init(iRun,iSetup,processname_,changed);
  if (!_hlt_initialized) {
  //if (!hltConfig_.init(iRun,iSetup,processname_,changed)) {
    processname_ = "FU";
    _hlt_initialized = hltConfig_.init(iRun,iSetup,processname_,changed);
    if(!_hlt_initialized){
      //if (!hltConfig_.init(iRun,iSetup,processname_,changed)){
      LogDebug("JetAnalyzer") << "HLTConfigProvider failed to initialize.";
    }
  }
 
}

// ***********************************************************
void JetAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
}

// ***********************************************************
void JetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // *** Fill lumisection ME
  int myLuminosityBlock;
  myLuminosityBlock = iEvent.luminosityBlock();
  lumisecME->Fill(myLuminosityBlock);

  if (myLuminosityBlock<_LSBegin) return;
  if (myLuminosityBlock>_LSEnd && _LSEnd>0) return;

  // **** Get the TriggerResults container
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  
  //bool bPhysicsDeclared = false;
  //if(bPhysicsDeclared!=false){
  //std::cout<<"wrong"<<std::endl;
  //}
  //if(!_doHLTPhysicsOn) bPhysicsDeclared = true;
  
  Int_t JetLoPass = 0;
  Int_t JetHiPass = 0;

  if (triggerResults.isValid()){
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);

    //if( triggerNames.triggerIndex("HLT_PhysicsDeclared") != triggerNames.size() )
    //if (triggerResults->accept(triggerNames.triggerIndex("HLT_PhysicsDeclared"))) {
    //	if(_doHLTPhysicsOn) bPhysicsDeclared = true;
    //}

    const unsigned int nTrig(triggerNames.size());
    for (unsigned int i=0;i<nTrig;++i)
      {
        if (triggerNames.triggerName(i).find(highPtJetExpr_[0].substr(0,highPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults->accept(i))
	  JetHiPass=1;
        else if (triggerNames.triggerName(i).find(lowPtJetExpr_[0].substr(0,lowPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults->accept(i))
	  JetLoPass=1;
      }

  }

  if (DEBUG)  std::cout << "trigger label " << theTriggerResultsLabel << std::endl;


  if (DEBUG) {
    std::cout << ">>> Trigger  Lo = " <<  JetLoPass
	      <<             " Hi = " <<  JetHiPass
	      << std::endl;
  }

  // ==========================================================
  //Vertex information

  int _numPV = 0;
  bool bPrimaryVertex = true;
  if(_doPVCheck){
    bPrimaryVertex = false;
    Handle<reco::VertexCollection> vertexHandle;
    iEvent.getByToken(vertexToken_, vertexHandle);

    if (!vertexHandle.isValid()) {
      LogInfo("JetAnalyzer") << "JetAnalyzer: Could not find vertex collection" << std::endl;
      if (DEBUG) std::cout << "JetAnalyzer: Could not find vertex collection" << std::endl;
    }

    if ( vertexHandle.isValid() ){
      reco::VertexCollection vertexCollection = *(vertexHandle.product());
      int vertex_number     = vertexCollection.size();
      reco::VertexCollection::const_iterator v = vertexCollection.begin();
      for ( ; v != vertexCollection.end(); ++v) {
	double vertex_chi2    = v->normalizedChi2();
	double vertex_ndof    = v->ndof();
	bool   fakeVtx        = v->isFake();
	double vertex_Z       = v->z();

	if (  !fakeVtx
	      && vertex_number>=_nvtx_min
	      && vertex_ndof   >_vtxndof_min
	      && vertex_chi2   <_vtxchi2_max
	      && fabs(vertex_Z)<_vtxz_max ) {
	  bPrimaryVertex = true;
	  ++_numPV;
	}
      }
    }
  }
  verticesME->Fill(_numPV);
  // ==========================================================


  edm::Handle<L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  iEvent.getByToken(gtToken_, gtReadoutRecord);


  if (!gtReadoutRecord.isValid()) {
    LogInfo("JetAnalyzer") << "JetAnalyzer: Could not find GT readout record" << std::endl;
    if (DEBUG) std::cout << "JetAnalyzer: Could not find GT readout record product" << std::endl;
  }
  DCSFilterAll->filter(iEvent, iSetup);
  if (bPrimaryVertex) cleanupME->Fill(0.5);
  if ( DCSFilterAll->passPIX      ) cleanupME->Fill(1.5);
  if ( DCSFilterAll->passSiStrip  ) cleanupME->Fill(2.5);
  if ( DCSFilterAll->passECAL     ) cleanupME->Fill(3.5);
  if ( DCSFilterAll->passES       ) cleanupME->Fill(4.5);
  if ( DCSFilterAll->passHBHE     ) cleanupME->Fill(5.5);
  if ( DCSFilterAll->passHF       ) cleanupME->Fill(6.5);
  if ( DCSFilterAll->passHO       ) cleanupME->Fill(7.5);
  if ( DCSFilterAll->passMuon     ) cleanupME->Fill(8.5);

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection> jptJets;
  edm::Handle<PFJetCollection> pfJets;



  if (isCaloJet) iEvent.getByToken(caloJetsToken_, caloJets);
  if (isJPTJet) iEvent.getByToken(jptJetsToken_, jptJets);
  if (isPFJet) iEvent.getByToken(pfJetsToken_, pfJets);


  //take already primary vertex in
  //check for collections AND DCS filters
  if (isCaloJet && !caloJets.isValid()) return;
  if (isJPTJet && !jptJets.isValid()) return;
  if (isPFJet && !pfJets.isValid()) return;
  if(theJetCleaningFlag){
    if (isCaloJet && (!caloJets.isValid() || !bPrimaryVertex || !(DCSFilterCalo->filter(iEvent, iSetup)))) return;
    if (isJPTJet && (!jptJets.isValid() || !bPrimaryVertex || !(DCSFilterJPT->filter(iEvent, iSetup)))) return;
    if (isPFJet && (!pfJets.isValid() || !bPrimaryVertex || !(DCSFilterPF->filter(iEvent, iSetup)))) return;
  }
  //recheck everything here -> pfjets wrong at the moment
  if(theDiJetSelectionFlag){
    if (isCaloJet && (!caloJets.isValid() || !bPrimaryVertex || !(DCSFilterCalo->filter(iEvent, iSetup)))) return;
    if (isJPTJet && (!jptJets.isValid() || !bPrimaryVertex || !(DCSFilterJPT->filter(iEvent, iSetup)))) return;
    if (isPFJet && (!pfJets.isValid() || !bPrimaryVertex || !(DCSFilterPF->filter(iEvent, iSetup)))) return;
  }
  unsigned int collSize=-1;
  


  if (isCaloJet)
    {
      collSize=caloJets->size();
      //for (unsigned ijet=0; ijet<caloJets->size(); ijet++)
      //recoJets.push_back((*caloJets)[ijet]);
    }
  
  if (isJPTJet)
    {
      collSize=jptJets->size();
      if(collSize>0){
	//update the track propagator and strip noise calculator
	trackPropagator_->update(iSetup);
	//sOverNCalculator_->update(iSetup);
      }
      //for (unsigned ijet=0; ijet<jptJets->size(); ijet++)
      //recoJets.push_back((*jptJets)[ijet]);
    }
  
  if (isPFJet) {
    collSize=pfJets->size();
    //for (unsigned ijet=0; ijet<pfJets->size(); ijet++)
    //recoJets.push_back((*pfJets)[ijet]);
  }
  

  double scale=-1;
  
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

  //now start changes for jets
  std::vector<Jet> recoJets;
  recoJets.clear();

  int numofjets=0;

  for (unsigned ijet=0; ijet<collSize; ijet++) {
    //bool thiscleaned=false;
    Jet correctedJet;
    bool pass_uncorrected=false;
    bool pass_corrected=false;
    if (isCaloJet){
      correctedJet=(*caloJets)[ijet];
    }
    if (isJPTJet){
      correctedJet=(*jptJets)[ijet];
    }
    if (isPFJet){
      correctedJet=(*pfJets)[ijet];
    }
    if(correctedJet.pt()>_ptThresholdUnc){
      pass_uncorrected=true;
    }
    if (!JetCorrectionService.empty())
      {
	const JetCorrector* corrector = JetCorrector::getJetCorrector(JetCorrectionService, iSetup);
	//for (unsigned ijet=0; ijet<recoJets.size(); ijet++) {
	
	if (isCaloJet){
	  scale = corrector->correction((*caloJets)[ijet], iEvent, iSetup);
	}
	if (isJPTJet){
	  scale = corrector->correction((*jptJets)[ijet], iEvent, iSetup);
	}
	if (isPFJet){ 
	  scale = corrector->correction((*pfJets)[ijet], iEvent, iSetup);
	}
	correctedJet.scaleEnergy(scale);	    
      }


    if(correctedJet.pt()>pt1){
      pt3=pt2;
      ind3=ind2;
      pt2=pt1;
      ind2=ind1;
      pt1=correctedJet.pt();
      ind1=ijet;
    }else if(correctedJet.pt()>pt2){
      pt3=pt2;
      ind3=ind2;
      pt2=correctedJet.pt();
      ind2=ijet;
    }else if(correctedJet.pt()>pt3){
      pt3=correctedJet.pt();
      ind3=ijet;
    }

    if(correctedJet.pt()> _ptThreshold){
      pass_corrected=true;
    }
    
    if (!pass_corrected && !pass_uncorrected) continue;
    //fill only corrected jets -> check ID for uncorrected jets
    if(pass_corrected){
      recoJets.push_back(correctedJet);
    }
    if(makedijetselection!=1){
      bool Thiscleaned=false;
      bool Loosecleaned=false;
      bool Tightcleaned=false;
      //jet ID for calojets
      if (isCaloJet) {
	bool thisemfclean=true;
	bool emfcleanLoose=true;
	bool emfcleanTight=true;
	jetID->calculate(iEvent, (*caloJets)[ijet]);
	thisemfclean=true;
	if(jetID->restrictedEMF()<_resEMFMin && fabs((*caloJets)[ijet].eta())<2.6) thisemfclean=false;
	if(jetID->n90Hits()>=_n90HitsMin && jetID->fHPD()<_fHPDMax && thisemfclean) Thiscleaned=true;
	//loose and tight cleaning, used to fill the JetIDPAssFraction histos
	if(jetID->n90Hits()>=_n90HitsMinLoose && jetID->fHPD()<_fHPDMaxLoose && emfcleanLoose) Loosecleaned=true;
	if(jetID->n90Hits()>=_n90HitsMinTight && jetID->fHPD()<_fHPDMaxTight && sqrt((*caloJets)[ijet].etaetaMoment())>_sigmaEtaMinTight && sqrt((*caloJets)[ijet].phiphiMoment())>_sigmaPhiMinTight && emfcleanTight) Tightcleaned=true;
	if(Thiscleaned && pass_uncorrected){
	  if (mPt_uncor)   mPt_uncor->Fill ((*caloJets)[ijet].pt());
	  if (mEta_uncor)  mEta_uncor->Fill ((*caloJets)[ijet].eta());
	  if (mPhi_uncor)  mPhi_uncor->Fill ((*caloJets)[ijet].phi());
	  if(!isJPTJet){
	    if (mConstituents_uncor) mConstituents_uncor->Fill ((*caloJets)[ijet].nConstituents());
	  }
	}
	//now do calojet specific fractions and histograms ->H and E fracs
	if(Thiscleaned && pass_corrected){//at the moment softer than loose ID
	  if (mHFrac)        mHFrac->Fill ((*caloJets)[ijet].energyFractionHadronic());
	  if (mEFrac)        mEFrac->Fill ((*caloJets)[ijet].emEnergyFraction());
	  if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*caloJets)[ijet].energyFractionHadronic());
	  if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, (*caloJets)[ijet].emEnergyFraction());
	  if (fabs((*caloJets)[ijet].eta()) <= 1.3) {	
	    if (mHFrac_Barrel)           mHFrac_Barrel->Fill((*caloJets)[ijet].energyFractionHadronic());	
	    if (mEFrac_Barrel)           mEFrac_Barrel->Fill((*caloJets)[ijet].emEnergyFraction());	
	  }else if(fabs((*caloJets)[ijet].eta()) <3.0){
	    if (mHFrac_EndCap)           mHFrac_EndCap->Fill((*caloJets)[ijet].energyFractionHadronic());	
	    if (mEFrac_EndCap)           mEFrac_EndCap->Fill((*caloJets)[ijet].emEnergyFraction());
	  }else{
	    if (mHFrac_Forward)           mHFrac_Forward->Fill((*caloJets)[ijet].energyFractionHadronic());	
	    if (mEFrac_Forward)           mEFrac_Forward->Fill((*caloJets)[ijet].emEnergyFraction());
	  }
	  if (mMaxEInEmTowers)  mMaxEInEmTowers->Fill ((*caloJets)[ijet].maxEInEmTowers());
	  if (mMaxEInHadTowers) mMaxEInHadTowers->Fill ((*caloJets)[ijet].maxEInHadTowers());
	  
	  if (mHadEnergyInHO)   mHadEnergyInHO->Fill ((*caloJets)[ijet].hadEnergyInHO());
	  if (mHadEnergyInHB)   mHadEnergyInHB->Fill ((*caloJets)[ijet].hadEnergyInHB());
	  if (mHadEnergyInHF)   mHadEnergyInHF->Fill ((*caloJets)[ijet].hadEnergyInHF());
	  if (mHadEnergyInHE)   mHadEnergyInHE->Fill ((*caloJets)[ijet].hadEnergyInHE());
	  if (mEmEnergyInEB)    mEmEnergyInEB->Fill ((*caloJets)[ijet].emEnergyInEB());
	  if (mEmEnergyInEE)    mEmEnergyInEE->Fill ((*caloJets)[ijet].emEnergyInEE());
	  if (mEmEnergyInHF)    mEmEnergyInHF->Fill ((*caloJets)[ijet].emEnergyInHF());
	  
	  if (mN90Hits)         mN90Hits->Fill (jetID->n90Hits());
	  if (mfHPD)            mfHPD->Fill (jetID->fHPD());
	  if (mresEMF)         mresEMF->Fill (jetID->restrictedEMF());
	  if (mfRBX)            mfRBX->Fill (jetID->fRBX());
	}
      }
      if(isJPTJet){
	const edm::RefToBase<reco::Jet>&  rawJet = (*jptJets)[ijet].getCaloJetRef();
	//change that step here
	//check jet is correctable by JPT
	//if ( fabs(rawJet->eta()) > 2.1) return;
	
	try {
	  const reco::CaloJet *rawCaloJet = dynamic_cast<const reco::CaloJet*>(&*rawJet);
	  jetID->calculate(iEvent,*rawCaloJet);
	  bool thisemfclean=true;
	  bool emfcleanLoose=true;
	  bool emfcleanTight=true;
	  thisemfclean=true;
	  if(jetID->restrictedEMF()<_resEMFMin && fabs((*rawCaloJet).eta())<2.6) thisemfclean=false;
	  if(jetID->n90Hits()>=_n90HitsMin && jetID->fHPD()<_fHPDMax && thisemfclean) Thiscleaned=true;
	  //loose and tight cleaning, used to fill the JetIDPAssFraction histos
	  if(jetID->n90Hits()>=_n90HitsMinLoose && jetID->fHPD()<_fHPDMaxLoose && emfcleanLoose) Loosecleaned=true;
	  if(jetID->n90Hits()>=_n90HitsMinTight && jetID->fHPD()<_fHPDMaxTight && sqrt((*rawCaloJet).etaetaMoment())>_sigmaEtaMinTight && sqrt((*rawCaloJet).phiphiMoment())>_sigmaPhiMinTight && emfcleanTight) Tightcleaned=true;
	} catch (const std::bad_cast&) {
	  edm::LogError("JetPlusTrackDQM") << "Failed to cast raw jet to CaloJet. JPT Jet does not appear to have been built from a CaloJet. "
						<< "Histograms not filled. ";
	  return;
	}
	//plot JPT specific variables for <2.1 jets
	if(Thiscleaned && pass_uncorrected&&  ( fabs(rawJet->eta()) < 2.1) ){
	  if (mPt_uncor)   mPt_uncor->Fill ((*jptJets)[ijet].pt());
	  if (mEta_uncor)  mEta_uncor->Fill ((*jptJets)[ijet].eta());
	  if (mPhi_uncor)  mPhi_uncor->Fill ((*jptJets)[ijet].phi());
	  if(!isJPTJet){
	    if (mConstituents_uncor) mConstituents_uncor->Fill ((*jptJets)[ijet].nConstituents());
	  }
	}
	if(Thiscleaned &&  ( fabs(rawJet->eta()) < 2.1) && pass_corrected){
	  if (mHFrac)        mHFrac->Fill ((*jptJets)[ijet].chargedHadronEnergyFraction()+(*jptJets)[ijet].neutralHadronEnergyFraction());
	  //if (mEFrac)        mEFrac->Fill ((*jptJets)[ijet].chargedEmEnergyFraction() +(*jptJets)[ijet].neutralEmEnergyFraction());
	  if (mEFrac)        mEFrac->Fill (1.-(*jptJets)[ijet].chargedHadronEnergyFraction()-(*jptJets)[ijet].neutralHadronEnergyFraction());
	  if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*jptJets)[ijet].chargedHadronEnergyFraction()+(*jptJets)[ijet].neutralHadronEnergyFraction());
	  if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, 1.-(*jptJets)[ijet].chargedHadronEnergyFraction()-(*jptJets)[ijet].neutralHadronEnergyFraction());
	  if (fabs((*jptJets)[ijet].eta()) <= 1.3) {	
	    if (mHFrac_Barrel)           mHFrac_Barrel->Fill((*jptJets)[ijet].chargedHadronEnergyFraction()+(*jptJets)[ijet].neutralHadronEnergyFraction());	
	    if (mEFrac_Barrel)           mEFrac_Barrel->Fill(1.-(*jptJets)[ijet].chargedHadronEnergyFraction()-(*jptJets)[ijet].neutralHadronEnergyFraction());	
	  }else if(fabs((*jptJets)[ijet].eta()) <3.0){
	    if (mHFrac_EndCap)           mHFrac_EndCap->Fill((*jptJets)[ijet].chargedHadronEnergyFraction()+(*jptJets)[ijet].neutralHadronEnergyFraction());	
	    if (mEFrac_EndCap)           mEFrac_EndCap->Fill(1.-(*jptJets)[ijet].chargedHadronEnergyFraction()-(*jptJets)[ijet].neutralHadronEnergyFraction());
	  }else{
	    if (mHFrac_Forward)           mHFrac_Forward->Fill((*jptJets)[ijet].chargedHadronEnergyFraction()+(*jptJets)[ijet].neutralHadronEnergyFraction());	
	    if (mEFrac_Forward)           mEFrac_Forward->Fill(1.-(*jptJets)[ijet].chargedHadronEnergyFraction()-(*jptJets)[ijet].neutralHadronEnergyFraction());
	  }
	  if (mE) mE->Fill ((*jptJets)[ijet].energy());	
	  if (mPx) mE->Fill ((*jptJets)[ijet].px());	
	  if (mPy) mE->Fill ((*jptJets)[ijet].py());	
	  if (mPz) mE->Fill ((*jptJets)[ijet].pz());	
	  if (mP) mE->Fill ((*jptJets)[ijet].p());	
	  if (mEt) mE->Fill ((*jptJets)[ijet].et());
	  if(mnTracks) mnTracks->Fill((*jptJets)[ijet].chargedMultiplicity());
	  if(mnTracksVsJetPt) mnTracksVsJetPt->Fill(rawJet->pt(),(*jptJets)[ijet].chargedMultiplicity());

	  if(mnTracksVsJetEta)mnTracksVsJetEta->Fill(rawJet->eta(),(*jptJets)[ijet].chargedMultiplicity());
	  const reco::TrackRefVector& pionsInVertexInCalo = (*jptJets)[ijet].getPionsInVertexInCalo();
	  const reco::TrackRefVector& pionsInVertexOutCalo = (*jptJets)[ijet].getPionsInVertexOutCalo();
	  const reco::TrackRefVector& pionsOutVertexInCalo = (*jptJets)[ijet].getPionsOutVertexInCalo();
	  const reco::TrackRefVector& muonsInVertexInCalo = (*jptJets)[ijet].getMuonsInVertexInCalo();
	  const reco::TrackRefVector& muonsInVertexOutCalo = (*jptJets)[ijet].getMuonsInVertexOutCalo();
	  const reco::TrackRefVector& muonsOutVertexInCalo = (*jptJets)[ijet].getMuonsOutVertexInCalo();
	  const reco::TrackRefVector& electronsInVertexInCalo = (*jptJets)[ijet].getElecsInVertexInCalo();
	  const reco::TrackRefVector& electronsInVertexOutCalo = (*jptJets)[ijet].getElecsInVertexOutCalo();
	  const reco::TrackRefVector& electronsOutVertexInCalo = (*jptJets)[ijet].getElecsOutVertexInCalo();
	  if(mnallPionsTracksPerJet) mnallPionsTracksPerJet->Fill(pionsInVertexInCalo.size()+pionsInVertexOutCalo.size()+pionsOutVertexInCalo.size());
	  if(mnInVertexInCaloPionsTracksPerJet) mnInVertexInCaloPionsTracksPerJet->Fill(pionsInVertexInCalo.size());
	  if(mnOutVertexInCaloPionsTracksPerJet) mnOutVertexInCaloPionsTracksPerJet->Fill(pionsOutVertexInCalo.size());
	  if(mnInVertexOutCaloPionsTracksPerJet) mnInVertexOutCaloPionsTracksPerJet->Fill(pionsInVertexOutCalo.size());
	  for (reco::TrackRefVector::const_iterator iTrack = pionsInVertexInCalo.begin(); iTrack != pionsInVertexInCalo.end(); ++iTrack) {
	    if(mallPionsTracksPt)mallPionsTracksPt->Fill((*iTrack)->pt());
	    if(mallPionsTracksEta)mallPionsTracksEta->Fill((*iTrack)->eta());
	    if(mallPionsTracksPhi)mallPionsTracksPhi->Fill((*iTrack)->phi());
	    if(mallPionsTracksPtVsEta)mallPionsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mInVertexInCaloPionsTracksPt)mInVertexInCaloPionsTracksPt->Fill((*iTrack)->pt());
	    if(mInVertexInCaloPionsTracksEta)mInVertexInCaloPionsTracksEta->Fill((*iTrack)->eta());
	    if(mInVertexInCaloPionsTracksPhi)mInVertexInCaloPionsTracksPhi->Fill((*iTrack)->phi());
	    if(mInVertexInCaloPionsTracksPtVsEta)mInVertexInCaloPionsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mInCaloTrackDirectionJetDRHisto_)mInCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mInVertexTrackImpactPointJetDRHisto_)mInVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }
	  for (reco::TrackRefVector::const_iterator iTrack = pionsInVertexOutCalo.begin(); iTrack != pionsInVertexOutCalo.end(); ++iTrack) {
	    if(mallPionsTracksPt)mallPionsTracksPt->Fill((*iTrack)->pt());
	    if(mallPionsTracksEta)mallPionsTracksEta->Fill((*iTrack)->eta());
	    if(mallPionsTracksPhi)mallPionsTracksPhi->Fill((*iTrack)->phi());
	    if(mallPionsTracksPtVsEta)mallPionsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mInVertexOutCaloPionsTracksPt)mInVertexOutCaloPionsTracksPt->Fill((*iTrack)->pt());
	    if(mInVertexOutCaloPionsTracksEta)mInVertexOutCaloPionsTracksEta->Fill((*iTrack)->eta());
	    if(mInVertexOutCaloPionsTracksPhi)mInVertexOutCaloPionsTracksPhi->Fill((*iTrack)->phi());
	    if(mInVertexOutCaloPionsTracksPtVsEta)mInVertexOutCaloPionsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mOutCaloTrackDirectionJetDRHisto_)mOutCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mInVertexTrackImpactPointJetDRHisto_)mInVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }
	  for (reco::TrackRefVector::const_iterator iTrack = pionsOutVertexInCalo.begin(); iTrack != pionsOutVertexInCalo.end(); ++iTrack) {
	    if(mallPionsTracksPt)mallPionsTracksPt->Fill((*iTrack)->pt());
	    if(mallPionsTracksEta)mallPionsTracksEta->Fill((*iTrack)->eta());
	    if(mallPionsTracksPhi)mallPionsTracksPhi->Fill((*iTrack)->phi());
	    if(mallPionsTracksPtVsEta)mallPionsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mOutVertexInCaloPionsTracksPt)mOutVertexInCaloPionsTracksPt->Fill((*iTrack)->pt());
	    if(mOutVertexInCaloPionsTracksEta)mOutVertexInCaloPionsTracksEta->Fill((*iTrack)->eta());
	    if(mOutVertexInCaloPionsTracksPhi)mOutVertexInCaloPionsTracksPhi->Fill((*iTrack)->phi());
	    if(mOutVertexInCaloPionsTracksPtVsEta)mOutVertexInCaloPionsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mInCaloTrackDirectionJetDRHisto_)mInCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mOutVertexTrackImpactPointJetDRHisto_)mOutVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }//muon track histos

	  if(mnallMuonsTracksPerJet) mnallMuonsTracksPerJet->Fill(muonsInVertexInCalo.size()+muonsInVertexOutCalo.size()+muonsOutVertexInCalo.size());
	  if(mnInVertexInCaloMuonsTracksPerJet) mnInVertexInCaloMuonsTracksPerJet->Fill(muonsInVertexInCalo.size());
	  if(mnOutVertexInCaloMuonsTracksPerJet) mnOutVertexInCaloMuonsTracksPerJet->Fill(muonsOutVertexInCalo.size());
	  if(mnInVertexOutCaloMuonsTracksPerJet) mnInVertexOutCaloMuonsTracksPerJet->Fill(muonsInVertexOutCalo.size());
	  for (reco::TrackRefVector::const_iterator iTrack = muonsInVertexInCalo.begin(); iTrack != muonsInVertexInCalo.end(); ++iTrack) {
	    if(mallMuonsTracksPt)mallMuonsTracksPt->Fill((*iTrack)->pt());
	    if(mallMuonsTracksEta)mallMuonsTracksEta->Fill((*iTrack)->eta());
	    if(mallMuonsTracksPhi)mallMuonsTracksPhi->Fill((*iTrack)->phi());
	    if(mallMuonsTracksPtVsEta)mallMuonsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mInVertexInCaloMuonsTracksPt)mInVertexInCaloMuonsTracksPt->Fill((*iTrack)->pt());
	    if(mInVertexInCaloMuonsTracksEta)mInVertexInCaloMuonsTracksEta->Fill((*iTrack)->eta());
	    if(mInVertexInCaloMuonsTracksPhi)mInVertexInCaloMuonsTracksPhi->Fill((*iTrack)->phi());
	    if(mInVertexInCaloMuonsTracksPtVsEta)mInVertexInCaloMuonsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mInCaloTrackDirectionJetDRHisto_)mInCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mInVertexTrackImpactPointJetDRHisto_)mInVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }
	  for (reco::TrackRefVector::const_iterator iTrack = muonsInVertexOutCalo.begin(); iTrack != muonsInVertexOutCalo.end(); ++iTrack) {
	    if(mallMuonsTracksPt)mallMuonsTracksPt->Fill((*iTrack)->pt());
	    if(mallMuonsTracksEta)mallMuonsTracksEta->Fill((*iTrack)->eta());
	    if(mallMuonsTracksPhi)mallMuonsTracksPhi->Fill((*iTrack)->phi());
	    if(mallMuonsTracksPtVsEta)mallMuonsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mInVertexOutCaloMuonsTracksPt)mInVertexOutCaloMuonsTracksPt->Fill((*iTrack)->pt());
	    if(mInVertexOutCaloMuonsTracksEta)mInVertexOutCaloMuonsTracksEta->Fill((*iTrack)->eta());
	    if(mInVertexOutCaloMuonsTracksPhi)mInVertexOutCaloMuonsTracksPhi->Fill((*iTrack)->phi());
	    if(mInVertexOutCaloMuonsTracksPtVsEta)mInVertexOutCaloMuonsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mOutCaloTrackDirectionJetDRHisto_)mOutCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mInVertexTrackImpactPointJetDRHisto_)mInVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }
	  for (reco::TrackRefVector::const_iterator iTrack = muonsOutVertexInCalo.begin(); iTrack != muonsOutVertexInCalo.end(); ++iTrack) {
	    if(mallMuonsTracksPt)mallMuonsTracksPt->Fill((*iTrack)->pt());
	    if(mallMuonsTracksEta)mallMuonsTracksEta->Fill((*iTrack)->eta());
	    if(mallMuonsTracksPhi)mallMuonsTracksPhi->Fill((*iTrack)->phi());
	    if(mallMuonsTracksPtVsEta)mallMuonsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mOutVertexInCaloMuonsTracksPt)mOutVertexInCaloMuonsTracksPt->Fill((*iTrack)->pt());
	    if(mOutVertexInCaloMuonsTracksEta)mOutVertexInCaloMuonsTracksEta->Fill((*iTrack)->eta());
	    if(mOutVertexInCaloMuonsTracksPhi)mOutVertexInCaloMuonsTracksPhi->Fill((*iTrack)->phi());
	    if(mOutVertexInCaloMuonsTracksPtVsEta)mOutVertexInCaloMuonsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mInCaloTrackDirectionJetDRHisto_)mInCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mOutVertexTrackImpactPointJetDRHisto_)mOutVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }//electron track histos
	  if(mnallElectronsTracksPerJet) mnallElectronsTracksPerJet->Fill(electronsInVertexInCalo.size()+electronsInVertexOutCalo.size()+electronsOutVertexInCalo.size());
	  if(mnInVertexInCaloElectronsTracksPerJet) mnInVertexInCaloElectronsTracksPerJet->Fill(electronsInVertexInCalo.size());
	  if(mnOutVertexInCaloElectronsTracksPerJet) mnOutVertexInCaloElectronsTracksPerJet->Fill(electronsOutVertexInCalo.size());
	  if(mnInVertexOutCaloElectronsTracksPerJet) mnInVertexOutCaloElectronsTracksPerJet->Fill(electronsInVertexOutCalo.size());
	  for (reco::TrackRefVector::const_iterator iTrack = electronsInVertexInCalo.begin(); iTrack != electronsInVertexInCalo.end(); ++iTrack) {
	    if(mallElectronsTracksPt)mallElectronsTracksPt->Fill((*iTrack)->pt());
	    if(mallElectronsTracksEta)mallElectronsTracksEta->Fill((*iTrack)->eta());
	    if(mallElectronsTracksPhi)mallElectronsTracksPhi->Fill((*iTrack)->phi());
	    if(mallElectronsTracksPtVsEta)mallElectronsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mInVertexInCaloElectronsTracksPt)mInVertexInCaloElectronsTracksPt->Fill((*iTrack)->pt());
	    if(mInVertexInCaloElectronsTracksEta)mInVertexInCaloElectronsTracksEta->Fill((*iTrack)->eta());
	    if(mInVertexInCaloElectronsTracksPhi)mInVertexInCaloElectronsTracksPhi->Fill((*iTrack)->phi());
	    if(mInVertexInCaloElectronsTracksPtVsEta)mInVertexInCaloElectronsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mInCaloTrackDirectionJetDRHisto_)mInCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mInVertexTrackImpactPointJetDRHisto_)mInVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }
	  for (reco::TrackRefVector::const_iterator iTrack = electronsInVertexOutCalo.begin(); iTrack != electronsInVertexOutCalo.end(); ++iTrack) {
	    if(mallElectronsTracksPt)mallElectronsTracksPt->Fill((*iTrack)->pt());
	    if(mallElectronsTracksEta)mallElectronsTracksEta->Fill((*iTrack)->eta());
	    if(mallElectronsTracksPhi)mallElectronsTracksPhi->Fill((*iTrack)->phi());
	    if(mallElectronsTracksPtVsEta)mallElectronsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mInVertexOutCaloElectronsTracksPt)mInVertexOutCaloElectronsTracksPt->Fill((*iTrack)->pt());
	    if(mInVertexOutCaloElectronsTracksEta)mInVertexOutCaloElectronsTracksEta->Fill((*iTrack)->eta());
	    if(mInVertexOutCaloElectronsTracksPhi)mInVertexOutCaloElectronsTracksPhi->Fill((*iTrack)->phi());
	    if(mInVertexOutCaloElectronsTracksPtVsEta)mInVertexOutCaloElectronsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mOutCaloTrackDirectionJetDRHisto_)mOutCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mInVertexTrackImpactPointJetDRHisto_)mInVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }
	  for (reco::TrackRefVector::const_iterator iTrack = electronsOutVertexInCalo.begin(); iTrack != electronsOutVertexInCalo.end(); ++iTrack) {
	    if(mallElectronsTracksPt)mallElectronsTracksPt->Fill((*iTrack)->pt());
	    if(mallElectronsTracksEta)mallElectronsTracksEta->Fill((*iTrack)->eta());
	    if(mallElectronsTracksPhi)mallElectronsTracksPhi->Fill((*iTrack)->phi());
	    if(mallElectronsTracksPtVsEta)mallElectronsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    if(mOutVertexInCaloElectronsTracksPt)mOutVertexInCaloElectronsTracksPt->Fill((*iTrack)->pt());
	    if(mOutVertexInCaloElectronsTracksEta)mOutVertexInCaloElectronsTracksEta->Fill((*iTrack)->eta());
	    if(mOutVertexInCaloElectronsTracksPhi)mOutVertexInCaloElectronsTracksPhi->Fill((*iTrack)->phi());
	    if(mOutVertexInCaloElectronsTracksPtVsEta)mOutVertexInCaloElectronsTracksPtVsEta->Fill((*iTrack)->eta(),(*iTrack)->pt());
	    const double trackDirectionJetDR = deltaR(rawJet->eta(),rawJet->phi(),(*iTrack)->eta(),(*iTrack)->phi());
	    if(mInCaloTrackDirectionJetDRHisto_)mInCaloTrackDirectionJetDRHisto_->Fill(trackDirectionJetDR);
	    math::XYZPoint point =trackPropagator_->impactPoint(**iTrack);
	    const double impactPointJetDR = deltaR(rawJet->eta(),rawJet->phi(), point.Eta(),point.Phi());
	    if(mOutVertexTrackImpactPointJetDRHisto_)mOutVertexTrackImpactPointJetDRHisto_->Fill(impactPointJetDR);
	  }
	}
      }
      if(isPFJet){
	//calculate the jetID
	bool ThisCHFcleaned=true;
	bool LooseCHFcleaned=true;
	bool TightCHFcleaned=true;
	if(((*pfJets)[ijet].chargedHadronEnergy()/(*pfJets)[ijet].energy())<=_ThisCHFMin && fabs((*pfJets)[ijet].eta())<2.4) ThisCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	if(((*pfJets)[ijet].chargedHadronEnergy()/(*pfJets)[ijet].energy())<=_LooseCHFMin && fabs((*pfJets)[ijet].eta())<2.4) LooseCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	if(((*pfJets)[ijet].chargedHadronEnergy()/(*pfJets)[ijet].energy())<=_TightCHFMin && fabs((*pfJets)[ijet].eta())<2.4) TightCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	if(ThisCHFcleaned && ((*pfJets)[ijet].neutralHadronEnergy()/(*pfJets)[ijet].energy())<_ThisNHFMax && ((*pfJets)[ijet].chargedEmEnergy()/(*pfJets)[ijet].energy())<_ThisCEFMax && ((*pfJets)[ijet].neutralEmEnergy()/(*pfJets)[ijet].energy())<_ThisNEFMax) Thiscleaned=true;
	if(LooseCHFcleaned && ((*pfJets)[ijet].neutralHadronEnergy()/(*pfJets)[ijet].energy())<_LooseNHFMax && ((*pfJets)[ijet].chargedEmEnergy()/(*pfJets)[ijet].energy())<_LooseCEFMax && ((*pfJets)[ijet].neutralEmEnergy()/(*pfJets)[ijet].energy())<_LooseNEFMax) Loosecleaned=true;
	if(TightCHFcleaned && ((*pfJets)[ijet].neutralHadronEnergy()/(*pfJets)[ijet].energy())<_TightNHFMax && ((*pfJets)[ijet].chargedEmEnergy()/(*pfJets)[ijet].energy())<_TightCEFMax && ((*pfJets)[ijet].neutralEmEnergy()/(*pfJets)[ijet].energy())<_TightNEFMax) Tightcleaned=true;
	if(Thiscleaned && pass_uncorrected){
	  if (mPt_uncor)   mPt_uncor->Fill ((*pfJets)[ijet].pt());
	  if (mEta_uncor)  mEta_uncor->Fill ((*pfJets)[ijet].eta());
	  if (mPhi_uncor)  mPhi_uncor->Fill ((*pfJets)[ijet].phi());
	  if(!isJPTJet){
	    if (mConstituents_uncor) mConstituents_uncor->Fill ((*pfJets)[ijet].nConstituents());
	  }
	}
	if(Thiscleaned && pass_corrected){
	  if (mHFrac)        mHFrac->Fill ((*pfJets)[ijet].chargedHadronEnergyFraction()+(*pfJets)[ijet].neutralHadronEnergyFraction());
	  if (mEFrac)        mEFrac->Fill ((*pfJets)[ijet].chargedEmEnergyFraction() +(*pfJets)[ijet].neutralEmEnergyFraction());
	  if ((*pfJets)[ijet].pt()<= 50) {
	    if (mCHFracVSeta_lowPt) mCHFracVSeta_lowPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	    if (mNHFracVSeta_lowPt) mNHFracVSeta_lowPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	    if (mPhFracVSeta_lowPt) mPhFracVSeta_lowPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralEmEnergyFraction());
	    if (mElFracVSeta_lowPt) mElFracVSeta_lowPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedEmEnergyFraction());
	    if (mMuFracVSeta_lowPt) mMuFracVSeta_lowPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedMuEnergyFraction());
	  }
	  if ((*pfJets)[ijet].pt()>50. && (*pfJets)[ijet].pt()<=140.) {
	    if (mCHFracVSeta_mediumPt) mCHFracVSeta_mediumPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	    if (mNHFracVSeta_mediumPt) mNHFracVSeta_mediumPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	    if (mPhFracVSeta_mediumPt) mPhFracVSeta_mediumPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralEmEnergyFraction());
	    if (mElFracVSeta_mediumPt) mElFracVSeta_mediumPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedEmEnergyFraction());
	    if (mMuFracVSeta_mediumPt) mMuFracVSeta_mediumPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedMuEnergyFraction());
	  }
	  if ((*pfJets)[ijet].pt()>140.) {
	    if (mCHFracVSeta_highPt) mCHFracVSeta_highPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	    if (mNHFracVSeta_highPt) mNHFracVSeta_highPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	    if (mPhFracVSeta_highPt) mPhFracVSeta_highPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].neutralEmEnergyFraction());
	    if (mElFracVSeta_highPt) mElFracVSeta_highPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedEmEnergyFraction());
	    if (mMuFracVSeta_highPt) mMuFracVSeta_highPt->Fill((*pfJets)[ijet].eta(),(*pfJets)[ijet].chargedMuEnergyFraction());
	  }
	  if (fabs((*pfJets)[ijet].eta()) <= 1.3) {
	    if (mHFrac_Barrel)           mHFrac_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergyFraction() + (*pfJets)[ijet].neutralHadronEnergyFraction() );
	    if (mEFrac_Barrel)           mEFrac->Fill ((*pfJets)[ijet].chargedEmEnergyFraction() + (*pfJets)[ijet].neutralEmEnergyFraction());	
	    //fractions
	    if ((*pfJets)[ijet].pt()<=50.) {
	      if (mCHFrac_lowPt_Barrel) mCHFrac_lowPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	      if (mNHFrac_lowPt_Barrel) mNHFrac_lowPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	      if (mPhFrac_lowPt_Barrel) mPhFrac_lowPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	      if (mElFrac_lowPt_Barrel) mElFrac_lowPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
	      if (mMuFrac_lowPt_Barrel) mMuFrac_lowPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergyFraction());
	      //
	      if (mCHEn_lowPt_Barrel) mCHEn_lowPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergy());
	      if (mNHEn_lowPt_Barrel) mNHEn_lowPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergy());
	      if (mPhEn_lowPt_Barrel) mPhEn_lowPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergy());
	      if (mElEn_lowPt_Barrel) mElEn_lowPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergy());
	      if (mMuEn_lowPt_Barrel) mMuEn_lowPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergy());
	    }
	    if ((*pfJets)[ijet].pt()>50. && (*pfJets)[ijet].pt()<=140.) {
	      if (mCHFrac_mediumPt_Barrel) mCHFrac_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	      if (mNHFrac_mediumPt_Barrel) mNHFrac_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	      if (mPhFrac_mediumPt_Barrel) mPhFrac_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	      if (mElFrac_mediumPt_Barrel) mElFrac_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
	      if (mMuFrac_mediumPt_Barrel) mMuFrac_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergyFraction());
	      //
	      if (mCHEn_mediumPt_Barrel) mCHEn_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergy());
	      if (mNHEn_mediumPt_Barrel) mNHEn_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergy());
	      if (mPhEn_mediumPt_Barrel) mPhEn_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergy());
	      if (mElEn_mediumPt_Barrel) mElEn_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergy());
	      if (mMuEn_mediumPt_Barrel) mMuEn_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergy());
	    }
	    if ((*pfJets)[ijet].pt()>140.) {
	      if (mCHFrac_highPt_Barrel) mCHFrac_highPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	      if (mNHFrac_highPt_Barrel) mNHFrac_highPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	      if (mPhFrac_highPt_Barrel) mPhFrac_highPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	      if (mElFrac_highPt_Barrel) mElFrac_highPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
	      if (mMuFrac_highPt_Barrel) mMuFrac_highPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergyFraction());
	      //
	      if (mCHEn_highPt_Barrel) mCHEn_highPt_Barrel->Fill((*pfJets)[ijet].chargedHadronEnergy());
	      if (mNHEn_highPt_Barrel) mNHEn_highPt_Barrel->Fill((*pfJets)[ijet].neutralHadronEnergy());
	      if (mPhEn_highPt_Barrel) mPhEn_highPt_Barrel->Fill((*pfJets)[ijet].neutralEmEnergy());
	      if (mElEn_highPt_Barrel) mElEn_highPt_Barrel->Fill((*pfJets)[ijet].chargedEmEnergy());
	      if (mMuEn_highPt_Barrel) mMuEn_highPt_Barrel->Fill((*pfJets)[ijet].chargedMuEnergy());
	    }
	    if(mChMultiplicity_lowPt_Barrel)  mChMultiplicity_lowPt_Barrel->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_lowPt_Barrel)  mNeuMultiplicity_lowPt_Barrel->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_lowPt_Barrel)  mMuMultiplicity_lowPt_Barrel->Fill((*pfJets)[ijet].muonMultiplicity());
	    if(mChMultiplicity_mediumPt_Barrel)  mChMultiplicity_mediumPt_Barrel->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_mediumPt_Barrel)  mNeuMultiplicity_mediumPt_Barrel->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_mediumPt_Barrel)  mMuMultiplicity_mediumPt_Barrel->Fill((*pfJets)[ijet].muonMultiplicity());
	    if(mChMultiplicity_highPt_Barrel)  mChMultiplicity_highPt_Barrel->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_highPt_Barrel)  mNeuMultiplicity_highPt_Barrel->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_highPt_Barrel)  mMuMultiplicity_highPt_Barrel->Fill((*pfJets)[ijet].muonMultiplicity());
	    //
	    if (mCHFracVSpT_Barrel) mCHFracVSpT_Barrel->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	    if (mNHFracVSpT_Barrel) mNHFracVSpT_Barrel->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	    if (mPhFracVSpT_Barrel) mPhFracVSpT_Barrel->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].neutralEmEnergyFraction());
	    if (mElFracVSpT_Barrel) mElFracVSpT_Barrel->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].chargedEmEnergyFraction());
	    if (mMuFracVSpT_Barrel) mMuFracVSpT_Barrel->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].chargedMuEnergyFraction());
	  }else if(fabs((*pfJets)[ijet].eta()) <= 3) {
	    if (mHFrac_EndCap)           mHFrac_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergyFraction() + (*pfJets)[ijet].neutralHadronEnergyFraction());
	    if (mEFrac_EndCap)           mEFrac->Fill ((*pfJets)[ijet].chargedEmEnergyFraction() + (*pfJets)[ijet].neutralEmEnergyFraction());
	    //fractions
	    if ((*pfJets)[ijet].pt()<=50.) {
	      if (mCHFrac_lowPt_EndCap) mCHFrac_lowPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	      if (mNHFrac_lowPt_EndCap) mNHFrac_lowPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	      if (mPhFrac_lowPt_EndCap) mPhFrac_lowPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	      if (mElFrac_lowPt_EndCap) mElFrac_lowPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
	      if (mMuFrac_lowPt_EndCap) mMuFrac_lowPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergyFraction());
	      //
	      if (mCHEn_lowPt_EndCap) mCHEn_lowPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergy());
	      if (mNHEn_lowPt_EndCap) mNHEn_lowPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergy());
	      if (mPhEn_lowPt_EndCap) mPhEn_lowPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergy());
	      if (mElEn_lowPt_EndCap) mElEn_lowPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergy());
	      if (mMuEn_lowPt_EndCap) mMuEn_lowPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergy());
	    }else if ((*pfJets)[ijet].pt()>50. && (*pfJets)[ijet].pt()<=140.) {
	      if (mCHFrac_mediumPt_EndCap) mCHFrac_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	      if (mNHFrac_mediumPt_EndCap) mNHFrac_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	      if (mPhFrac_mediumPt_EndCap) mPhFrac_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	      if (mElFrac_mediumPt_EndCap) mElFrac_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
	      if (mMuFrac_mediumPt_EndCap) mMuFrac_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergyFraction());
	      //
	      if (mCHEn_mediumPt_EndCap) mCHEn_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergy());
	      if (mNHEn_mediumPt_EndCap) mNHEn_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergy());
	      if (mPhEn_mediumPt_EndCap) mPhEn_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergy());
	      if (mElEn_mediumPt_EndCap) mElEn_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergy());
	      if (mMuEn_mediumPt_EndCap) mMuEn_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergy());
	    }else if ((*pfJets)[ijet].pt()>140.) {
	      if (mCHFrac_highPt_EndCap) mCHFrac_highPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergyFraction());
	      if (mNHFrac_highPt_EndCap) mNHFrac_highPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergyFraction());
	      if (mPhFrac_highPt_EndCap) mPhFrac_highPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergyFraction());
	      if (mElFrac_highPt_EndCap) mElFrac_highPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergyFraction());
	      if (mMuFrac_highPt_EndCap) mMuFrac_highPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergyFraction());
	      //
	      if (mCHEn_highPt_EndCap) mCHEn_highPt_EndCap->Fill((*pfJets)[ijet].chargedHadronEnergy());
	      if (mNHEn_highPt_EndCap) mNHEn_highPt_EndCap->Fill((*pfJets)[ijet].neutralHadronEnergy());
	      if (mPhEn_highPt_EndCap) mPhEn_highPt_EndCap->Fill((*pfJets)[ijet].neutralEmEnergy());
	      if (mElEn_highPt_EndCap) mElEn_highPt_EndCap->Fill((*pfJets)[ijet].chargedEmEnergy());
	      if (mMuEn_highPt_EndCap) mMuEn_highPt_EndCap->Fill((*pfJets)[ijet].chargedMuEnergy());
	    }
	    if(mChMultiplicity_lowPt_EndCap)  mChMultiplicity_lowPt_EndCap->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_lowPt_EndCap)  mNeuMultiplicity_lowPt_EndCap->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_lowPt_EndCap)  mMuMultiplicity_lowPt_EndCap->Fill((*pfJets)[ijet].muonMultiplicity());
	    if(mChMultiplicity_mediumPt_EndCap)  mChMultiplicity_mediumPt_EndCap->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_mediumPt_EndCap)  mNeuMultiplicity_mediumPt_EndCap->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_mediumPt_EndCap)  mMuMultiplicity_mediumPt_EndCap->Fill((*pfJets)[ijet].muonMultiplicity());
	    if(mChMultiplicity_highPt_EndCap)  mChMultiplicity_highPt_EndCap->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_highPt_EndCap)  mNeuMultiplicity_highPt_EndCap->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_highPt_EndCap)  mMuMultiplicity_highPt_EndCap->Fill((*pfJets)[ijet].muonMultiplicity());
	    //
	    if (mCHFracVSpT_EndCap) mCHFracVSpT_EndCap->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].chargedHadronEnergyFraction());
	    if (mNHFracVSpT_EndCap) mNHFracVSpT_EndCap->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].neutralHadronEnergyFraction());
	    if (mPhFracVSpT_EndCap) mPhFracVSpT_EndCap->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].neutralEmEnergyFraction());
	    if (mElFracVSpT_EndCap) mElFracVSpT_EndCap->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].chargedEmEnergyFraction());
	    if (mMuFracVSpT_EndCap) mMuFracVSpT_EndCap->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].chargedMuEnergyFraction());
	    
	  }else{
	    if (mHFrac_Forward)           mHFrac_Forward->Fill((*pfJets)[ijet].chargedHadronEnergyFraction() + (*pfJets)[ijet].neutralHadronEnergyFraction());	
	    if (mEFrac_Forward)           mEFrac->Fill ((*pfJets)[ijet].chargedEmEnergyFraction() + (*pfJets)[ijet].neutralEmEnergyFraction());
	    //fractions
	    if ((*pfJets)[ijet].pt()<=50.) {
	      if(mHFEFrac_lowPt_Forward) mHFEFrac_lowPt_Forward->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	      if(mHFHFrac_lowPt_Forward) mHFHFrac_lowPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	      //
	      if(mHFEEn_lowPt_Forward) mHFEEn_lowPt_Forward->Fill((*pfJets)[ijet].HFEMEnergy());
	      if(mHFHEn_lowPt_Forward) mHFHEn_lowPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergy());
	    }
	    if ((*pfJets)[ijet].pt()>50. && (*pfJets)[ijet].pt()<=140.) {
	      if(mHFEFrac_mediumPt_Forward) mHFEFrac_mediumPt_Forward->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	      if(mHFHFrac_mediumPt_Forward) mHFHFrac_mediumPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	      //
	      if(mHFEEn_mediumPt_Forward) mHFEEn_mediumPt_Forward->Fill((*pfJets)[ijet].HFEMEnergy());
	      if(mHFHEn_mediumPt_Forward) mHFHEn_mediumPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergy());
	    }
	    if ((*pfJets)[ijet].pt()>140.) {
	      if(mHFEFrac_highPt_Forward) mHFEFrac_highPt_Forward->Fill((*pfJets)[ijet].HFEMEnergyFraction());
	      if(mHFHFrac_highPt_Forward) mHFHFrac_highPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergyFraction());
	      //
	      if(mHFEEn_highPt_Forward) mHFEEn_highPt_Forward->Fill((*pfJets)[ijet].HFEMEnergy());
	      if(mHFHEn_highPt_Forward) mHFHEn_highPt_Forward->Fill((*pfJets)[ijet].HFHadronEnergy());
	    }
	    if(mChMultiplicity_lowPt_Forward)  mChMultiplicity_lowPt_Forward->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_lowPt_Forward)  mNeuMultiplicity_lowPt_Forward->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_lowPt_Forward)  mMuMultiplicity_lowPt_Forward->Fill((*pfJets)[ijet].muonMultiplicity());
	    if(mChMultiplicity_mediumPt_Forward)  mChMultiplicity_mediumPt_Forward->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_mediumPt_Forward)  mNeuMultiplicity_mediumPt_Forward->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_mediumPt_Forward)  mMuMultiplicity_mediumPt_Forward->Fill((*pfJets)[ijet].muonMultiplicity());
	    if(mChMultiplicity_highPt_Forward)  mChMultiplicity_highPt_Forward->Fill((*pfJets)[ijet].chargedMultiplicity());
	    if(mNeuMultiplicity_highPt_Forward)  mNeuMultiplicity_highPt_Forward->Fill((*pfJets)[ijet].neutralMultiplicity());
	    if(mMuMultiplicity_highPt_Forward)  mMuMultiplicity_highPt_Forward->Fill((*pfJets)[ijet].muonMultiplicity());
	    if(mHFHFracVSpT_Forward) mHFHFracVSpT_Forward->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].HFHadronEnergyFraction());
	    if(mHFEFracVSpT_Forward) mHFEFracVSpT_Forward->Fill((*pfJets)[ijet].pt(),(*pfJets)[ijet].HFEMEnergyFraction());
	  }
	  if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill ((*pfJets)[ijet].chargedHadronEnergy());
	  if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill ((*pfJets)[ijet].neutralHadronEnergy());
	  if (mChargedEmEnergy) mChargedEmEnergy->Fill((*pfJets)[ijet].chargedEmEnergy());
	  if (mChargedMuEnergy) mChargedMuEnergy->Fill ((*pfJets)[ijet].chargedMuEnergy ());
	  if (mNeutralEmEnergy) mNeutralEmEnergy->Fill((*pfJets)[ijet].neutralEmEnergy());
	  if (mChargedMultiplicity ) mChargedMultiplicity->Fill((*pfJets)[ijet].chargedMultiplicity());
	  if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill((*pfJets)[ijet].neutralMultiplicity());
	  if (mMuonMultiplicity )mMuonMultiplicity->Fill ((*pfJets)[ijet]. muonMultiplicity());
	  //_______________________________________________________
	  if (mNeutralFraction) mNeutralFraction->Fill ((double)(*pfJets)[ijet].neutralMultiplicity()/(double)(*pfJets)[ijet].nConstituents());

	  if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*pfJets)[ijet].chargedHadronEnergyFraction() + (*pfJets)[ijet].neutralHadronEnergyFraction());
	  if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, (*pfJets)[ijet].chargedEmEnergyFraction()     + (*pfJets)[ijet].neutralEmEnergyFraction());
	  
	  if (mChargedHadronEnergy_profile) mChargedHadronEnergy_profile->Fill(_numPV, (*pfJets)[ijet].chargedHadronEnergy());
	  if (mNeutralHadronEnergy_profile) mNeutralHadronEnergy_profile->Fill(_numPV, (*pfJets)[ijet].neutralHadronEnergy());
	  if (mChargedEmEnergy_profile)     mChargedEmEnergy_profile    ->Fill(_numPV, (*pfJets)[ijet].chargedEmEnergy());
	  if (mChargedMuEnergy_profile)     mChargedMuEnergy_profile    ->Fill(_numPV, (*pfJets)[ijet].chargedMuEnergy ());
	  if (mNeutralEmEnergy_profile)     mNeutralEmEnergy_profile    ->Fill(_numPV, (*pfJets)[ijet].neutralEmEnergy());
	  if (mChargedMultiplicity_profile) mChargedMultiplicity_profile->Fill(_numPV, (*pfJets)[ijet].chargedMultiplicity());
	  if (mNeutralMultiplicity_profile) mNeutralMultiplicity_profile->Fill(_numPV, (*pfJets)[ijet].neutralMultiplicity());
	  if (mMuonMultiplicity_profile)    mMuonMultiplicity_profile   ->Fill(_numPV, (*pfJets)[ijet].muonMultiplicity());

	}//cleaned PFJets
      }//PFJet specific loop
      //if only uncorrected jets but no corrected jets over threshold pass on
      if(!pass_corrected){
	continue;
      }        
      //after jettype specific variables are filled -> perform histograms for all jets
      if(fillJIDPassFrac==1) {
	if(Loosecleaned) {
	  mLooseJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	  mLooseJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	} else {
	  mLooseJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	  mLooseJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	}
	//TIGHT
	if(Tightcleaned) {
	  mTightJIDPassFractionVSeta->Fill(correctedJet.eta(),1.);
	  mTightJIDPassFractionVSpt->Fill(correctedJet.pt(),1.);
	} else {
	  mTightJIDPassFractionVSeta->Fill(correctedJet.eta(),0.);
	  mTightJIDPassFractionVSpt->Fill(correctedJet.pt(),0.);
	}
      }
      //here we so far consider calojets ->check for PFJets and JPT jets again
      if(Thiscleaned && pass_corrected){//might be softer than loose jet ID 
	numofjets++;
	if(isCaloJet){
	  jetME->Fill(1);
	  mJetEnergyCorr->Fill(correctedJet.pt()/(*caloJets)[ijet].pt());
	  mJetEnergyCorrVsEta->Fill(correctedJet.eta(),correctedJet.pt()/(*caloJets)[ijet].pt());
	}
	if(isPFJet){
	  jetME->Fill(2);
	  mJetEnergyCorr->Fill(correctedJet.pt()/(*pfJets)[ijet].pt());
	  mJetEnergyCorrVsEta->Fill(correctedJet.eta(),correctedJet.pt()/(*pfJets)[ijet].pt());
	}
	if(isJPTJet){
	  jetME->Fill(3);
	  mJetEnergyCorr->Fill(correctedJet.pt()/(*jptJets)[ijet].pt());
	  mJetEnergyCorrVsEta->Fill(correctedJet.eta(),correctedJet.pt()/(*jptJets)[ijet].pt());
	}
	// --- Event passed the low pt jet trigger
	if (_JetLoPass == 1) {
	  if (mPhi_Lo) mPhi_Lo->Fill (correctedJet.phi());
	  if (mPt_Lo)  mPt_Lo->Fill (correctedJet.pt());
	}
	// --- Event passed the high pt jet trigger
	if (_JetHiPass == 1) {
	  if (fabs(correctedJet.eta()) <= 1.3) {
	    if (mPt_Barrel_Hi && correctedJet.pt()>100.)          mPt_Barrel_Hi->Fill(correctedJet.pt());
	    if (mEta_Hi && correctedJet.pt()>100.)          mEta_Hi->Fill(correctedJet.eta());
	    if (mPhi_Barrel_Hi)          mPhi_Barrel_Hi->Fill(correctedJet.phi());
	  }else if (fabs(correctedJet.eta()) <= 3) {
	    if (mPt_EndCap_Hi && correctedJet.pt()>100.)           mPt_EndCap_Hi->Fill(correctedJet.pt());
	    if (mEta_Hi && correctedJet.pt()>100.)          mEta_Hi->Fill(correctedJet.eta());
	    if (mPhi_EndCap_Hi)          mPhi_EndCap_Hi->Fill(correctedJet.phi());
	  }else{
	    if (mPt_Forward_Hi && correctedJet.pt()>100.)           mPt_Forward_Hi->Fill(correctedJet.pt());
	    if (mEta_Hi && correctedJet.pt()>100.)          mEta_Hi->Fill(correctedJet.eta());
	    if (mPhi_Forward_Hi)          mPhi_Forward_Hi->Fill(correctedJet.phi());	
	  }	    
	  if (mEta_Hi && correctedJet.pt()>100.) mEta_Hi->Fill (correctedJet.eta());
	  if (mPhi_Hi) mPhi_Hi->Fill (correctedJet.phi());
	  if (mPt_Hi)  mPt_Hi->Fill (correctedJet.pt());
	}
	if (mPt)   mPt->Fill (correctedJet.pt());
	if (mPt_1) mPt_1->Fill (correctedJet.pt());
	if (mPt_2) mPt_2->Fill (correctedJet.pt());
	if (mPt_3) mPt_3->Fill (correctedJet.pt());
	if (mEta)  mEta->Fill (correctedJet.eta());
	if (mPhi)  mPhi->Fill (correctedJet.phi());	 
	
	if (mPhiVSEta) mPhiVSEta->Fill(correctedJet.eta(),correctedJet.phi());
	if(!isJPTJet){
	  if (mConstituents) mConstituents->Fill (correctedJet.nConstituents());
	}
	// Fill NPV profiles
	//--------------------------------------------------------------------
	if (mPt_profile)           mPt_profile          ->Fill(_numPV, correctedJet.pt());
	if (mEta_profile)          mEta_profile         ->Fill(_numPV, correctedJet.eta());
	if (mPhi_profile)          mPhi_profile         ->Fill(_numPV, correctedJet.phi());
	if(!isJPTJet){
	  if (mConstituents_profile) mConstituents_profile->Fill(_numPV, correctedJet.nConstituents());
	}
	if (fabs(correctedJet.eta()) <= 1.3) {
	  if (mPt_Barrel)   mPt_Barrel->Fill (correctedJet.pt());
	  if (mPhi_Barrel)  mPhi_Barrel->Fill (correctedJet.phi());
	  //if (mE_Barrel)    mE_Barrel->Fill (correctedJet.energy());
	  if(!isJPTJet){
	    if (mConstituents_Barrel)    mConstituents_Barrel->Fill(correctedJet.nConstituents());
	  }
	}else if (fabs(correctedJet.eta()) <= 3) {
	  if (mPt_EndCap)   mPt_EndCap->Fill (correctedJet.pt());
	  if (mPhi_EndCap)  mPhi_EndCap->Fill (correctedJet.phi());
	  //if (mE_EndCap)    mE_EndCap->Fill (correctedJet.energy());
	  if(!isJPTJet){
	    if (mConstituents_EndCap)    mConstituents_EndCap->Fill(correctedJet.nConstituents());
	  }
	}else{
	  if (mPt_Forward)   mPt_Forward->Fill (correctedJet.pt());
	  if (mPhi_Forward)  mPhi_Forward->Fill (correctedJet.phi());
	  //if (mE_Forward)    mE_Forward->Fill (correctedJet.energy());
	  if(!isJPTJet){
	    if (mConstituents_Forward)    mConstituents_Forward->Fill(correctedJet.nConstituents());
	  }
	}
      }  
    }// the selection of all jets --> analysis loop if makedijetselection==1
  }//loop over uncorrected jets 
  if(makedijetselection!=1){
    if (mNJets) mNJets->Fill (numofjets);
    if (mNJets_profile) mNJets_profile->Fill(_numPV, numofjets);
  }
  sort(recoJets.begin(),recoJets.end(),jetSortingRule);

  //for non dijet selection, otherwise numofjets==0
  if(numofjets>0){
 
    //check ID of the leading jet
    bool Thiscleaned=false;
    bool thisemfclean=true;
    bool ThisCHFcleaned=true;
    if(isCaloJet){
      jetID->calculate(iEvent, (*caloJets)[ind1]);
      thisemfclean=true;
      if(jetID->restrictedEMF()<_resEMFMin && fabs((*caloJets)[ind1].eta())<2.6) thisemfclean=false;
      if(jetID->n90Hits()>=_n90HitsMin && jetID->fHPD()<_fHPDMax && thisemfclean) Thiscleaned=true;

    }
    if(isJPTJet){
      const edm::RefToBase<reco::Jet>&  rawJet1 = (*jptJets)[ind1].getCaloJetRef();
      //change that step here
      //check jet is correctable by JPT
      //if ( fabs(rawJet->eta()) > 2.1) return;
      
      try {
	const reco::CaloJet *rawCaloJet1 = dynamic_cast<const reco::CaloJet*>(&*rawJet1);
	jetID->calculate(iEvent,*rawCaloJet1);
	  if(jetID->restrictedEMF()<_resEMFMin && fabs((*rawCaloJet1).eta())<2.6) thisemfclean=false;
	  if(jetID->n90Hits()>=_n90HitsMin && jetID->fHPD()<_fHPDMax && thisemfclean) Thiscleaned=true;
      } catch (const std::bad_cast&) {
	edm::LogError("JetPlusTrackDQM") << "Failed to cast raw jet to CaloJet. JPT Jet does not appear to have been built from a CaloJet. "
					 << "Histograms not filled. ";
	return;
      }
    }
    if(isPFJet){
      ThisCHFcleaned=true;
      //LooseCHFcleaned=true;
      //TightCHFcleaned=true;
      if(((*pfJets)[ind1].chargedHadronEnergy()/(*pfJets)[ind1].energy())<=_ThisCHFMin && fabs((*pfJets)[ind1].eta())<2.4) ThisCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
      //if(((*pfJets)[ind1].chargedHadronEnergy()/(*pfJets)[ind1].energy())<=_LooseCHFMin && fabs((*pfJets)[ind1].eta())<2.4) LooseCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
      //if(((*pfJets)[ind1].chargedHadronEnergy()/(*pfJets)[ind1].energy())<=_TightCHFMin && fabs((*pfJets)[ind1].eta())<2.4) TightCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
      if(ThisCHFcleaned && ((*pfJets)[ind1].neutralHadronEnergy()/(*pfJets)[ind1].energy())<_ThisNHFMax && ((*pfJets)[ind1].chargedEmEnergy()/(*pfJets)[ind1].energy())<_ThisCEFMax && ((*pfJets)[ind1].neutralEmEnergy()/(*pfJets)[ind1].energy())<_ThisNEFMax) Thiscleaned=true;
    }
    if(Thiscleaned){
      if (mEtaFirst) mEtaFirst->Fill ((recoJets)[0].eta());
      if (mPhiFirst) mPhiFirst->Fill ((recoJets)[0].phi());
      if (mPtFirst)  mPtFirst->Fill ((recoJets)[0].pt());
      if(numofjets>1) {
	//first jet fine -> now check second jet
	Thiscleaned=false;
	if(isCaloJet){
	  jetID->calculate(iEvent, (*caloJets)[ind2]);
	  thisemfclean=true;
	  if(jetID->restrictedEMF()<_resEMFMin && fabs((*caloJets)[ind2].eta())<2.6) thisemfclean=false;
	  if(jetID->n90Hits()>=_n90HitsMin && jetID->fHPD()<_fHPDMax && thisemfclean) Thiscleaned=true;
	}
	if(isJPTJet){
	  const edm::RefToBase<reco::Jet>&  rawJet2 = (*jptJets)[ind2].getCaloJetRef();
	  //change that step here
	  //check jet is correctable by JPT
	  //if ( fabs(rawJet->eta()) > 2.1) return;
	  
	  try {
	    const reco::CaloJet *rawCaloJet2 = dynamic_cast<const reco::CaloJet*>(&*rawJet2);
	    jetID->calculate(iEvent,*rawCaloJet2);
	    if(jetID->restrictedEMF()<_resEMFMin && fabs((*rawCaloJet2).eta())<2.6) thisemfclean=false;
	    if(jetID->n90Hits()>=_n90HitsMin && jetID->fHPD()<_fHPDMax && thisemfclean) Thiscleaned=true;
	  } catch (const std::bad_cast&) {
	    edm::LogError("JetPlusTrackDQM") << "Failed to cast raw jet to CaloJet. JPT Jet does not appear to have been built from a CaloJet. "
					     << "Histograms not filled. ";
	    return;
	  }
	  if(Thiscleaned){
	    mPtSecond->Fill(recoJets[1].pt());
	  }
	  if(numofjets>2){
	    const edm::RefToBase<reco::Jet>&  rawJet3 = (*jptJets)[ind3].getCaloJetRef();
	    //change that step here
	    //check jet is correctable by JPT
	    //if ( fabs(rawJet->eta()) > 2.1) return;
	    
	    try {
	      thisemfclean=true;
	      const reco::CaloJet *rawCaloJet3 = dynamic_cast<const reco::CaloJet*>(&*rawJet3);
	      jetID->calculate(iEvent,*rawCaloJet3);
	      if(jetID->restrictedEMF()<_resEMFMin && fabs((*rawCaloJet3).eta())<2.6) thisemfclean=false;
	      if(jetID->n90Hits()>=_n90HitsMin && jetID->fHPD()<_fHPDMax && thisemfclean) Thiscleaned=true;
	    } catch (const std::bad_cast&) {
	      edm::LogError("JetPlusTrackDQM") << "Failed to cast raw jet to CaloJet. JPT Jet does not appear to have been built from a CaloJet. "
					       << "Histograms not filled. ";
	      return;
	    }
	    if(Thiscleaned){
	      mPtThird->Fill(recoJets[2].pt());
	    }
	  }
	}
	if(isPFJet){
	  ThisCHFcleaned=true;
	  //LooseCHFcleaned=false;
	  //TightCHFcleaned=false;
	  if(((*pfJets)[ind2].chargedHadronEnergy()/(*pfJets)[ind2].energy())<=_ThisCHFMin && fabs((*pfJets)[ind2].eta())<2.4) ThisCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	  //if(((*pfJets)[ind2].chargedHadronEnergy()/(*pfJets)[ind2].energy())<=_LooseCHFMin && fabs((*pfJets)[ind2].eta())<2.4) LooseCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	  //if(((*pfJets)[ind2].chargedHadronEnergy()/(*pfJets)[ind2].energy())<=_TightCHFMin && fabs((*pfJets)[ind2].eta())<2.4) TightCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	  if(ThisCHFcleaned && ((*pfJets)[ind2].neutralHadronEnergy()/(*pfJets)[ind2].energy())<_ThisNHFMax && ((*pfJets)[ind2].chargedEmEnergy()/(*pfJets)[ind2].energy())<_ThisCEFMax && ((*pfJets)[ind2].neutralEmEnergy()/(*pfJets)[ind2].energy())<_ThisNEFMax) Thiscleaned=true;
	}
	if(Thiscleaned){
	  double dphi=fabs((recoJets)[0].phi()-(recoJets)[1].phi());
	  if(dphi>acos(-1.)){
	    dphi=2*acos(-1.)-dphi;
	  }
	  if (mDPhi) mDPhi->Fill (dphi);
	  //try here now the three jet dijet asymmetry stuff
	  if (fabs(recoJets[0].eta() < 1.4)) {
	    double pt_dijet = (recoJets[0].pt() + recoJets[1].pt())/2;      
	    double dPhiDJ = fabs(recoJets[0].phi()-recoJets[1].phi());
	    if(dPhiDJ>acos(-1.)){
	      dPhiDJ=2*acos(-1.)-dPhiDJ;
	    }
	    if (dPhiDJ > 2.7) {

	      double pt_probe;
	      double pt_barrel;
	      int jet1, jet2;
	      int randJet = rand() % 2;
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
		  if (pt3 > _asymmetryThirdJetCut){
		    thirdJetCut=false;
		  } 
		}
		if(thirdJetCut){
		  double dijetAsymmetry =(recoJets[jet1].pt() - recoJets[jet2].pt()) / (recoJets[jet1].pt() + recoJets[jet2].pt());
		  mDijetAsymmetry->Fill(dijetAsymmetry);
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
		if (pt3/pt_dijet > _balanceThirdJetCut){ 
		  thirdJetCut = false;
		}
	      }
	      if (thirdJetCut) {
		double dijetBalance = (pt_probe - pt_barrel) / pt_dijet;
		mDijetBalance->Fill(dijetBalance);
	      }// end restriction on third jet pt ratio in balance calculation
	    }// dPhi > 2.7
	  }// leading jet eta cut for asymmetry and balance calculations
	  //end here test try for 3 jet asymmetry blabla
	}	
      }
    }
  }//leading jet histograms for non dijet selection
 

  //dijet selection -> recoJets selection ensures jet hard enough and corrected (if possible)
  if(recoJets.size()>1 && makedijetselection==1){
    //pt threshold checked before filling
    if(fabs(recoJets[0].eta())<3. && fabs(recoJets[1].eta())<3. ){
      //calculate dphi
      double dphi=fabs((recoJets)[0].phi()-(recoJets)[1].phi());
      if(dphi>acos(-1.)){
	dphi=2*acos(-1.)-dphi;
      } 
      if (mDPhi) mDPhi->Fill (dphi);
      //dphi cut
      if(fabs(dphi)>2.1){
	bool LoosecleanedFirstJet = false;
	bool LoosecleanedSecondJet = false;
	bool TightcleanedFirstJet = false;
	bool TightcleanedSecondJet = false;
	
	if(isCaloJet){
	  bool emfcleanLooseFirstJet=true;
	  bool emfcleanTightFirstJet=true;
	  jetID->calculate(iEvent, (*caloJets)[ind1]);
	  if(jetID->restrictedEMF()<_resEMFMinLoose && fabs((*caloJets)[ind1].eta())<2.6) emfcleanLooseFirstJet=false;
	  if(jetID->restrictedEMF()<_resEMFMinTight && fabs((*caloJets)[ind1].eta())<2.6) emfcleanTightFirstJet=false;
	  if(jetID->n90Hits()>=_n90HitsMinLoose && jetID->fHPD()<_fHPDMaxLoose && emfcleanLooseFirstJet) LoosecleanedFirstJet=true;
	  if(jetID->n90Hits()>=_n90HitsMinTight && jetID->fHPD()<_fHPDMaxTight && sqrt((*caloJets)[ind1].etaetaMoment())>_sigmaEtaMinTight && sqrt((*caloJets)[ind1].phiphiMoment())>_sigmaPhiMinTight && emfcleanTightFirstJet) TightcleanedFirstJet=true;
	  if (mN90Hits)         mN90Hits->Fill (jetID->n90Hits());
	  if (mfHPD)            mfHPD->Fill (jetID->fHPD());
	  if (mresEMF)         mresEMF->Fill (jetID->restrictedEMF());
	  if (mfRBX)            mfRBX->Fill (jetID->fRBX());	  
	  bool emfcleanLooseSecondJet=true;
	  bool emfcleanTightSecondJet=true;
 	  jetID->calculate(iEvent, (*caloJets)[ind2]);
	  if(jetID->restrictedEMF()<_resEMFMinLoose && fabs((*caloJets)[ind2].eta())<2.6) emfcleanLooseSecondJet=false;
	  if(jetID->restrictedEMF()<_resEMFMinTight && fabs((*caloJets)[ind2].eta())<2.6) emfcleanTightSecondJet=false;
	  if(jetID->n90Hits()>=_n90HitsMinLoose && jetID->fHPD()<_fHPDMaxLoose && emfcleanLooseSecondJet) LoosecleanedSecondJet=true;
	  if(jetID->n90Hits()>=_n90HitsMinTight && jetID->fHPD()<_fHPDMaxTight && sqrt((*caloJets)[ind2].etaetaMoment())>_sigmaEtaMinTight && sqrt((*caloJets)[ind2].phiphiMoment())>_sigmaPhiMinTight && emfcleanTightSecondJet) TightcleanedSecondJet=true;
	  if (mN90Hits)         mN90Hits->Fill (jetID->n90Hits());
	  if (mfHPD)            mfHPD->Fill (jetID->fHPD());
	  if (mresEMF)         mresEMF->Fill (jetID->restrictedEMF());
	  if (mfRBX)            mfRBX->Fill (jetID->fRBX());
	}
	if(isJPTJet){
	  const edm::RefToBase<reco::Jet>&  rawJet1 = (*jptJets)[ind1].getCaloJetRef();	
	  try {
	    const reco::CaloJet *rawCaloJet1 = dynamic_cast<const reco::CaloJet*>(&*rawJet1);
	    jetID->calculate(iEvent,*rawCaloJet1);
	    bool emfcleanLooseFirstJet=true;
	    bool emfcleanTightFirstJet=true;
	    //loose and tight cleaning, used to fill the JetIDPAssFraction histos
	    if(jetID->restrictedEMF()<_resEMFMinLoose && fabs((*rawCaloJet1).eta())<2.6) emfcleanLooseFirstJet=false;
	    if(jetID->restrictedEMF()<_resEMFMinTight && fabs((*rawCaloJet1).eta())<2.6) emfcleanTightFirstJet=false;
	    if(jetID->n90Hits()>=_n90HitsMinLoose && jetID->fHPD()<_fHPDMaxLoose && emfcleanLooseFirstJet) LoosecleanedFirstJet=true;
	    if(jetID->n90Hits()>=_n90HitsMinTight && jetID->fHPD()<_fHPDMaxTight && sqrt((*rawCaloJet1).etaetaMoment())>_sigmaEtaMinTight && sqrt((*rawCaloJet1).phiphiMoment())>_sigmaPhiMinTight && emfcleanTightFirstJet) TightcleanedFirstJet=true;
	  } catch (const std::bad_cast&) {
	    edm::LogError("JetPlusTrackDQM") << "Failed to cast raw jet to CaloJet lead jet in dijet selection. JPT Jet does not appear to have been built from a CaloJet. "
					     << "Histograms not filled. ";
	    return;
	  }
	  const edm::RefToBase<reco::Jet>&  rawJet2 = (*jptJets)[ind2].getCaloJetRef();	
	  try {
	    const reco::CaloJet *rawCaloJet2 = dynamic_cast<const reco::CaloJet*>(&*rawJet2);
	    jetID->calculate(iEvent,*rawCaloJet2);
	    bool emfcleanLooseSecondJet=true;
	    bool emfcleanTightSecondJet=true;
	    //loose and tight cleaning, used to fill the JetIDPAssFraction histos
	    if(jetID->restrictedEMF()<_resEMFMinLoose && fabs((*rawCaloJet2).eta())<2.6) emfcleanLooseSecondJet=false;
	    if(jetID->restrictedEMF()<_resEMFMinTight && fabs((*rawCaloJet2).eta())<2.6) emfcleanTightSecondJet=false;
	    if(jetID->n90Hits()>=_n90HitsMinLoose && jetID->fHPD()<_fHPDMaxLoose && emfcleanLooseSecondJet) LoosecleanedSecondJet=true;
	    if(jetID->n90Hits()>=_n90HitsMinTight && jetID->fHPD()<_fHPDMaxTight && sqrt((*rawCaloJet2).etaetaMoment())>_sigmaEtaMinTight && sqrt((*rawCaloJet2).phiphiMoment())>_sigmaPhiMinTight && emfcleanTightSecondJet) TightcleanedSecondJet=true;
	  } catch (const std::bad_cast&) {
	    edm::LogError("JetPlusTrackDQM") << "Failed to cast raw jet to CaloJet lead jet in dijet selection. JPT Jet does not appear to have been built from a CaloJet. "
					     << "Histograms not filled. ";
	    return;
	  }

	}
	if(isPFJet){
	  bool LooseCHFcleanedFirstJet =false;
	  bool LooseCHFcleanedSecondJet=false;
	  bool TightCHFcleanedFirstJet =false;
	  bool TightCHFcleanedSecondJet=false;
	  LooseCHFcleanedFirstJet=true;
	  TightCHFcleanedFirstJet=true;
	  if(((*pfJets)[ind1].chargedHadronEnergy()/(*pfJets)[ind1].energy())<=_LooseCHFMin && fabs((*pfJets)[ind1].eta())<2.4) LooseCHFcleanedFirstJet=false; //apply CHF>0 only if |eta|<2.4
	  if(((*pfJets)[ind1].chargedHadronEnergy()/(*pfJets)[ind1].energy())<=_TightCHFMin && fabs((*pfJets)[ind1].eta())<2.4) TightCHFcleanedFirstJet=false; //apply CHF>0 only if |eta|<2.4
	  if(LooseCHFcleanedFirstJet && ((*pfJets)[ind1].neutralHadronEnergy()/(*pfJets)[ind1].energy())<_LooseNHFMax && ((*pfJets)[ind1].chargedEmEnergy()/(*pfJets)[ind1].energy())<_LooseCEFMax && ((*pfJets)[ind1].neutralEmEnergy()/(*pfJets)[ind1].energy())<_LooseNEFMax) LoosecleanedFirstJet=true;
	  if(TightCHFcleanedFirstJet && ((*pfJets)[ind1].neutralHadronEnergy()/(*pfJets)[ind1].energy())<_TightNHFMax && ((*pfJets)[ind1].chargedEmEnergy()/(*pfJets)[ind1].energy())<_TightCEFMax && ((*pfJets)[ind1].neutralEmEnergy()/(*pfJets)[ind1].energy())<_TightNEFMax) TightcleanedFirstJet=true;
	  
	  //second jet
	  LooseCHFcleanedSecondJet=true;
	  TightCHFcleanedSecondJet=true;
	  if(((*pfJets)[ind2].chargedHadronEnergy()/(*pfJets)[ind2].energy())<=_LooseCHFMin && fabs((*pfJets)[ind2].eta())<2.4) LooseCHFcleanedSecondJet=false; //apply CHF>0 only if |eta|<2.4
	  if(((*pfJets)[ind2].chargedHadronEnergy()/(*pfJets)[ind2].energy())<=_TightCHFMin && fabs((*pfJets)[ind2].eta())<2.4) TightCHFcleanedSecondJet=false; //apply CHF>0 only if |eta|<2.4
	  if(LooseCHFcleanedSecondJet && ((*pfJets)[ind2].neutralHadronEnergy()/(*pfJets)[ind2].energy())<_LooseNHFMax && ((*pfJets)[ind2].chargedEmEnergy()/(*pfJets)[ind2].energy())<_LooseCEFMax && ((*pfJets)[ind2].neutralEmEnergy()/(*pfJets)[ind2].energy())<_LooseNEFMax) LoosecleanedSecondJet=true;
	  if(TightCHFcleanedSecondJet && ((*pfJets)[ind2].neutralHadronEnergy()/(*pfJets)[ind2].energy())<_TightNHFMax && ((*pfJets)[ind2].chargedEmEnergy()/(*pfJets)[ind2].energy())<_TightCEFMax && ((*pfJets)[ind2].neutralEmEnergy()/(*pfJets)[ind2].energy())<_TightNEFMax) TightcleanedSecondJet=true;
	  
	}	  
	if(LoosecleanedFirstJet && LoosecleanedSecondJet) { //only if both jets are (loose) cleaned
	  //fill histos for first jet
	  if (mPt)   mPt->Fill (recoJets[0].pt());
	  if (mEta)  mEta->Fill (recoJets[0].eta());
	  if (mPhi)  mPhi->Fill (recoJets[0].phi());
	  if (mPhiVSEta) mPhiVSEta->Fill(recoJets[0].eta(),recoJets[0].phi());
	  if(!isJPTJet){
	    if (mConstituents) mConstituents->Fill (recoJets[0].nConstituents());
	  }
	  if (mPt)   mPt->Fill (recoJets[1].pt());
	  if (mEta)  mEta->Fill (recoJets[1].eta());
	  if (mPhi)  mPhi->Fill (recoJets[1].phi());
	  if (mPhiVSEta) mPhiVSEta->Fill(recoJets[1].eta(),recoJets[1].phi());
	  if(!isJPTJet){
	    if (mConstituents) mConstituents->Fill (recoJets[1].nConstituents());
	  }
	  //PV profiles 
	  if (mPt_profile)           mPt_profile          ->Fill(_numPV, recoJets[0].pt());
	  if (mEta_profile)          mEta_profile         ->Fill(_numPV, recoJets[0].eta());
	  if (mPhi_profile)          mPhi_profile         ->Fill(_numPV, recoJets[0].phi());
	  if(!isJPTJet){
	    if (mConstituents_profile) mConstituents_profile->Fill(_numPV, recoJets[0].nConstituents());
	  }
	  if (mPt_profile)           mPt_profile          ->Fill(_numPV, recoJets[1].pt());
	  if (mEta_profile)          mEta_profile         ->Fill(_numPV, recoJets[1].eta());
	  if (mPhi_profile)          mPhi_profile         ->Fill(_numPV, recoJets[1].phi());
	  if(!isJPTJet){
	    if (mConstituents_profile) mConstituents_profile->Fill(_numPV, recoJets[1].nConstituents());
	  }
	  if(isCaloJet){
	    if (mHFrac)        mHFrac->Fill ((*caloJets)[ind1].energyFractionHadronic());
	    if (mEFrac)        mEFrac->Fill ((*caloJets)[ind1].emEnergyFraction());
	    if (mMaxEInEmTowers)  mMaxEInEmTowers->Fill ((*caloJets)[ind1].maxEInEmTowers());
	    if (mMaxEInHadTowers) mMaxEInHadTowers->Fill ((*caloJets)[ind1].maxEInHadTowers());
	    if (mHFrac)        mHFrac->Fill ((*caloJets)[ind2].energyFractionHadronic());
	    if (mEFrac)        mEFrac->Fill ((*caloJets)[ind2].emEnergyFraction());
	    if (mMaxEInEmTowers)  mMaxEInEmTowers->Fill ((*caloJets)[ind2].maxEInEmTowers());
	    if (mMaxEInHadTowers) mMaxEInHadTowers->Fill ((*caloJets)[ind2].maxEInHadTowers());  

	    if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*caloJets)[ind1].energyFractionHadronic());
	    if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, (*caloJets)[ind1].emEnergyFraction());
	    if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*caloJets)[ind2].energyFractionHadronic());
	    if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, (*caloJets)[ind2].emEnergyFraction());
	  }
	  if(isJPTJet){
	    if (mHFrac)        mHFrac->Fill ((*jptJets)[ind1].chargedHadronEnergyFraction()+(*jptJets)[ind1].neutralHadronEnergyFraction());
	    //if (mEFrac)        mEFrac->Fill ((*jptJets)[ind1].chargedEmEnergyFraction() +(*jptJets)[ind1].neutralEmEnergyFraction());
	    if (mEFrac)        mEFrac->Fill (1.-(*jptJets)[ind1].chargedHadronEnergyFraction()-(*jptJets)[ind1].neutralHadronEnergyFraction());
	    if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*jptJets)[ind1].chargedHadronEnergyFraction()+(*jptJets)[ind1].neutralHadronEnergyFraction());
	    if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, 1.-(*jptJets)[ind1].chargedHadronEnergyFraction()-(*jptJets)[ind1].neutralHadronEnergyFraction());
	    if (fabs((*jptJets)[ind1].eta()) <= 1.3) {	
	      if (mHFrac_Barrel)           mHFrac_Barrel->Fill((*jptJets)[ind1].chargedHadronEnergyFraction()+(*jptJets)[ind1].neutralHadronEnergyFraction());	
	      if (mEFrac_Barrel)           mEFrac_Barrel->Fill(1.-(*jptJets)[ind1].chargedHadronEnergyFraction()-(*jptJets)[ind1].neutralHadronEnergyFraction());	
	    }else if(fabs((*jptJets)[ind1].eta()) <3.0){
	      if (mHFrac_EndCap)           mHFrac_EndCap->Fill((*jptJets)[ind1].chargedHadronEnergyFraction()+(*jptJets)[ind1].neutralHadronEnergyFraction());	
	      if (mEFrac_EndCap)           mEFrac_EndCap->Fill(1.-(*jptJets)[ind1].chargedHadronEnergyFraction()-(*jptJets)[ind1].neutralHadronEnergyFraction());
	    }else{
	      if (mHFrac_Forward)           mHFrac_Forward->Fill((*jptJets)[ind1].chargedHadronEnergyFraction()+(*jptJets)[ind1].neutralHadronEnergyFraction());	
	      if (mEFrac_Forward)           mEFrac_Forward->Fill(1.-(*jptJets)[ind1].chargedHadronEnergyFraction()-(*jptJets)[ind1].neutralHadronEnergyFraction());
	    }
	    if (mHFrac)        mHFrac->Fill ((*jptJets)[ind2].chargedHadronEnergyFraction()+(*jptJets)[ind2].neutralHadronEnergyFraction());
	    if (mEFrac)        mEFrac->Fill (1.-(*jptJets)[ind2].chargedHadronEnergyFraction()-(*jptJets)[ind2].neutralHadronEnergyFraction());
	    if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*jptJets)[ind2].chargedHadronEnergyFraction()+(*jptJets)[ind2].neutralHadronEnergyFraction());
	    if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, 1.-(*jptJets)[ind2].chargedHadronEnergyFraction()-(*jptJets)[ind2].neutralHadronEnergyFraction());
	    if (fabs((*jptJets)[ind2].eta()) <= 1.3) {	
	      if (mHFrac_Barrel)           mHFrac_Barrel->Fill((*jptJets)[ind2].chargedHadronEnergyFraction()+(*jptJets)[ind2].neutralHadronEnergyFraction());	
	      if (mEFrac_Barrel)           mEFrac_Barrel->Fill(1.-(*jptJets)[ind2].chargedHadronEnergyFraction()-(*jptJets)[ind2].neutralHadronEnergyFraction());	
	    }else if(fabs((*jptJets)[ind2].eta()) <3.0){
	      if (mHFrac_EndCap)           mHFrac_EndCap->Fill((*jptJets)[ind2].chargedHadronEnergyFraction()+(*jptJets)[ind2].neutralHadronEnergyFraction());	
	      if (mEFrac_EndCap)           mEFrac_EndCap->Fill(1.-(*jptJets)[ind2].chargedHadronEnergyFraction()-(*jptJets)[ind2].neutralHadronEnergyFraction());
	    }else{
	      if (mHFrac_Forward)           mHFrac_Forward->Fill((*jptJets)[ind2].chargedHadronEnergyFraction()+(*jptJets)[ind2].neutralHadronEnergyFraction());	
	      if (mEFrac_Forward)           mEFrac_Forward->Fill(1.-(*jptJets)[ind2].chargedHadronEnergyFraction()-(*jptJets)[ind2].neutralHadronEnergyFraction());
	    }
	  }
	  if(isPFJet){
	    if (mHFrac)        mHFrac->Fill ((*pfJets)[ind1].chargedHadronEnergyFraction()+(*pfJets)[ind1].neutralHadronEnergyFraction());
	    if (mEFrac)        mEFrac->Fill ((*pfJets)[ind1].chargedEmEnergyFraction() +(*pfJets)[ind1].neutralEmEnergyFraction());
	    
	    if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill ((*pfJets)[ind1].chargedHadronEnergy());
	    if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill ((*pfJets)[ind1].neutralHadronEnergy());
	    if (mChargedEmEnergy) mChargedEmEnergy->Fill((*pfJets)[ind1].chargedEmEnergy());
	    if (mChargedMuEnergy) mChargedMuEnergy->Fill ((*pfJets)[ind1].chargedMuEnergy ());
	    if (mNeutralEmEnergy) mNeutralEmEnergy->Fill((*pfJets)[ind1].neutralEmEnergy());
	    if (mChargedMultiplicity ) mChargedMultiplicity->Fill((*pfJets)[ind1].chargedMultiplicity());
	    if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill((*pfJets)[ind1].neutralMultiplicity());
	    if (mMuonMultiplicity )mMuonMultiplicity->Fill ((*pfJets)[ind1]. muonMultiplicity());
	    //_______________________________________________________
	    if (mNeutralFraction) mNeutralFraction->Fill ((double)(*pfJets)[ind1].neutralMultiplicity()/(double)(*pfJets)[ind1].nConstituents());
	    //Filling variables for second jet
	    if (mHFrac)        mHFrac->Fill ((*pfJets)[ind2].chargedHadronEnergyFraction()+(*pfJets)[ind2].neutralHadronEnergyFraction());
	    if (mEFrac)        mEFrac->Fill ((*pfJets)[ind2].chargedEmEnergyFraction() +(*pfJets)[ind2].neutralEmEnergyFraction());
            
	    if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill ((*pfJets)[ind2].chargedHadronEnergy());
	    if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill ((*pfJets)[ind2].neutralHadronEnergy());
	    if (mChargedEmEnergy) mChargedEmEnergy->Fill((*pfJets)[ind2].chargedEmEnergy());
	    if (mChargedMuEnergy) mChargedMuEnergy->Fill ((*pfJets)[ind2].chargedMuEnergy ());
	    if (mNeutralEmEnergy) mNeutralEmEnergy->Fill((*pfJets)[ind2].neutralEmEnergy());
	    if (mChargedMultiplicity ) mChargedMultiplicity->Fill((*pfJets)[ind2].chargedMultiplicity());
	    if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill((*pfJets)[ind2].neutralMultiplicity());
	    if (mMuonMultiplicity )mMuonMultiplicity->Fill ((*pfJets)[ind2]. muonMultiplicity());
	    //_______________________________________________________
	    if (mNeutralFraction) mNeutralFraction->Fill ((double)(*pfJets)[ind2].neutralMultiplicity()/(double)(*pfJets)[ind2].nConstituents());

	    if (mPt_profile)           mPt_profile          ->Fill(_numPV, (*pfJets)[ind1].pt());
	    if (mEta_profile)          mEta_profile         ->Fill(_numPV, (*pfJets)[ind1].eta());
	    if (mPhi_profile)          mPhi_profile         ->Fill(_numPV, (*pfJets)[ind1].phi());
	    if (mConstituents_profile) mConstituents_profile->Fill(_numPV, (*pfJets)[ind1].nConstituents());
	    if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*pfJets)[ind1].chargedHadronEnergyFraction() + (*pfJets)[ind1].neutralHadronEnergyFraction());
	    if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, (*pfJets)[ind1].chargedEmEnergyFraction()     + (*pfJets)[ind1].neutralEmEnergyFraction());
	    
	    if (mChargedHadronEnergy_profile) mChargedHadronEnergy_profile->Fill(_numPV, (*pfJets)[ind1].chargedHadronEnergy());
	    if (mNeutralHadronEnergy_profile) mNeutralHadronEnergy_profile->Fill(_numPV, (*pfJets)[ind1].neutralHadronEnergy());
	    if (mChargedEmEnergy_profile)     mChargedEmEnergy_profile    ->Fill(_numPV, (*pfJets)[ind1].chargedEmEnergy());
	    if (mChargedMuEnergy_profile)     mChargedMuEnergy_profile    ->Fill(_numPV, (*pfJets)[ind1].chargedMuEnergy());
	    if (mNeutralEmEnergy_profile)     mNeutralEmEnergy_profile    ->Fill(_numPV, (*pfJets)[ind1].neutralEmEnergy());
	    if (mChargedMultiplicity_profile) mChargedMultiplicity_profile->Fill(_numPV, (*pfJets)[ind1].chargedMultiplicity());
	    if (mNeutralMultiplicity_profile) mNeutralMultiplicity_profile->Fill(_numPV, (*pfJets)[ind1].neutralMultiplicity());
	    if (mMuonMultiplicity_profile)    mMuonMultiplicity_profile   ->Fill(_numPV, (*pfJets)[ind1].muonMultiplicity());

	    if (mPt_profile)           mPt_profile          ->Fill(_numPV, (*pfJets)[ind2].pt());
	    if (mEta_profile)          mEta_profile         ->Fill(_numPV, (*pfJets)[ind2].eta());
	    if (mPhi_profile)          mPhi_profile         ->Fill(_numPV, (*pfJets)[ind2].phi());
	    if (mConstituents_profile) mConstituents_profile->Fill(_numPV, (*pfJets)[ind2].nConstituents());
	    if (mHFrac_profile)        mHFrac_profile       ->Fill(_numPV, (*pfJets)[ind2].chargedHadronEnergyFraction() + (*pfJets)[ind2].neutralHadronEnergyFraction());
	    if (mEFrac_profile)        mEFrac_profile       ->Fill(_numPV, (*pfJets)[ind2].chargedEmEnergyFraction()     + (*pfJets)[ind2].neutralEmEnergyFraction());
	    
	    if (mChargedHadronEnergy_profile) mChargedHadronEnergy_profile->Fill(_numPV, (*pfJets)[ind2].chargedHadronEnergy());
	    if (mNeutralHadronEnergy_profile) mNeutralHadronEnergy_profile->Fill(_numPV, (*pfJets)[ind2].neutralHadronEnergy());
	    if (mChargedEmEnergy_profile)     mChargedEmEnergy_profile    ->Fill(_numPV, (*pfJets)[ind2].chargedEmEnergy());
	    if (mChargedMuEnergy_profile)     mChargedMuEnergy_profile    ->Fill(_numPV, (*pfJets)[ind2].chargedMuEnergy());
	    if (mNeutralEmEnergy_profile)     mNeutralEmEnergy_profile    ->Fill(_numPV, (*pfJets)[ind2].neutralEmEnergy());
	    if (mChargedMultiplicity_profile) mChargedMultiplicity_profile->Fill(_numPV, (*pfJets)[ind2].chargedMultiplicity());
	    if (mNeutralMultiplicity_profile) mNeutralMultiplicity_profile->Fill(_numPV, (*pfJets)[ind2].neutralMultiplicity());
	    if (mMuonMultiplicity_profile)    mMuonMultiplicity_profile   ->Fill(_numPV, (*pfJets)[ind2].muonMultiplicity());
	  }
	  if(fillJIDPassFrac==1) {
	    if(LoosecleanedFirstJet) {
	      mLooseJIDPassFractionVSeta->Fill(recoJets[0].eta(),1.);
	      mLooseJIDPassFractionVSpt->Fill(recoJets[0].pt(),1.);
	    } else {
	      mLooseJIDPassFractionVSeta->Fill(recoJets[0].eta(),0.);
	      mLooseJIDPassFractionVSpt->Fill(recoJets[0].pt(),0.);
	    }
	    if(LoosecleanedSecondJet) {
	      mLooseJIDPassFractionVSeta->Fill(recoJets[1].eta(),1.);
	      mLooseJIDPassFractionVSpt->Fill(recoJets[1].pt(),1.);
	    } else {
	      mLooseJIDPassFractionVSeta->Fill(recoJets[1].eta(),0.);
	      mLooseJIDPassFractionVSpt->Fill(recoJets[1].pt(),0.);
	    }
	    //TIGHT JID
	    if(TightcleanedFirstJet) {
	      mTightJIDPassFractionVSeta->Fill(recoJets[0].eta(),1.);
	      mTightJIDPassFractionVSpt->Fill(recoJets[0].pt(),1.);
	    } else {
	      mTightJIDPassFractionVSeta->Fill(recoJets[0].eta(),0.);
	      mTightJIDPassFractionVSpt->Fill(recoJets[0].pt(),0.);
	    }
	    if(TightcleanedSecondJet) {
	      mTightJIDPassFractionVSeta->Fill(recoJets[1].eta(),1.);
	      mTightJIDPassFractionVSpt->Fill(recoJets[1].pt(),1.);
	    } else {
	      mTightJIDPassFractionVSeta->Fill(recoJets[1].eta(),0.);
	      mTightJIDPassFractionVSpt->Fill(recoJets[1].pt(),0.);
	    }

         }//if fillJIDPassFrac
	}//two leading jets loose cleaned
      }//DPhi cut
    }//leading jets within endcaps -> extend to FWD?
    //now do the dijet balance and asymmetry calculations
    if (fabs(recoJets[0].eta() < 1.4)) {
      double pt_dijet = (recoJets[0].pt() + recoJets[1].pt())/2;      
      double dPhiDJ = fabs(recoJets[0].phi()-recoJets[1].phi());
      if(dPhiDJ>acos(-1.)){
	dPhiDJ=2*acos(-1.)-dPhiDJ;
      }
      if (dPhiDJ > 2.7) {
	double pt_probe;
	double pt_barrel;
	int jet1, jet2;
	int randJet = rand() % 2;
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
	    if (pt3 > _asymmetryThirdJetCut){
	      thirdJetCut=false;
	    } 
	  }
	  if(thirdJetCut){
	    double dijetAsymmetry =(recoJets[jet1].pt() - recoJets[jet2].pt()) / (recoJets[jet1].pt() + recoJets[jet2].pt());
	    mDijetAsymmetry->Fill(dijetAsymmetry);
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
	  if (pt3/pt_dijet > _balanceThirdJetCut){ 
	    thirdJetCut = false;
	  }
	}
	if (thirdJetCut) {
	  double dijetBalance = (pt_probe - pt_barrel) / pt_dijet;
	  mDijetBalance->Fill(dijetBalance);
	}// end restriction on third jet pt ratio in balance calculation
       
      }// dPhi > 2.7
    }// leading jet eta cut for asymmetry and balance calculations
  
  }//at least two hard corrected jets
  
}

// ***********************************************************
void JetAnalyzer::endJob(void) {
  
  if(isJPTJet || isCaloJet){
    delete jetID;
  }
  LogTrace(metname)<<"[JetAnalyzer] Saving the histos";


  //--- Jet


  if(!mOutputFile.empty() && &*edm::Service<DQMStore>()){
      //dbe->save(mOutputFile);
    edm::Service<DQMStore>()->save(mOutputFile);
  }
  
}


namespace jetAnalysis {
  
  TrackPropagatorToCalo::TrackPropagatorToCalo()
    : magneticField_(NULL),
      propagator_(NULL),
      magneticFieldCacheId_(0),
      propagatorCacheId_(0)
    {}
  
  void TrackPropagatorToCalo::update(const edm::EventSetup& eventSetup)
  {
    //update magnetic filed if necessary
    const IdealMagneticFieldRecord& magneticFieldRecord = eventSetup.get<IdealMagneticFieldRecord>();
    const uint32_t newMagneticFieldCacheId = magneticFieldRecord.cacheIdentifier();
    if ((newMagneticFieldCacheId != magneticFieldCacheId_) || !magneticField_) {
      edm::ESHandle<MagneticField> magneticFieldHandle;
      magneticFieldRecord.get(magneticFieldHandle);
      magneticField_ = magneticFieldHandle.product();
      magneticFieldCacheId_ = newMagneticFieldCacheId;
    }
    //update propagator if necessary
    const TrackingComponentsRecord& trackingComponentsRecord = eventSetup.get<TrackingComponentsRecord>();
    const uint32_t newPropagatorCacheId = trackingComponentsRecord.cacheIdentifier();
    if ((propagatorCacheId_ != newPropagatorCacheId) || !propagator_) {
      edm::ESHandle<Propagator> propagatorHandle;
      trackingComponentsRecord.get("SteppingHelixPropagatorAlong",propagatorHandle);
      propagator_ = propagatorHandle.product();
      propagatorCacheId_ = newPropagatorCacheId;
    }
  }
  
  inline math::XYZPoint TrackPropagatorToCalo::impactPoint(const reco::Track& track) const
  {
    return JetTracksAssociationDRCalo::propagateTrackToCalorimeter(track,*magneticField_,*propagator_);
  }
  /*
  StripSignalOverNoiseCalculator::StripSignalOverNoiseCalculator(const std::string& theQualityLabel)
    : qualityLabel_(theQualityLabel),
      quality_(NULL),
      noise_(NULL),
      gain_(NULL),
      qualityCacheId_(0),
      noiseCacheId_(0),
      gainCacheId_(0)
    {}
  
  void StripSignalOverNoiseCalculator::update(const edm::EventSetup& eventSetup)
  {
    //update the quality if necessary
    const SiStripQualityRcd& qualityRecord = eventSetup.get<SiStripQualityRcd>();
    const uint32_t newQualityCacheId = qualityRecord.cacheIdentifier();
    if ((newQualityCacheId != qualityCacheId_) || !quality_) {
      edm::ESHandle<SiStripQuality> qualityHandle;
      qualityRecord.get(qualityLabel_,qualityHandle);
      quality_ = qualityHandle.product();
      qualityCacheId_ = newQualityCacheId;
    }
    //update the noise if necessary
    const SiStripNoisesRcd& noiseRecord = eventSetup.get<SiStripNoisesRcd>();
    const uint32_t newNoiseCacheId = noiseRecord.cacheIdentifier();
    if ((newNoiseCacheId != noiseCacheId_) || !noise_) {
      edm::ESHandle<SiStripNoises> noiseHandle;
      noiseRecord.get(noiseHandle);
      noise_ = noiseHandle.product();
      noiseCacheId_ = newNoiseCacheId;
    }
    //update the gain if necessary
    const SiStripGainRcd& gainRecord = eventSetup.get<SiStripGainRcd>();
    const uint32_t newGainCacheId = gainRecord.cacheIdentifier();
    if ((newGainCacheId != gainCacheId_) || !gain_) {
      edm::ESHandle<SiStripGain> gainHandle;
      gainRecord.get(gainHandle);
      gain_ = gainHandle.product();
      gainCacheId_ = newGainCacheId;
    }
    }*/
  /*
  double StripSignalOverNoiseCalculator::signalOverNoise(const SiStripCluster& cluster,
							 const uint32_t& detId) const
  {
    //const uint32_t detId = cluster.geographicalId();
    
    const uint16_t firstStrip = cluster.firstStrip();
    const SiStripQuality::Range& qualityRange = quality_->getRange(detId);
    const SiStripNoises::Range& noiseRange = noise_->getRange(detId);
    const SiStripApvGain::Range& gainRange = gain_->getRange(detId);
    double signal = 0;
    double noise2 = 0;
    unsigned int nNonZeroStrips = 0;
    const std::vector<uint8_t>& clusterAmplitudes = cluster.amplitudes();
    const std::vector<uint8_t>::const_iterator clusterAmplitudesEnd = clusterAmplitudes.end();
    const std::vector<uint8_t>::const_iterator clusterAmplitudesBegin = clusterAmplitudes.begin();
    for (std::vector<uint8_t>::const_iterator iAmp = clusterAmplitudesBegin; iAmp != clusterAmplitudesEnd; ++iAmp) {
      const uint8_t adc = *iAmp;
      const uint16_t strip = iAmp-clusterAmplitudesBegin+firstStrip;
      const bool stripBad = quality_->IsStripBad(qualityRange,strip);
      const double noise = noise_->getNoise(strip,noiseRange);
      const double gain = gain_->getStripGain(strip,gainRange);
      signal += adc;
      if (adc) ++nNonZeroStrips;
      const double noiseContrib = (stripBad ? 0 : noise/gain);
      noise2 += noiseContrib*noiseContrib;
    }
    const double noise = sqrt(noise2/nNonZeroStrips);
    if (noise) return signal/noise;
    else return 0;
  }
  */ 
}

