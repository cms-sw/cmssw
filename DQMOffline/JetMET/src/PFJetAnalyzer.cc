/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/04/12 15:42:58 $
 *  $Revision: 1.24 $
 *  \author F. Chlebana - Fermilab
 */

#include "DQMOffline/JetMET/interface/PFJetAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/PFJet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace edm;


PFJetAnalyzer::PFJetAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;
  _leadJetFlag = 0;
  _JetLoPass   = 0;
  _JetHiPass   = 0;
  _ptThreshold = 5.;
  _asymmetryThirdJetCut = 30.;
  _balanceThirdJetCut   = 0.2;
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

}


PFJetAnalyzer::~PFJetAnalyzer() { }


void PFJetAnalyzer::beginJob(DQMStore * dbe) {

  metname = "pFJetAnalyzer";

  LogTrace(metname)<<"[PFJetAnalyzer] Parameters initialization";
  //dbe->setCurrentFolder("JetMET/Jet/PFJets");//old version, now name set to source, which 
  //can be set for each instance of PFJetAnalyzer called inside JetMETAnalyzer. Useful, e.g., to 
  //name differently the dir for all jets and cleaned jets 
  dbe->setCurrentFolder("JetMET/Jet/"+_source);
  // dbe->setCurrentFolder("JetMET/Jet/PFJets");
			
  jetME = dbe->book1D("jetReco", "jetReco", 3, 1, 4);
  jetME->setBinLabel(2,"PFJets",1);

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

  _ptThreshold = parameters.getParameter<double>("ptThreshold");
  _asymmetryThirdJetCut = parameters.getParameter<double>("asymmetryThirdJetCut");
  _balanceThirdJetCut   = parameters.getParameter<double>("balanceThirdJetCut");

  _TightCHFMin = parameters.getParameter<double>("TightCHFMin");
  _TightNHFMax = parameters.getParameter<double>("TightNHFMax");
  _TightCEFMax = parameters.getParameter<double>("TightCEFMax");
  _TightNEFMax = parameters.getParameter<double>("TightNEFMax");
  _LooseCHFMin = parameters.getParameter<double>("LooseCHFMin");
  _LooseNHFMax = parameters.getParameter<double>("LooseNHFMax");
  _LooseCEFMax = parameters.getParameter<double>("LooseCEFMax");
  _LooseNEFMax = parameters.getParameter<double>("LooseNEFMax");

  fillpfJIDPassFrac  = parameters.getParameter<int>("fillpfJIDPassFrac");
  makedijetselection = parameters.getParameter<int>("makedijetselection");

  _ThisCHFMin = parameters.getParameter<double>("ThisCHFMin");
  _ThisNHFMax = parameters.getParameter<double>("ThisNHFMax");
  _ThisCEFMax = parameters.getParameter<double>("ThisCEFMax");
  _ThisNEFMax = parameters.getParameter<double>("ThisNEFMax");

  // Generic Jet Parameters
  mPt                      = dbe->book1D("Pt",  "Pt", ptBin, ptMin, ptMax);
  mEta                     = dbe->book1D("Eta", "Eta", etaBin, etaMin, etaMax);
  mPhi                     = dbe->book1D("Phi", "Phi", phiBin, phiMin, phiMax);
  mConstituents            = dbe->book1D("Constituents", "# of constituents", 50, 0, 100);
  mHFrac                   = dbe->book1D("HFrac", "HFrac", 120, -0.1, 1.1);
  mEFrac                   = dbe->book1D("EFrac", "EFrac", 120, -0.1, 1.1);


  // Book NPV profiles
  //----------------------------------------------------------------------------
  mPt_profile           = dbe->bookProfile("Pt_profile",           "pt",                nbinsPV, PVlow, PVup,  ptBin,  ptMin,  ptMax);
  mEta_profile          = dbe->bookProfile("Eta_profile",          "eta",               nbinsPV, PVlow, PVup, etaBin, etaMin, etaMax);
  mPhi_profile          = dbe->bookProfile("Phi_profile",          "phi",               nbinsPV, PVlow, PVup, phiBin, phiMin, phiMax);
  mConstituents_profile = dbe->bookProfile("Constituents_profile", "# of constituents", nbinsPV, PVlow, PVup,     50,      0,    100);
  mHFrac_profile        = dbe->bookProfile("HFrac_profile",        "Hfrac",             nbinsPV, PVlow, PVup,    120,   -0.1,    1.1);
  mEFrac_profile        = dbe->bookProfile("EFrac_profile",        "Efrac",             nbinsPV, PVlow, PVup,    120,   -0.1,    1.1);


  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  mPt_profile          ->setAxisTitle("nvtx",1);
  mEta_profile         ->setAxisTitle("nvtx",1);
  mPhi_profile         ->setAxisTitle("nvtx",1);
  mConstituents_profile->setAxisTitle("nvtx",1);
  mHFrac_profile       ->setAxisTitle("nvtx",1);
  mEFrac_profile       ->setAxisTitle("nvtx",1);


  //mE                       = dbe->book1D("E", "E", eBin, eMin, eMax);
  //mP                       = dbe->book1D("P", "P", pBin, pMin, pMax);
  //mMass                    = dbe->book1D("Mass", "Mass", 100, 0, 25);
  //
  mPhiVSEta                = dbe->book2D("PhiVSEta", "PhiVSEta", 50, etaMin, etaMax, 24, phiMin, phiMax);
  
  if(makedijetselection!=1) {
    mPt_1                    = dbe->book1D("Pt1", "Pt1", 50, 0, 100);
    mPt_2                    = dbe->book1D("Pt2", "Pt2", 60, 0, 300);  
    mPt_3                    = dbe->book1D("Pt3", "Pt3", 100, 0, 5000);
    
    // Low and high pt trigger paths
    mPt_Lo                  = dbe->book1D("Pt_Lo", "Pt (Pass Low Pt Jet Trigger)", 20, 0, 100);
    //mEta_Lo                 = dbe->book1D("Eta_Lo", "Eta (Pass Low Pt Jet Trigger)", etaBin, etaMin, etaMax);
    mPhi_Lo                 = dbe->book1D("Phi_Lo", "Phi (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
    
    mPt_Hi                  = dbe->book1D("Pt_Hi", "Pt (Pass Hi Pt Jet Trigger)", 100, 0, 300);
    mEta_Hi                 = dbe->book1D("Eta_Hi", "Eta (Pass Hi Pt Jet Trigger)", etaBin, etaMin, etaMax);
    mPhi_Hi                 = dbe->book1D("Phi_Hi", "Phi (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);
    
    mNJets                   = dbe->book1D("NJets", "number of jets", 100, 0, 100);
    
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
    
    //mPt_Barrel_Lo            = dbe->book1D("Pt_Barrel_Lo", "Pt Barrel (Pass Low Pt Jet Trigger)", 20, 0, 100);
    //mPhi_Barrel_Lo           = dbe->book1D("Phi_Barrel_Lo", "Phi Barrel (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
    mConstituents_Barrel     = dbe->book1D("Constituents_Barrel", "Constituents Barrel", 50, 0, 100);
    mHFrac_Barrel            = dbe->book1D("HFrac_Barrel", "HFrac Barrel", 100, 0, 1);
    mEFrac_Barrel            = dbe->book1D("EFrac_Barrel", "EFrac Barrel", 110, -0.05, 1.05);
    
    //mPt_EndCap_Lo            = dbe->book1D("Pt_EndCap_Lo", "Pt EndCap (Pass Low Pt Jet Trigger)", 20, 0, 100);
    //mPhi_EndCap_Lo           = dbe->book1D("Phi_EndCap_Lo", "Phi EndCap (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
    mConstituents_EndCap     = dbe->book1D("Constituents_EndCap", "Constituents EndCap", 50, 0, 100);
    mHFrac_EndCap            = dbe->book1D("HFrac_Endcap", "HFrac EndCap", 100, 0, 1);
    mEFrac_EndCap            = dbe->book1D("EFrac_Endcap", "EFrac EndCap", 110, -0.05, 1.05);
    
    //mPt_Forward_Lo           = dbe->book1D("Pt_Forward_Lo", "Pt Forward (Pass Low Pt Jet Trigger)", 20, 0, 100);
    //mPhi_Forward_Lo          = dbe->book1D("Phi_Forward_Lo", "Phi Forward (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
    mConstituents_Forward    = dbe->book1D("Constituents_Forward", "Constituents Forward", 50, 0, 100);
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

    mPhi_EndCap              = dbe->book1D("Phi_EndCap", "Phi_EndCap", phiBin, phiMin, phiMax);
    //mE_EndCap                = dbe->book1D("E_EndCap", "E_EndCap", eBin, eMin, eMax);
    mPt_EndCap               = dbe->book1D("Pt_EndCap", "Pt_EndCap", ptBin, ptMin, ptMax);
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

    mPhi_Forward             = dbe->book1D("Phi_Forward", "Phi_Forward", phiBin, phiMin, phiMax);
    //mE_Forward               = dbe->book1D("E_Forward", "E_Forward", eBin, eMin, eMax);
    mPt_Forward              = dbe->book1D("Pt_Forward", "Pt_Forward", ptBin, ptMin, ptMax);
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



    // Leading Jet Parameters
    mEtaFirst                = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5);
    mPhiFirst                = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5);
    //mEFirst                  = dbe->book1D("EFirst", "EFirst", 100, 0, 1000);
    mPtFirst                 = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500);
    if(fillpfJIDPassFrac==1) {
      mLooseJIDPassFractionVSeta= dbe->bookProfile("LooseJIDPassFractionVSeta","LooseJIDPassFractionVSeta",etaBin, etaMin, etaMax,0.,1.2);
      mLooseJIDPassFractionVSpt= dbe->bookProfile("LooseJIDPassFractionVSpt","LooseJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
      mTightJIDPassFractionVSeta= dbe->bookProfile("TightJIDPassFractionVSeta","TightJIDPassFractionVSeta",etaBin, etaMin, etaMax,0.,1.2);
      mTightJIDPassFractionVSpt= dbe->bookProfile("TightJIDPassFractionVSpt","TightJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
    }
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
  mChargedHadronEnergy_profile = dbe->bookProfile("mChargedHadronEnergy_profile", "charged HAD energy",   nbinsPV, PVlow, PVup, 100, 0, 100);
  mNeutralHadronEnergy_profile = dbe->bookProfile("mNeutralHadronEnergy_profile", "neutral HAD energy",   nbinsPV, PVlow, PVup, 100, 0, 100);
  mChargedEmEnergy_profile     = dbe->bookProfile("mChargedEmEnergy_profile",     "charged EM energy",    nbinsPV, PVlow, PVup, 100, 0, 100);
  mChargedMuEnergy_profile     = dbe->bookProfile("mChargedMuEnergy_profile",     "charged Mu energy",    nbinsPV, PVlow, PVup, 100, 0, 100);
  mNeutralEmEnergy_profile     = dbe->bookProfile("mNeutralEmEnergy_profile",     "neutral EM energy",    nbinsPV, PVlow, PVup, 100, 0, 100);
  mChargedMultiplicity_profile = dbe->bookProfile("mChargedMultiplicity_profile", "charged multiplicity", nbinsPV, PVlow, PVup, 100, 0, 100);
  mNeutralMultiplicity_profile = dbe->bookProfile("mNeutralMultiplicity_profile", "neutral multiplicity", nbinsPV, PVlow, PVup, 100, 0, 100);
  mMuonMultiplicity_profile    = dbe->bookProfile("mMuonMultiplicity_profile",    "muon multiplicity",    nbinsPV, PVlow, PVup, 100, 0, 100);

  if (makedijetselection != 1) {
    mNJets_profile = dbe->bookProfile("NJets_profile", "number of jets", nbinsPV, PVlow, PVup, 100, 0, 100);
  }


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

  if (makedijetselection != 1) {
    mNJets_profile->setAxisTitle("nvtx",1);
  }


  //__________________________________________________
  mNeutralFraction     = dbe->book1D("NeutralFraction","Neutral Fraction",100,0,1);
  //}
  
  mDPhi                = dbe->book1D("DPhi", "dPhi btw the two leading jets", 100, 0., acos(-1.));
  
  if(makedijetselection==1) {
    mDijetAsymmetry                   = dbe->book1D("DijetAsymmetry", "DijetAsymmetry", 100, -1., 1.);
    mDijetBalance                     = dbe->book1D("DijetBalance",   "DijetBalance",   100, -2., 2.);
    if (fillpfJIDPassFrac==1) {
      mLooseJIDPassFractionVSeta  = dbe->bookProfile("LooseJIDPassFractionVSeta","LooseJIDPassFractionVSeta",50, -3., 3.,0.,1.2);
      mLooseJIDPassFractionVSpt   = dbe->bookProfile("LooseJIDPassFractionVSpt","LooseJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
      mTightJIDPassFractionVSeta  = dbe->bookProfile("TightJIDPassFractionVSeta","TightJIDPassFractionVSeta",50, -3., 3.,0.,1.2);
      mTightJIDPassFractionVSpt   = dbe->bookProfile("TightJIDPassFractionVSpt","TightJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
      
    }
  }
  
}

void PFJetAnalyzer::endJob() {
  
}

void PFJetAnalyzer::analyze(const edm::Event&            iEvent,
			    const edm::EventSetup&       iSetup,
			    const reco::PFJetCollection& pfJets,
			    const int                    numPV)
{
  int numofjets=0;
  double  fstPhi=0.;
  double  sndPhi=0.;
  double  diff = 0.;
  double  corr = 0.;
  double  dphi = -999. ;

  bool Thiscleaned=false; 
  bool Loosecleaned=false; 
  bool Tightcleaned=false; 
  bool ThisCHFcleaned=false;
  bool LooseCHFcleaned=false;
  bool TightCHFcleaned=false;

  srand( iEvent.id().event() % 10000);


  if (makedijetselection == 1) {
    //Dijet selection - careful: the pT is uncorrected!
    //if(makedijetselection==1 && pfJets.size()>=2){
    if(pfJets.size()>=2){
      double  dphiDJ = -999. ;

      bool LoosecleanedFirstJet =false; 
      bool LoosecleanedSecondJet=false; 
      bool TightcleanedFirstJet =false; 
      bool TightcleanedSecondJet=false; 
      bool LooseCHFcleanedFirstJet =false;
      bool LooseCHFcleanedSecondJet=false;
      bool TightCHFcleanedFirstJet =false;
      bool TightCHFcleanedSecondJet=false;

      //both jets pass pt threshold
      if ((pfJets.at(0)).pt() > _ptThreshold && (pfJets.at(1)).pt() > _ptThreshold ) {
	if(fabs((pfJets.at(0)).eta())<3. && fabs((pfJets.at(1)).eta())<3. ){
	  //calculate dphi
	  dphiDJ = fabs((pfJets.at(0)).phi()-(pfJets.at(1)).phi());
	  if (dphiDJ > 3.14) dphiDJ=fabs(dphiDJ -6.28 );
	  //fill DPhi histo (before cutting)
	  if (mDPhi) mDPhi->Fill (dphiDJ);
	  //dphi cut
	  if(fabs(dphiDJ)>2.1){
	    //first jet
	    LooseCHFcleanedFirstJet=true;
	    TightCHFcleanedFirstJet=true;
	    if((pfJets.at(0).chargedHadronEnergy()/pfJets.at(0).energy())<=_LooseCHFMin && fabs(pfJets.at(0).eta())<2.4) LooseCHFcleanedFirstJet=false; //apply CHF>0 only if |eta|<2.4
	    if((pfJets.at(0).chargedHadronEnergy()/pfJets.at(0).energy())<=_TightCHFMin && fabs(pfJets.at(0).eta())<2.4) TightCHFcleanedFirstJet=false; //apply CHF>0 only if |eta|<2.4
	    if(LooseCHFcleanedFirstJet && (pfJets.at(0).neutralHadronEnergy()/pfJets.at(0).energy())<_LooseNHFMax && (pfJets.at(0).chargedEmEnergy()/pfJets.at(0).energy())<_LooseCEFMax && (pfJets.at(0).neutralEmEnergy()/pfJets.at(0).energy())<_LooseNEFMax) LoosecleanedFirstJet=true;
	    if(TightCHFcleanedFirstJet && (pfJets.at(0).neutralHadronEnergy()/pfJets.at(0).energy())<_TightNHFMax && (pfJets.at(0).chargedEmEnergy()/pfJets.at(0).energy())<_TightCEFMax && (pfJets.at(0).neutralEmEnergy()/pfJets.at(0).energy())<_TightNEFMax) TightcleanedFirstJet=true;

	    //second jet
	    LooseCHFcleanedSecondJet=true;
	    TightCHFcleanedSecondJet=true;
	    if((pfJets.at(1).chargedHadronEnergy()/pfJets.at(1).energy())<=_LooseCHFMin && fabs(pfJets.at(1).eta())<2.4) LooseCHFcleanedSecondJet=false; //apply CHF>0 only if |eta|<2.4
	    if((pfJets.at(1).chargedHadronEnergy()/pfJets.at(1).energy())<=_TightCHFMin && fabs(pfJets.at(1).eta())<2.4) TightCHFcleanedSecondJet=false; //apply CHF>0 only if |eta|<2.4
	    if(LooseCHFcleanedSecondJet && (pfJets.at(1).neutralHadronEnergy()/pfJets.at(1).energy())<_LooseNHFMax && (pfJets.at(1).chargedEmEnergy()/pfJets.at(1).energy())<_LooseCEFMax && (pfJets.at(1).neutralEmEnergy()/pfJets.at(1).energy())<_LooseNEFMax) LoosecleanedSecondJet=true;
	    if(TightCHFcleanedSecondJet && (pfJets.at(1).neutralHadronEnergy()/pfJets.at(1).energy())<_TightNHFMax && (pfJets.at(1).chargedEmEnergy()/pfJets.at(1).energy())<_TightCEFMax && (pfJets.at(1).neutralEmEnergy()/pfJets.at(1).energy())<_TightNEFMax) TightcleanedSecondJet=true;
      
	    if(fillpfJIDPassFrac==1) {
	      //fill the profile for jid efficiency 
	      if(LoosecleanedFirstJet) {
		mLooseJIDPassFractionVSeta->Fill(pfJets.at(0).eta(),1.);
		mLooseJIDPassFractionVSpt->Fill(pfJets.at(0).pt(),1.);
	      } else {
		mLooseJIDPassFractionVSeta->Fill(pfJets.at(0).eta(),0.);
		mLooseJIDPassFractionVSpt->Fill(pfJets.at(0).pt(),0.);
	      }
	      if(TightcleanedFirstJet) {
		mTightJIDPassFractionVSeta->Fill(pfJets.at(0).eta(),1.);
		mTightJIDPassFractionVSpt->Fill(pfJets.at(0).pt(),1.);
	      } else {
		mTightJIDPassFractionVSeta->Fill(pfJets.at(0).eta(),0.);
		mTightJIDPassFractionVSpt->Fill(pfJets.at(0).pt(),0.);
	      }

	      if(LoosecleanedSecondJet) {
		mLooseJIDPassFractionVSeta->Fill(pfJets.at(1).eta(),1.);
		mLooseJIDPassFractionVSpt->Fill(pfJets.at(1).pt(),1.);
	      } else {
		mLooseJIDPassFractionVSeta->Fill(pfJets.at(1).eta(),0.);
		mLooseJIDPassFractionVSpt->Fill(pfJets.at(1).pt(),0.);
	      }
	      if(TightcleanedSecondJet) {
		mTightJIDPassFractionVSeta->Fill(pfJets.at(1).eta(),1.);
		mTightJIDPassFractionVSpt->Fill(pfJets.at(1).pt(),1.);
	      } else {
		mTightJIDPassFractionVSeta->Fill(pfJets.at(1).eta(),0.);
		mTightJIDPassFractionVSpt->Fill(pfJets.at(1).pt(),0.);
	      }
	    }
	    
	    if(LoosecleanedFirstJet && LoosecleanedSecondJet) {
	      //Filling variables for first jet
	      if (mPt)   mPt->Fill (pfJets.at(0).pt());
	      if (mEta)  mEta->Fill (pfJets.at(0).eta());
	      if (mPhi)  mPhi->Fill (pfJets.at(0).phi());
	      if (mPhiVSEta) mPhiVSEta->Fill(pfJets.at(0).eta(),pfJets.at(0).phi());
	      
	      if (mConstituents) mConstituents->Fill (pfJets.at(0).nConstituents());
	      if (mHFrac)        mHFrac->Fill (pfJets.at(0).chargedHadronEnergyFraction()+pfJets.at(0).neutralHadronEnergyFraction());
	      if (mEFrac)        mEFrac->Fill (pfJets.at(0).chargedEmEnergyFraction() +pfJets.at(0).neutralEmEnergyFraction());
	      
	      //if (mE) mE->Fill (pfJets.at(0).energy());
	      //if (mP) mP->Fill (pfJets.at(0).p());
	      //if (mMass) mMass->Fill (pfJets.at(0).mass());
            
	      if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill (pfJets.at(0).chargedHadronEnergy());
	      if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill (pfJets.at(0).neutralHadronEnergy());
	      if (mChargedEmEnergy) mChargedEmEnergy->Fill(pfJets.at(0).chargedEmEnergy());
	      if (mChargedMuEnergy) mChargedMuEnergy->Fill (pfJets.at(0).chargedMuEnergy ());
	      if (mNeutralEmEnergy) mNeutralEmEnergy->Fill(pfJets.at(0).neutralEmEnergy());
	      if (mChargedMultiplicity ) mChargedMultiplicity->Fill(pfJets.at(0).chargedMultiplicity());
	      if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill(pfJets.at(0).neutralMultiplicity());
	      if (mMuonMultiplicity )mMuonMultiplicity->Fill (pfJets.at(0). muonMultiplicity());
	      //_______________________________________________________
	      if (mNeutralFraction) mNeutralFraction->Fill (pfJets.at(0).neutralMultiplicity()/pfJets.at(0).nConstituents());


	      //Filling variables for second jet
	      if (mPt)   mPt->Fill (pfJets.at(1).pt());
	      if (mEta)  mEta->Fill (pfJets.at(1).eta());
	      if (mPhi)  mPhi->Fill (pfJets.at(1).phi());
	      if (mPhiVSEta) mPhiVSEta->Fill(pfJets.at(1).eta(),pfJets.at(1).phi());
	      
	      if (mConstituents) mConstituents->Fill (pfJets.at(1).nConstituents());
	      if (mHFrac)        mHFrac->Fill (pfJets.at(1).chargedHadronEnergyFraction()+pfJets.at(1).neutralHadronEnergyFraction());
	      if (mEFrac)        mEFrac->Fill (pfJets.at(1).chargedEmEnergyFraction() +pfJets.at(1).neutralEmEnergyFraction());
	      
	      //if (mE) mE->Fill (pfJets.at(1).energy());
	      //if (mP) mP->Fill (pfJets.at(1).p());
	      //if (mMass) mMass->Fill (pfJets.at(1).mass());
            
	      if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill (pfJets.at(1).chargedHadronEnergy());
	      if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill (pfJets.at(1).neutralHadronEnergy());
	      if (mChargedEmEnergy) mChargedEmEnergy->Fill(pfJets.at(1).chargedEmEnergy());
	      if (mChargedMuEnergy) mChargedMuEnergy->Fill (pfJets.at(1).chargedMuEnergy ());
	      if (mNeutralEmEnergy) mNeutralEmEnergy->Fill(pfJets.at(1).neutralEmEnergy());
	      if (mChargedMultiplicity ) mChargedMultiplicity->Fill(pfJets.at(1).chargedMultiplicity());
	      if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill(pfJets.at(1).neutralMultiplicity());
	      if (mMuonMultiplicity )mMuonMultiplicity->Fill (pfJets.at(1). muonMultiplicity());
	      //_______________________________________________________
	      if (mNeutralFraction) mNeutralFraction->Fill (pfJets.at(1).neutralMultiplicity()/pfJets.at(1).nConstituents());


	      // Fill NPV profiles
	      //----------------------------------------------------------------
	      for (int iJet=0; iJet<2; iJet++) {

		if (mPt_profile)           mPt_profile          ->Fill(numPV, pfJets.at(iJet).pt());
		if (mEta_profile)          mEta_profile         ->Fill(numPV, pfJets.at(iJet).eta());
		if (mPhi_profile)          mPhi_profile         ->Fill(numPV, pfJets.at(iJet).phi());
		if (mConstituents_profile) mConstituents_profile->Fill(numPV, pfJets.at(iJet).nConstituents());
		if (mHFrac_profile)        mHFrac_profile       ->Fill(numPV, pfJets.at(iJet).chargedHadronEnergyFraction() + pfJets.at(iJet).neutralHadronEnergyFraction());
		if (mEFrac_profile)        mEFrac_profile       ->Fill(numPV, pfJets.at(iJet).chargedEmEnergyFraction()     + pfJets.at(iJet).neutralEmEnergyFraction());
	      
		if (mChargedHadronEnergy_profile) mChargedHadronEnergy_profile->Fill(numPV, pfJets.at(iJet).chargedHadronEnergy());
		if (mNeutralHadronEnergy_profile) mNeutralHadronEnergy_profile->Fill(numPV, pfJets.at(iJet).neutralHadronEnergy());
		if (mChargedEmEnergy_profile)     mChargedEmEnergy_profile    ->Fill(numPV, pfJets.at(iJet).chargedEmEnergy());
		if (mChargedMuEnergy_profile)     mChargedMuEnergy_profile    ->Fill(numPV, pfJets.at(iJet).chargedMuEnergy());
		if (mNeutralEmEnergy_profile)     mNeutralEmEnergy_profile    ->Fill(numPV, pfJets.at(iJet).neutralEmEnergy());
		if (mChargedMultiplicity_profile) mChargedMultiplicity_profile->Fill(numPV, pfJets.at(iJet).chargedMultiplicity());
		if (mNeutralMultiplicity_profile) mNeutralMultiplicity_profile->Fill(numPV, pfJets.at(iJet).neutralMultiplicity());
		if (mMuonMultiplicity_profile)    mMuonMultiplicity_profile   ->Fill(numPV, pfJets.at(iJet).muonMultiplicity());
	      }


	    }// loose cleaned jets 1 and 2
	  }// fabs dphi < 2.1
	}// fabs eta < 3
      }// pt jets > threshold
      //now do the dijet balance and asymmetry calculations
      if (fabs(pfJets.at(0).eta() < 1.4)) {
	double pt_dijet = (pfJets.at(0).pt() + pfJets.at(1).pt())/2;

	double dPhi = fabs((pfJets.at(0)).phi()-(pfJets.at(1)).phi());
	if (dPhi > 3.14) dPhi=fabs(dPhi -6.28 );
	
	if (dPhi > 2.7) {
	  double pt_probe;
	  double pt_barrel;
	  int jet1, jet2;

	  int randJet = rand() % 2;

	  if (fabs(pfJets.at(1).eta() < 1.4)) {
	    if (randJet) {
	      jet1 = 0;
	      jet2 = 1;
	    }
	    else {
	      jet1 = 1;
	      jet2 = 0;
	    }
	  
	    /***Di-Jet Asymmetry****
	     * leading jets eta < 1.4
	     * leading jets dphi > 2.7
	     * pt_third jet < threshold
	     * A = (pt_1 - pt_2)/(pt_1 + pt_2)
	     * jets 1 and two are randomly ordered
	     */
	    bool thirdJetCut = true;
	    for (unsigned int third = 2; third < pfJets.size(); ++third) 
	      if (pfJets.at(third).pt() > _asymmetryThirdJetCut) 
		thirdJetCut = false;
	    if (thirdJetCut) {
	      double dijetAsymmetry = (pfJets.at(jet1).pt() - pfJets.at(jet2).pt()) / (pfJets.at(jet1).pt() + pfJets.at(jet2).pt());
	      mDijetAsymmetry->Fill(dijetAsymmetry);
	    }// end restriction on third jet pt in asymmetry calculation
	      
	  }
	  else {
	    jet1 = 0;
	    jet2 = 1;
	  }
	  
	  pt_barrel = pfJets.at(jet1).pt();
	  pt_probe  = pfJets.at(jet2).pt();
	  
	  //dijet balance cuts
	  /***Di-Jet Balance****
	   * pt_dijet = (pt_probe+pt_barrel)/2
	   * leading jets dphi > 2.7
	   * reject evnets where pt_third/pt_dijet > 0.2
	   * pv selection
	   * B = (pt_probe - pt_barrel)/pt_dijet
	   * select probe randomly from 2 jets if both leading jets are in the barrel
	   */
	  bool thirdJetCut = true;
	  for (unsigned int third = 2; third < pfJets.size(); ++third) 
	    if (pfJets.at(third).pt()/pt_dijet > _balanceThirdJetCut) 
	      thirdJetCut = false;
	  if (thirdJetCut) {
	    double dijetBalance = (pt_probe - pt_barrel) / pt_dijet;
	    mDijetBalance->Fill(dijetBalance);
	  }// end restriction on third jet pt ratio in balance calculation
	}// dPhi > 2.7
      }// leading jet eta cut for asymmetry and balance calculations
    }// jet size >= 2
  } // do dijet selection
  else{
    for (reco::PFJetCollection::const_iterator jet = pfJets.begin(); jet!=pfJets.end(); ++jet){
      LogTrace(metname)<<"[JetAnalyzer] Analyze PFJet";
 
      Thiscleaned=false;
      Loosecleaned=false;
      Tightcleaned=false;

      if (jet == pfJets.begin()) {
	fstPhi = jet->phi();
	_leadJetFlag = 1;
      } else {
	_leadJetFlag = 0;
      }

      if (jet == (pfJets.begin()+1)) sndPhi = jet->phi();
      //  if (jet->pt() < _ptThreshold) return;
      if (jet->pt() > _ptThreshold) {
	numofjets++ ;
	jetME->Fill(2);
            
	//calculate the jetID
	ThisCHFcleaned=true;
	LooseCHFcleaned=true;
	TightCHFcleaned=true;
	if((jet->chargedHadronEnergy()/jet->energy())<=_ThisCHFMin && fabs(jet->eta())<2.4) ThisCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	if((jet->chargedHadronEnergy()/jet->energy())<=_LooseCHFMin && fabs(jet->eta())<2.4) LooseCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	if((jet->chargedHadronEnergy()/jet->energy())<=_TightCHFMin && fabs(jet->eta())<2.4) TightCHFcleaned=false; //apply CHF>0 only if |eta|<2.4
	if(ThisCHFcleaned && (jet->neutralHadronEnergy()/jet->energy())<_ThisNHFMax && (jet->chargedEmEnergy()/jet->energy())<_ThisCEFMax && (jet->neutralEmEnergy()/jet->energy())<_ThisNEFMax) Thiscleaned=true;
	if(LooseCHFcleaned && (jet->neutralHadronEnergy()/jet->energy())<_LooseNHFMax && (jet->chargedEmEnergy()/jet->energy())<_LooseCEFMax && (jet->neutralEmEnergy()/jet->energy())<_LooseNEFMax) Loosecleaned=true;
	if(TightCHFcleaned && (jet->neutralHadronEnergy()/jet->energy())<_TightNHFMax && (jet->chargedEmEnergy()/jet->energy())<_TightCEFMax && (jet->neutralEmEnergy()/jet->energy())<_TightNEFMax) Tightcleaned=true;
      
	if(fillpfJIDPassFrac==1) {
	  //fill the profile for jid efficiency 
	  if(Loosecleaned) {
	    mLooseJIDPassFractionVSeta->Fill(jet->eta(),1.);
	    mLooseJIDPassFractionVSpt->Fill(jet->pt(),1.);
	  } else {
	    mLooseJIDPassFractionVSeta->Fill(jet->eta(),0.);
	    mLooseJIDPassFractionVSpt->Fill(jet->pt(),0.);
	  }
	  if(Tightcleaned) {
	    mTightJIDPassFractionVSeta->Fill(jet->eta(),1.);
	    mTightJIDPassFractionVSpt->Fill(jet->pt(),1.);
	  } else {
	    mTightJIDPassFractionVSeta->Fill(jet->eta(),0.);
	    mTightJIDPassFractionVSpt->Fill(jet->pt(),0.);
	  }
	}
      
	if(!Thiscleaned) continue;
      
	// Leading jet
	// Histograms are filled once per event
	if (_leadJetFlag == 1) { 
	
	  if (mEtaFirst) mEtaFirst->Fill (jet->eta());
	  if (mPhiFirst) mPhiFirst->Fill (jet->phi());
	  //if (mEFirst)   mEFirst->Fill (jet->energy());
	  if (mPtFirst)  mPtFirst->Fill (jet->pt());
	}
      
	// --- Passed the low pt jet trigger (no longer used)
	if (_JetLoPass == 1) {
	/*  if (fabs(jet->eta()) <= 1.3) {
	    if (mPt_Barrel_Lo)           mPt_Barrel_Lo->Fill(jet->pt());
	    if (mEta_Lo)          mEta_Lo->Fill(jet->eta());
	    if (mPhi_Barrel_Lo)          mPhi_Barrel_Lo->Fill(jet->phi());
	  }
	  if ( (fabs(jet->eta()) > 1.3) && (fabs(jet->eta()) <= 3) ) {
	    if (mPt_EndCap_Lo)           mPt_EndCap_Lo->Fill(jet->pt());
	    if (mEta_Lo)          mEta_Lo->Fill(jet->eta());
	    if (mPhi_EndCap_Lo)          mPhi_EndCap_Lo->Fill(jet->phi());
	  }
	  if (fabs(jet->eta()) > 3.0) {
	    if (mPt_Forward_Lo)           mPt_Forward_Lo->Fill(jet->pt());
	    if (mEta_Lo)          mEta_Lo->Fill(jet->eta());
	    if (mPhi_Forward_Lo)          mPhi_Forward_Lo->Fill(jet->phi());
	  }*/
	  //if (mEta_Lo) mEta_Lo->Fill (jet->eta());
	  if (mPhi_Lo) mPhi_Lo->Fill (jet->phi());
	  if (mPt_Lo)  mPt_Lo->Fill (jet->pt());
	}
      
	// --- Passed the high pt jet trigger
	if (_JetHiPass == 1) {
	  if (fabs(jet->eta()) <= 1.3) {
	    if (mPt_Barrel_Hi && jet->pt()>100.)           mPt_Barrel_Hi->Fill(jet->pt());
	    if (mEta_Hi && jet->pt()>100.)          mEta_Hi->Fill(jet->eta());
	    if (mPhi_Barrel_Hi)          mPhi_Barrel_Hi->Fill(jet->phi());
	    //if (mConstituents_Barrel_Hi) mConstituents_Barrel_Hi->Fill(jet->nConstituents());	
	    //if (mHFrac_Barrel_Hi)        mHFrac_Barrel_Hi->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());	
	  }
	  if ( (fabs(jet->eta()) > 1.3) && (fabs(jet->eta()) <= 3) ) {
	    if (mPt_EndCap_Hi && jet->pt()>100.)           mPt_EndCap_Hi->Fill(jet->pt());
	    if (mEta_Hi && jet->pt()>100.)          mEta_Hi->Fill(jet->eta());
	    if (mPhi_EndCap_Hi)          mPhi_EndCap_Hi->Fill(jet->phi());
	    //if (mConstituents_EndCap_Hi) mConstituents_EndCap_Hi->Fill(jet->nConstituents());	
	    //if (mHFrac_EndCap_Hi)        mHFrac_EndCap_Hi->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());	
	  }
	  if (fabs(jet->eta()) > 3.0) {
	    if (mPt_Forward_Hi && jet->pt()>100.)           mPt_Forward_Hi->Fill(jet->pt());
	    if (mEta_Hi && jet->pt()>100.)          mEta_Hi->Fill(jet->eta());
	    if (mPhi_Forward_Hi)          mPhi_Forward_Hi->Fill(jet->phi());
	    //if (mConstituents_Forward_Hi) mConstituents_Forward_Hi->Fill(jet->nConstituents());	
	    //if (mHFrac_Forward_Hi)        mHFrac_Forward_Hi->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());	
	  }
	
	  if (mEta_Hi && jet->pt()>100.) mEta_Hi->Fill (jet->eta());
	  if (mPhi_Hi) mPhi_Hi->Fill (jet->phi());
	  if (mPt_Hi)  mPt_Hi->Fill (jet->pt());
	}
      
	if (mPt)   mPt->Fill (jet->pt());
	if (mPt_1) mPt_1->Fill (jet->pt());
	if (mPt_2) mPt_2->Fill (jet->pt());
	if (mPt_3) mPt_3->Fill (jet->pt());
	if (mEta)  mEta->Fill (jet->eta());
	if (mPhi)  mPhi->Fill (jet->phi());
	if (mPhiVSEta) mPhiVSEta->Fill(jet->eta(),jet->phi());
      
	if (mConstituents) mConstituents->Fill (jet->nConstituents());
	if (mHFrac)        mHFrac->Fill (jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());
	if (mEFrac)        mEFrac->Fill (jet->chargedEmEnergyFraction() +jet->neutralEmEnergyFraction());
      
	if (jet->pt()<= 50) {
	  if (mCHFracVSeta_lowPt) mCHFracVSeta_lowPt->Fill(jet->eta(),jet->chargedHadronEnergyFraction());
	  if (mNHFracVSeta_lowPt) mNHFracVSeta_lowPt->Fill(jet->eta(),jet->neutralHadronEnergyFraction());
	  if (mPhFracVSeta_lowPt) mPhFracVSeta_lowPt->Fill(jet->eta(),jet->neutralEmEnergyFraction());
	  if (mElFracVSeta_lowPt) mElFracVSeta_lowPt->Fill(jet->eta(),jet->chargedEmEnergyFraction());
	  if (mMuFracVSeta_lowPt) mMuFracVSeta_lowPt->Fill(jet->eta(),jet->chargedMuEnergyFraction());
	}
	if (jet->pt()>50. && jet->pt()<=140.) {
	  if (mCHFracVSeta_mediumPt) mCHFracVSeta_mediumPt->Fill(jet->eta(),jet->chargedHadronEnergyFraction());
	  if (mNHFracVSeta_mediumPt) mNHFracVSeta_mediumPt->Fill(jet->eta(),jet->neutralHadronEnergyFraction());
	  if (mPhFracVSeta_mediumPt) mPhFracVSeta_mediumPt->Fill(jet->eta(),jet->neutralEmEnergyFraction());
	  if (mElFracVSeta_mediumPt) mElFracVSeta_mediumPt->Fill(jet->eta(),jet->chargedEmEnergyFraction());
	  if (mMuFracVSeta_mediumPt) mMuFracVSeta_mediumPt->Fill(jet->eta(),jet->chargedMuEnergyFraction());
	}
	if (jet->pt()>140.) {
	  if (mCHFracVSeta_highPt) mCHFracVSeta_highPt->Fill(jet->eta(),jet->chargedHadronEnergyFraction());
	  if (mNHFracVSeta_highPt) mNHFracVSeta_highPt->Fill(jet->eta(),jet->neutralHadronEnergyFraction());
	  if (mPhFracVSeta_highPt) mPhFracVSeta_highPt->Fill(jet->eta(),jet->neutralEmEnergyFraction());
	  if (mElFracVSeta_highPt) mElFracVSeta_highPt->Fill(jet->eta(),jet->chargedEmEnergyFraction());
	  if (mMuFracVSeta_highPt) mMuFracVSeta_highPt->Fill(jet->eta(),jet->chargedMuEnergyFraction());
	}

	if (fabs(jet->eta()) <= 1.3) {
	  if (mPt_Barrel)   mPt_Barrel->Fill (jet->pt());
	  if (mPhi_Barrel)  mPhi_Barrel->Fill (jet->phi());
	  //if (mE_Barrel)    mE_Barrel->Fill (jet->energy());
    if (mConstituents_Barrel)    mConstituents_Barrel->Fill(jet->nConstituents());	
    if (mHFrac_Barrel)           mHFrac_Barrel->Fill(jet->chargedHadronEnergyFraction() + jet->neutralHadronEnergyFraction() );
    if (mEFrac_Barrel)           mEFrac->Fill (jet->chargedEmEnergyFraction() + jet->neutralEmEnergyFraction());	
	  //fractions
	  if (jet->pt()<=50.) {
	    if (mCHFrac_lowPt_Barrel) mCHFrac_lowPt_Barrel->Fill(jet->chargedHadronEnergyFraction());
	    if (mNHFrac_lowPt_Barrel) mNHFrac_lowPt_Barrel->Fill(jet->neutralHadronEnergyFraction());
	    if (mPhFrac_lowPt_Barrel) mPhFrac_lowPt_Barrel->Fill(jet->neutralEmEnergyFraction());
	    if (mElFrac_lowPt_Barrel) mElFrac_lowPt_Barrel->Fill(jet->chargedEmEnergyFraction());
	    if (mMuFrac_lowPt_Barrel) mMuFrac_lowPt_Barrel->Fill(jet->chargedMuEnergyFraction());
	    //
	    if (mCHEn_lowPt_Barrel) mCHEn_lowPt_Barrel->Fill(jet->chargedHadronEnergy());
	    if (mNHEn_lowPt_Barrel) mNHEn_lowPt_Barrel->Fill(jet->neutralHadronEnergy());
	    if (mPhEn_lowPt_Barrel) mPhEn_lowPt_Barrel->Fill(jet->neutralEmEnergy());
	    if (mElEn_lowPt_Barrel) mElEn_lowPt_Barrel->Fill(jet->chargedEmEnergy());
	    if (mMuEn_lowPt_Barrel) mMuEn_lowPt_Barrel->Fill(jet->chargedMuEnergy());
	  }
	  if (jet->pt()>50. && jet->pt()<=140.) {
	    if (mCHFrac_mediumPt_Barrel) mCHFrac_mediumPt_Barrel->Fill(jet->chargedHadronEnergyFraction());
	    if (mNHFrac_mediumPt_Barrel) mNHFrac_mediumPt_Barrel->Fill(jet->neutralHadronEnergyFraction());
	    if (mPhFrac_mediumPt_Barrel) mPhFrac_mediumPt_Barrel->Fill(jet->neutralEmEnergyFraction());
	    if (mElFrac_mediumPt_Barrel) mElFrac_mediumPt_Barrel->Fill(jet->chargedEmEnergyFraction());
	    if (mMuFrac_mediumPt_Barrel) mMuFrac_mediumPt_Barrel->Fill(jet->chargedMuEnergyFraction());
	    //
	    if (mCHEn_mediumPt_Barrel) mCHEn_mediumPt_Barrel->Fill(jet->chargedHadronEnergy());
	    if (mNHEn_mediumPt_Barrel) mNHEn_mediumPt_Barrel->Fill(jet->neutralHadronEnergy());
	    if (mPhEn_mediumPt_Barrel) mPhEn_mediumPt_Barrel->Fill(jet->neutralEmEnergy());
	    if (mElEn_mediumPt_Barrel) mElEn_mediumPt_Barrel->Fill(jet->chargedEmEnergy());
	    if (mMuEn_mediumPt_Barrel) mMuEn_mediumPt_Barrel->Fill(jet->chargedMuEnergy());
	  }
	  if (jet->pt()>140.) {
	    if (mCHFrac_highPt_Barrel) mCHFrac_highPt_Barrel->Fill(jet->chargedHadronEnergyFraction());
	    if (mNHFrac_highPt_Barrel) mNHFrac_highPt_Barrel->Fill(jet->neutralHadronEnergyFraction());
	    if (mPhFrac_highPt_Barrel) mPhFrac_highPt_Barrel->Fill(jet->neutralEmEnergyFraction());
	    if (mElFrac_highPt_Barrel) mElFrac_highPt_Barrel->Fill(jet->chargedEmEnergyFraction());
	    if (mMuFrac_highPt_Barrel) mMuFrac_highPt_Barrel->Fill(jet->chargedMuEnergyFraction());
	    //
	    if (mCHEn_highPt_Barrel) mCHEn_highPt_Barrel->Fill(jet->chargedHadronEnergy());
	    if (mNHEn_highPt_Barrel) mNHEn_highPt_Barrel->Fill(jet->neutralHadronEnergy());
	    if (mPhEn_highPt_Barrel) mPhEn_highPt_Barrel->Fill(jet->neutralEmEnergy());
	    if (mElEn_highPt_Barrel) mElEn_highPt_Barrel->Fill(jet->chargedEmEnergy());
	    if (mMuEn_highPt_Barrel) mMuEn_highPt_Barrel->Fill(jet->chargedMuEnergy());
	  }
	  if(mChMultiplicity_lowPt_Barrel)  mChMultiplicity_lowPt_Barrel->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_lowPt_Barrel)  mNeuMultiplicity_lowPt_Barrel->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_lowPt_Barrel)  mMuMultiplicity_lowPt_Barrel->Fill(jet->muonMultiplicity());
	  if(mChMultiplicity_mediumPt_Barrel)  mChMultiplicity_mediumPt_Barrel->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_mediumPt_Barrel)  mNeuMultiplicity_mediumPt_Barrel->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_mediumPt_Barrel)  mMuMultiplicity_mediumPt_Barrel->Fill(jet->muonMultiplicity());
	  if(mChMultiplicity_highPt_Barrel)  mChMultiplicity_highPt_Barrel->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_highPt_Barrel)  mNeuMultiplicity_highPt_Barrel->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_highPt_Barrel)  mMuMultiplicity_highPt_Barrel->Fill(jet->muonMultiplicity());
	  //
	  if (mCHFracVSpT_Barrel) mCHFracVSpT_Barrel->Fill(jet->pt(),jet->chargedHadronEnergyFraction());
	  if (mNHFracVSpT_Barrel) mNHFracVSpT_Barrel->Fill(jet->pt(),jet->neutralHadronEnergyFraction());
	  if (mPhFracVSpT_Barrel) mPhFracVSpT_Barrel->Fill(jet->pt(),jet->neutralEmEnergyFraction());
	  if (mElFracVSpT_Barrel) mElFracVSpT_Barrel->Fill(jet->pt(),jet->chargedEmEnergyFraction());
	  if (mMuFracVSpT_Barrel) mMuFracVSpT_Barrel->Fill(jet->pt(),jet->chargedMuEnergyFraction());
	}
	if ( (fabs(jet->eta()) > 1.3) && (fabs(jet->eta()) <= 3) ) {
	  if (mPt_EndCap)   mPt_EndCap->Fill (jet->pt());
	  if (mPhi_EndCap)  mPhi_EndCap->Fill (jet->phi());
	  //if (mE_EndCap)    mE_EndCap->Fill (jet->energy());
    if (mConstituents_EndCap)    mConstituents_EndCap->Fill(jet->nConstituents());	
    if (mHFrac_EndCap)           mHFrac_EndCap->Fill(jet->chargedHadronEnergyFraction() + jet->neutralHadronEnergyFraction());
    if (mEFrac_EndCap)           mEFrac->Fill (jet->chargedEmEnergyFraction() + jet->neutralEmEnergyFraction());
	  //fractions
	  if (jet->pt()<=50.) {
	    if (mCHFrac_lowPt_EndCap) mCHFrac_lowPt_EndCap->Fill(jet->chargedHadronEnergyFraction());
	    if (mNHFrac_lowPt_EndCap) mNHFrac_lowPt_EndCap->Fill(jet->neutralHadronEnergyFraction());
	    if (mPhFrac_lowPt_EndCap) mPhFrac_lowPt_EndCap->Fill(jet->neutralEmEnergyFraction());
	    if (mElFrac_lowPt_EndCap) mElFrac_lowPt_EndCap->Fill(jet->chargedEmEnergyFraction());
	    if (mMuFrac_lowPt_EndCap) mMuFrac_lowPt_EndCap->Fill(jet->chargedMuEnergyFraction());
	    //
	    if (mCHEn_lowPt_EndCap) mCHEn_lowPt_EndCap->Fill(jet->chargedHadronEnergy());
	    if (mNHEn_lowPt_EndCap) mNHEn_lowPt_EndCap->Fill(jet->neutralHadronEnergy());
	    if (mPhEn_lowPt_EndCap) mPhEn_lowPt_EndCap->Fill(jet->neutralEmEnergy());
	    if (mElEn_lowPt_EndCap) mElEn_lowPt_EndCap->Fill(jet->chargedEmEnergy());
	    if (mMuEn_lowPt_EndCap) mMuEn_lowPt_EndCap->Fill(jet->chargedMuEnergy());
	  }
	  if (jet->pt()>50. && jet->pt()<=140.) {
	    if (mCHFrac_mediumPt_EndCap) mCHFrac_mediumPt_EndCap->Fill(jet->chargedHadronEnergyFraction());
	    if (mNHFrac_mediumPt_EndCap) mNHFrac_mediumPt_EndCap->Fill(jet->neutralHadronEnergyFraction());
	    if (mPhFrac_mediumPt_EndCap) mPhFrac_mediumPt_EndCap->Fill(jet->neutralEmEnergyFraction());
	    if (mElFrac_mediumPt_EndCap) mElFrac_mediumPt_EndCap->Fill(jet->chargedEmEnergyFraction());
	    if (mMuFrac_mediumPt_EndCap) mMuFrac_mediumPt_EndCap->Fill(jet->chargedMuEnergyFraction());
	    //
	    if (mCHEn_mediumPt_EndCap) mCHEn_mediumPt_EndCap->Fill(jet->chargedHadronEnergy());
	    if (mNHEn_mediumPt_EndCap) mNHEn_mediumPt_EndCap->Fill(jet->neutralHadronEnergy());
	    if (mPhEn_mediumPt_EndCap) mPhEn_mediumPt_EndCap->Fill(jet->neutralEmEnergy());
	    if (mElEn_mediumPt_EndCap) mElEn_mediumPt_EndCap->Fill(jet->chargedEmEnergy());
	    if (mMuEn_mediumPt_EndCap) mMuEn_mediumPt_EndCap->Fill(jet->chargedMuEnergy());
	  }
	  if (jet->pt()>140.) {
	    if (mCHFrac_highPt_EndCap) mCHFrac_highPt_EndCap->Fill(jet->chargedHadronEnergyFraction());
	    if (mNHFrac_highPt_EndCap) mNHFrac_highPt_EndCap->Fill(jet->neutralHadronEnergyFraction());
	    if (mPhFrac_highPt_EndCap) mPhFrac_highPt_EndCap->Fill(jet->neutralEmEnergyFraction());
	    if (mElFrac_highPt_EndCap) mElFrac_highPt_EndCap->Fill(jet->chargedEmEnergyFraction());
	    if (mMuFrac_highPt_EndCap) mMuFrac_highPt_EndCap->Fill(jet->chargedMuEnergyFraction());
	    //
	    if (mCHEn_highPt_EndCap) mCHEn_highPt_EndCap->Fill(jet->chargedHadronEnergy());
	    if (mNHEn_highPt_EndCap) mNHEn_highPt_EndCap->Fill(jet->neutralHadronEnergy());
	    if (mPhEn_highPt_EndCap) mPhEn_highPt_EndCap->Fill(jet->neutralEmEnergy());
	    if (mElEn_highPt_EndCap) mElEn_highPt_EndCap->Fill(jet->chargedEmEnergy());
	    if (mMuEn_highPt_EndCap) mMuEn_highPt_EndCap->Fill(jet->chargedMuEnergy());
	  }
	  if(mChMultiplicity_lowPt_EndCap)  mChMultiplicity_lowPt_EndCap->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_lowPt_EndCap)  mNeuMultiplicity_lowPt_EndCap->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_lowPt_EndCap)  mMuMultiplicity_lowPt_EndCap->Fill(jet->muonMultiplicity());
	  if(mChMultiplicity_mediumPt_EndCap)  mChMultiplicity_mediumPt_EndCap->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_mediumPt_EndCap)  mNeuMultiplicity_mediumPt_EndCap->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_mediumPt_EndCap)  mMuMultiplicity_mediumPt_EndCap->Fill(jet->muonMultiplicity());
	  if(mChMultiplicity_highPt_EndCap)  mChMultiplicity_highPt_EndCap->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_highPt_EndCap)  mNeuMultiplicity_highPt_EndCap->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_highPt_EndCap)  mMuMultiplicity_highPt_EndCap->Fill(jet->muonMultiplicity());
	  //
	  if (mCHFracVSpT_EndCap) mCHFracVSpT_EndCap->Fill(jet->pt(),jet->chargedHadronEnergyFraction());
	  if (mNHFracVSpT_EndCap) mNHFracVSpT_EndCap->Fill(jet->pt(),jet->neutralHadronEnergyFraction());
	  if (mPhFracVSpT_EndCap) mPhFracVSpT_EndCap->Fill(jet->pt(),jet->neutralEmEnergyFraction());
	  if (mElFracVSpT_EndCap) mElFracVSpT_EndCap->Fill(jet->pt(),jet->chargedEmEnergyFraction());
	  if (mMuFracVSpT_EndCap) mMuFracVSpT_EndCap->Fill(jet->pt(),jet->chargedMuEnergyFraction());
	}
	if (fabs(jet->eta()) > 3.0) {
	  if (mPt_Forward)   mPt_Forward->Fill (jet->pt());
	  if (mPhi_Forward)  mPhi_Forward->Fill (jet->phi());
	  //if (mE_Forward)    mE_Forward->Fill (jet->energy());
    if (mConstituents_Forward)    mConstituents_Forward->Fill(jet->nConstituents());	
    if (mHFrac_Forward)           mHFrac_Forward->Fill(jet->chargedHadronEnergyFraction() + jet->neutralHadronEnergyFraction());	
    if (mEFrac_Forward)           mEFrac->Fill (jet->chargedEmEnergyFraction() + jet->neutralEmEnergyFraction());
	  //fractions
	  if (jet->pt()<=50.) {
	    if(mHFEFrac_lowPt_Forward) mHFEFrac_lowPt_Forward->Fill(jet->HFEMEnergyFraction());
	    if(mHFHFrac_lowPt_Forward) mHFHFrac_lowPt_Forward->Fill(jet->HFHadronEnergyFraction());
	    //
	    if(mHFEEn_lowPt_Forward) mHFEEn_lowPt_Forward->Fill(jet->HFEMEnergy());
	    if(mHFHEn_lowPt_Forward) mHFHEn_lowPt_Forward->Fill(jet->HFHadronEnergy());
	  }
	  if (jet->pt()>50. && jet->pt()<=140.) {
	    if(mHFEFrac_mediumPt_Forward) mHFEFrac_mediumPt_Forward->Fill(jet->HFEMEnergyFraction());
	    if(mHFHFrac_mediumPt_Forward) mHFHFrac_mediumPt_Forward->Fill(jet->HFHadronEnergyFraction());
	    //
	    if(mHFEEn_mediumPt_Forward) mHFEEn_mediumPt_Forward->Fill(jet->HFEMEnergy());
	    if(mHFHEn_mediumPt_Forward) mHFHEn_mediumPt_Forward->Fill(jet->HFHadronEnergy());
	  }
	  if (jet->pt()>140.) {
	    if(mHFEFrac_highPt_Forward) mHFEFrac_highPt_Forward->Fill(jet->HFEMEnergyFraction());
	    if(mHFHFrac_highPt_Forward) mHFHFrac_highPt_Forward->Fill(jet->HFHadronEnergyFraction());
	    //
	    if(mHFEEn_highPt_Forward) mHFEEn_highPt_Forward->Fill(jet->HFEMEnergy());
	    if(mHFHEn_highPt_Forward) mHFHEn_highPt_Forward->Fill(jet->HFHadronEnergy());
	  }
	  if(mChMultiplicity_lowPt_Forward)  mChMultiplicity_lowPt_Forward->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_lowPt_Forward)  mNeuMultiplicity_lowPt_Forward->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_lowPt_Forward)  mMuMultiplicity_lowPt_Forward->Fill(jet->muonMultiplicity());
	  if(mChMultiplicity_mediumPt_Forward)  mChMultiplicity_mediumPt_Forward->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_mediumPt_Forward)  mNeuMultiplicity_mediumPt_Forward->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_mediumPt_Forward)  mMuMultiplicity_mediumPt_Forward->Fill(jet->muonMultiplicity());
	  if(mChMultiplicity_highPt_Forward)  mChMultiplicity_highPt_Forward->Fill(jet->chargedMultiplicity());
	  if(mNeuMultiplicity_highPt_Forward)  mNeuMultiplicity_highPt_Forward->Fill(jet->neutralMultiplicity());
	  if(mMuMultiplicity_highPt_Forward)  mMuMultiplicity_highPt_Forward->Fill(jet->muonMultiplicity());
	  if(mHFHFracVSpT_Forward) mHFHFracVSpT_Forward->Fill(jet->pt(),jet->HFHadronEnergyFraction());
	  if(mHFEFracVSpT_Forward) mHFEFracVSpT_Forward->Fill(jet->pt(),jet->HFEMEnergyFraction());
	}
	//if (mE) mE->Fill (jet->energy());
	//if (mP) mP->Fill (jet->p());
	//if (mMass) mMass->Fill (jet->mass());
            
	if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill (jet->chargedHadronEnergy());
	if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill (jet->neutralHadronEnergy());
	if (mChargedEmEnergy) mChargedEmEnergy->Fill(jet->chargedEmEnergy());
	if (mChargedMuEnergy) mChargedMuEnergy->Fill (jet->chargedMuEnergy ());
	if (mNeutralEmEnergy) mNeutralEmEnergy->Fill(jet->neutralEmEnergy());
	if (mChargedMultiplicity ) mChargedMultiplicity->Fill(jet->chargedMultiplicity());
	if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill(jet->neutralMultiplicity());
	if (mMuonMultiplicity )mMuonMultiplicity->Fill (jet-> muonMultiplicity());
	//_______________________________________________________
	if (mNeutralFraction) mNeutralFraction->Fill (jet->neutralMultiplicity()/jet->nConstituents());


        // Fill NPV profiles
	//----------------------------------------------------------------------
        if (mPt_profile)           mPt_profile          ->Fill(numPV, jet->pt());
        if (mEta_profile)          mEta_profile         ->Fill(numPV, jet->eta());
        if (mPhi_profile)          mPhi_profile         ->Fill(numPV, jet->phi());
        if (mConstituents_profile) mConstituents_profile->Fill(numPV, jet->nConstituents());
        if (mHFrac_profile)        mHFrac_profile       ->Fill(numPV, jet->chargedHadronEnergyFraction() + jet->neutralHadronEnergyFraction());
        if (mEFrac_profile)        mEFrac_profile       ->Fill(numPV, jet->chargedEmEnergyFraction()     + jet->neutralEmEnergyFraction());

        if (mChargedHadronEnergy_profile) mChargedHadronEnergy_profile->Fill(numPV, jet->chargedHadronEnergy());
        if (mNeutralHadronEnergy_profile) mNeutralHadronEnergy_profile->Fill(numPV, jet->neutralHadronEnergy());
        if (mChargedEmEnergy_profile)     mChargedEmEnergy_profile    ->Fill(numPV, jet->chargedEmEnergy());
        if (mChargedMuEnergy_profile)     mChargedMuEnergy_profile    ->Fill(numPV, jet->chargedMuEnergy ());
        if (mNeutralEmEnergy_profile)     mNeutralEmEnergy_profile    ->Fill(numPV, jet->neutralEmEnergy());
        if (mChargedMultiplicity_profile) mChargedMultiplicity_profile->Fill(numPV, jet->chargedMultiplicity());
        if (mNeutralMultiplicity_profile) mNeutralMultiplicity_profile->Fill(numPV, jet->neutralMultiplicity());
        if (mMuonMultiplicity_profile)    mMuonMultiplicity_profile   ->Fill(numPV, jet->muonMultiplicity());


	//calculate correctly the dphi
	if(numofjets>1) {
	  diff = fabs(fstPhi - sndPhi);
	  corr = 2*acos(-1.) - diff;
	  if(diff < acos(-1.)) { 
	    dphi = diff; 
	  } else { 
	    dphi = corr;
	  }
	} // numofjets>1
      } // JetPt>_ptThreshold
    } // PF jet loop
    if (mNJets)   mNJets->Fill (numofjets);
    if (mDPhi)    mDPhi->Fill (dphi);


    if (mNJets_profile) mNJets_profile->Fill(numPV, numofjets);


  } // non dijet selection
}
