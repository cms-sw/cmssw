/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/10/15 13:49:55 $
 *  $Revision: 1.15 $
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

  _TightCHFMin = parameters.getParameter<double>("TightCHFMin");
  _TightNHFMax = parameters.getParameter<double>("TightNHFMax");
  _TightCEFMax = parameters.getParameter<double>("TightCEFMax");
  _TightNEFMax = parameters.getParameter<double>("TightNEFMax");
  _LooseCHFMin = parameters.getParameter<double>("LooseCHFMin");
  _LooseNHFMax = parameters.getParameter<double>("LooseNHFMax");
  _LooseCEFMax = parameters.getParameter<double>("LooseCEFMax");
  _LooseNEFMax = parameters.getParameter<double>("LooseNEFMax");

  fillpfJIDPassFrac = parameters.getParameter<int>("fillpfJIDPassFrac");

  _ThisCHFMin = parameters.getParameter<double>("ThisCHFMin");
  _ThisNHFMax = parameters.getParameter<double>("ThisNHFMax");
  _ThisCEFMax = parameters.getParameter<double>("ThisCEFMax");
  _ThisNEFMax = parameters.getParameter<double>("ThisNEFMax");

  // Generic Jet Parameters
  mPt                      = dbe->book1D("Pt",  "Pt", ptBin, ptMin, ptMax);
  mPt_1                    = dbe->book1D("Pt1", "Pt1", 100, 0, 100);
  mPt_2                    = dbe->book1D("Pt2", "Pt2", 100, 0, 300);
  mPt_3                    = dbe->book1D("Pt3", "Pt3", 100, 0, 5000);
  mEta                     = dbe->book1D("Eta", "Eta", etaBin, etaMin, etaMax);
  mPhi                     = dbe->book1D("Phi", "Phi", phiBin, phiMin, phiMax);
  mConstituents            = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100);
  mHFrac                   = dbe->book1D("HFrac", "HFrac", 120, -0.1, 1.1);
  mEFrac                   = dbe->book1D("EFrac", "EFrac", 120, -0.1, 1.1);
 //
  mPhiVSEta                     = dbe->book2D("PhiVSEta", "PhiVSEta", 50, etaMin, etaMax, 24, phiMin, phiMax);

  // Low and high pt trigger paths
  mPt_Lo                  = dbe->book1D("Pt_Lo", "Pt (Pass Low Pt Jet Trigger)", 100, 0, 100);
  mEta_Lo                 = dbe->book1D("Eta_Lo", "Eta (Pass Low Pt Jet Trigger)", etaBin, etaMin, etaMax);
  mPhi_Lo                 = dbe->book1D("Phi_Lo", "Phi (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);

  mPt_Hi                  = dbe->book1D("Pt_Hi", "Pt (Pass Hi Pt Jet Trigger)", 100, 0, 300);
  mEta_Hi                 = dbe->book1D("Eta_Hi", "Eta (Pass Hi Pt Jet Trigger)", etaBin, etaMin, etaMax);
  mPhi_Hi                 = dbe->book1D("Phi_Hi", "Phi (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);

  mE                       = dbe->book1D("E", "E", eBin, eMin, eMax);
  mP                       = dbe->book1D("P", "P", pBin, pMin, pMax);
  mMass                    = dbe->book1D("Mass", "Mass", 100, 0, 25);
  mNJets                   = dbe->book1D("NJets", "Number of Jets", 100, 0, 100);

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

  mPt_Barrel_Lo            = dbe->book1D("Pt_Barrel_Lo", "Pt Barrel (Pass Low Pt Jet Trigger)", 100, 0, 100);
  mPhi_Barrel_Lo           = dbe->book1D("Phi_Barrel_Lo", "Phi Barrel (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
  mConstituents_Barrel_Lo  = dbe->book1D("Constituents_Barrel_Lo", "Constituents Barrel (Pass Low Pt Jet Trigger)", 100, 0, 100);
  mHFrac_Barrel_Lo         = dbe->book1D("HFrac_Barrel_Lo", "HFrac Barrel (Pass Low Pt Jet Trigger)", 100, 0, 1);

  mPt_EndCap_Lo            = dbe->book1D("Pt_EndCap_Lo", "Pt EndCap (Pass Low Pt Jet Trigger)", 100, 0, 100);
  mPhi_EndCap_Lo           = dbe->book1D("Phi_EndCap_Lo", "Phi EndCap (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
  mConstituents_EndCap_Lo  = dbe->book1D("Constituents_EndCap_Lo", "Constituents EndCap (Pass Low Pt Jet Trigger)", 100, 0, 100);
  mHFrac_EndCap_Lo         = dbe->book1D("HFrac_Endcap_Lo", "HFrac EndCap (Pass Low Pt Jet Trigger)", 100, 0, 1);

  mPt_Forward_Lo           = dbe->book1D("Pt_Forward_Lo", "Pt Forward (Pass Low Pt Jet Trigger)", 100, 0, 100);
  mPhi_Forward_Lo          = dbe->book1D("Phi_Forward_Lo", "Phi Forward (Pass Low Pt Jet Trigger)", phiBin, phiMin, phiMax);
  mConstituents_Forward_Lo = dbe->book1D("Constituents_Forward_Lo", "Constituents Forward (Pass Low Pt Jet Trigger)", 100, 0, 100);
  mHFrac_Forward_Lo        = dbe->book1D("HFrac_Forward_Lo", "HFrac Forward (Pass Low Pt Jet Trigger)", 100, 0, 1);

  mPt_Barrel_Hi            = dbe->book1D("Pt_Barrel_Hi", "Pt Barrel (Pass Hi Pt Jet Trigger)", 100, 0, 300);
  mPhi_Barrel_Hi           = dbe->book1D("Phi_Barrel_Hi", "Phi Barrel (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);
  mConstituents_Barrel_Hi  = dbe->book1D("Constituents_Barrel_Hi", "Constituents Barrel (Pass Hi Pt Jet Trigger)", 100, 0, 100);
  mHFrac_Barrel_Hi         = dbe->book1D("HFrac_Barrel_Hi", "HFrac Barrel (Pass Hi Pt Jet Trigger)", 100, 0, 1);

  mPt_EndCap_Hi            = dbe->book1D("Pt_EndCap_Hi", "Pt EndCap (Pass Hi Pt Jet Trigger)", 100, 0, 300);
  mPhi_EndCap_Hi           = dbe->book1D("Phi_EndCap_Hi", "Phi EndCap (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);
  mConstituents_EndCap_Hi  = dbe->book1D("Constituents_EndCap_Hi", "Constituents EndCap (Pass Hi Pt Jet Trigger)", 100, 0, 100);
  mHFrac_EndCap_Hi         = dbe->book1D("HFrac_EndCap_Hi", "HFrac EndCap (Pass Hi Pt Jet Trigger)", 100, 0, 1);

  mPt_Forward_Hi           = dbe->book1D("Pt_Forward_Hi", "Pt Forward (Pass Hi Pt Jet Trigger)", 100, 0, 300);
  mPhi_Forward_Hi          = dbe->book1D("Phi_Forward_Hi", "Phi Forward (Pass Hi Pt Jet Trigger)", phiBin, phiMin, phiMax);
  mConstituents_Forward_Hi = dbe->book1D("Constituents_Forward_Hi", "Constituents Forward (Pass Hi Pt Jet Trigger)", 100, 0, 100);
  mHFrac_Forward_Hi        = dbe->book1D("HFrac_Forward_Hi", "HFrac Forward (Pass Hi Pt Jet Trigger)", 100, 0, 1);

  mPhi_Barrel              = dbe->book1D("Phi_Barrel", "Phi_Barrel", phiBin, phiMin, phiMax);
  mE_Barrel                = dbe->book1D("E_Barrel", "E_Barrel", eBin, eMin, eMax);
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
  mE_EndCap                = dbe->book1D("E_EndCap", "E_EndCap", eBin, eMin, eMax);
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
  mE_Forward               = dbe->book1D("E_Forward", "E_Forward", eBin, eMin, eMax);
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
  mChMultiplicity_lowPt_Forward    = dbe->book1D("ChMultiplicity_lowPt_Forward", "ChMultiplicity_lowPt_Forward", 30,0,30);
  mNeuMultiplicity_lowPt_Forward   = dbe->book1D("NeuMultiplicity_lowPt_Forward", "NeuMultiplicity_lowPt_Forward", 30,0,30);
  mMuMultiplicity_lowPt_Forward    = dbe->book1D("MuMultiplicity_lowPt_Forward", "MuMultiplicity_lowPt_Forward", 30,0,30);
  mChMultiplicity_mediumPt_Forward    = dbe->book1D("ChMultiplicity_mediumPt_Forward", "ChMultiplicity_mediumPt_Forward", 30,0,30);
  mNeuMultiplicity_mediumPt_Forward   = dbe->book1D("NeuMultiplicity_mediumPt_Forward", "NeuMultiplicity_mediumPt_Forward", 30,0,30);
  mMuMultiplicity_mediumPt_Forward    = dbe->book1D("MuMultiplicity_mediumPt_Forward", "MuMultiplicity_mediumPt_Forward", 30,0,30);
  mChMultiplicity_highPt_Forward    = dbe->book1D("ChMultiplicity_highPt_Forward", "ChMultiplicity_highPt_Forward", 30,0,30);
  mNeuMultiplicity_highPt_Forward   = dbe->book1D("NeuMultiplicity_highPt_Forward", "NeuMultiplicity_highPt_Forward", 30,0,30);
  mMuMultiplicity_highPt_Forward    = dbe->book1D("MuMultiplicity_highPt_Forward", "MuMultiplicity_highPt_Forward", 30,0,30);



  // Leading Jet Parameters
  mEtaFirst                = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5);
  mPhiFirst                = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5);
  mEFirst                  = dbe->book1D("EFirst", "EFirst", 100, 0, 1000);
  mPtFirst                 = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500);

  mDPhi                   = dbe->book1D("DPhi", "dPhi btw the two leading jets", 100, 0., acos(-1.));

  //
  mChargedHadronEnergy = dbe->book1D("mChargedHadronEnergy", "mChargedHadronEnergy", 100, 0, 100);
  mNeutralHadronEnergy = dbe->book1D("mNeutralHadronEnergy", "mNeutralHadronEnergy", 100, 0, 100);
  mChargedEmEnergy= dbe->book1D("mChargedEmEnergy ", "mChargedEmEnergy ", 100, 0, 100);
  mChargedMuEnergy = dbe->book1D("mChargedMuEnergy", "mChargedMuEnergy", 100, 0, 100);
  mNeutralEmEnergy= dbe->book1D("mNeutralEmEnergy", "mNeutralEmEnergy", 100, 0, 100);
  mChargedMultiplicity= dbe->book1D("mChargedMultiplicity ", "mChargedMultiplicity ", 100, 0, 100);
  mNeutralMultiplicity = dbe->book1D(" mNeutralMultiplicity", "mNeutralMultiplicity", 100, 0, 100);
  mMuonMultiplicity= dbe->book1D("mMuonMultiplicity", "mMuonMultiplicity", 100, 0, 100);
  //__________________________________________________
  mNeutralFraction = dbe->book1D("NeutralFraction","Neutral Fraction",100,0,1);
  if(fillpfJIDPassFrac==1) {
    mLooseJIDPassFractionVSeta= dbe->bookProfile("LooseJIDPassFractionVSeta","LooseJIDPassFractionVSeta",etaBin, etaMin, etaMax,0.,1.2);
    mLooseJIDPassFractionVSpt= dbe->bookProfile("LooseJIDPassFractionVSpt","LooseJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
    mTightJIDPassFractionVSeta= dbe->bookProfile("TightJIDPassFractionVSeta","TightJIDPassFractionVSeta",etaBin, etaMin, etaMax,0.,1.2);
    mTightJIDPassFractionVSpt= dbe->bookProfile("TightJIDPassFractionVSpt","TightJIDPassFractionVSpt",ptBin, ptMin, ptMax,0.,1.2);
  }

  
}

void PFJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::PFJetCollection& pfJets) {

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
	if (mEFirst)   mEFirst->Fill (jet->energy());
	if (mPtFirst)  mPtFirst->Fill (jet->pt());
      }
      
      // --- Passed the low pt jet trigger
      if (_JetLoPass == 1) {
	if (fabs(jet->eta()) <= 1.3) {
	  if (mPt_Barrel_Lo)           mPt_Barrel_Lo->Fill(jet->pt());
	  if (mEta_Lo)          mEta_Lo->Fill(jet->eta());
	  if (mPhi_Barrel_Lo)          mPhi_Barrel_Lo->Fill(jet->phi());
	  if (mConstituents_Barrel_Lo) mConstituents_Barrel_Lo->Fill(jet->nConstituents());	
	  if (mHFrac_Barrel_Lo)        mHFrac_Barrel_Lo->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction() );	
	}
	if ( (fabs(jet->eta()) > 1.3) && (fabs(jet->eta()) <= 3) ) {
	  if (mPt_EndCap_Lo)           mPt_EndCap_Lo->Fill(jet->pt());
	  if (mEta_Lo)          mEta_Lo->Fill(jet->eta());
	  if (mPhi_EndCap_Lo)          mPhi_EndCap_Lo->Fill(jet->phi());
	  if (mConstituents_EndCap_Lo) mConstituents_EndCap_Lo->Fill(jet->nConstituents());	
	  if (mHFrac_EndCap_Lo)        mHFrac_EndCap_Lo->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());	
	}
	if (fabs(jet->eta()) > 3.0) {
	  if (mPt_Forward_Lo)           mPt_Forward_Lo->Fill(jet->pt());
	  if (mEta_Lo)          mEta_Lo->Fill(jet->eta());
	  if (mPhi_Forward_Lo)          mPhi_Forward_Lo->Fill(jet->phi());
	  if (mConstituents_Forward_Lo) mConstituents_Forward_Lo->Fill(jet->nConstituents());	
	  if (mHFrac_Forward_Lo)        mHFrac_Forward_Lo->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());	
	}
	if (mEta_Lo) mEta_Lo->Fill (jet->eta());
	if (mPhi_Lo) mPhi_Lo->Fill (jet->phi());
	if (mPt_Lo)  mPt_Lo->Fill (jet->pt());
      }
      
      // --- Passed the high pt jet trigger
      if (_JetHiPass == 1) {
	if (fabs(jet->eta()) <= 1.3) {
	  if (mPt_Barrel_Hi)           mPt_Barrel_Hi->Fill(jet->pt());
	  if (mEta_Hi)          mEta_Hi->Fill(jet->eta());
	  if (mPhi_Barrel_Hi)          mPhi_Barrel_Hi->Fill(jet->phi());
	  if (mConstituents_Barrel_Hi) mConstituents_Barrel_Hi->Fill(jet->nConstituents());	
	  if (mHFrac_Barrel_Hi)        mHFrac_Barrel_Hi->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());	
	}
	if ( (fabs(jet->eta()) > 1.3) && (fabs(jet->eta()) <= 3) ) {
	  if (mPt_EndCap_Hi)           mPt_EndCap_Hi->Fill(jet->pt());
	  if (mEta_Hi)          mEta_Hi->Fill(jet->eta());
	  if (mPhi_EndCap_Hi)          mPhi_EndCap_Hi->Fill(jet->phi());
	  if (mConstituents_EndCap_Hi) mConstituents_EndCap_Hi->Fill(jet->nConstituents());	
	  if (mHFrac_EndCap_Hi)        mHFrac_EndCap_Hi->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());	
	}
	if (fabs(jet->eta()) > 3.0) {
	  if (mPt_Forward_Hi)           mPt_Forward_Hi->Fill(jet->pt());
	  if (mEta_Hi)          mEta_Hi->Fill(jet->eta());
	  if (mPhi_Forward_Hi)          mPhi_Forward_Hi->Fill(jet->phi());
	  if (mConstituents_Forward_Hi) mConstituents_Forward_Hi->Fill(jet->nConstituents());	
	  if (mHFrac_Forward_Hi)        mHFrac_Forward_Hi->Fill(jet->chargedHadronEnergyFraction()+jet->neutralHadronEnergyFraction());	
	}
	
	if (mEta_Hi) mEta_Hi->Fill (jet->eta());
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
	if (mE_Barrel)    mE_Barrel->Fill (jet->energy());
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
	if (mE_EndCap)    mE_EndCap->Fill (jet->energy());
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
	if (mE_Forward)    mE_Forward->Fill (jet->energy());
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
      if (mE) mE->Fill (jet->energy());
      if (mP) mP->Fill (jet->p());
      if (mMass) mMass->Fill (jet->mass());
            
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
  if (mNJets)    mNJets->Fill (numofjets);
  if (mDPhi)    mDPhi->Fill (dphi);
}
