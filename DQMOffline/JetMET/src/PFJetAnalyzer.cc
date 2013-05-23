/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/05/14 00:28:58 $
 *  $Revision: 1.14 $
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

  mPhi_EndCap              = dbe->book1D("Phi_EndCap", "Phi_EndCap", phiBin, phiMin, phiMax);
  mE_EndCap                = dbe->book1D("E_EndCap", "E_EndCap", eBin, eMin, eMax);
  mPt_EndCap               = dbe->book1D("Pt_EndCap", "Pt_EndCap", ptBin, ptMin, ptMax);

  mPhi_Forward             = dbe->book1D("Phi_Forward", "Phi_Forward", phiBin, phiMin, phiMax);
  mE_Forward               = dbe->book1D("E_Forward", "E_Forward", eBin, eMin, eMax);
  mPt_Forward              = dbe->book1D("Pt_Forward", "Pt_Forward", ptBin, ptMin, ptMax);

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
      
      if (fabs(jet->eta()) <= 1.3) {
	if (mPt_Barrel)   mPt_Barrel->Fill (jet->pt());
	if (mPhi_Barrel)  mPhi_Barrel->Fill (jet->phi());
	if (mE_Barrel)    mE_Barrel->Fill (jet->energy());
      }
      if ( (fabs(jet->eta()) > 1.3) && (fabs(jet->eta()) <= 3) ) {
	if (mPt_EndCap)   mPt_EndCap->Fill (jet->pt());
	if (mPhi_EndCap)  mPhi_EndCap->Fill (jet->phi());
	if (mE_EndCap)    mE_EndCap->Fill (jet->energy());
      }
      if (fabs(jet->eta()) > 3.0) {
	if (mPt_Forward)   mPt_Forward->Fill (jet->pt());
	if (mPhi_Forward)  mPhi_Forward->Fill (jet->phi());
	if (mE_Forward)    mE_Forward->Fill (jet->energy());
      }
      
      if (mE)    mE->Fill (jet->energy());
      if (mP)    mP->Fill (jet->p());
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
