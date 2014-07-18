#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/PhysicsHWW/interface/MuonIDMVA.h"
#include "DQM/PhysicsHWW/interface/muonSelections.h"
#include "DQM/PhysicsHWW/interface/trackSelections.h"

#include "TFile.h"
#include "TRandom3.h"
#include "TVector3.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

using namespace HWWFunctions;

//--------------------------------------------------------------------------------------------------
MuonIDMVA::MuonIDMVA() :
    fMethodname("BDTG method"),
    fIsInitialized(kFALSE)
{
    // Constructor.
    for(UInt_t i=0; i<6; ++i) {
        fTMVAReader[i] = 0;
    }
}


//--------------------------------------------------------------------------------------------------
MuonIDMVA::~MuonIDMVA()
{
    for(UInt_t i=0; i<6; ++i) {
        if (fTMVAReader[i]) delete fTMVAReader[i];
    }
}

//--------------------------------------------------------------------------------------------------
void MuonIDMVA::Initialize( TString methodName, unsigned int version,
			    TString Subdet0Pt10To14p5Weights , 
                            TString Subdet1Pt10To14p5Weights , 
                            TString Subdet0Pt14p5To20Weights,
                            TString Subdet1Pt14p5To20Weights, 
                            TString Subdet0Pt20ToInfWeights, 
                            TString Subdet1Pt20ToInfWeights) {

    if (version != 1) {
        edm::LogError("InvalidInput") << "[MuonIDMVA::Initialize] Version must be 1.  Aborting.";
        return;
    }

    fIsInitialized = kTRUE;
    fMethodname = methodName;

    for(UInt_t i=0; i<6; ++i) {
        if (fTMVAReader[i]) delete fTMVAReader[i];

	fTMVAReader[i] = new TMVA::Reader( "!Color:!Silent:Error" );  
	fTMVAReader[i]->SetVerbose(kTRUE);

        // order matters!
	if (version == 1) {
	  fTMVAReader[i]->AddVariable( "TkNchi2",              &fMVAVar_MuTkNchi2               );
	  fTMVAReader[i]->AddVariable( "GlobalNchi2",          &fMVAVar_MuGlobalNchi2           );
	  fTMVAReader[i]->AddVariable( "NValidHits",           &fMVAVar_MuNValidHits            );
	  fTMVAReader[i]->AddVariable( "NTrackerHits",         &fMVAVar_MuNTrackerHits          );
	  fTMVAReader[i]->AddVariable( "NPixelHits",           &fMVAVar_MuNPixelHits            );
	  fTMVAReader[i]->AddVariable( "NMatches",             &fMVAVar_MuNMatches              );
	  fTMVAReader[i]->AddVariable( "D0",                   &fMVAVar_MuD0                    );      
	  fTMVAReader[i]->AddVariable( "IP3d",                 &fMVAVar_MuIP3d                  );      
	  fTMVAReader[i]->AddVariable( "IP3dSig",              &fMVAVar_MuIP3dSig               );      
	  fTMVAReader[i]->AddVariable( "TrkKink",              &fMVAVar_MuTrkKink               );      
	  fTMVAReader[i]->AddVariable( "SegmentCompatibility", &fMVAVar_MuSegmentCompatibility  );      
	  fTMVAReader[i]->AddVariable( "CaloCompatibility",    &fMVAVar_MuCaloCompatibility     );      
	  fTMVAReader[i]->AddVariable( "HadEnergyOverPt",      &fMVAVar_MuHadEnergyOverPt       );      
	  fTMVAReader[i]->AddVariable( "EmEnergyOverPt",       &fMVAVar_MuEmEnergyOverPt        );      
	  fTMVAReader[i]->AddVariable( "HadS9EnergyOverPt",    &fMVAVar_MuHadS9EnergyOverPt     );      
	  fTMVAReader[i]->AddVariable( "EmS9EnergyOverPt",     &fMVAVar_MuEmS9EnergyOverPt      );      
	  fTMVAReader[i]->AddVariable( "TrkIso03OverPt",       &fMVAVar_MuTrkIso03OverPt        );
	  fTMVAReader[i]->AddVariable( "EMIso03OverPt",        &fMVAVar_MuEMIso03OverPt         );
	  fTMVAReader[i]->AddVariable( "HadIso03OverPt",       &fMVAVar_MuHadIso03OverPt        );
	  fTMVAReader[i]->AddVariable( "TrkIso05OverPt",       &fMVAVar_MuTrkIso05OverPt        );
	  fTMVAReader[i]->AddVariable( "EMIso05OverPt",        &fMVAVar_MuEMIso05OverPt         );
	  fTMVAReader[i]->AddVariable( "HadIso05OverPt",       &fMVAVar_MuHadIso05OverPt        ); 
	}
    
	if (i==0) fTMVAReader[i]->BookMVA(fMethodname , Subdet0Pt10To14p5Weights );
	if (i==1) fTMVAReader[i]->BookMVA(fMethodname , Subdet1Pt10To14p5Weights );
	if (i==2) fTMVAReader[i]->BookMVA(fMethodname , Subdet0Pt14p5To20Weights );
	if (i==3) fTMVAReader[i]->BookMVA(fMethodname , Subdet1Pt14p5To20Weights );
	if (i==4) fTMVAReader[i]->BookMVA(fMethodname , Subdet0Pt20ToInfWeights  );
	if (i==5) fTMVAReader[i]->BookMVA(fMethodname , Subdet1Pt20ToInfWeights  );
	
    }
}

//--------------------------------------------------------------------------------------------------
Double_t MuonIDMVA::MVAValue(Double_t MuPt , Double_t MuEta,
                             Double_t                   MuTkNchi2, 
                             Double_t                   MuGlobalNchi2, 
                             Double_t                   MuNValidHits, 
                             Double_t                   MuNTrackerHits, 
                             Double_t                   MuNPixelHits, 
                             Double_t                   MuNMatches, 
                             Double_t                   MuD0, 
                             Double_t                   MuIP3d, 
                             Double_t                   MuIP3dSig, 
                             Double_t                   MuTrkKink, 
                             Double_t                   MuSegmentCompatibility, 
                             Double_t                   MuCaloCompatibility, 
                             Double_t                   MuHadEnergyOverPt, 
                             Double_t                   MuHoEnergyOverPt, 
                             Double_t                   MuEmEnergyOverPt, 
                             Double_t                   MuHadS9EnergyOverPt, 
                             Double_t                   MuHoS9EnergyOverPt, 
                             Double_t                   MuEmS9EnergyOverPt,
                             Double_t                   MuTrkIso03OverPt,
                             Double_t                   MuEMIso03OverPt,
                             Double_t                   MuHadIso03OverPt,
                             Double_t                   MuTrkIso05OverPt,
                             Double_t                   MuEMIso05OverPt,
                             Double_t                   MuHadIso05OverPt,
                             Bool_t                     printDebug                            
  ) {
  
  if (!fIsInitialized) { 
    edm::LogError("NotInitialized") << "Error: MuonIDMVA not properly initialized."; 
    return -9999;
  }

  Int_t subdet = 0;
  if (fabs(MuEta) < 1.479) subdet = 0;
  else subdet = 1;
  Int_t ptBin = 0;
  if (MuPt > 14.5) ptBin = 1;
  if (MuPt > 20.0) ptBin = 2;

  
  //set all input variables
  fMVAVar_MuTkNchi2              = MuTkNchi2; 
  fMVAVar_MuGlobalNchi2          = MuGlobalNchi2; 
  fMVAVar_MuNValidHits           = MuNValidHits; 
  fMVAVar_MuNTrackerHits         = MuNTrackerHits; 
  fMVAVar_MuNPixelHits           = MuNPixelHits;  
  fMVAVar_MuNMatches             = MuNMatches; 
  fMVAVar_MuD0                   = MuD0; 
  fMVAVar_MuIP3d                 = MuIP3d; 
  fMVAVar_MuIP3dSig              = MuIP3dSig; 
  fMVAVar_MuTrkKink              = MuTrkKink; 
  fMVAVar_MuSegmentCompatibility = MuSegmentCompatibility; 
  fMVAVar_MuCaloCompatibility    = MuCaloCompatibility; 
  fMVAVar_MuHadEnergyOverPt      = MuHadEnergyOverPt; 
  fMVAVar_MuHoEnergyOverPt       = MuHoEnergyOverPt; 
  fMVAVar_MuEmEnergyOverPt       = MuEmEnergyOverPt; 
  fMVAVar_MuHadS9EnergyOverPt    = MuHadS9EnergyOverPt; 
  fMVAVar_MuHoS9EnergyOverPt     = MuHoS9EnergyOverPt; 
  fMVAVar_MuEmS9EnergyOverPt     = MuEmS9EnergyOverPt; 
  fMVAVar_MuTrkIso03OverPt       = MuTrkIso03OverPt; 
  fMVAVar_MuEMIso03OverPt        = MuEMIso03OverPt; 
  fMVAVar_MuHadIso03OverPt       = MuHadIso03OverPt; 
  fMVAVar_MuTrkIso05OverPt       = MuTrkIso05OverPt; 
  fMVAVar_MuEMIso05OverPt        = MuEMIso05OverPt; 
  fMVAVar_MuHadIso05OverPt       = MuHadIso05OverPt; 

  Double_t mva = -9999;  
  TMVA::Reader *reader = 0;

  Int_t MVABin = -1;
  if (subdet == 0 && ptBin == 0) MVABin = 0;
  if (subdet == 1 && ptBin == 0) MVABin = 1;
  if (subdet == 0 && ptBin == 1) MVABin = 2;
  if (subdet == 1 && ptBin == 1) MVABin = 3;
  if (subdet == 0 && ptBin == 2) MVABin = 4;
  if (subdet == 1 && ptBin == 2) MVABin = 5;
  assert(MVABin >= 0 && MVABin <= 5);
  reader = fTMVAReader[MVABin];
                                                
  mva = reader->EvaluateMVA( fMethodname );

  if (printDebug) {
    LogDebug("MuonIDMVA") << "Debug Muon MVA: "
	                        << MuPt << " " << MuEta << " --> MVABin " << MVABin << " : "     
	                        << fMVAVar_MuTkNchi2              << " " 
	                        << fMVAVar_MuGlobalNchi2          << " " 
	                        << fMVAVar_MuNValidHits           << " " 
	                        << fMVAVar_MuNTrackerHits         << " " 
	                        << fMVAVar_MuNPixelHits           << " "  
	                        << fMVAVar_MuNMatches             << " " 
	                        << fMVAVar_MuD0                   << " " 
	                        << fMVAVar_MuIP3d                 << " " 
	                        << fMVAVar_MuIP3dSig              << " " 
	                        << fMVAVar_MuTrkKink              << " " 
	                        << fMVAVar_MuSegmentCompatibility << " " 
	                        << fMVAVar_MuCaloCompatibility    << " " 
	                        << fMVAVar_MuHadEnergyOverPt      << " " 
	                        << fMVAVar_MuHoEnergyOverPt       << " " 
	                        << fMVAVar_MuEmEnergyOverPt       << " " 
	                        << fMVAVar_MuHadS9EnergyOverPt    << " " 
	                        << fMVAVar_MuHoS9EnergyOverPt     << " " 
	                        << fMVAVar_MuEmS9EnergyOverPt     << " " 
	                        << fMVAVar_MuTrkIso03OverPt   << " " 
	                        << fMVAVar_MuEMIso03OverPt   << " " 
	                        << fMVAVar_MuHadIso03OverPt   << " " 
	                        << fMVAVar_MuTrkIso05OverPt   << " " 
	                        << fMVAVar_MuEMIso05OverPt   << " " 
	                        << fMVAVar_MuHadIso05OverPt   << " " 
	                        << " === : === "
	                        << mva;
  }

  return mva;
}

//--------------------------------------------------------------------------------------------------
Double_t MuonIDMVA::MVAValue(HWW& hww, const unsigned int mu, const unsigned int vertex) {
  
  if (!fIsInitialized) { 
    edm::LogError("NotInialized") << "Error: MuonIDMVA not properly initialized."; 
    return -9999;
  }

  Double_t MuPt  = hww.mus_trk_p4().at(mu).pt();
  Double_t MuEta = hww.mus_trk_p4().at(mu).eta();
  Double_t Rho = hww.evt_ww_rho_vor();

  Int_t subdet = 0;
  if (fabs(MuEta) < 1.479) subdet = 0;
  else subdet = 1;
  Int_t ptBin = 0;
  if (MuPt > 14.5) ptBin = 1;
  if (MuPt > 20.0) ptBin = 2;

  //set all input variables
  fMVAVar_MuTkNchi2              = hww.mus_chi2().at(mu) / hww.mus_ndof().at(mu);
  fMVAVar_MuGlobalNchi2          = hww.mus_gfit_chi2().at(mu) / hww.mus_gfit_ndof().at(mu);
  fMVAVar_MuNValidHits           = hww.mus_gfit_validSTAHits().at(mu);
  fMVAVar_MuNTrackerHits         = hww.mus_validHits().at(mu);
  fMVAVar_MuNPixelHits           = hww.trks_valid_pixelhits().at(hww.mus_trkidx().at(mu));
  fMVAVar_MuNMatches             = hww.mus_nmatches().at(mu);
  fMVAVar_MuD0                   = mud0PV_smurfV3(hww, mu);

  const double mud0sign   = ( (trks_d0_pv(hww, hww.mus_trkidx().at(mu),0).first)   >=0 ) ? 1. : -1.;
  fMVAVar_MuIP3d =                   hww.mus_ip3d().at(mu)*mud0sign; 
  if (hww.mus_ip3derr().at(mu) == 0.0) fMVAVar_MuIP3dSig = 0.0;
  else fMVAVar_MuIP3dSig =           hww.mus_ip3d().at(mu)*mud0sign / hww.mus_ip3derr().at(mu); 
  fMVAVar_MuTrkKink              = hww.mus_trkKink().at(mu);
  fMVAVar_MuSegmentCompatibility = hww.mus_segmCompatibility().at(mu);
  fMVAVar_MuCaloCompatibility    = hww.mus_caloCompatibility().at(mu);
  fMVAVar_MuHadEnergyOverPt      = (hww.mus_e_had().at(mu)   - Rho*MuonEffectiveArea(MuonIDMVA::kMuHadEnergy,MuEta))/MuPt;
  fMVAVar_MuHoEnergyOverPt       = (hww.mus_e_ho().at(mu)    - Rho*MuonEffectiveArea(MuonIDMVA::kMuHoEnergy,MuEta))/MuPt;
  fMVAVar_MuEmEnergyOverPt       = (hww.mus_e_em().at(mu)    - Rho*MuonEffectiveArea(MuonIDMVA::kMuEmEnergy,MuEta))/MuPt;
  fMVAVar_MuHadS9EnergyOverPt    = (hww.mus_e_hadS9().at(mu) - Rho*MuonEffectiveArea(MuonIDMVA::kMuHadS9Energy,MuEta))/MuPt;
  fMVAVar_MuHoS9EnergyOverPt     = (hww.mus_e_hoS9().at(mu)  - Rho*MuonEffectiveArea(MuonIDMVA::kMuHoS9Energy,MuEta))/MuPt;
  fMVAVar_MuEmS9EnergyOverPt     = (hww.mus_e_emS9().at(mu)  - Rho*MuonEffectiveArea(MuonIDMVA::kMuEmS9Energy,MuEta))/MuPt;
  fMVAVar_MuTrkIso03OverPt       = (hww.mus_iso03_sumPt().at(mu) - Rho*MuonEffectiveArea(MuonIDMVA::kMuTrkIso03,MuEta))/MuPt;
  fMVAVar_MuEMIso03OverPt        = (hww.mus_iso03_emEt().at(mu)  - Rho*MuonEffectiveArea(MuonIDMVA::kMuEMIso03,MuEta))/MuPt;
  fMVAVar_MuHadIso03OverPt       = (hww.mus_iso03_hadEt().at(mu) - Rho*MuonEffectiveArea(MuonIDMVA::kMuHadIso03,MuEta))/MuPt;
  fMVAVar_MuTrkIso05OverPt       = (hww.mus_iso05_sumPt().at(mu) - Rho*MuonEffectiveArea(MuonIDMVA::kMuTrkIso05,MuEta))/MuPt;
  fMVAVar_MuEMIso05OverPt        = (hww.mus_iso05_emEt().at(mu)  - Rho*MuonEffectiveArea(MuonIDMVA::kMuEMIso05,MuEta))/MuPt;
  fMVAVar_MuHadIso05OverPt       = (hww.mus_iso05_hadEt().at(mu) - Rho*MuonEffectiveArea(MuonIDMVA::kMuHadIso05,MuEta))/MuPt;

  Double_t mva = -9999;  
  TMVA::Reader *reader = 0;

  Int_t MVABin = -1;
  if (subdet == 0 && ptBin == 0) MVABin = 0;
  if (subdet == 1 && ptBin == 0) MVABin = 1;
  if (subdet == 0 && ptBin == 1) MVABin = 2;
  if (subdet == 1 && ptBin == 1) MVABin = 3;
  if (subdet == 0 && ptBin == 2) MVABin = 4;
  if (subdet == 1 && ptBin == 2) MVABin = 5;
  assert(MVABin >= 0 && MVABin <= 5);
  reader = fTMVAReader[MVABin];

  mva = reader->EvaluateMVA( fMethodname );

  if (0) {
    LogDebug("MuonIDMVA") << "Debug Muon MVA: "
                          << hww.evt_run() << " " << hww.evt_lumiBlock() << " " << hww.evt_event() << " " << Rho << " "
                          << hww.mus_p4().at(mu).pt() << " " << hww.mus_p4().at(mu).eta() << " " << hww.mus_p4().at(mu).phi() << " : "
                          << MuPt << " " << MuEta << " --> MVABin " << MVABin << " : "     
                          << fMVAVar_MuTkNchi2              << " " 
                          << fMVAVar_MuGlobalNchi2          << " " 
                          << fMVAVar_MuNValidHits           << " " 
                          << fMVAVar_MuNTrackerHits         << " " 
                          << fMVAVar_MuNPixelHits           << " "  
                          << fMVAVar_MuNMatches             << " " 
                          << fMVAVar_MuD0                   << " " 
                          << fMVAVar_MuIP3d                 << " " 
                          << fMVAVar_MuIP3dSig              << " " 
                          << fMVAVar_MuTrkKink              << " " 
                          << fMVAVar_MuSegmentCompatibility << " " 
                          << fMVAVar_MuCaloCompatibility    << " " 
                          << fMVAVar_MuHadEnergyOverPt      << " " 
                          << fMVAVar_MuHoEnergyOverPt       << " " 
                          << fMVAVar_MuEmEnergyOverPt       << " " 
                          << fMVAVar_MuHadS9EnergyOverPt    << " " 
                          << fMVAVar_MuHoS9EnergyOverPt     << " " 
                          << fMVAVar_MuEmS9EnergyOverPt     << " " 
                          << fMVAVar_MuTrkIso03OverPt   << " " 
                          << fMVAVar_MuEMIso03OverPt   << " " 
                          << fMVAVar_MuHadIso03OverPt   << " " 
                          << fMVAVar_MuTrkIso05OverPt   << " " 
                          << fMVAVar_MuEMIso05OverPt   << " " 
                          << fMVAVar_MuHadIso05OverPt   << " " 
                          << " === : === "
                          << mva;
  }

  return mva;
}
