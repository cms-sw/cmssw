#include <TFile.h>
#include <TMath.h>
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA4GBR.h"

AntiElectronIDMVA4GBR::AntiElectronIDMVA4GBR()
  : isInitialized_(kFALSE),
    methodName_("BDTG")
{
    Var_NoEleMatch_woGwoGSF_Barrel_ = new Float_t[8];
    Var_NoEleMatch_woGwGSF_Barrel_ = new Float_t[14];
    Var_NoEleMatch_wGwoGSF_Barrel_ = new Float_t[12];
    Var_NoEleMatch_wGwGSF_Barrel_ = new Float_t[18];
    Var_woGwoGSF_Barrel_ = new Float_t[16];
    Var_woGwGSF_Barrel_ = new Float_t[22];
    Var_wGwoGSF_Barrel_ = new Float_t[20];
    Var_wGwGSF_Barrel_ = new Float_t[26];
    Var_NoEleMatch_woGwoGSF_Endcap_ = new Float_t[7];
    Var_NoEleMatch_woGwGSF_Endcap_ = new Float_t[13];
    Var_NoEleMatch_wGwoGSF_Endcap_ = new Float_t[11];
    Var_NoEleMatch_wGwGSF_Endcap_ = new Float_t[17];
    Var_woGwoGSF_Endcap_ = new Float_t[15];
    Var_woGwGSF_Endcap_ = new Float_t[21];
    Var_wGwoGSF_Endcap_ = new Float_t[19];
    Var_wGwGSF_Endcap_ = new Float_t[25];

    gbr_NoEleMatch_woGwoGSF_BL_ = 0;
    gbr_NoEleMatch_woGwGSF_BL_ = 0;
    gbr_NoEleMatch_wGwoGSF_BL_ = 0;
    gbr_NoEleMatch_wGwGSF_BL_ = 0;
    gbr_woGwoGSF_BL_= 0;
    gbr_woGwGSF_BL_ = 0;
    gbr_wGwoGSF_BL_ = 0;
    gbr_wGwGSF_BL_ = 0;
    gbr_NoEleMatch_woGwoGSF_EC_ = 0;
    gbr_NoEleMatch_woGwGSF_EC_ = 0;
    gbr_NoEleMatch_wGwoGSF_EC_ = 0;
    gbr_NoEleMatch_wGwGSF_EC_ = 0;
    gbr_woGwoGSF_EC_ = 0;
    gbr_woGwGSF_EC_ = 0;
    gbr_wGwoGSF_EC_ = 0;
    gbr_wGwGSF_EC_ = 0;

  verbosity_ = 1;
}

AntiElectronIDMVA4GBR::~AntiElectronIDMVA4GBR()
{
  delete [] Var_NoEleMatch_woGwoGSF_Barrel_;
  delete [] Var_NoEleMatch_woGwGSF_Barrel_;
  delete [] Var_NoEleMatch_wGwoGSF_Barrel_;
  delete [] Var_NoEleMatch_wGwGSF_Barrel_;
  delete [] Var_woGwoGSF_Barrel_;
  delete [] Var_woGwGSF_Barrel_;
  delete [] Var_wGwoGSF_Barrel_;
  delete [] Var_wGwGSF_Barrel_;
  delete [] Var_NoEleMatch_woGwoGSF_Endcap_;
  delete [] Var_NoEleMatch_woGwGSF_Endcap_;
  delete [] Var_NoEleMatch_wGwoGSF_Endcap_;
  delete [] Var_NoEleMatch_wGwGSF_Endcap_;
  delete [] Var_woGwoGSF_Endcap_;
  delete [] Var_woGwGSF_Endcap_;
  delete [] Var_wGwoGSF_Endcap_;
  delete [] Var_wGwGSF_Endcap_;
  delete fin_;
}


void AntiElectronIDMVA4GBR::Initialize_from_file(const std::string& methodName,
						 const std::string& gbrFile
						 )
{

  isInitialized_ = kTRUE;
  methodName_    = methodName;

  //open input root file
  fin_ = new TFile(gbrFile.data(),"READ");
  if(fin_->IsZombie()){
    std::cout << "No such file!" << std::endl;
    return;
  }

  //read GBRForest from file
  gbr_NoEleMatch_woGwoGSF_BL_ = (GBRForest *)(fin_->Get("gbr_NoEleMatch_woGwoGSF_BL"));
  gbr_NoEleMatch_woGwGSF_BL_ = (GBRForest *)(fin_->Get("gbr_NoEleMatch_woGwGSF_BL"));
  gbr_NoEleMatch_wGwoGSF_BL_ = (GBRForest *)(fin_->Get("gbr_NoEleMatch_wGwoGSF_BL"));
  gbr_NoEleMatch_wGwGSF_BL_ = (GBRForest *)(fin_->Get("gbr_NoEleMatch_wGwGSF_BL"));
  gbr_woGwoGSF_BL_ = (GBRForest *)(fin_->Get("gbr_woGwoGSF_BL"));
  gbr_woGwGSF_BL_ = (GBRForest *)(fin_->Get("gbr_woGwGSF_BL"));
  gbr_wGwoGSF_BL_ = (GBRForest *)(fin_->Get("gbr_wGwoGSF_BL"));
  gbr_wGwGSF_BL_ = (GBRForest *)(fin_->Get("gbr_wGwGSF_BL"));
  gbr_NoEleMatch_woGwoGSF_EC_ = (GBRForest *)(fin_->Get("gbr_NoEleMatch_woGwoGSF_EC"));
  gbr_NoEleMatch_woGwGSF_EC_ = (GBRForest *)(fin_->Get("gbr_NoEleMatch_woGwGSF_EC"));
  gbr_NoEleMatch_wGwoGSF_EC_ = (GBRForest *)(fin_->Get("gbr_NoEleMatch_wGwoGSF_EC"));
  gbr_NoEleMatch_wGwGSF_EC_ = (GBRForest *)(fin_->Get("gbr_NoEleMatch_wGwGSF_EC"));
  gbr_woGwoGSF_EC_ = (GBRForest *)(fin_->Get("gbr_woGwoGSF_EC"));
  gbr_woGwGSF_EC_ = (GBRForest *)(fin_->Get("gbr_woGwGSF_EC"));
  gbr_wGwoGSF_EC_ = (GBRForest *)(fin_->Get("gbr_wGwoGSF_EC"));
  gbr_wGwGSF_EC_ = (GBRForest *)(fin_->Get("gbr_wGwGSF_EC"));
  
}



double AntiElectronIDMVA4GBR::MVAValue(Float_t TauEtaAtEcalEntrance,
				    Float_t TauPt,
				    Float_t TaudCrackEta,
				    Float_t TaudCrackPhi,
				    Float_t TauEmFraction,
				    Float_t TauSignalPFGammaCands,
				    Float_t TauLeadPFChargedHadrHoP,
				    Float_t TauLeadPFChargedHadrEoP,
				    Float_t TauVisMass,
				    Float_t TauHadrMva,
				    const std::vector<Float_t>& GammasdEta,
				    const std::vector<Float_t>& GammasdPhi,
				    const std::vector<Float_t>& GammasPt,
				    Float_t TauKFNumHits,				   
				    Float_t TauGSFNumHits,				   
				    Float_t TauGSFChi2,				   
				    Float_t TauGSFTrackResol,
				    Float_t TauGSFTracklnPt,
				    Float_t TauGSFTrackEta,
				    Float_t TauPhi,
				    Float_t TauSignalPFChargedCands,
				    Float_t TauHasGsf,
				    Float_t ElecEta,
				    Float_t ElecPhi,
				    Float_t ElecPt,
				    Float_t ElecEe,
				    Float_t ElecEgamma,
				    Float_t ElecPin,
				    Float_t ElecPout,
				    Float_t ElecFbrem,
				    Float_t ElecChi2GSF,
				    Float_t ElecGSFNumHits,
				    Float_t ElecGSFTrackResol,
				    Float_t ElecGSFTracklnPt,
				    Float_t ElecGSFTrackEta)
{
  double sumPt  = 0.;
  double dEta   = 0.;
  double dEta2  = 0.;
  double dPhi   = 0.;
  double dPhi2  = 0.;
  double sumPt2 = 0.;
  for ( unsigned int i = 0 ; i < GammasPt.size() ; ++i ) {
    double pt_i  = GammasPt[i];
    double phi_i = GammasdPhi[i];
    if ( GammasdPhi[i] > TMath::Pi() ) phi_i = GammasdPhi[i] - 2*TMath::Pi();
    else if ( GammasdPhi[i] < -TMath::Pi() ) phi_i = GammasdPhi[i] + 2*TMath::Pi();
    double eta_i = GammasdEta[i];
    sumPt  +=  pt_i;
    sumPt2 += (pt_i*pt_i);
    dEta   += (pt_i*eta_i);
    dEta2  += (pt_i*eta_i*eta_i);
    dPhi   += (pt_i*phi_i);
    dPhi2  += (pt_i*phi_i*phi_i);
  }

  Float_t TauGammaEnFrac = sumPt/TauPt;

  if ( sumPt > 0. ) {
    dEta  /= sumPt;
    dPhi  /= sumPt;
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }

  Float_t TauGammaEtaMom = TMath::Sqrt(dEta2)*TMath::Sqrt(TauGammaEnFrac)*TauPt;
  Float_t TauGammaPhiMom = TMath::Sqrt(dPhi2)*TMath::Sqrt(TauGammaEnFrac)*TauPt;

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPt,
		  TaudCrackEta,
		  TaudCrackPhi,
		  TauEmFraction,
		  TauSignalPFGammaCands,				    
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauVisMass,
		  TauHadrMva,
		  TauGammaEtaMom,
		  TauGammaPhiMom,
		  TauGammaEnFrac,
		  TauKFNumHits,				   
		  TauGSFNumHits,				   
		  TauGSFChi2,				   
		  TauGSFTrackResol,
		  TauGSFTracklnPt,
		  TauGSFTrackEta,
		  TauPhi,
		  TauSignalPFChargedCands,
		  TauHasGsf,
		  ElecEta,
		  ElecPhi,
		  ElecPt,
		  ElecEe,
		  ElecEgamma,
		  ElecPin,
		  ElecPout,
		  ElecFbrem,
		  ElecChi2GSF,
		  ElecGSFNumHits,
		  ElecGSFTrackResol,
		  ElecGSFTracklnPt,
		  ElecGSFTrackEta);
}

double AntiElectronIDMVA4GBR::MVAValue(Float_t TauEtaAtEcalEntrance,
				    Float_t TauPt,
				    Float_t TaudCrackEta,
				    Float_t TaudCrackPhi,
				    Float_t TauEmFract,
				    Float_t TauSignalPFGammaCands,				    
				    Float_t TauLeadPFChargedHadrHoP,
				    Float_t TauLeadPFChargedHadrEoP,
				    Float_t TauVisMass,
				    Float_t TauHadrMva,
				    Float_t TauGammaEtaMom,
				    Float_t TauGammaPhiMom,
				    Float_t TauGammaEnFrac,
				    Float_t TauKFNumHits,				   
				    Float_t TauGSFNumHits,				   
				    Float_t TauGSFChi2,				   
				    Float_t TauGSFTrackResol,
				    Float_t TauGSFTracklnPt,
				    Float_t TauGSFTrackEta,
				    Float_t TauPhi,
				    Float_t TauSignalPFChargedCands,
				    Float_t TauHasGsf,
				    Float_t ElecEta,
				    Float_t ElecPhi,
				    Float_t ElecPt,
				    Float_t ElecEe,
				    Float_t ElecEgamma,
				    Float_t ElecPin,
				    Float_t ElecPout,
				    Float_t ElecFbrem,
				    Float_t ElecChi2GSF,
				    Float_t ElecGSFNumHits,
				    Float_t ElecGSFTrackResol,
				    Float_t ElecGSFTracklnPt,
				    Float_t ElecGSFTrackEta)
{

  if ( !isInitialized_ ) {
    std::cout << "Error: AntiElectronMVA not properly initialized.\n";
    return -99.;
  }

  Float_t TauEmFraction = TMath::Max(TauEmFract,float(0.0));
  Float_t TauNumHitsVariable = (TauGSFNumHits-TauKFNumHits)/(TauGSFNumHits+TauKFNumHits); 
  Float_t ElecEtotOverPin = (ElecEe + ElecEgamma)/ElecPin;
  Float_t ElecEgammaOverPdif = ElecEgamma/(ElecPin - ElecPout);

  double mva = -99.;
  if ( TauSignalPFChargedCands == 3 ) mva = 1.0;
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 &&  TauSignalPFGammaCands == 0 && TauHasGsf < 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ){
      Var_NoEleMatch_woGwoGSF_Barrel_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Barrel_[1] = TauPt;
      Var_NoEleMatch_woGwoGSF_Barrel_[2] = TauEmFraction;
      Var_NoEleMatch_woGwoGSF_Barrel_[3] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwoGSF_Barrel_[4] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwoGSF_Barrel_[5] = TauVisMass;
      Var_NoEleMatch_woGwoGSF_Barrel_[6] = TaudCrackEta;
      Var_NoEleMatch_woGwoGSF_Barrel_[7] = TaudCrackPhi;
      mva = gbr_NoEleMatch_woGwoGSF_BL_->GetClassifier(Var_NoEleMatch_woGwoGSF_Barrel_);
    }
    else {
      Var_NoEleMatch_woGwoGSF_Endcap_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Endcap_[1] = TauPt;
      Var_NoEleMatch_woGwoGSF_Endcap_[2] = TauEmFraction;
      Var_NoEleMatch_woGwoGSF_Endcap_[3] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwoGSF_Endcap_[4] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwoGSF_Endcap_[5] = TauVisMass;
      Var_NoEleMatch_woGwoGSF_Endcap_[6] = TaudCrackEta;
      mva = gbr_NoEleMatch_woGwoGSF_EC_->GetClassifier(Var_NoEleMatch_woGwoGSF_Endcap_);
    }
  } 
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 &&  TauSignalPFGammaCands == 0 && TauHasGsf > 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ){
      Var_NoEleMatch_woGwGSF_Barrel_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwGSF_Barrel_[1] = TauPt;
      Var_NoEleMatch_woGwGSF_Barrel_[2] = TauEmFraction;
      Var_NoEleMatch_woGwGSF_Barrel_[3] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwGSF_Barrel_[4] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwGSF_Barrel_[5] = TauVisMass;
      Var_NoEleMatch_woGwGSF_Barrel_[6] = TauHadrMva;
      Var_NoEleMatch_woGwGSF_Barrel_[7] = TauGSFChi2;
      Var_NoEleMatch_woGwGSF_Barrel_[8] = TauNumHitsVariable;
      Var_NoEleMatch_woGwGSF_Barrel_[9] = TauGSFTrackResol;
      Var_NoEleMatch_woGwGSF_Barrel_[10] = TauGSFTracklnPt;
      Var_NoEleMatch_woGwGSF_Barrel_[11] = TauGSFTrackEta;
      Var_NoEleMatch_woGwGSF_Barrel_[12] = TaudCrackEta;
      Var_NoEleMatch_woGwGSF_Barrel_[13] = TaudCrackPhi;
      mva = gbr_NoEleMatch_woGwGSF_BL_->GetClassifier(Var_NoEleMatch_woGwGSF_Barrel_);
    }
    else{
      Var_NoEleMatch_woGwGSF_Endcap_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_woGwGSF_Endcap_[1] = TauPt;
      Var_NoEleMatch_woGwGSF_Endcap_[2] = TauEmFraction;
      Var_NoEleMatch_woGwGSF_Endcap_[3] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_woGwGSF_Endcap_[4] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_woGwGSF_Endcap_[5] = TauVisMass;
      Var_NoEleMatch_woGwGSF_Endcap_[6] = TauHadrMva;
      Var_NoEleMatch_woGwGSF_Endcap_[7] = TauGSFChi2;
      Var_NoEleMatch_woGwGSF_Endcap_[8] = TauNumHitsVariable;
      Var_NoEleMatch_woGwGSF_Endcap_[9] = TauGSFTrackResol;
      Var_NoEleMatch_woGwGSF_Endcap_[10] = TauGSFTracklnPt;
      Var_NoEleMatch_woGwGSF_Endcap_[11] = TauGSFTrackEta;
      Var_NoEleMatch_woGwGSF_Endcap_[12] = TaudCrackEta;
      mva = gbr_NoEleMatch_woGwGSF_EC_->GetClassifier(Var_NoEleMatch_woGwGSF_Endcap_);
    }
  } 
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 &&  TauSignalPFGammaCands > 0 && TauHasGsf < 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ){
      Var_NoEleMatch_wGwoGSF_Barrel_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Barrel_[1] = TauPt;
      Var_NoEleMatch_wGwoGSF_Barrel_[2] = TauEmFraction;
      Var_NoEleMatch_wGwoGSF_Barrel_[3] = TauSignalPFGammaCands;
      Var_NoEleMatch_wGwoGSF_Barrel_[4] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwoGSF_Barrel_[5] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwoGSF_Barrel_[6] = TauVisMass;
      Var_NoEleMatch_wGwoGSF_Barrel_[7] = TauGammaEtaMom;
      Var_NoEleMatch_wGwoGSF_Barrel_[8] = TauGammaPhiMom;
      Var_NoEleMatch_wGwoGSF_Barrel_[9] = TauGammaEnFrac;
      Var_NoEleMatch_wGwoGSF_Barrel_[10] = TaudCrackEta;
      Var_NoEleMatch_wGwoGSF_Barrel_[11] = TaudCrackPhi;
      mva = gbr_NoEleMatch_wGwoGSF_BL_->GetClassifier(Var_NoEleMatch_wGwoGSF_Barrel_);
    }
    else {
      Var_NoEleMatch_wGwoGSF_Endcap_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Endcap_[1] = TauPt;
      Var_NoEleMatch_wGwoGSF_Endcap_[2] = TauEmFraction;
      Var_NoEleMatch_wGwoGSF_Endcap_[3] = TauSignalPFGammaCands;
      Var_NoEleMatch_wGwoGSF_Endcap_[4] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwoGSF_Endcap_[5] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwoGSF_Endcap_[6] = TauVisMass;
      Var_NoEleMatch_wGwoGSF_Endcap_[7] = TauGammaEtaMom;
      Var_NoEleMatch_wGwoGSF_Endcap_[8] = TauGammaPhiMom;
      Var_NoEleMatch_wGwoGSF_Endcap_[9] = TauGammaEnFrac;
      Var_NoEleMatch_wGwoGSF_Endcap_[10] = TaudCrackEta;
      mva = gbr_NoEleMatch_wGwoGSF_EC_->GetClassifier(Var_NoEleMatch_wGwoGSF_Endcap_);
    }
  } 
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 &&  TauSignalPFGammaCands > 0 && TauHasGsf > 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ){
      Var_NoEleMatch_wGwGSF_Barrel_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwGSF_Barrel_[1] = TauPt;
      Var_NoEleMatch_wGwGSF_Barrel_[2] = TauEmFraction;
      Var_NoEleMatch_wGwGSF_Barrel_[3] = TauSignalPFGammaCands;
      Var_NoEleMatch_wGwGSF_Barrel_[4] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwGSF_Barrel_[5] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwGSF_Barrel_[6] = TauVisMass;
      Var_NoEleMatch_wGwGSF_Barrel_[7] = TauHadrMva;
      Var_NoEleMatch_wGwGSF_Barrel_[8] = TauGammaEtaMom;
      Var_NoEleMatch_wGwGSF_Barrel_[9] = TauGammaPhiMom;
      Var_NoEleMatch_wGwGSF_Barrel_[10] = TauGammaEnFrac;
      Var_NoEleMatch_wGwGSF_Barrel_[11] = TauGSFChi2;
      Var_NoEleMatch_wGwGSF_Barrel_[12] = TauNumHitsVariable;
      Var_NoEleMatch_wGwGSF_Barrel_[13] = TauGSFTrackResol;
      Var_NoEleMatch_wGwGSF_Barrel_[14] = TauGSFTracklnPt;
      Var_NoEleMatch_wGwGSF_Barrel_[15] = TauGSFTrackEta;
      Var_NoEleMatch_wGwGSF_Barrel_[16] = TaudCrackEta;
      Var_NoEleMatch_wGwGSF_Barrel_[17] = TaudCrackPhi;
      mva =  gbr_NoEleMatch_wGwGSF_BL_->GetClassifier(Var_NoEleMatch_wGwGSF_Barrel_);
    }
    else{
      Var_NoEleMatch_wGwGSF_Endcap_[0] = TauEtaAtEcalEntrance;
      Var_NoEleMatch_wGwGSF_Endcap_[1] = TauPt;
      Var_NoEleMatch_wGwGSF_Endcap_[2] = TauEmFraction;
      Var_NoEleMatch_wGwGSF_Endcap_[3] = TauSignalPFGammaCands;
      Var_NoEleMatch_wGwGSF_Endcap_[4] = TauLeadPFChargedHadrHoP;
      Var_NoEleMatch_wGwGSF_Endcap_[5] = TauLeadPFChargedHadrEoP;
      Var_NoEleMatch_wGwGSF_Endcap_[6] = TauVisMass;
      Var_NoEleMatch_wGwGSF_Endcap_[7] = TauHadrMva;
      Var_NoEleMatch_wGwGSF_Endcap_[8] = TauGammaEtaMom;
      Var_NoEleMatch_wGwGSF_Endcap_[9] = TauGammaPhiMom;
      Var_NoEleMatch_wGwGSF_Endcap_[10] = TauGammaEnFrac;
      Var_NoEleMatch_wGwGSF_Endcap_[11] = TauGSFChi2;
      Var_NoEleMatch_wGwGSF_Endcap_[12] = TauNumHitsVariable;
      Var_NoEleMatch_wGwGSF_Endcap_[13] = TauGSFTrackResol;
      Var_NoEleMatch_wGwGSF_Endcap_[14] = TauGSFTracklnPt;
      Var_NoEleMatch_wGwGSF_Endcap_[15] = TauGSFTrackEta;
      Var_NoEleMatch_wGwGSF_Endcap_[16] = TaudCrackEta;
      mva = gbr_NoEleMatch_wGwGSF_EC_->GetClassifier(Var_NoEleMatch_wGwGSF_Endcap_);
    } 
  }
  else if ( TauSignalPFGammaCands == 0 && TauHasGsf < 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) {
      Var_woGwoGSF_Barrel_[0] = ElecEtotOverPin;
      Var_woGwoGSF_Barrel_[1] = ElecEgammaOverPdif;
      Var_woGwoGSF_Barrel_[2] = ElecFbrem;
      Var_woGwoGSF_Barrel_[3] = ElecChi2GSF;
      Var_woGwoGSF_Barrel_[4] = ElecGSFNumHits;
      Var_woGwoGSF_Barrel_[5] = ElecGSFTrackResol;
      Var_woGwoGSF_Barrel_[6] = ElecGSFTracklnPt;
      Var_woGwoGSF_Barrel_[7] = ElecGSFTrackEta;
      Var_woGwoGSF_Barrel_[8] = TauEtaAtEcalEntrance;
      Var_woGwoGSF_Barrel_[9] = TauPt;
      Var_woGwoGSF_Barrel_[10] = TauEmFraction;
      Var_woGwoGSF_Barrel_[11] = TauLeadPFChargedHadrHoP;
      Var_woGwoGSF_Barrel_[12] = TauLeadPFChargedHadrEoP;
      Var_woGwoGSF_Barrel_[13] = TauVisMass;
      Var_woGwoGSF_Barrel_[14] = TaudCrackEta;
      Var_woGwoGSF_Barrel_[15] = TaudCrackPhi;
      mva = gbr_woGwoGSF_BL_->GetClassifier(Var_woGwoGSF_Barrel_);
    }
    else{
      Var_woGwoGSF_Endcap_[0] = ElecEtotOverPin;
      Var_woGwoGSF_Endcap_[1] = ElecEgammaOverPdif;
      Var_woGwoGSF_Endcap_[2] = ElecFbrem;
      Var_woGwoGSF_Endcap_[3] = ElecChi2GSF;
      Var_woGwoGSF_Endcap_[4] = ElecGSFNumHits;
      Var_woGwoGSF_Endcap_[5] = ElecGSFTrackResol;
      Var_woGwoGSF_Endcap_[6] = ElecGSFTracklnPt;
      Var_woGwoGSF_Endcap_[7] = ElecGSFTrackEta;
      Var_woGwoGSF_Endcap_[8] = TauEtaAtEcalEntrance;
      Var_woGwoGSF_Endcap_[9] = TauPt;
      Var_woGwoGSF_Endcap_[10] = TauEmFraction;
      Var_woGwoGSF_Endcap_[11] = TauLeadPFChargedHadrHoP;
      Var_woGwoGSF_Endcap_[12] = TauLeadPFChargedHadrEoP;
      Var_woGwoGSF_Endcap_[13] = TauVisMass;
      Var_woGwoGSF_Endcap_[14] = TaudCrackEta;
      mva = gbr_woGwoGSF_EC_->GetClassifier(Var_woGwoGSF_Endcap_);
    }
  } 
  else if ( TauSignalPFGammaCands == 0 && TauHasGsf > 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ){
      Var_woGwGSF_Barrel_[0] = ElecEtotOverPin;
      Var_woGwGSF_Barrel_[1] = ElecEgammaOverPdif;
      Var_woGwGSF_Barrel_[2] = ElecFbrem;
      Var_woGwGSF_Barrel_[3] = ElecChi2GSF;
      Var_woGwGSF_Barrel_[4] = ElecGSFNumHits;
      Var_woGwGSF_Barrel_[5] = ElecGSFTrackResol;
      Var_woGwGSF_Barrel_[6] = ElecGSFTracklnPt;
      Var_woGwGSF_Barrel_[7] = ElecGSFTrackEta;
      Var_woGwGSF_Barrel_[8] = TauEtaAtEcalEntrance;
      Var_woGwGSF_Barrel_[9] = TauPt;
      Var_woGwGSF_Barrel_[10] = TauEmFraction;
      Var_woGwGSF_Barrel_[11] = TauLeadPFChargedHadrHoP;
      Var_woGwGSF_Barrel_[12] = TauLeadPFChargedHadrEoP;
      Var_woGwGSF_Barrel_[13] = TauVisMass;
      Var_woGwGSF_Barrel_[14] = TauHadrMva;
      Var_woGwGSF_Barrel_[15] = TauGSFChi2;
      Var_woGwGSF_Barrel_[16] = TauNumHitsVariable;
      Var_woGwGSF_Barrel_[17] = TauGSFTrackResol;
      Var_woGwGSF_Barrel_[18] = TauGSFTracklnPt;
      Var_woGwGSF_Barrel_[19] = TauGSFTrackEta;
      Var_woGwGSF_Barrel_[20] = TaudCrackEta;
      Var_woGwGSF_Barrel_[21] = TaudCrackPhi;
      mva = gbr_woGwGSF_BL_->GetClassifier(Var_woGwGSF_Barrel_);
    }
    else{
      Var_woGwGSF_Endcap_[0] = ElecEtotOverPin;
      Var_woGwGSF_Endcap_[1] = ElecEgammaOverPdif;
      Var_woGwGSF_Endcap_[2] = ElecFbrem;
      Var_woGwGSF_Endcap_[3] = ElecChi2GSF;
      Var_woGwGSF_Endcap_[4] = ElecGSFNumHits;
      Var_woGwGSF_Endcap_[5] = ElecGSFTrackResol;
      Var_woGwGSF_Endcap_[6] = ElecGSFTracklnPt;
      Var_woGwGSF_Endcap_[7] = ElecGSFTrackEta;
      Var_woGwGSF_Endcap_[8] = TauEtaAtEcalEntrance;
      Var_woGwGSF_Endcap_[9] = TauPt;
      Var_woGwGSF_Endcap_[10] = TauEmFraction;
      Var_woGwGSF_Endcap_[11] = TauLeadPFChargedHadrHoP;
      Var_woGwGSF_Endcap_[12] = TauLeadPFChargedHadrEoP;
      Var_woGwGSF_Endcap_[13] = TauVisMass;
      Var_woGwGSF_Endcap_[14] = TauHadrMva;
      Var_woGwGSF_Endcap_[15] = TauGSFChi2;
      Var_woGwGSF_Endcap_[16] = TauNumHitsVariable;
      Var_woGwGSF_Endcap_[17] = TauGSFTrackResol;
      Var_woGwGSF_Endcap_[18] = TauGSFTracklnPt;
      Var_woGwGSF_Endcap_[19] = TauGSFTrackEta;
      Var_woGwGSF_Endcap_[20] = TaudCrackEta;
      mva = gbr_woGwGSF_EC_->GetClassifier(Var_woGwGSF_Endcap_);
    } 
  }
  else if ( TauSignalPFGammaCands > 0 && TauHasGsf < 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ){
      Var_wGwoGSF_Barrel_[0] = ElecEtotOverPin;
      Var_wGwoGSF_Barrel_[1] = ElecEgammaOverPdif;
      Var_wGwoGSF_Barrel_[2] = ElecFbrem;
      Var_wGwoGSF_Barrel_[3] = ElecChi2GSF;
      Var_wGwoGSF_Barrel_[4] = ElecGSFNumHits;
      Var_wGwoGSF_Barrel_[5] = ElecGSFTrackResol;
      Var_wGwoGSF_Barrel_[6] = ElecGSFTracklnPt;
      Var_wGwoGSF_Barrel_[7] = ElecGSFTrackEta;
      Var_wGwoGSF_Barrel_[8] = TauEtaAtEcalEntrance;
      Var_wGwoGSF_Barrel_[9] = TauPt;
      Var_wGwoGSF_Barrel_[10] = TauEmFraction;
      Var_wGwoGSF_Barrel_[11] = TauSignalPFGammaCands;
      Var_wGwoGSF_Barrel_[12] = TauLeadPFChargedHadrHoP;
      Var_wGwoGSF_Barrel_[13] = TauLeadPFChargedHadrEoP;
      Var_wGwoGSF_Barrel_[14] = TauVisMass;
      Var_wGwoGSF_Barrel_[15] = TauGammaEtaMom;
      Var_wGwoGSF_Barrel_[16] = TauGammaPhiMom;
      Var_wGwoGSF_Barrel_[17] = TauGammaEnFrac;
      Var_wGwoGSF_Barrel_[18] = TaudCrackEta;
      Var_wGwoGSF_Barrel_[19] = TaudCrackPhi;
      mva = gbr_wGwoGSF_BL_->GetClassifier(Var_wGwoGSF_Barrel_);
    }
    else {
      Var_wGwoGSF_Endcap_[0] = ElecEtotOverPin;
      Var_wGwoGSF_Endcap_[1] = ElecEgammaOverPdif;
      Var_wGwoGSF_Endcap_[2] = ElecFbrem;
      Var_wGwoGSF_Endcap_[3] = ElecChi2GSF;
      Var_wGwoGSF_Endcap_[4] = ElecGSFNumHits;
      Var_wGwoGSF_Endcap_[5] = ElecGSFTrackResol;
      Var_wGwoGSF_Endcap_[6] = ElecGSFTracklnPt;
      Var_wGwoGSF_Endcap_[7] = ElecGSFTrackEta;
      Var_wGwoGSF_Endcap_[8] = TauEtaAtEcalEntrance;
      Var_wGwoGSF_Endcap_[9] = TauPt;
      Var_wGwoGSF_Endcap_[10] = TauEmFraction;
      Var_wGwoGSF_Endcap_[11] = TauSignalPFGammaCands;
      Var_wGwoGSF_Endcap_[12] = TauLeadPFChargedHadrHoP;
      Var_wGwoGSF_Endcap_[13] = TauLeadPFChargedHadrEoP;
      Var_wGwoGSF_Endcap_[14] = TauVisMass;
      Var_wGwoGSF_Endcap_[15] = TauGammaEtaMom;
      Var_wGwoGSF_Endcap_[16] = TauGammaPhiMom;
      Var_wGwoGSF_Endcap_[17] = TauGammaEnFrac;
      Var_wGwoGSF_Endcap_[18] = TaudCrackEta;
      mva = gbr_wGwoGSF_EC_->GetClassifier(Var_wGwoGSF_Endcap_);
    }
  }
  else if ( TauSignalPFGammaCands > 0 && TauHasGsf > 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ){
      Var_wGwGSF_Barrel_[0] = ElecEtotOverPin;
      Var_wGwGSF_Barrel_[1] = ElecEgammaOverPdif;
      Var_wGwGSF_Barrel_[2] = ElecFbrem;
      Var_wGwGSF_Barrel_[3] = ElecChi2GSF;
      Var_wGwGSF_Barrel_[4] = ElecGSFNumHits;
      Var_wGwGSF_Barrel_[5] = ElecGSFTrackResol;
      Var_wGwGSF_Barrel_[6] = ElecGSFTracklnPt;
      Var_wGwGSF_Barrel_[7] = ElecGSFTrackEta;
      Var_wGwGSF_Barrel_[8] = TauEtaAtEcalEntrance;
      Var_wGwGSF_Barrel_[9] = TauPt;
      Var_wGwGSF_Barrel_[10] = TauEmFraction;
      Var_wGwGSF_Barrel_[11] = TauSignalPFGammaCands;
      Var_wGwGSF_Barrel_[12] = TauLeadPFChargedHadrHoP;
      Var_wGwGSF_Barrel_[13] = TauLeadPFChargedHadrEoP;
      Var_wGwGSF_Barrel_[14] = TauVisMass;
      Var_wGwGSF_Barrel_[15] = TauHadrMva;
      Var_wGwGSF_Barrel_[16] = TauGammaEtaMom;
      Var_wGwGSF_Barrel_[17] = TauGammaPhiMom;
      Var_wGwGSF_Barrel_[18] = TauGammaEnFrac;
      Var_wGwGSF_Barrel_[19] = TauGSFChi2;
      Var_wGwGSF_Barrel_[20] = TauNumHitsVariable;
      Var_wGwGSF_Barrel_[21] = TauGSFTrackResol;
      Var_wGwGSF_Barrel_[22] = TauGSFTracklnPt;
      Var_wGwGSF_Barrel_[23] = TauGSFTrackEta;
      Var_wGwGSF_Barrel_[24] = TaudCrackEta;
      Var_wGwGSF_Barrel_[25] = TaudCrackPhi;
      mva = gbr_wGwGSF_BL_->GetClassifier(Var_wGwGSF_Barrel_);
    }
    else{
      Var_wGwGSF_Endcap_[0] = ElecEtotOverPin;
      Var_wGwGSF_Endcap_[1] = ElecEgammaOverPdif;
      Var_wGwGSF_Endcap_[2] = ElecFbrem;
      Var_wGwGSF_Endcap_[3] = ElecChi2GSF;
      Var_wGwGSF_Endcap_[4] = ElecGSFNumHits;
      Var_wGwGSF_Endcap_[5] = ElecGSFTrackResol;
      Var_wGwGSF_Endcap_[6] = ElecGSFTracklnPt;
      Var_wGwGSF_Endcap_[7] = ElecGSFTrackEta;
      Var_wGwGSF_Endcap_[8] = TauEtaAtEcalEntrance;
      Var_wGwGSF_Endcap_[9] = TauPt;
      Var_wGwGSF_Endcap_[10] = TauEmFraction;
      Var_wGwGSF_Endcap_[11] = TauSignalPFGammaCands;
      Var_wGwGSF_Endcap_[12] = TauLeadPFChargedHadrHoP;
      Var_wGwGSF_Endcap_[13] = TauLeadPFChargedHadrEoP;
      Var_wGwGSF_Endcap_[14] = TauVisMass;
      Var_wGwGSF_Endcap_[15] = TauHadrMva;
      Var_wGwGSF_Endcap_[16] = TauGammaEtaMom;
      Var_wGwGSF_Endcap_[17] = TauGammaPhiMom;
      Var_wGwGSF_Endcap_[18] = TauGammaEnFrac;
      Var_wGwGSF_Endcap_[19] = TauGSFChi2;
      Var_wGwGSF_Endcap_[20] = TauNumHitsVariable;
      Var_wGwGSF_Endcap_[21] = TauGSFTrackResol;
      Var_wGwGSF_Endcap_[22] = TauGSFTracklnPt;
      Var_wGwGSF_Endcap_[23] = TauGSFTrackEta;
      Var_wGwGSF_Endcap_[24] = TaudCrackEta;
      mva = gbr_wGwGSF_EC_->GetClassifier(Var_wGwGSF_Endcap_);
    } 
  }
  return mva;
}


double AntiElectronIDMVA4GBR::MVAValue(const reco::PFTau& thePFTau,
				    const reco::GsfElectron& theGsfEle)

{
  Float_t TauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0;
  float sumEnergy = 0;
  for(unsigned int j = 0 ; j < (thePFTau.signalPFCands()).size() ; j++){
    reco::PFCandidatePtr pfcandidate = (thePFTau.signalPFCands()).at(j);
    sumEtaTimesEnergy += pfcandidate->positionAtECALEntrance().eta()*pfcandidate->energy();
    sumEnergy += pfcandidate->energy();
  }
  if(sumEnergy>0)TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
  
  Float_t TauPt = thePFTau.pt();
  Float_t TauEmFraction = TMath::Max(thePFTau.emFraction(), (Float_t)0.);
  Float_t TauSignalPFGammaCands = thePFTau.signalPFGammaCands().size();
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
  }
  Float_t TauVisMass = thePFTau.mass();
  Float_t TauHadrMva = TMath::Max(thePFTau.electronPreIDOutput(),float(-1.0));
  std::vector<Float_t> GammasdEta;
  std::vector<Float_t> GammasdPhi;
  std::vector<Float_t> GammasPt;
  for ( unsigned i = 0 ; i < thePFTau.signalPFGammaCands().size(); ++i ) {
    reco::PFCandidatePtr gamma = thePFTau.signalPFGammaCands().at(i);
    if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
      GammasdEta.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
      GammasdPhi.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
    } else {
      GammasdEta.push_back(gamma->eta() - thePFTau.eta());
      GammasdPhi.push_back(gamma->phi() - thePFTau.phi());
    }
    GammasPt.push_back(gamma->pt());
  }
  Float_t TauKFNumHits = -99.;
  if((thePFTau.leadPFChargedHadrCand()->trackRef()).isNonnull()){
    TauKFNumHits = thePFTau.leadPFChargedHadrCand()->trackRef()->numberOfValidHits();
  }
  Float_t TauGSFNumHits = -99.;
  Float_t TauGSFChi2 = -99.;
  Float_t TauGSFTrackResol = -99.;
  Float_t TauGSFTracklnPt = -99.;
  Float_t TauGSFTrackEta = -99.;
  if((thePFTau.leadPFChargedHadrCand()->gsfTrackRef()).isNonnull()){
      TauGSFChi2 = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->normalizedChi2();
      TauGSFNumHits = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->numberOfValidHits();
      if ( thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt() > 0. ) {
	TauGSFTrackResol = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->ptError()/thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt();
	TauGSFTracklnPt = log(thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt())*TMath::Ln10();
      }
      TauGSFTrackEta = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->eta();
  }
  Float_t TauPhi = thePFTau.phi();
  float sumPhiTimesEnergy = 0;
  float sumEnergyPhi = 0;
  for(unsigned int j = 0 ; j < (thePFTau.signalPFCands()).size() ; j++){
    reco::PFCandidatePtr pfcandidate = (thePFTau.signalPFCands()).at(j);
    sumPhiTimesEnergy += pfcandidate->positionAtECALEntrance().phi()*pfcandidate->energy();
    sumEnergyPhi += pfcandidate->energy();
  }
  if(sumEnergy>0)TauPhi = sumPhiTimesEnergy/sumEnergyPhi;
  Float_t TaudCrackPhi = dCrackPhi(TauPhi,TauEtaAtEcalEntrance) ;
  Float_t TaudCrackEta = dCrackEta(TauEtaAtEcalEntrance) ;
  Float_t TauSignalPFChargedCands = thePFTau.signalPFChargedHadrCands().size();
  Float_t TauHasGsf = thePFTau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull();

  Float_t ElecEta = theGsfEle.eta();
  Float_t ElecPhi = theGsfEle.phi();
  Float_t ElecPt = theGsfEle.pt();
  //Variables related to the electron Cluster
  Float_t ElecEe = 0.;
  Float_t ElecEgamma = 0.;
  reco::SuperClusterRef pfSuperCluster = theGsfEle.pflowSuperCluster();
  if ( pfSuperCluster.isNonnull() && pfSuperCluster.isAvailable() ) {
    for ( reco::CaloCluster_iterator pfCluster = pfSuperCluster->clustersBegin();
	  pfCluster != pfSuperCluster->clustersEnd(); ++pfCluster ) {
      double pfClusterEn = (*pfCluster)->energy();
      if ( pfCluster == pfSuperCluster->clustersBegin() ) ElecEe += pfClusterEn;
      else ElecEgamma += pfClusterEn;
    }
  }
  Float_t ElecPin = TMath::Sqrt(theGsfEle.trackMomentumAtVtx().Mag2());
  Float_t ElecPout = TMath::Sqrt(theGsfEle.trackMomentumOut().Mag2());
  Float_t ElecFbrem = theGsfEle.fbrem();
  //Variables related to the GsfTrack
  Float_t ElecChi2GSF = -99.;
  Float_t ElecGSFNumHits = -99.;
  Float_t ElecGSFTrackResol = -99.;
  Float_t ElecGSFTracklnPt = -99.;
  Float_t ElecGSFTrackEta = -99.;
  if ( theGsfEle.gsfTrack().isNonnull() ) {
    ElecChi2GSF = (theGsfEle).gsfTrack()->normalizedChi2();
    ElecGSFNumHits = (theGsfEle).gsfTrack()->numberOfValidHits();
    if ( theGsfEle.gsfTrack()->pt() > 0. ) {
      ElecGSFTrackResol = theGsfEle.gsfTrack()->ptError()/theGsfEle.gsfTrack()->pt();
      ElecGSFTracklnPt = log(theGsfEle.gsfTrack()->pt())*TMath::Ln10();
    }
    ElecGSFTrackEta = theGsfEle.gsfTrack()->eta();
  }

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPt,
		  TaudCrackEta,
		  TaudCrackPhi,
		  TauEmFraction,
		  TauSignalPFGammaCands,
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauVisMass,
		  TauHadrMva,
		  GammasdEta,
		  GammasdPhi,
		  GammasPt,
		  TauKFNumHits,				   
		  TauGSFNumHits,				   
		  TauGSFChi2,				   
		  TauGSFTrackResol,
		  TauGSFTracklnPt,
		  TauGSFTrackEta,
		  TauPhi,
		  TauSignalPFChargedCands,
		  TauHasGsf,
		  ElecEta,
		  ElecPhi,
		  ElecPt,
		  ElecEe,
		  ElecEgamma,
		  ElecPin,
		  ElecPout,
		  ElecFbrem,
		  ElecChi2GSF,
		  ElecGSFNumHits,
		  ElecGSFTrackResol,
		  ElecGSFTracklnPt,
		  ElecGSFTrackEta);
}

double AntiElectronIDMVA4GBR::MVAValue(const reco::PFTau& thePFTau)
{
  Float_t TauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0;
  float sumEnergy = 0;
  for(unsigned int j = 0 ; j < (thePFTau.signalPFCands()).size() ; j++){
    reco::PFCandidatePtr pfcandidate = (thePFTau.signalPFCands()).at(j);
    sumEtaTimesEnergy += pfcandidate->positionAtECALEntrance().eta()*pfcandidate->energy();
    sumEnergy += pfcandidate->energy();
  }
  if(sumEnergy>0)TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;
  
  Float_t TauPt = thePFTau.pt();
  Float_t TauEmFraction = TMath::Max(thePFTau.emFraction(), (Float_t)0.);
  Float_t TauSignalPFGammaCands = thePFTau.signalPFGammaCands().size();
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
  }
  Float_t TauVisMass = thePFTau.mass();
  Float_t TauHadrMva = TMath::Max(thePFTau.electronPreIDOutput(),float(-1.0));
  std::vector<Float_t> GammasdEta;
  std::vector<Float_t> GammasdPhi;
  std::vector<Float_t> GammasPt;
  for ( unsigned i = 0 ; i < thePFTau.signalPFGammaCands().size(); ++i ) {
    reco::PFCandidatePtr gamma = thePFTau.signalPFGammaCands().at(i);
    if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
      GammasdEta.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
      GammasdPhi.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
    } else {
      GammasdEta.push_back(gamma->eta() - thePFTau.eta());
      GammasdPhi.push_back(gamma->phi() - thePFTau.phi());
    }
    GammasPt.push_back(gamma->pt());
  }
  Float_t TauKFNumHits = -99.;
  if((thePFTau.leadPFChargedHadrCand()->trackRef()).isNonnull()){
    TauKFNumHits = thePFTau.leadPFChargedHadrCand()->trackRef()->numberOfValidHits();
  }
  Float_t TauGSFNumHits = -99.;
  Float_t TauGSFChi2 = -99.;
  Float_t TauGSFTrackResol = -99.;
  Float_t TauGSFTracklnPt = -99.;
  Float_t TauGSFTrackEta = -99.;
  if((thePFTau.leadPFChargedHadrCand()->gsfTrackRef()).isNonnull()){
      TauGSFChi2 = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->normalizedChi2();
      TauGSFNumHits = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->numberOfValidHits();
      if ( thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt() > 0. ) {
	TauGSFTrackResol = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->ptError()/thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt();
	TauGSFTracklnPt = log(thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->pt())*TMath::Ln10();
      }
      TauGSFTrackEta = thePFTau.leadPFChargedHadrCand()->gsfTrackRef()->eta();
  }
  Float_t TauPhi = thePFTau.phi();
  float sumPhiTimesEnergy = 0;
  float sumEnergyPhi = 0;
  for(unsigned int j = 0 ; j < (thePFTau.signalPFCands()).size() ; j++){
    reco::PFCandidatePtr pfcandidate = (thePFTau.signalPFCands()).at(j);
    sumPhiTimesEnergy += pfcandidate->positionAtECALEntrance().phi()*pfcandidate->energy();
    sumEnergyPhi += pfcandidate->energy();
  }
  if(sumEnergy>0)TauPhi = sumPhiTimesEnergy/sumEnergyPhi;
  Float_t TaudCrackPhi = dCrackPhi(TauPhi,TauEtaAtEcalEntrance) ;
  Float_t TaudCrackEta = dCrackEta(TauEtaAtEcalEntrance) ;
  Float_t TauSignalPFChargedCands = thePFTau.signalPFChargedHadrCands().size();
  Float_t TauHasGsf = thePFTau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull();

  Float_t dummyElecEta = 9.9;

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPt,
		  TaudCrackEta,
		  TaudCrackPhi,
		  TauEmFraction,
		  TauSignalPFGammaCands,
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauVisMass,
		  TauHadrMva,
		  GammasdEta,
		  GammasdPhi,
		  GammasPt,
		  TauKFNumHits,				   
		  TauGSFNumHits,				   
		  TauGSFChi2,				   
		  TauGSFTrackResol,
		  TauGSFTracklnPt,
		  TauGSFTrackEta,
		  TauPhi,
		  TauSignalPFChargedCands,
		  TauHasGsf,
		  dummyElecEta,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.,
		  0.);
}


double
AntiElectronIDMVA4GBR::minimum(double a,double b){
  if(TMath::Abs(b)<TMath::Abs(a)) return b;
  else return a;
}

//compute the unsigned distance to the closest phi-crack in the barrel
double
AntiElectronIDMVA4GBR:: dCrackPhi(double phi, double eta){

  double pi= TMath::Pi();// 3.14159265358979323846;
  
  //Location of the 18 phi-cracks
  static std::vector<double> cPhi;
  if(cPhi.size()==0)
    {
      cPhi.resize(18,0);
      cPhi[0]=2.97025;
      for(unsigned i=1;i<=17;++i) cPhi[i]=cPhi[0]-2*i*pi/18;
    }

  //Shift of this location if eta<0
  double delta_cPhi=0.00638;

  double m; //the result

  if (eta>=- 1.47464 && eta<= 1.47464){

    //the location is shifted
    if(eta<0) phi +=delta_cPhi;

    if (phi>=-pi && phi<=pi){

      //the problem of the extrema
      if (phi<cPhi[17] || phi>=cPhi[0]){
	if (phi<0) phi+= 2*pi;
	m = minimum(phi -cPhi[0],phi-cPhi[17]-2*pi);        	
      }

      //between these extrema...
      else{
	bool OK = false;
	unsigned i=16;
	while(!OK){
	  if (phi<cPhi[i]){
	    m=minimum(phi-cPhi[i+1],phi-cPhi[i]);
	    OK=true;
	  }
	  else i-=1;
	}
      }
    }
    else{
      m=0.;        //if there is a problem, we assum that we are in a crack
      std::cout<<"Problem in dminphi"<<std::endl;
    }
  }
  else{
    return -99.;       
    std::cout<<"Encap region"<<std::endl;
  }
  
  return TMath::Abs(m);
}

//compute the unsigned distance to the closest phi-crack in the barrel
double
AntiElectronIDMVA4GBR:: dCrackEta(double eta){
  
  //Location of the eta-cracks
  double cracks[5] = {0, 4.44747e-01, 7.92824e-01, 1.14090e+00, 1.47464e+00};
  
  double m=99.; //the result
  
  for(int i=0;i<5;i++){
    double d = minimum(eta-cracks[i], eta+cracks[i]);
    if (TMath::Abs(d)<TMath::Abs(m)){
      m=d;
    }
  }

  return TMath::Abs(m);
}
