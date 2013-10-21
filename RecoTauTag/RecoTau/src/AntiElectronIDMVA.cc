#include <TFile.h>
#include <TMath.h>
#include "RecoTauTag/RecoTau//interface/AntiElectronIDMVA.h"
#include "RecoTauTag/RecoTau/interface/TMVAZipReader.h"

AntiElectronIDMVA::AntiElectronIDMVA():
  isInitialized_(kFALSE),
  methodName_("BDT")
{
  for(UInt_t i=0; i<6; ++i) {
    fTMVAReader_[i] = 0;
  }

  verbosity_ = 1;
}


AntiElectronIDMVA::~AntiElectronIDMVA()
{
  for(UInt_t i=0; i<6; ++i) {
    if (fTMVAReader_[i]) delete fTMVAReader_[i];
  }
}

void AntiElectronIDMVA::Initialize(std::string methodName,
				   std::string oneProng0Pi0_BL,
				   std::string oneProng1pi0wGSF_BL,
				   std::string oneProng1pi0woGSF_BL,
				   std::string oneProng0Pi0_EC,
				   std::string oneProng1pi0wGSF_EC,
				   std::string oneProng1pi0woGSF_EC
				   ){
  TauSignalPFGammaCands_ = 0.; 
  TauVisMass_ = 0.; 
  GammadEta_ = 0.; 
  GammadPhi_ = 0.; 
  GammadPt_ = 0.;
  TauLeadPFChargedHadrMva_ = 0.;
  TauLeadPFChargedHadrHoP_ = 0.;
  TauLeadPFChargedHadrEoP_ = 0.;
  TauEmFraction_ = 0.;

  for(UInt_t i=0; i<6; ++i) {
    if (fTMVAReader_[i]) delete fTMVAReader_[i];
  }

  isInitialized_ = kTRUE;
  methodName_    = methodName;

  //TMVA::Tools::Instance();

  TMVA::Reader *readerX0BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerX0BL->AddVariable("HoP",       &TauLeadPFChargedHadrHoP_);
  readerX0BL->AddVariable("EoP",       &TauLeadPFChargedHadrEoP_);
  //readerX0BL->AddVariable("emFraction",&TauEmFraction_);
  readerX0BL->SetVerbose(verbosity_);
  reco::details::loadTMVAWeights(readerX0BL,  methodName_, oneProng0Pi0_BL );

  TMVA::Reader *reader11BL = new TMVA::Reader( "!Color:Silent:Error" );
  reader11BL->AddVariable("mva",               &TauLeadPFChargedHadrMva_);
  reader11BL->AddVariable("visMass",           &TauVisMass_);
  reader11BL->AddVariable("etaMom2*TMath::Sqrt(gammaFrac)*pt", &GammadEta_);
  reader11BL->AddVariable("phiMom2*TMath::Sqrt(gammaFrac)*pt", &GammadPhi_);
  reader11BL->AddVariable("gammaFrac",         &GammadPt_);
  reader11BL->SetVerbose(verbosity_);
  reco::details::loadTMVAWeights(reader11BL,  methodName_, oneProng1pi0wGSF_BL );

  TMVA::Reader *reader01BL = new TMVA::Reader( "!Color:Silent:Error" );
  reader01BL->AddVariable("visMass",           &TauVisMass_);
  reader01BL->AddVariable("etaMom2*TMath::Sqrt(gammaFrac)*pt", &GammadEta_);
  reader01BL->AddVariable("phiMom2*TMath::Sqrt(gammaFrac)*pt", &GammadPhi_);
  reader01BL->AddVariable("gammaFrac",         &GammadPt_);
  reader01BL->SetVerbose(verbosity_);
  reco::details::loadTMVAWeights(reader01BL,  methodName_, oneProng1pi0woGSF_BL );

  //////////////////

  TMVA::Reader *readerX0EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerX0EC->AddVariable("HoP",       &TauLeadPFChargedHadrHoP_);
  readerX0EC->AddVariable("EoP",       &TauLeadPFChargedHadrEoP_);
  //readerX0EC->AddVariable("emFraction",&TauEmFraction_);
  readerX0EC->SetVerbose(verbosity_);
  reco::details::loadTMVAWeights(readerX0EC,  methodName_, oneProng0Pi0_EC );

  TMVA::Reader *reader11EC = new TMVA::Reader( "!Color:Silent:Error" );
  reader11EC->AddVariable("mva",               &TauLeadPFChargedHadrMva_);
  reader11EC->AddVariable("visMass",           &TauVisMass_);
  reader11EC->AddVariable("etaMom2*TMath::Sqrt(gammaFrac)*pt", &GammadEta_);
  reader11EC->AddVariable("phiMom2*TMath::Sqrt(gammaFrac)*pt", &GammadPhi_);
  reader11EC->AddVariable("gammaFrac",         &GammadPt_);
  reader11EC->SetVerbose(verbosity_);
  reco::details::loadTMVAWeights(reader11EC,  methodName_, oneProng1pi0wGSF_EC );

  TMVA::Reader *reader01EC = new TMVA::Reader( "!Color:Silent:Error" );
  reader01EC->AddVariable("visMass",           &TauVisMass_);
  reader01EC->AddVariable("etaMom2*TMath::Sqrt(gammaFrac)*pt", &GammadEta_);
  reader01EC->AddVariable("phiMom2*TMath::Sqrt(gammaFrac)*pt", &GammadPhi_);
  reader01EC->AddVariable("gammaFrac",         &GammadPt_);
  reader01EC->SetVerbose(verbosity_);
  reco::details::loadTMVAWeights(reader01EC,  methodName_, oneProng1pi0woGSF_EC );

  fTMVAReader_[0] = readerX0BL;
  fTMVAReader_[1] = reader11BL;
  fTMVAReader_[2] = reader01BL;
  fTMVAReader_[3] = readerX0EC;
  fTMVAReader_[4] = reader11EC;
  fTMVAReader_[5] = reader01EC;
}


double AntiElectronIDMVA::MVAValue(Float_t TauEta, Float_t TauPt,
				   Float_t TauSignalPFChargedCands, Float_t TauSignalPFGammaCands,
				   Float_t TauLeadPFChargedHadrMva,
				   Float_t TauLeadPFChargedHadrHoP, Float_t TauLeadPFChargedHadrEoP,
				   Float_t TauHasGsf, Float_t TauVisMass,  Float_t TauEmFraction,
				   std::vector<Float_t>* GammasdEta, std::vector<Float_t>* GammasdPhi, std::vector<Float_t>* GammasPt
				   ){

  if (!isInitialized_) {
    std::cout << "Error: AntiElectronMVA with method 1 not properly initialized.\n";
    return -999;
  }

  double mva;

  TauVisMass_              = TauVisMass;
  TauLeadPFChargedHadrMva_ = TMath::Max(TauLeadPFChargedHadrMva,float(-1.0));
  TauLeadPFChargedHadrHoP_ = TauLeadPFChargedHadrHoP;
  TauLeadPFChargedHadrEoP_ = TauLeadPFChargedHadrEoP;
  TauEmFraction_           = TMath::Max(TauEmFraction,float(0.0));

  float sumPt  = 0;
  float dEta   = 0;
  float dEta2  = 0;
  float dPhi   = 0;
  float dPhi2  = 0;
  float sumPt2 = 0;

  for(unsigned int k = 0 ; k < GammasPt->size() ; k++){
    float pt_k  = (*GammasPt)[k];
    float phi_k = (*GammasdPhi)[k];
    if ((*GammasdPhi)[k] > TMath::Pi()) phi_k = (*GammasdPhi)[k] - 2*TMath::Pi();
    else if((*GammasdPhi)[k] < -TMath::Pi()) phi_k = (*GammasdPhi)[k] + 2*TMath::Pi();
    float eta_k = (*GammasdEta)[k];
    sumPt  +=  pt_k;
    sumPt2 += (pt_k*pt_k);
    dEta   += (pt_k*eta_k);
    dEta2  += (pt_k*eta_k*eta_k);
    dPhi   += (pt_k*phi_k);
    dPhi2  += (pt_k*phi_k*phi_k);
  }

  GammadPt_ = sumPt/TauPt;

  if(sumPt>0){
    dEta  /= sumPt;
    dPhi  /= sumPt;
    dEta2 /= sumPt;
    dPhi2 /= sumPt;

  }

  //GammadEta_ = dEta;
  //GammadPhi_ = dPhi;

  GammadEta_ = TMath::Sqrt(dEta2)*TMath::Sqrt(GammadPt_)*TauPt;
  GammadPhi_ = TMath::Sqrt(dPhi2)*TMath::Sqrt(GammadPt_)*TauPt;


  if( TauSignalPFChargedCands==3 )
    mva = 1.0;
  else if( TauSignalPFChargedCands==1 && TauSignalPFGammaCands==0){
    if(TMath::Abs(TauEta)<1.5)
      mva = fTMVAReader_[0]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[3]->EvaluateMVA( methodName_ );
  }
  else if( TauSignalPFChargedCands==1 && TauSignalPFGammaCands>0 && TauHasGsf>0.5){
    if(TMath::Abs(TauEta)<1.5)
      mva = fTMVAReader_[1]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[4]->EvaluateMVA( methodName_ );
  }
  else if( TauSignalPFChargedCands==1 && TauSignalPFGammaCands>0 && TauHasGsf<0.5){
    if(TMath::Abs(TauEta)<1.5)
      mva = fTMVAReader_[2]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[5]->EvaluateMVA( methodName_ );
  }
  else{
    mva = -99;
  }

  return mva;

}

double AntiElectronIDMVA::MVAValue(Float_t TauEta, Float_t TauPt,
				   Float_t TauSignalPFChargedCands, Float_t TauSignalPFGammaCands,
				   Float_t TauLeadPFChargedHadrMva,
				   Float_t TauLeadPFChargedHadrHoP, Float_t TauLeadPFChargedHadrEoP,
				   Float_t TauHasGsf, Float_t TauVisMass,  Float_t TauEmFraction,
				   Float_t GammaEtaMom, Float_t GammaPhiMom, Float_t GammaEnFrac
				   ){

  if (!isInitialized_) {
    std::cout << "Error: AntiElectronMVA with method 2 not properly initialized.\n";
    return -999;
  }


  double mva;

  TauVisMass_              = TauVisMass;
  TauLeadPFChargedHadrMva_ = TMath::Max(TauLeadPFChargedHadrMva,float(-1.0));
  TauLeadPFChargedHadrHoP_ = TauLeadPFChargedHadrHoP;
  TauLeadPFChargedHadrEoP_ = TauLeadPFChargedHadrEoP;
  TauEmFraction_           = TMath::Max(TauEmFraction,float(0.0));
  GammadPt_                = GammaEnFrac;
  GammadEta_               = GammaEtaMom;
  GammadPhi_               = GammaPhiMom;

  if( TauSignalPFChargedCands==3 )
    mva = 1.0;
  else if( TauSignalPFChargedCands==1 && TauSignalPFGammaCands==0){
    if(TMath::Abs(TauEta)<1.5)
      mva = fTMVAReader_[0]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[3]->EvaluateMVA( methodName_ );
  }
  else if( TauSignalPFChargedCands==1 && TauSignalPFGammaCands>0 && TauHasGsf>0.5){
    if(TMath::Abs(TauEta)<1.5)
      mva = fTMVAReader_[1]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[4]->EvaluateMVA( methodName_ );
  }
  else if( TauSignalPFChargedCands==1 && TauSignalPFGammaCands>0 && TauHasGsf<0.5){
    if(TMath::Abs(TauEta)<1.5)
      mva = fTMVAReader_[2]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[5]->EvaluateMVA( methodName_ );
  }
  else{
    mva = -99.;
  }

  return mva;

}



double AntiElectronIDMVA::MVAValue(const reco::PFTauRef& thePFTauRef){

  if (!isInitialized_) {
    std::cout << "Error: AntiElectronMVA with method 3 not properly initialized.\n";
    return -999;
  }

  double mva;

  TauVisMass_              = (*thePFTauRef).mass();
  TauLeadPFChargedHadrMva_ = TMath::Max((*thePFTauRef).electronPreIDOutput(),float(-1.0));
  TauLeadPFChargedHadrHoP_ = ((*thePFTauRef).leadPFChargedHadrCand())->hcalEnergy()/(*thePFTauRef).leadPFChargedHadrCand()->p();
  TauLeadPFChargedHadrEoP_ = ((*thePFTauRef).leadPFChargedHadrCand())->ecalEnergy()/(*thePFTauRef).leadPFChargedHadrCand()->p();
  TauEmFraction_           = TMath::Max((*thePFTauRef).emFraction(),float(0.0));

  std::vector<float> GammasdEta;
  std::vector<float> GammasdPhi;
  std::vector<float> GammasPt;

  for(unsigned int k = 0 ; k < ((*thePFTauRef).signalPFGammaCands()).size() ; k++){
    reco::PFCandidateRef gamma = ((*thePFTauRef).signalPFGammaCands()).at(k);
    if( ((*thePFTauRef).leadPFChargedHadrCand()).isNonnull() ){
      GammasdEta.push_back( gamma->eta() - (*thePFTauRef).leadPFChargedHadrCand()->eta() );
      GammasdPhi.push_back( gamma->phi() - (*thePFTauRef).leadPFChargedHadrCand()->phi() );
    }
    else{
      GammasdEta.push_back( gamma->eta() - (*thePFTauRef).eta() );
      GammasdPhi.push_back( gamma->phi() - (*thePFTauRef).phi() );
    }
    GammasPt.push_back(  gamma->pt() );
  }

  float sumPt  = 0;
  float dEta   = 0;
  float dEta2  = 0;
  float dPhi   = 0;
  float dPhi2  = 0;
  float sumPt2 = 0;

  for(unsigned int k = 0 ; k < GammasPt.size() ; k++){
    float pt_k  = GammasPt[k];
    float phi_k = GammasdPhi[k];
    if (GammasdPhi[k] > TMath::Pi()) phi_k = GammasdPhi[k] - 2*TMath::Pi();
    else if(GammasdPhi[k] < -TMath::Pi()) phi_k = GammasdPhi[k] + 2*TMath::Pi();
    float eta_k = GammasdEta[k];
    sumPt  +=  pt_k;
    sumPt2 += (pt_k*pt_k);
    dEta   += (pt_k*eta_k);
    dEta2  += (pt_k*eta_k*eta_k);
    dPhi   += (pt_k*phi_k);
    dPhi2  += (pt_k*phi_k*phi_k);
  }

  GammadPt_ = sumPt/(*thePFTauRef).pt();

  if(sumPt>0){
    dEta  /= sumPt;
    dPhi  /= sumPt;
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }

  //GammadEta_ = dEta;
  //GammadPhi_ = dPhi;

  GammadEta_ = TMath::Sqrt(dEta2)*TMath::Sqrt(GammadPt_)*(*thePFTauRef).pt();
  GammadPhi_ = TMath::Sqrt(dPhi2)*TMath::Sqrt(GammadPt_)*(*thePFTauRef).pt();

  if( ((*thePFTauRef).signalPFChargedHadrCands()).size() == 3)
    mva = 1.0;
  else if( ((*thePFTauRef).signalPFChargedHadrCands()).size()==1 && ((*thePFTauRef).signalPFGammaCands()).size()==0){
    if(TMath::Abs((*thePFTauRef).eta())<1.5)
      mva = fTMVAReader_[0]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[3]->EvaluateMVA( methodName_ );
  }
  else if( ((*thePFTauRef).signalPFChargedHadrCands()).size()==1 && ((*thePFTauRef).signalPFGammaCands()).size()>0 && (((*thePFTauRef).leadPFChargedHadrCand())->gsfTrackRef()).isNonnull()){
    if(TMath::Abs((*thePFTauRef).eta())<1.5)
      mva = fTMVAReader_[1]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[4]->EvaluateMVA( methodName_ );
  }
  else if( ((*thePFTauRef).signalPFChargedHadrCands()).size()==1 && ((*thePFTauRef).signalPFGammaCands()).size()>0 && !(((*thePFTauRef).leadPFChargedHadrCand())->gsfTrackRef()).isNonnull()){
    if(TMath::Abs((*thePFTauRef).eta())<1.5)
      mva = fTMVAReader_[2]->EvaluateMVA( methodName_ );
    else
      mva = fTMVAReader_[5]->EvaluateMVA( methodName_ );
  }
  else{
    mva = -99;
  }

  return mva;

}
