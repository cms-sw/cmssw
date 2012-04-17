#include <TFile.h>
#include <TMath.h>
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA2.h"

AntiElectronIDMVA2::AntiElectronIDMVA2():
  isInitialized_(kFALSE),
  methodName_("BDT")
{
  for(UInt_t i=0; i<8; ++i) {
    fTMVAReader_[i] = 0;
  }
}


AntiElectronIDMVA2::~AntiElectronIDMVA2()
{
  for(UInt_t i=0; i<8; ++i) {
    if (fTMVAReader_[i]) delete fTMVAReader_[i];
  }
}


void AntiElectronIDMVA2::Initialize(std::string methodName,
				    std::string oneProng0Pi0_BL,
				    std::string oneProng1pi0woGSF_BL,
				    std::string oneProng1pi0wGSFwoPfEleMva_BL,
				    std::string oneProng1pi0wGSFwPfEleMva_BL,
				    std::string oneProng0Pi0_EC,
				    std::string oneProng1pi0woGSF_EC,
				    std::string oneProng1pi0wGSFwoPfEleMva_EC,
				    std::string oneProng1pi0wGSFwPfEleMva_EC
				   ){

  for(UInt_t i=0; i<8; ++i) {
    if (fTMVAReader_[i]) delete fTMVAReader_[i];
  }

  isInitialized_ = kTRUE;
  methodName_    = methodName;

//   //TMVA::Tools::Instance();


  TMVA::Reader *readerwoG_BL = new TMVA::Reader( "!Color:!Silent:Error" );  
  readerwoG_BL->AddVariable("Elec_Pt",&Elec_Pt_);   
  readerwoG_BL->AddVariable("Elec_AbsEta",&Elec_AbsEta_);
  readerwoG_BL->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwoG_BL->AddVariable("Elec_EarlyBrem",&Elec_EarlyBrem_);
  readerwoG_BL->AddVariable("Elec_LateBrem",&Elec_LateBrem_);
  readerwoG_BL->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwoG_BL->AddVariable("Elec_Chi2KF",&Elec_Chi2KF_);
  readerwoG_BL->AddVariable("Elec_NumHits",&Elec_NumHits_);
  readerwoG_BL->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwoG_BL->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwoG_BL->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwoG_BL->AddVariable("Tau_AbsEta",&Tau_AbsEta_);
  readerwoG_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwoG_BL->AddVariable("Tau_HasGsf",&Tau_HasGsf_);
  readerwoG_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwoG_BL->AddVariable("Tau_NumChargedCands",&Tau_NumChargedCands_);
  readerwoG_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwoG_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwoG_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwoG_BL->SetVerbose(kTRUE);
  readerwoG_BL->BookMVA("BDT",oneProng0Pi0_BL);


  TMVA::Reader *readerwGwoGSF_BL = new TMVA::Reader( "!Color:!Silent:Error" ); 
  readerwGwoGSF_BL->AddVariable("Elec_Pt",&Elec_Pt_);   
  readerwGwoGSF_BL->AddVariable("Elec_AbsEta",&Elec_AbsEta_);
  readerwGwoGSF_BL->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwGwoGSF_BL->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwGwoGSF_BL->AddVariable("Elec_EarlyBrem",&Elec_EarlyBrem_);
  readerwGwoGSF_BL->AddVariable("Elec_LateBrem",&Elec_LateBrem_);
  readerwGwoGSF_BL->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwGwoGSF_BL->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwoGSF_BL->AddVariable("Elec_NumHits",&Elec_NumHits_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwoGSF_BL->AddVariable("Tau_AbsEta",&Tau_AbsEta_);
  readerwGwoGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwoGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwoGSF_BL->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwoGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwoGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwoGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwoGSF_BL->SetVerbose(kTRUE);
  readerwGwoGSF_BL->BookMVA("BDT",oneProng1pi0woGSF_BL);   
  
  TMVA::Reader *readerwGwGSFwoPFMVA_BL = new TMVA::Reader( "!Color:!Silent:Error" ); 
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_Pt",&Elec_Pt_);   
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_AbsEta",&Elec_AbsEta_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_Chi2KF",&Elec_Chi2KF_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_NumHits",&Elec_NumHits_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_AbsEta",&Tau_AbsEta_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwGSFwoPFMVA_BL->SetVerbose(kTRUE);
  readerwGwGSFwoPFMVA_BL->BookMVA("BDT",oneProng1pi0wGSFwoPfEleMva_BL); 
  
  TMVA::Reader *readerwGwGSFwPFMVA_BL = new TMVA::Reader( "!Color:!Silent:Error" ); 
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_Pt",&Elec_Pt_);   
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_AbsEta",&Elec_AbsEta_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_EeOverPout",&Elec_EeOverPout_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_EarlyBrem",&Elec_EarlyBrem_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_LateBrem",&Elec_LateBrem_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_NumHits",&Elec_NumHits_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_AbsEta",&Tau_AbsEta_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwGSFwPFMVA_BL->SetVerbose(kTRUE);
  readerwGwGSFwPFMVA_BL->BookMVA("BDT",oneProng1pi0wGSFwPfEleMva_BL);  
  
  //////////////////

  TMVA::Reader *readerwoG_EC = new TMVA::Reader( "!Color:!Silent:Error" ); 
  readerwoG_EC->AddVariable("Elec_Pt",&Elec_Pt_);   
  readerwoG_EC->AddVariable("Elec_AbsEta",&Elec_AbsEta_);
  readerwoG_EC->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwoG_EC->AddVariable("Elec_EarlyBrem",&Elec_EarlyBrem_);
  readerwoG_EC->AddVariable("Elec_LateBrem",&Elec_LateBrem_);
  readerwoG_EC->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwoG_EC->AddVariable("Elec_Chi2KF",&Elec_Chi2KF_);
  readerwoG_EC->AddVariable("Elec_NumHits",&Elec_NumHits_);
  readerwoG_EC->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwoG_EC->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwoG_EC->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwoG_EC->AddVariable("Tau_AbsEta",&Tau_AbsEta_);
  readerwoG_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwoG_EC->AddVariable("Tau_HasGsf",&Tau_HasGsf_);
  readerwoG_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwoG_EC->AddVariable("Tau_NumChargedCands",&Tau_NumChargedCands_);
  readerwoG_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwoG_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwoG_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwoG_EC->SetVerbose(kTRUE);
  readerwoG_EC->BookMVA("BDT",oneProng0Pi0_EC);
  
  TMVA::Reader *readerwGwoGSF_EC = new TMVA::Reader( "!Color:!Silent:Error" );
  readerwGwoGSF_EC->AddVariable("Elec_Pt",&Elec_Pt_);   
  readerwGwoGSF_EC->AddVariable("Elec_AbsEta",&Elec_AbsEta_);
  readerwGwoGSF_EC->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwGwoGSF_EC->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwGwoGSF_EC->AddVariable("Elec_EarlyBrem",&Elec_EarlyBrem_);
  readerwGwoGSF_EC->AddVariable("Elec_LateBrem",&Elec_LateBrem_);
  readerwGwoGSF_EC->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwGwoGSF_EC->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwoGSF_EC->AddVariable("Elec_NumHits",&Elec_NumHits_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwoGSF_EC->AddVariable("Tau_AbsEta",&Tau_AbsEta_);
  readerwGwoGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwoGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwoGSF_EC->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwoGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwoGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwoGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwoGSF_EC->SetVerbose(kTRUE);
  readerwGwoGSF_EC->BookMVA("BDT",oneProng1pi0woGSF_EC); 

  TMVA::Reader *readerwGwGSFwoPFMVA_EC = new TMVA::Reader( "!Color:!Silent:Error" );   
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_Pt",&Elec_Pt_);   
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_AbsEta",&Elec_AbsEta_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_Chi2KF",&Elec_Chi2KF_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_NumHits",&Elec_NumHits_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_AbsEta",&Tau_AbsEta_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwGSFwoPFMVA_EC->SetVerbose(kTRUE);
  readerwGwGSFwoPFMVA_EC->BookMVA("BDT",oneProng1pi0wGSFwoPfEleMva_EC); 

  TMVA::Reader *readerwGwGSFwPFMVA_EC = new TMVA::Reader( "!Color:!Silent:Error" ); 
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_Pt",&Elec_Pt_);   
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_AbsEta",&Elec_AbsEta_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_EeOverPout",&Elec_EeOverPout_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_EarlyBrem",&Elec_EarlyBrem_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_LateBrem",&Elec_LateBrem_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_NumHits",&Elec_NumHits_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_AbsEta",&Tau_AbsEta_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwGSFwPFMVA_EC->SetVerbose(kTRUE);
  readerwGwGSFwPFMVA_EC->BookMVA("BDT",oneProng1pi0wGSFwPfEleMva_EC); 


  fTMVAReader_[0] = readerwoG_BL;
  fTMVAReader_[1] = readerwGwoGSF_BL;
  fTMVAReader_[2] = readerwGwGSFwoPFMVA_BL;
  fTMVAReader_[3] = readerwGwGSFwPFMVA_BL;
  fTMVAReader_[4] = readerwoG_EC;
  fTMVAReader_[5] = readerwGwoGSF_EC;
  fTMVAReader_[6] = readerwGwGSFwoPFMVA_EC;
  fTMVAReader_[7] = readerwGwGSFwPFMVA_EC;


}


double AntiElectronIDMVA2::MVAValue(Float_t TauEta,
				    Float_t TauPhi,
				    Float_t TauPt,
				    Float_t TauSignalPFChargedCands, 
				    Float_t TauSignalPFGammaCands, 
				    Float_t TauLeadPFChargedHadrHoP, 
				    Float_t TauLeadPFChargedHadrEoP, 
				    Float_t TauHasGsf, 
				    Float_t TauVisMass,  
				    Float_t TauEmFraction,
				    vector<Float_t>* GammasdEta, 
				    vector<Float_t>* GammasdPhi, 
				    vector<Float_t>* GammasPt,
				    Float_t TauLeadPFChargedHadrMva,
				    Float_t ElecEta,
				    Float_t ElecPhi,
				    Float_t ElecPt,
				    Float_t ElecPFMvaOutput,
				    Float_t ElecEe,
				    Float_t ElecEgamma,
				    Float_t ElecPin,
				    Float_t ElecPout,
				    Float_t ElecEarlyBrem,
				    Float_t ElecLateBrem,
				    Float_t ElecLogsihih,
				    Float_t ElecDeltaEta,
				    Float_t ElecHoHplusE,
				    Float_t ElecFbrem,
				    Float_t ElecChi2KF,
				    Float_t ElecChi2GSF,
				    Float_t ElecNumHits,
				    Float_t ElecGSFTrackResol,
				    Float_t ElecGSFTracklnPt,
				    Float_t ElecGSFTrackEta
				   )
{

  if (!isInitialized_) { 
    std::cout << "Error: AntiElectronMVA with method 1 not properly initialized.\n"; 
    return -999;
  }

  if (deltaR(TauEta,TauPhi,ElecEta,ElecPhi)<0.3){
    std::cout << "Electron and Tau not matched.\n"; 
    return -999;
  }

  double mva;

  Tau_AbsEta_ = TMath::Abs(TauEta) ;
  Tau_Pt_ = TauPt;
  Tau_HasGsf_ = TauHasGsf; 
  Tau_EmFraction_ = TMath::Max(TauEmFraction,float(0.0)); 
  Tau_NumChargedCands_ = TauSignalPFChargedCands;
  Tau_NumGammaCands_ = TauSignalPFGammaCands; 
  Tau_HadrHoP_ = TauLeadPFChargedHadrHoP; 
  Tau_HadrEoP_ = TauLeadPFChargedHadrEoP; 
  Tau_VisMass_ = TauVisMass; 
  Tau_HadrMva_ = TMath::Max(TauLeadPFChargedHadrMva,float(-1.0)); 

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
  
  GammadEta_ = TMath::Sqrt(dEta2)*TMath::Sqrt(GammadPt_)*TauPt;
  GammadPhi_ = TMath::Sqrt(dPhi2)*TMath::Sqrt(GammadPt_)*TauPt;
  
  Tau_GammaEtaMom_ = GammadEta_;
  Tau_GammaPhiMom_ = GammadPhi_;
  Tau_GammaEnFrac_ = GammadPt_;
  
  Elec_AbsEta_ = TMath::Abs(ElecEta);
  Elec_Pt_ = ElecPt;
  Elec_EtotOverPin_ = (ElecEe+ElecEgamma)/ElecPin;
  Elec_EgammaOverPdif_ = ElecEgamma/(ElecPin-ElecPout);
  Elec_EarlyBrem_ = ElecEarlyBrem;
  Elec_LateBrem_ = ElecLateBrem;
  Elec_Fbrem_ = ElecFbrem;
  Elec_Chi2KF_ = ElecChi2KF;
  Elec_Chi2GSF_  = ElecChi2GSF;
  Elec_NumHits_ = ElecNumHits;
  Elec_GSFTrackResol_ = ElecGSFTrackResol;
  Elec_GSFTracklnPt_ = ElecGSFTracklnPt;
  Elec_GSFTrackEta_ = ElecGSFTrackEta;
  
  if(TauSignalPFGammaCands==0){
    if(TMath::Abs(TauEta)<1.5) 
      mva = fTMVAReader_[0]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[4]->EvaluateMVA( methodName_ );
  }
  else if(TauSignalPFGammaCands>0 && TauHasGsf<0.5){
    if(TMath::Abs(TauEta)<1.5) 
      mva = fTMVAReader_[1]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[5]->EvaluateMVA( methodName_ );
  }
  else if(TauSignalPFGammaCands>0 && TauHasGsf>0.5 && ElecPFMvaOutput<-0.1){
    if(TMath::Abs(TauEta)<1.5) 
      mva = fTMVAReader_[2]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[6]->EvaluateMVA( methodName_ );
  }
  else if(TauSignalPFGammaCands>0 && TauHasGsf>0.5 && ElecPFMvaOutput>-0.1){
    if(TMath::Abs(TauEta)<1.5) 
      mva = fTMVAReader_[3]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[7]->EvaluateMVA( methodName_ );
  }
  else{
    mva = -99;
  }

  return mva;

}






double AntiElectronIDMVA2::MVAValue(Float_t TauEta,
				    Float_t TauPhi,
				    Float_t TauPt,
				    Float_t TauSignalPFChargedCands, 
				    Float_t TauSignalPFGammaCands, 
				    Float_t TauLeadPFChargedHadrHoP, 
				    Float_t TauLeadPFChargedHadrEoP, 
				    Float_t TauHasGsf, 
				    Float_t TauVisMass,  
				    Float_t TauEmFraction,
				    Float_t GammaEtaMom,
				    Float_t GammaPhiMom,
				    Float_t GammaEnFrac,
				    Float_t TauLeadPFChargedHadrMva,
				    Float_t ElecEta,
				    Float_t ElecPhi,
				    Float_t ElecPt,
				    Float_t ElecPFMvaOutput,
				    Float_t ElecEe,
				    Float_t ElecEgamma,
				    Float_t ElecPin,
				    Float_t ElecPout,
				    Float_t ElecEarlyBrem,
				    Float_t ElecLateBrem,
				    Float_t ElecLogsihih,
				    Float_t ElecDeltaEta,
				    Float_t ElecHoHplusE,
				    Float_t ElecFbrem,
				    Float_t ElecChi2KF,
				    Float_t ElecChi2GSF,
				    Float_t ElecNumHits,
				    Float_t ElecGSFTrackResol,
				    Float_t ElecGSFTracklnPt,
				    Float_t ElecGSFTrackEta
				   )
{

  if (!isInitialized_) { 
    std::cout << "Error: AntiElectronMVA with method 1 not properly initialized.\n"; 
    return -999;
  }

  if (deltaR(TauEta,TauPhi,ElecEta,ElecPhi)<0.3){
    std::cout << "Electron and Tau not matched.\n"; 
    return -999;
  }

  double mva;

  Tau_AbsEta_ = TMath::Abs(TauEta) ;
  Tau_Pt_ = TauPt;
  Tau_HasGsf_ = TauHasGsf; 
  Tau_EmFraction_ = TMath::Max(TauEmFraction,float(0.0)); 
  Tau_NumChargedCands_ = TauSignalPFChargedCands;
  Tau_NumGammaCands_ = TauSignalPFGammaCands; 
  Tau_HadrHoP_ = TauLeadPFChargedHadrHoP; 
  Tau_HadrEoP_ = TauLeadPFChargedHadrEoP; 
  Tau_VisMass_ = TauVisMass; 
  Tau_HadrMva_ = TMath::Max(TauLeadPFChargedHadrMva,float(-1.0)); 
  Tau_GammaEtaMom_ = GammaEtaMom;
  Tau_GammaPhiMom_ = GammaPhiMom;
  Tau_GammaEnFrac_ = GammaEnFrac;
  
  Elec_AbsEta_ = TMath::Abs(ElecEta);
  Elec_Pt_ = ElecPt;
  Elec_EtotOverPin_ = (ElecEe+ElecEgamma)/ElecPin;
  Elec_EgammaOverPdif_ = ElecEgamma/(ElecPin-ElecPout);
  Elec_EarlyBrem_ = ElecEarlyBrem;
  Elec_LateBrem_ = ElecLateBrem;
  Elec_Fbrem_ = ElecFbrem;
  Elec_Chi2KF_ = ElecChi2KF;
  Elec_Chi2GSF_  = ElecChi2GSF;
  Elec_NumHits_ = ElecNumHits;
  Elec_GSFTrackResol_ = ElecGSFTrackResol;
  Elec_GSFTracklnPt_ = ElecGSFTracklnPt;
  Elec_GSFTrackEta_ = ElecGSFTrackEta;
  
  if(TauSignalPFGammaCands==0){
    if(TMath::Abs(TauEta)<1.5) 
      mva = fTMVAReader_[0]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[4]->EvaluateMVA( methodName_ );
  }
  else if(TauSignalPFGammaCands>0 && TauHasGsf<0.5){
    if(TMath::Abs(TauEta)<1.5) 
      mva = fTMVAReader_[1]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[5]->EvaluateMVA( methodName_ );
  }
  else if(TauSignalPFGammaCands>0 && TauHasGsf>0.5 && ElecPFMvaOutput<-0.1){
    if(TMath::Abs(TauEta)<1.5) 
      mva = fTMVAReader_[2]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[6]->EvaluateMVA( methodName_ );
  }
  else if(TauSignalPFGammaCands>0 && TauHasGsf>0.5 && ElecPFMvaOutput>-0.1){
    if(TMath::Abs(TauEta)<1.5) 
      mva = fTMVAReader_[3]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[7]->EvaluateMVA( methodName_ );
  }
  else{
    mva = -99;
  }

  return mva;

}








double AntiElectronIDMVA2::MVAValue(const reco::PFTau& thePFTau,
				    const reco::GsfElectron& theGsfEle
				    )
{

  if (!isInitialized_) { 
    std::cout << "Error: AntiElectronMVA with method 3 not properly initialized.\n"; 
    return -999;
  }

  if (deltaR((thePFTau).eta(),(thePFTau).phi(),(theGsfEle).eta(),(theGsfEle).phi())>0.3){
    std::cout << "DeltaR : "<<deltaR((thePFTau).eta(),(thePFTau).phi(),(theGsfEle).eta(),(theGsfEle).phi())<<std::endl; 
    std::cout << "Electron and Tau not matched.\n"; 
    return -999;
  }
  
  double mva;

  Tau_AbsEta_ = TMath::Abs((thePFTau).eta()) ;
  Tau_Pt_ = (thePFTau).pt();
  Tau_HasGsf_ = (((thePFTau).leadPFChargedHadrCand())->gsfTrackRef()).isNonnull();
  Tau_EmFraction_ = TMath::Max((thePFTau).emFraction(),float(0.0)); 
  Tau_NumChargedCands_ = (thePFTau).signalPFChargedHadrCands().size();
  Tau_NumGammaCands_ = (thePFTau).signalPFGammaCands().size();
  Tau_HadrHoP_ = (thePFTau).leadPFChargedHadrCand()->hcalEnergy()/(thePFTau).leadPFChargedHadrCand()->p();
  Tau_HadrEoP_ = (thePFTau).leadPFChargedHadrCand()->ecalEnergy()/(thePFTau).leadPFChargedHadrCand()->p();
  Tau_VisMass_ = (thePFTau).mass(); 
  Tau_HadrMva_ = TMath::Max((thePFTau).electronPreIDOutput(),float(-1.0)); 

  vector<float> GammasdEta;
  vector<float> GammasdPhi;
  vector<float> GammasPt;

  for(unsigned int k = 0 ; k < ((thePFTau).signalPFGammaCands()).size() ; k++){
    reco::PFCandidateRef gamma = ((thePFTau).signalPFGammaCands()).at(k);
    if( ((thePFTau).leadPFChargedHadrCand()).isNonnull() ){
      GammasdEta.push_back( gamma->eta() - (thePFTau).leadPFChargedHadrCand()->eta() );
      GammasdPhi.push_back( gamma->phi() - (thePFTau).leadPFChargedHadrCand()->phi() );
    }
    else{
      GammasdEta.push_back( gamma->eta() - (thePFTau).eta() );
      GammasdPhi.push_back( gamma->phi() - (thePFTau).phi() );
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

  GammadPt_ = sumPt/(thePFTau).pt();

  if(sumPt>0){
    dEta  /= sumPt;
    dPhi  /= sumPt;
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }

  //GammadEta_ = dEta;
  //GammadPhi_ = dPhi;
  
  GammadEta_ = TMath::Sqrt(dEta2)*TMath::Sqrt(GammadPt_)*(thePFTau).pt();
  GammadPhi_ = TMath::Sqrt(dPhi2)*TMath::Sqrt(GammadPt_)*(thePFTau).pt();

  Tau_GammaEtaMom_ = GammadEta_;
  Tau_GammaPhiMom_ = GammadPhi_;
  Tau_GammaEnFrac_ = GammadPt_;

  Elec_AbsEta_ = TMath::Abs((theGsfEle).eta());
  Elec_Pt_ = (theGsfEle).pt();

  //Variables related to the SC
  Elec_Ee_ = -99;
  Elec_Egamma_ = -99;
  Elec_Pin_ = -99;
  Elec_Pout_ = -99;
  Elec_EtotOverPin_ = -99;
  Elec_EgammaOverPdif_ = -99;

  reco::SuperClusterRef pfSuperCluster = (theGsfEle).pflowSuperCluster();
  if(pfSuperCluster.isNonnull() && pfSuperCluster.isAvailable()){
    Elec_Ee_ = 0.;
    Elec_Egamma_ = 0.;
    for (reco::CaloCluster_iterator pfCluster = pfSuperCluster->clustersBegin();
	 pfCluster != pfSuperCluster->clustersEnd(); ++pfCluster ) {
      float pfClusterEn = (*pfCluster)->energy();
      if ( pfCluster == pfSuperCluster->clustersBegin() ) Elec_Ee_ += pfClusterEn;
      else Elec_Egamma_ += pfClusterEn;
    }
    Elec_Pin_ = TMath::Sqrt((theGsfEle).trackMomentumAtVtx().Mag2());
    Elec_Pout_ = TMath::Sqrt((theGsfEle).trackMomentumOut().Mag2()); 
    Elec_EtotOverPin_ = (Elec_Ee_+Elec_Egamma_)/Elec_Pin_;
    Elec_EgammaOverPdif_ = Elec_Egamma_/(Elec_Pin_-Elec_Pout_);
  }

  Elec_EarlyBrem_ = (theGsfEle).mvaInput().earlyBrem;
  Elec_LateBrem_= (theGsfEle).mvaInput().lateBrem;
  Elec_Fbrem_ = (theGsfEle).fbrem();

  //Variables related to the CtfTrack
  Elec_Chi2KF_ = -99;
  Elec_NumHits_ = -99;
  if ((theGsfEle).closestCtfTrackRef().isNonnull()){
    Elec_Chi2KF_ = (theGsfEle).closestCtfTrackRef()->normalizedChi2();
    Elec_NumHits_ = (theGsfEle).closestCtfTrackRef()->numberOfValidHits();
  }

  //Variables related to the GsfTrack
  Elec_Chi2GSF_ = -99;
  Elec_GSFTrackResol_ = -99;
  Elec_GSFTracklnPt_ = -99;
  Elec_GSFTrackEta_ = -99;
  if((theGsfEle).gsfTrack().isNonnull()){
    Elec_Chi2GSF_ = (theGsfEle).gsfTrack()->normalizedChi2();
    Elec_GSFTrackResol_ = (theGsfEle).gsfTrack()->ptError()/(theGsfEle).gsfTrack()->pt();
    Elec_GSFTracklnPt_ = log((theGsfEle).gsfTrack()->pt())*TMath::Ln10();
    Elec_GSFTrackEta_ = (theGsfEle).gsfTrack()->eta();
  }


  if((thePFTau).signalPFGammaCands().size()==0){
    if(TMath::Abs((thePFTau).eta())<1.5) 
      mva = fTMVAReader_[0]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[4]->EvaluateMVA( methodName_ );
  }
  else if((thePFTau).signalPFGammaCands().size()>0 && !((thePFTau).leadPFChargedHadrCand()->gsfTrackRef()).isNonnull()){
    if(TMath::Abs((thePFTau).eta())<1.5) 
      mva = fTMVAReader_[1]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[5]->EvaluateMVA( methodName_ );
  }
  else if((thePFTau).signalPFGammaCands().size()>0 && ((thePFTau).leadPFChargedHadrCand()->gsfTrackRef()).isNonnull() && TMath::Max((theGsfEle).mvaOutput().mva,float(-1.0))<-0.1){
    if(TMath::Abs((thePFTau).eta())<1.5) 
      mva = fTMVAReader_[2]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[6]->EvaluateMVA( methodName_ );
  }
	  else if((thePFTau).signalPFGammaCands().size()>0 && ((thePFTau).leadPFChargedHadrCand()->gsfTrackRef()).isNonnull() && TMath::Max((theGsfEle).mvaOutput().mva,float(-1.0))>-0.1){
    if(TMath::Abs((thePFTau).eta())<1.5) 
      mva = fTMVAReader_[3]->EvaluateMVA( methodName_ );
    else  
      mva = fTMVAReader_[7]->EvaluateMVA( methodName_ );
  }
  else{
    mva = -99;
  }

  return mva;


}
