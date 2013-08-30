#include <TFile.h>
#include <TMath.h>
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA3.h"
#include "RecoTauTag/RecoTau/interface/TMVAZipReader.h"

AntiElectronIDMVA3::AntiElectronIDMVA3()
  : isInitialized_(kFALSE),
    methodName_("BDTG")
{
  for ( unsigned i = 0; i < 16; ++i ) {
    fTMVAReader_[i] = 0;
  }

  verbosity_ = 1;
}

AntiElectronIDMVA3::~AntiElectronIDMVA3()
{
  for ( unsigned i = 0; i < 16; ++i ) {
    if ( fTMVAReader_[i] ) delete fTMVAReader_[i];
  }
}

enum {  k_NoEleMatchwoGwoGSF_BL,
	k_NoEleMatchwoGwGSF_BL,
	k_NoEleMatchwGwoGSF_BL,
	k_NoEleMatchwGwGSF_BL,
	k_woGwoGSF_BL,
	k_woGwGSF_BL,
	k_wGwoGSF_BL,
	k_wGwGSF_BL,
	k_NoEleMatchwoGwoGSF_EC,
	k_NoEleMatchwoGwGSF_EC,
	k_NoEleMatchwGwoGSF_EC,
	k_NoEleMatchwGwGSF_EC,
	k_woGwoGSF_EC,
	k_woGwGSF_EC,
	k_wGwoGSF_EC,
	k_wGwGSF_EC};


void AntiElectronIDMVA3::Initialize_from_file(const std::string& methodName,
					      const std::string& oneProngNoEleMatch0Pi0woGSF_BL,
					      const std::string& oneProngNoEleMatch0Pi0wGSF_BL,
					      const std::string& oneProngNoEleMatch1Pi0woGSF_BL,
					      const std::string& oneProngNoEleMatch1Pi0wGSF_BL,
					      const std::string& oneProng0Pi0woGSF_BL,
					      const std::string& oneProng0Pi0wGSF_BL,
					      const std::string& oneProng1Pi0woGSF_BL,
					      const std::string& oneProng1Pi0wGSF_BL,
					      const std::string& oneProngNoEleMatch0Pi0woGSF_EC,
					      const std::string& oneProngNoEleMatch0Pi0wGSF_EC,
					      const std::string& oneProngNoEleMatch1Pi0woGSF_EC,
					      const std::string& oneProngNoEleMatch1Pi0wGSF_EC,
					      const std::string& oneProng0Pi0woGSF_EC,
					      const std::string& oneProng0Pi0wGSF_EC,
					      const std::string& oneProng1Pi0woGSF_EC,
					      const std::string& oneProng1Pi0wGSF_EC)
{
  for ( unsigned i = 0; i < 16; ++i ) {
    if ( fTMVAReader_[i] ) delete fTMVAReader_[i];
  }

  isInitialized_ = kTRUE;
  methodName_    = methodName;

  bookMVAs();

  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatchwoGwoGSF_BL], "BDTG", oneProngNoEleMatch0Pi0woGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatchwoGwGSF_BL], "BDTG", oneProngNoEleMatch0Pi0wGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatchwGwoGSF_BL], "BDTG", oneProngNoEleMatch1Pi0woGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatchwGwGSF_BL], "BDTG", oneProngNoEleMatch1Pi0wGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_woGwoGSF_BL], "BDTG", oneProng0Pi0woGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_woGwGSF_BL], "BDTG", oneProng0Pi0wGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwoGSF_BL], "BDTG", oneProng1Pi0woGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwGSF_BL], "BDTG", oneProng1Pi0wGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatchwoGwoGSF_EC], "BDTG", oneProngNoEleMatch0Pi0woGSF_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatchwoGwGSF_EC], "BDTG", oneProngNoEleMatch0Pi0wGSF_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatchwGwoGSF_EC], "BDTG", oneProngNoEleMatch1Pi0woGSF_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatchwGwGSF_EC], "BDTG", oneProngNoEleMatch1Pi0wGSF_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_woGwoGSF_EC], "BDTG", oneProng0Pi0woGSF_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_woGwGSF_EC], "BDTG", oneProng0Pi0wGSF_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwoGSF_EC], "BDTG", oneProng1Pi0woGSF_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwGSF_EC], "BDTG", oneProng1Pi0wGSF_EC);

}

void AntiElectronIDMVA3::Initialize_from_string(const std::string& methodName,
						const std::string& oneProngNoEleMatch0Pi0woGSF_BL,
						const std::string& oneProngNoEleMatch0Pi0wGSF_BL,
						const std::string& oneProngNoEleMatch1Pi0woGSF_BL,
						const std::string& oneProngNoEleMatch1Pi0wGSF_BL,
						const std::string& oneProng0Pi0woGSF_BL,
						const std::string& oneProng0Pi0wGSF_BL,
						const std::string& oneProng1Pi0woGSF_BL,
						const std::string& oneProng1Pi0wGSF_BL,
						const std::string& oneProngNoEleMatch0Pi0woGSF_EC,
						const std::string& oneProngNoEleMatch0Pi0wGSF_EC,
						const std::string& oneProngNoEleMatch1Pi0woGSF_EC,
						const std::string& oneProngNoEleMatch1Pi0wGSF_EC,
						const std::string& oneProng0Pi0woGSF_EC,
						const std::string& oneProng0Pi0wGSF_EC,
						const std::string& oneProng1Pi0woGSF_EC,
						const std::string& oneProng1Pi0wGSF_EC)
{
  for ( unsigned i = 0; i < 16; ++i ) {
    if ( fTMVAReader_[i] ) delete fTMVAReader_[i];
  }

  isInitialized_ = kTRUE;
  methodName_    = methodName;

  bookMVAs();

  fTMVAReader_[k_NoEleMatchwoGwoGSF_BL]->BookMVA(TMVA::Types::kBDT, oneProngNoEleMatch0Pi0woGSF_BL.data());
  fTMVAReader_[k_NoEleMatchwoGwGSF_BL]->BookMVA(TMVA::Types::kBDT, oneProngNoEleMatch0Pi0wGSF_BL.data());
  fTMVAReader_[k_NoEleMatchwGwoGSF_BL]->BookMVA(TMVA::Types::kBDT, oneProngNoEleMatch1Pi0woGSF_BL.data());
  fTMVAReader_[k_NoEleMatchwGwGSF_BL]->BookMVA(TMVA::Types::kBDT, oneProngNoEleMatch1Pi0wGSF_BL.data());
  fTMVAReader_[k_woGwoGSF_BL]->BookMVA(TMVA::Types::kBDT, oneProng0Pi0woGSF_BL.data());
  fTMVAReader_[k_woGwGSF_BL]->BookMVA(TMVA::Types::kBDT, oneProng0Pi0wGSF_BL.data());
  fTMVAReader_[k_wGwoGSF_BL]->BookMVA(TMVA::Types::kBDT, oneProng1Pi0woGSF_BL.data());
  fTMVAReader_[k_wGwGSF_BL]->BookMVA(TMVA::Types::kBDT, oneProng1Pi0wGSF_BL.data());
  fTMVAReader_[k_NoEleMatchwoGwoGSF_EC]->BookMVA(TMVA::Types::kBDT, oneProngNoEleMatch0Pi0woGSF_EC.data());
  fTMVAReader_[k_NoEleMatchwoGwGSF_EC]->BookMVA(TMVA::Types::kBDT, oneProngNoEleMatch0Pi0wGSF_EC.data());
  fTMVAReader_[k_NoEleMatchwGwoGSF_EC]->BookMVA(TMVA::Types::kBDT, oneProngNoEleMatch1Pi0woGSF_EC.data());
  fTMVAReader_[k_NoEleMatchwGwGSF_EC]->BookMVA(TMVA::Types::kBDT, oneProngNoEleMatch1Pi0wGSF_EC.data());
  fTMVAReader_[k_woGwoGSF_EC]->BookMVA(TMVA::Types::kBDT, oneProng0Pi0woGSF_EC.data());
  fTMVAReader_[k_woGwGSF_EC]->BookMVA(TMVA::Types::kBDT, oneProng0Pi0wGSF_EC.data());
  fTMVAReader_[k_wGwoGSF_EC]->BookMVA(TMVA::Types::kBDT, oneProng1Pi0woGSF_EC.data());
  fTMVAReader_[k_wGwGSF_EC]->BookMVA(TMVA::Types::kBDT, oneProng1Pi0wGSF_EC.data());

}

void AntiElectronIDMVA3::bookMVAs()
{
  //TMVA::Tools::Instance();

  TMVA::Reader *readerNoEleMatchwoGwoGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatchwoGwoGSF_BL->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerNoEleMatchwoGwoGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerNoEleMatchwoGwoGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerNoEleMatchwoGwoGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerNoEleMatchwoGwoGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerNoEleMatchwoGwoGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerNoEleMatchwoGwoGSF_BL->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerNoEleMatchwoGwoGSF_BL->AddVariable("Tau_dCrackPhi",&Tau_dCrackPhi_);
  readerNoEleMatchwoGwoGSF_BL->SetVerbose(verbosity_);

  TMVA::Reader *readerNoEleMatchwoGwGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_HadrMva",&Tau_HadrMva_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_GSFChi2",&Tau_GSFChi2_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)",&Tau_NumHitsVariable_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_GSFTrackResol",&Tau_GSFTrackResol_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_GSFTracklnPt",&Tau_GSFTracklnPt_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_GSFTrackEta",&Tau_GSFTrackEta_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerNoEleMatchwoGwGSF_BL->AddVariable("Tau_dCrackPhi",&Tau_dCrackPhi_);
  readerNoEleMatchwoGwGSF_BL->SetVerbose(verbosity_);

  TMVA::Reader *readerNoEleMatchwGwoGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerNoEleMatchwGwoGSF_BL->AddVariable("Tau_dCrackPhi",&Tau_dCrackPhi_);
  readerNoEleMatchwGwoGSF_BL->SetVerbose(verbosity_);

  TMVA::Reader *readerNoEleMatchwGwGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_HadrMva",&Tau_HadrMva_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_GSFChi2",&Tau_GSFChi2_);
  readerNoEleMatchwGwGSF_BL->AddVariable("(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)",&Tau_NumHitsVariable_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_GSFTrackResol",&Tau_GSFTrackResol_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_GSFTracklnPt",&Tau_GSFTracklnPt_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_GSFTrackEta",&Tau_GSFTrackEta_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerNoEleMatchwGwGSF_BL->AddVariable("Tau_dCrackPhi",&Tau_dCrackPhi_);
  readerNoEleMatchwGwGSF_BL->SetVerbose(verbosity_);

  TMVA::Reader *readerwoGwoGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerwoGwoGSF_BL->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwoGwoGSF_BL->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwoGwoGSF_BL->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwoGwoGSF_BL->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwoGwoGSF_BL->AddVariable("Elec_GSFNumHits",&Elec_GSFNumHits_);
  readerwoGwoGSF_BL->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwoGwoGSF_BL->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwoGwoGSF_BL->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwoGwoGSF_BL->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerwoGwoGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwoGwoGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwoGwoGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwoGwoGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwoGwoGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwoGwoGSF_BL->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerwoGwoGSF_BL->AddVariable("Tau_dCrackPhi",&Tau_dCrackPhi_);
  readerwoGwoGSF_BL->SetVerbose(verbosity_);

  TMVA::Reader *readerwoGwGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerwoGwGSF_BL->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwoGwGSF_BL->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwoGwGSF_BL->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwoGwGSF_BL->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwoGwGSF_BL->AddVariable("Elec_GSFNumHits",&Elec_GSFNumHits_);
  readerwoGwGSF_BL->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwoGwGSF_BL->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwoGwGSF_BL->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwoGwGSF_BL->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerwoGwGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwoGwGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwoGwGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwoGwGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwoGwGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwoGwGSF_BL->AddVariable("Tau_HadrMva",&Tau_HadrMva_);
  readerwoGwGSF_BL->AddVariable("Tau_GSFChi2",&Tau_GSFChi2_);
  readerwoGwGSF_BL->AddVariable("(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)",&Tau_NumHitsVariable_);
  readerwoGwGSF_BL->AddVariable("Tau_GSFTrackResol",&Tau_GSFTrackResol_);
  readerwoGwGSF_BL->AddVariable("Tau_GSFTracklnPt",&Tau_GSFTracklnPt_);
  readerwoGwGSF_BL->AddVariable("Tau_GSFTrackEta",&Tau_GSFTrackEta_);
  readerwoGwGSF_BL->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerwoGwGSF_BL->AddVariable("Tau_dCrackPhi",&Tau_dCrackPhi_);
  readerwoGwGSF_BL->SetVerbose(verbosity_);

  TMVA::Reader *readerwGwoGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwoGSF_BL->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwGwoGSF_BL->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwGwoGSF_BL->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwGwoGSF_BL->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFNumHits",&Elec_GSFNumHits_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwoGSF_BL->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerwGwoGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwoGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwoGSF_BL->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwoGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwoGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwoGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwoGSF_BL->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerwGwoGSF_BL->AddVariable("Tau_dCrackPhi",&Tau_dCrackPhi_);
  readerwGwoGSF_BL->SetVerbose(verbosity_);

  TMVA::Reader *readerwGwGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwGSF_BL->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwGwGSF_BL->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwGwGSF_BL->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwGwGSF_BL->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwGSF_BL->AddVariable("Elec_GSFNumHits",&Elec_GSFNumHits_);
  readerwGwGSF_BL->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwGSF_BL->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwGSF_BL->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwGSF_BL->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerwGwGSF_BL->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwGSF_BL->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwGSF_BL->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwGSF_BL->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwGSF_BL->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwGSF_BL->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwGSF_BL->AddVariable("Tau_HadrMva",&Tau_HadrMva_);
  readerwGwGSF_BL->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwGSF_BL->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwGSF_BL->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwGSF_BL->AddVariable("Tau_GSFChi2",&Tau_GSFChi2_);
  readerwGwGSF_BL->AddVariable("(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)",&Tau_NumHitsVariable_);
  readerwGwGSF_BL->AddVariable("Tau_GSFTrackResol",&Tau_GSFTrackResol_);
  readerwGwGSF_BL->AddVariable("Tau_GSFTracklnPt",&Tau_GSFTracklnPt_);
  readerwGwGSF_BL->AddVariable("Tau_GSFTrackEta",&Tau_GSFTrackEta_);
  readerwGwGSF_BL->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerwGwGSF_BL->AddVariable("Tau_dCrackPhi",&Tau_dCrackPhi_);
  readerwGwGSF_BL->SetVerbose(verbosity_);

  ////////////////////////

  TMVA::Reader *readerNoEleMatchwoGwoGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatchwoGwoGSF_EC->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerNoEleMatchwoGwoGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerNoEleMatchwoGwoGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerNoEleMatchwoGwoGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerNoEleMatchwoGwoGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerNoEleMatchwoGwoGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerNoEleMatchwoGwoGSF_EC->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerNoEleMatchwoGwoGSF_EC->SetVerbose(verbosity_);

  TMVA::Reader *readerNoEleMatchwoGwGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_HadrMva",&Tau_HadrMva_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_GSFChi2",&Tau_GSFChi2_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)",&Tau_NumHitsVariable_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_GSFTrackResol",&Tau_GSFTrackResol_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_GSFTracklnPt",&Tau_GSFTracklnPt_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_GSFTrackEta",&Tau_GSFTrackEta_);
  readerNoEleMatchwoGwGSF_EC->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerNoEleMatchwoGwGSF_EC->SetVerbose(verbosity_);

  TMVA::Reader *readerNoEleMatchwGwoGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerNoEleMatchwGwoGSF_EC->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerNoEleMatchwGwoGSF_EC->SetVerbose(verbosity_);

  TMVA::Reader *readerNoEleMatchwGwGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_HadrMva",&Tau_HadrMva_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_GSFChi2",&Tau_GSFChi2_);
  readerNoEleMatchwGwGSF_EC->AddVariable("(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)",&Tau_NumHitsVariable_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_GSFTrackResol",&Tau_GSFTrackResol_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_GSFTracklnPt",&Tau_GSFTracklnPt_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_GSFTrackEta",&Tau_GSFTrackEta_);
  readerNoEleMatchwGwGSF_EC->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerNoEleMatchwGwGSF_EC->SetVerbose(verbosity_);

  TMVA::Reader *readerwoGwoGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerwoGwoGSF_EC->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwoGwoGSF_EC->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwoGwoGSF_EC->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwoGwoGSF_EC->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwoGwoGSF_EC->AddVariable("Elec_GSFNumHits",&Elec_GSFNumHits_);
  readerwoGwoGSF_EC->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwoGwoGSF_EC->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwoGwoGSF_EC->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwoGwoGSF_EC->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerwoGwoGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwoGwoGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwoGwoGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwoGwoGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwoGwoGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwoGwoGSF_EC->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerwoGwoGSF_EC->SetVerbose(verbosity_);

  TMVA::Reader *readerwoGwGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerwoGwGSF_EC->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwoGwGSF_EC->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwoGwGSF_EC->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwoGwGSF_EC->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwoGwGSF_EC->AddVariable("Elec_GSFNumHits",&Elec_GSFNumHits_);
  readerwoGwGSF_EC->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwoGwGSF_EC->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwoGwGSF_EC->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwoGwGSF_EC->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerwoGwGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwoGwGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwoGwGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwoGwGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwoGwGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwoGwGSF_EC->AddVariable("Tau_HadrMva",&Tau_HadrMva_);
  readerwoGwGSF_EC->AddVariable("Tau_GSFChi2",&Tau_GSFChi2_);
  readerwoGwGSF_EC->AddVariable("(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)",&Tau_NumHitsVariable_);
  readerwoGwGSF_EC->AddVariable("Tau_GSFTrackResol",&Tau_GSFTrackResol_);
  readerwoGwGSF_EC->AddVariable("Tau_GSFTracklnPt",&Tau_GSFTracklnPt_);
  readerwoGwGSF_EC->AddVariable("Tau_GSFTrackEta",&Tau_GSFTrackEta_);
  readerwoGwGSF_EC->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerwoGwGSF_EC->SetVerbose(verbosity_);

  TMVA::Reader *readerwGwoGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwoGSF_EC->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwGwoGSF_EC->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwGwoGSF_EC->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwGwoGSF_EC->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFNumHits",&Elec_GSFNumHits_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwoGSF_EC->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerwGwoGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwoGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwoGSF_EC->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwoGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwoGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwoGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwoGSF_EC->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerwGwoGSF_EC->SetVerbose(verbosity_);

  TMVA::Reader *readerwGwGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwGSF_EC->AddVariable("Elec_EtotOverPin",&Elec_EtotOverPin_);
  readerwGwGSF_EC->AddVariable("Elec_EgammaOverPdif",&Elec_EgammaOverPdif_);
  readerwGwGSF_EC->AddVariable("Elec_Fbrem",&Elec_Fbrem_);
  readerwGwGSF_EC->AddVariable("Elec_Chi2GSF",&Elec_Chi2GSF_);
  readerwGwGSF_EC->AddVariable("Elec_GSFNumHits",&Elec_GSFNumHits_);
  readerwGwGSF_EC->AddVariable("Elec_GSFTrackResol",&Elec_GSFTrackResol_);
  readerwGwGSF_EC->AddVariable("Elec_GSFTracklnPt",&Elec_GSFTracklnPt_);
  readerwGwGSF_EC->AddVariable("Elec_GSFTrackEta",&Elec_GSFTrackEta_);
  readerwGwGSF_EC->AddVariable("Tau_EtaAtEcalEntrance",&Tau_EtaAtEcalEntrance_);
  readerwGwGSF_EC->AddVariable("Tau_Pt",&Tau_Pt_);
  readerwGwGSF_EC->AddVariable("Tau_EmFraction",&Tau_EmFraction_);
  readerwGwGSF_EC->AddVariable("Tau_NumGammaCands",&Tau_NumGammaCands_);
  readerwGwGSF_EC->AddVariable("Tau_HadrHoP",&Tau_HadrHoP_);
  readerwGwGSF_EC->AddVariable("Tau_HadrEoP",&Tau_HadrEoP_);
  readerwGwGSF_EC->AddVariable("Tau_VisMass",&Tau_VisMass_);
  readerwGwGSF_EC->AddVariable("Tau_HadrMva",&Tau_HadrMva_);
  readerwGwGSF_EC->AddVariable("Tau_GammaEtaMom",&Tau_GammaEtaMom_);
  readerwGwGSF_EC->AddVariable("Tau_GammaPhiMom",&Tau_GammaPhiMom_);
  readerwGwGSF_EC->AddVariable("Tau_GammaEnFrac",&Tau_GammaEnFrac_);
  readerwGwGSF_EC->AddVariable("Tau_GSFChi2",&Tau_GSFChi2_);
  readerwGwGSF_EC->AddVariable("(Tau_GSFNumHits - Tau_KFNumHits)/(Tau_GSFNumHits + Tau_KFNumHits)",&Tau_NumHitsVariable_);
  readerwGwGSF_EC->AddVariable("Tau_GSFTrackResol",&Tau_GSFTrackResol_);
  readerwGwGSF_EC->AddVariable("Tau_GSFTracklnPt",&Tau_GSFTracklnPt_);
  readerwGwGSF_EC->AddVariable("Tau_GSFTrackEta",&Tau_GSFTrackEta_);
  readerwGwGSF_EC->AddVariable("Tau_dCrackEta",&Tau_dCrackEta_);
  readerwGwGSF_EC->SetVerbose(verbosity_);

  fTMVAReader_[k_NoEleMatchwoGwoGSF_BL] = readerNoEleMatchwoGwoGSF_BL;
  fTMVAReader_[k_NoEleMatchwoGwGSF_BL] = readerNoEleMatchwoGwGSF_BL;
  fTMVAReader_[k_NoEleMatchwGwoGSF_BL] = readerNoEleMatchwGwoGSF_BL;
  fTMVAReader_[k_NoEleMatchwGwGSF_BL] = readerNoEleMatchwGwGSF_BL;
  fTMVAReader_[k_woGwoGSF_BL] = readerwoGwoGSF_BL;
  fTMVAReader_[k_woGwGSF_BL] = readerwoGwGSF_BL;
  fTMVAReader_[k_wGwoGSF_BL] = readerwGwoGSF_BL;
  fTMVAReader_[k_wGwGSF_BL] = readerwGwGSF_BL;
  fTMVAReader_[k_NoEleMatchwoGwoGSF_EC] = readerNoEleMatchwoGwoGSF_EC;
  fTMVAReader_[k_NoEleMatchwoGwGSF_EC] = readerNoEleMatchwoGwGSF_EC;
  fTMVAReader_[k_NoEleMatchwGwoGSF_EC] = readerNoEleMatchwGwoGSF_EC;
  fTMVAReader_[k_NoEleMatchwGwGSF_EC] = readerNoEleMatchwGwGSF_EC;
  fTMVAReader_[k_woGwoGSF_EC] = readerwoGwoGSF_EC;
  fTMVAReader_[k_woGwGSF_EC] = readerwoGwGSF_EC;
  fTMVAReader_[k_wGwoGSF_EC] = readerwGwoGSF_EC;
  fTMVAReader_[k_wGwGSF_EC] = readerwGwGSF_EC;

}

double AntiElectronIDMVA3::MVAValue(Float_t TauEtaAtEcalEntrance,
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

double AntiElectronIDMVA3::MVAValue(Float_t TauEtaAtEcalEntrance,
				    Float_t TauPt,
				    Float_t TaudCrackEta,
				    Float_t TaudCrackPhi,
				    Float_t TauEmFraction,
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
    //std::cout << "Error: AntiElectronMVA not properly initialized.\n";
    return -99.;
  }

  Tau_EtaAtEcalEntrance_ = TauEtaAtEcalEntrance;
  Tau_Pt_ = TauPt;
  Tau_dCrackEta_ = TaudCrackEta;
  Tau_dCrackPhi_ = TaudCrackPhi;
  Tau_EmFraction_ = TMath::Max(TauEmFraction,float(0.0));
  Tau_NumGammaCands_ = TauSignalPFGammaCands;
  Tau_HadrHoP_ = TauLeadPFChargedHadrHoP;
  Tau_HadrEoP_ = TauLeadPFChargedHadrEoP;
  Tau_VisMass_ = TauVisMass;
  Tau_HadrMva_ = TauHadrMva;
  Tau_GammaEtaMom_ = TauGammaEtaMom;
  Tau_GammaPhiMom_ = TauGammaPhiMom;
  Tau_GammaEnFrac_ = TauGammaEnFrac;
  Tau_GSFChi2_ = TauGSFChi2;
  Tau_NumHitsVariable_ = (TauGSFNumHits-TauKFNumHits)/(TauGSFNumHits+TauKFNumHits);
  Tau_GSFTrackResol_ = TauGSFTrackResol;
  Tau_GSFTracklnPt_ = TauGSFTracklnPt;
  Tau_GSFTrackEta_ = TauGSFTrackEta;

  Elec_EtotOverPin_ = (ElecEe + ElecEgamma)/ElecPin;
  Elec_EgammaOverPdif_ = ElecEgamma/(ElecPin - ElecPout);
  Elec_Fbrem_ = ElecFbrem;
  Elec_Chi2GSF_ = ElecChi2GSF;
  Elec_GSFNumHits_ = ElecGSFNumHits;
  Elec_GSFTrackResol_ = ElecGSFTrackResol;
  Elec_GSFTracklnPt_ = ElecGSFTracklnPt;
  Elec_GSFTrackEta_ = ElecGSFTrackEta;

  double mva = -99.;
  if ( TauSignalPFChargedCands == 3 ) mva = 1.0;
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 &&  TauSignalPFGammaCands == 0 && TauHasGsf < 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_NoEleMatchwoGwoGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_NoEleMatchwoGwoGSF_EC]->EvaluateMVA(methodName_);
  }
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 &&  TauSignalPFGammaCands == 0 && TauHasGsf > 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_NoEleMatchwoGwGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_NoEleMatchwoGwGSF_EC]->EvaluateMVA(methodName_);
  }
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 &&  TauSignalPFGammaCands > 0 && TauHasGsf < 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_NoEleMatchwGwoGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_NoEleMatchwGwoGSF_EC]->EvaluateMVA(methodName_);
  }
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 &&  TauSignalPFGammaCands > 0 && TauHasGsf > 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_NoEleMatchwGwGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_NoEleMatchwGwGSF_EC]->EvaluateMVA(methodName_);
  }
  else if ( TauSignalPFGammaCands == 0 && TauHasGsf < 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_woGwoGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_woGwoGSF_EC]->EvaluateMVA(methodName_);
  }
  else if ( TauSignalPFGammaCands == 0 && TauHasGsf > 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_woGwGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_woGwGSF_EC]->EvaluateMVA(methodName_);
  }
  else if ( TauSignalPFGammaCands > 0 && TauHasGsf < 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_wGwoGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_wGwoGSF_EC]->EvaluateMVA(methodName_);
  }
  else if ( TauSignalPFGammaCands > 0 && TauHasGsf > 0.5) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_wGwGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_wGwGSF_EC]->EvaluateMVA(methodName_);
  }
  return mva;
}


double AntiElectronIDMVA3::MVAValue(const reco::PFTau& thePFTau,
				    const reco::GsfElectron& theGsfEle)

{
  Float_t TauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0;
  float sumEnergy = 0;
  for(unsigned int j = 0 ; j < (thePFTau.signalPFCands()).size() ; j++){
    reco::PFCandidateRef pfcandidate = (thePFTau.signalPFCands()).at(j);
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
    reco::PFCandidateRef gamma = thePFTau.signalPFGammaCands().at(i);
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
    reco::PFCandidateRef pfcandidate = (thePFTau.signalPFCands()).at(j);
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

double AntiElectronIDMVA3::MVAValue(const reco::PFTau& thePFTau)
{
  Float_t TauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0;
  float sumEnergy = 0;
  for(unsigned int j = 0 ; j < (thePFTau.signalPFCands()).size() ; j++){
    reco::PFCandidateRef pfcandidate = (thePFTau.signalPFCands()).at(j);
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
    reco::PFCandidateRef gamma = thePFTau.signalPFGammaCands().at(i);
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
    reco::PFCandidateRef pfcandidate = (thePFTau.signalPFCands()).at(j);
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
AntiElectronIDMVA3::minimum(double a,double b){
  if(TMath::Abs(b)<TMath::Abs(a)) return b;
  else return a;
}

//compute the unsigned distance to the closest phi-crack in the barrel
double
AntiElectronIDMVA3:: dCrackPhi(double phi, double eta){

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
      //std::cout<<"Problem in dminphi"<<std::endl;
    }
  }
  else{
    return -99.;
    //std::cout<<"Encap region"<<std::endl;
  }

  return TMath::Abs(m);
}

//compute the unsigned distance to the closest phi-crack in the barrel
double
AntiElectronIDMVA3:: dCrackEta(double eta){

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
