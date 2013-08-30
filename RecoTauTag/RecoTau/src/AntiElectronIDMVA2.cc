#include <TFile.h>
#include <TMath.h>
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA2.h"
#include "RecoTauTag/RecoTau/interface/TMVAZipReader.h"
#include "FWCore/Utilities/interface/Exception.h"

AntiElectronIDMVA2::AntiElectronIDMVA2()
  : isInitialized_(kFALSE),
    methodName_("BDT")
{
  for ( unsigned i = 0; i < 10; ++i ) {
    fTMVAReader_[i] = 0;
  }

  verbosity_ = 1;
}

AntiElectronIDMVA2::~AntiElectronIDMVA2()
{
  for ( unsigned i = 0; i < 10; ++i ) {
    if ( fTMVAReader_[i] ) delete fTMVAReader_[i];
  }
}

enum {  k_NoEleMatch_BL,
	k_woG_BL,
	k_wGwoGSF_BL,
	k_wGwGSFwoPFMVA_BL,
	k_wGwGSFwPFMVA_BL,
	k_NoEleMatch_EC,
	k_woG_EC,
	k_wGwoGSF_EC,
	k_wGwGSFwoPFMVA_EC,
	k_wGwGSFwPFMVA_EC };


void AntiElectronIDMVA2::Initialize_from_file(const std::string& methodName,
					      const std::string& oneProngNoEleMatch_BL,
					      const std::string& oneProng0Pi0_BL,
					      const std::string& oneProng1pi0woGSF_BL,
					      const std::string& oneProng1pi0wGSFwoPfEleMva_BL,
					      const std::string& oneProng1pi0wGSFwPfEleMva_BL,
					      const std::string& oneProngNoEleMatch_EC,
					      const std::string& oneProng0Pi0_EC,
					      const std::string& oneProng1pi0woGSF_EC,
					      const std::string& oneProng1pi0wGSFwoPfEleMva_EC,
					      const std::string& oneProng1pi0wGSFwPfEleMva_EC)
{
  for ( unsigned i = 0; i < 10; ++i ) {
    if ( fTMVAReader_[i] ) delete fTMVAReader_[i];
  }

  isInitialized_ = kTRUE;
  methodName_ = methodName;

  bookMVAs();

  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatch_BL], methodName_, oneProngNoEleMatch_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_woG_BL], methodName_, oneProng0Pi0_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwoGSF_BL], methodName_, oneProng1pi0woGSF_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwGSFwoPFMVA_BL], methodName_, oneProng1pi0wGSFwoPfEleMva_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwGSFwPFMVA_BL], methodName_, oneProng1pi0wGSFwPfEleMva_BL);
  reco::details::loadTMVAWeights(fTMVAReader_[k_NoEleMatch_EC], methodName_, oneProngNoEleMatch_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_woG_EC], methodName_, oneProng0Pi0_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwoGSF_EC], methodName_, oneProng1pi0woGSF_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwGSFwoPFMVA_EC], methodName_, oneProng1pi0wGSFwoPfEleMva_EC);
  reco::details::loadTMVAWeights(fTMVAReader_[k_wGwGSFwPFMVA_EC], methodName_, oneProng1pi0wGSFwPfEleMva_EC);
}

void AntiElectronIDMVA2::Initialize_from_string(const std::string& methodName,
						const std::string& oneProngNoEleMatch_BL,
						const std::string& oneProng0Pi0_BL,
						const std::string& oneProng1pi0woGSF_BL,
						const std::string& oneProng1pi0wGSFwoPfEleMva_BL,
						const std::string& oneProng1pi0wGSFwPfEleMva_BL,
						const std::string& oneProngNoEleMatch_EC,
						const std::string& oneProng0Pi0_EC,
						const std::string& oneProng1pi0woGSF_EC,
						const std::string& oneProng1pi0wGSFwoPfEleMva_EC,
	 					const std::string& oneProng1pi0wGSFwPfEleMva_EC)
{
  for ( unsigned i = 0; i < 10; ++i ) {
    if ( fTMVAReader_[i] ) delete fTMVAReader_[i];
  }

  isInitialized_ = kTRUE;
  methodName_ = methodName;

  bookMVAs();

  int methodName_int = -1;
  if ( methodName_ == "BDT" ) methodName_int = TMVA::Types::kBDT;
  else throw cms::Exception("InvalidParameter") 
    << "Invalid TMVA method name = " << methodName_ << " !!\n";

  fTMVAReader_[k_NoEleMatch_BL]->BookMVA(methodName_int, oneProngNoEleMatch_BL.data());
  fTMVAReader_[k_woG_BL]->BookMVA(methodName_int, oneProng0Pi0_BL.data());
  fTMVAReader_[k_wGwoGSF_BL]->BookMVA(methodName_int, oneProng1pi0woGSF_BL.data());
  fTMVAReader_[k_wGwGSFwoPFMVA_BL]->BookMVA(methodName_int, oneProng1pi0wGSFwoPfEleMva_BL.data());
  fTMVAReader_[k_wGwGSFwPFMVA_BL]->BookMVA(methodName_int, oneProng1pi0wGSFwPfEleMva_BL.data());
  fTMVAReader_[k_NoEleMatch_EC]->BookMVA(methodName_int, oneProngNoEleMatch_EC.data());
  fTMVAReader_[k_woG_EC]->BookMVA(methodName_int, oneProng0Pi0_EC.data());
  fTMVAReader_[k_wGwoGSF_EC]->BookMVA(methodName_int, oneProng1pi0woGSF_EC.data());
  fTMVAReader_[k_wGwGSFwoPFMVA_EC]->BookMVA(methodName_int, oneProng1pi0wGSFwoPfEleMva_EC.data());
  fTMVAReader_[k_wGwGSFwPFMVA_EC]->BookMVA(methodName_int, oneProng1pi0wGSFwPfEleMva_EC.data());
}

void AntiElectronIDMVA2::bookMVAs()
{
  //TMVA::Tools::Instance();

  TMVA::Reader* readerNoEleMatch_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatch_BL->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerNoEleMatch_BL->AddVariable("Tau_Pt", &Tau_Pt_);
  readerNoEleMatch_BL->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerNoEleMatch_BL->AddVariable("Tau_NumGammaCands", &Tau_NumGammaCands_);
  readerNoEleMatch_BL->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerNoEleMatch_BL->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerNoEleMatch_BL->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerNoEleMatch_BL->AddVariable("Tau_GammaEtaMom", &Tau_GammaEtaMom_);
  readerNoEleMatch_BL->AddVariable("Tau_GammaPhiMom", &Tau_GammaPhiMom_);
  readerNoEleMatch_BL->AddVariable("Tau_GammaEnFrac", &Tau_GammaEnFrac_);
  readerNoEleMatch_BL->SetVerbose(verbosity_);

  TMVA::Reader* readerwoG_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerwoG_BL->AddVariable("Elec_EtotOverPin", &Elec_EtotOverPin_);
  readerwoG_BL->AddVariable("Elec_LateBrem", &Elec_LateBrem_);
  readerwoG_BL->AddVariable("Elec_Fbrem", &Elec_Fbrem_);
  readerwoG_BL->AddVariable("Elec_Chi2KF", &Elec_Chi2KF_);
  readerwoG_BL->AddVariable("Elec_GSFTrackResol", &Elec_GSFTrackResol_);
  readerwoG_BL->AddVariable("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_);
  readerwoG_BL->AddVariable("Elec_GSFTrackEta", &Elec_GSFTrackEta_);
  readerwoG_BL->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerwoG_BL->AddVariable("Tau_Pt", &Tau_Pt_);
  readerwoG_BL->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerwoG_BL->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerwoG_BL->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerwoG_BL->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerwoG_BL->SetVerbose(verbosity_);

  TMVA::Reader* readerwGwoGSF_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwoGSF_BL->AddVariable("Elec_EtotOverPin", &Elec_EtotOverPin_);
  readerwGwoGSF_BL->AddVariable("Elec_EgammaOverPdif", &Elec_EgammaOverPdif_);
  readerwGwoGSF_BL->AddVariable("Elec_LateBrem", &Elec_LateBrem_);
  readerwGwoGSF_BL->AddVariable("Elec_Fbrem", &Elec_Fbrem_);
  readerwGwoGSF_BL->AddVariable("Elec_Chi2GSF", &Elec_Chi2GSF_);
  readerwGwoGSF_BL->AddVariable("Elec_NumHits", &Elec_NumHits_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTrackResol", &Elec_GSFTrackResol_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_);
  readerwGwoGSF_BL->AddVariable("Elec_GSFTrackEta", &Elec_GSFTrackEta_);
  readerwGwoGSF_BL->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerwGwoGSF_BL->AddVariable("Tau_Pt", &Tau_Pt_);
  readerwGwoGSF_BL->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerwGwoGSF_BL->AddVariable("Tau_NumGammaCands", &Tau_NumGammaCands_);
  readerwGwoGSF_BL->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerwGwoGSF_BL->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerwGwoGSF_BL->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaEtaMom", &Tau_GammaEtaMom_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaPhiMom", &Tau_GammaPhiMom_);
  readerwGwoGSF_BL->AddVariable("Tau_GammaEnFrac", &Tau_GammaEnFrac_);
  readerwGwoGSF_BL->SetVerbose(verbosity_);

  TMVA::Reader* readerwGwGSFwoPFMVA_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_Fbrem", &Elec_Fbrem_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_Chi2KF", &Elec_Chi2KF_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_Chi2GSF", &Elec_Chi2GSF_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_NumHits", &Elec_NumHits_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_GSFTrackResol", &Elec_GSFTrackResol_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Elec_GSFTrackEta", &Elec_GSFTrackEta_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_Pt", &Tau_Pt_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_NumGammaCands", &Tau_NumGammaCands_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_GammaEtaMom", &Tau_GammaEtaMom_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_GammaPhiMom", &Tau_GammaPhiMom_);
  readerwGwGSFwoPFMVA_BL->AddVariable("Tau_GammaEnFrac", &Tau_GammaEnFrac_);
  readerwGwGSFwoPFMVA_BL->SetVerbose(verbosity_);

  TMVA::Reader* readerwGwGSFwPFMVA_BL = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_EtotOverPin", &Elec_EtotOverPin_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_EeOverPout", &Elec_EeOverPout_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_LateBrem", &Elec_LateBrem_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_Chi2GSF", &Elec_Chi2GSF_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_NumHits", &Elec_NumHits_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_GSFTrackResol", &Elec_GSFTrackResol_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_);
  readerwGwGSFwPFMVA_BL->AddVariable("Elec_GSFTrackEta", &Elec_GSFTrackEta_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_Pt", &Tau_Pt_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_NumGammaCands", &Tau_NumGammaCands_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_GammaEtaMom", &Tau_GammaEtaMom_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_GammaPhiMom", &Tau_GammaPhiMom_);
  readerwGwGSFwPFMVA_BL->AddVariable("Tau_GammaEnFrac", &Tau_GammaEnFrac_);
  readerwGwGSFwPFMVA_BL->SetVerbose(verbosity_);

  TMVA::Reader* readerNoEleMatch_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerNoEleMatch_EC->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerNoEleMatch_EC->AddVariable("Tau_Pt", &Tau_Pt_);
  readerNoEleMatch_EC->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerNoEleMatch_EC->AddVariable("Tau_NumGammaCands", &Tau_NumGammaCands_);
  readerNoEleMatch_EC->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerNoEleMatch_EC->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerNoEleMatch_EC->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerNoEleMatch_EC->AddVariable("Tau_GammaEtaMom", &Tau_GammaEtaMom_);
  readerNoEleMatch_EC->AddVariable("Tau_GammaPhiMom", &Tau_GammaPhiMom_);
  readerNoEleMatch_EC->AddVariable("Tau_GammaEnFrac", &Tau_GammaEnFrac_);
  readerNoEleMatch_EC->SetVerbose(verbosity_);

  TMVA::Reader* readerwoG_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerwoG_EC->AddVariable("Elec_EtotOverPin", &Elec_EtotOverPin_);
  readerwoG_EC->AddVariable("Elec_LateBrem", &Elec_LateBrem_);
  readerwoG_EC->AddVariable("Elec_Fbrem", &Elec_Fbrem_);
  readerwoG_EC->AddVariable("Elec_Chi2KF", &Elec_Chi2KF_);
  readerwoG_EC->AddVariable("Elec_GSFTrackResol", &Elec_GSFTrackResol_);
  readerwoG_EC->AddVariable("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_);
  readerwoG_EC->AddVariable("Elec_GSFTrackEta", &Elec_GSFTrackEta_);
  readerwoG_EC->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerwoG_EC->AddVariable("Tau_Pt", &Tau_Pt_);
  readerwoG_EC->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerwoG_EC->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerwoG_EC->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerwoG_EC->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerwoG_EC->SetVerbose(verbosity_);

  TMVA::Reader* readerwGwoGSF_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwoGSF_EC->AddVariable("Elec_EtotOverPin", &Elec_EtotOverPin_);
  readerwGwoGSF_EC->AddVariable("Elec_EgammaOverPdif", &Elec_EgammaOverPdif_);
  readerwGwoGSF_EC->AddVariable("Elec_LateBrem", &Elec_LateBrem_);
  readerwGwoGSF_EC->AddVariable("Elec_Fbrem", &Elec_Fbrem_);
  readerwGwoGSF_EC->AddVariable("Elec_Chi2GSF", &Elec_Chi2GSF_);
  readerwGwoGSF_EC->AddVariable("Elec_NumHits", &Elec_NumHits_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTrackResol", &Elec_GSFTrackResol_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_);
  readerwGwoGSF_EC->AddVariable("Elec_GSFTrackEta", &Elec_GSFTrackEta_);
  readerwGwoGSF_EC->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerwGwoGSF_EC->AddVariable("Tau_Pt", &Tau_Pt_);
  readerwGwoGSF_EC->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerwGwoGSF_EC->AddVariable("Tau_NumGammaCands", &Tau_NumGammaCands_);
  readerwGwoGSF_EC->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerwGwoGSF_EC->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerwGwoGSF_EC->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaEtaMom", &Tau_GammaEtaMom_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaPhiMom", &Tau_GammaPhiMom_);
  readerwGwoGSF_EC->AddVariable("Tau_GammaEnFrac", &Tau_GammaEnFrac_);
  readerwGwoGSF_EC->SetVerbose(verbosity_);

  TMVA::Reader* readerwGwGSFwoPFMVA_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_Fbrem", &Elec_Fbrem_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_Chi2KF", &Elec_Chi2KF_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_Chi2GSF", &Elec_Chi2GSF_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_NumHits", &Elec_NumHits_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_GSFTrackResol", &Elec_GSFTrackResol_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Elec_GSFTrackEta", &Elec_GSFTrackEta_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_Pt", &Tau_Pt_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_NumGammaCands", &Tau_NumGammaCands_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_GammaEtaMom", &Tau_GammaEtaMom_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_GammaPhiMom", &Tau_GammaPhiMom_);
  readerwGwGSFwoPFMVA_EC->AddVariable("Tau_GammaEnFrac", &Tau_GammaEnFrac_);
  readerwGwGSFwoPFMVA_EC->SetVerbose(verbosity_);

  TMVA::Reader* readerwGwGSFwPFMVA_EC = new TMVA::Reader( "!Color:Silent:Error" );
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_EtotOverPin", &Elec_EtotOverPin_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_EeOverPout", &Elec_EeOverPout_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_LateBrem", &Elec_LateBrem_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_Chi2GSF", &Elec_Chi2GSF_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_NumHits", &Elec_NumHits_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_GSFTrackResol", &Elec_GSFTrackResol_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_GSFTracklnPt", &Elec_GSFTracklnPt_);
  readerwGwGSFwPFMVA_EC->AddVariable("Elec_GSFTrackEta", &Elec_GSFTrackEta_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_EtaAtEcalEntrance", &Tau_EtaAtEcalEntrance_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_Pt", &Tau_Pt_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_EmFraction", &Tau_EmFraction_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_NumGammaCands", &Tau_NumGammaCands_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_HadrHoP", &Tau_HadrHoP_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_HadrEoP", &Tau_HadrEoP_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_VisMass", &Tau_VisMass_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_GammaEtaMom", &Tau_GammaEtaMom_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_GammaPhiMom", &Tau_GammaPhiMom_);
  readerwGwGSFwPFMVA_EC->AddVariable("Tau_GammaEnFrac", &Tau_GammaEnFrac_);
  readerwGwGSFwPFMVA_EC->SetVerbose(verbosity_);

  fTMVAReader_[k_NoEleMatch_BL] = readerNoEleMatch_BL;
  fTMVAReader_[k_woG_BL] = readerwoG_BL;
  fTMVAReader_[k_wGwoGSF_BL] = readerwGwoGSF_BL;
  fTMVAReader_[k_wGwGSFwoPFMVA_BL] = readerwGwGSFwoPFMVA_BL;
  fTMVAReader_[k_wGwGSFwPFMVA_BL] = readerwGwGSFwPFMVA_BL;
  fTMVAReader_[k_NoEleMatch_EC] = readerNoEleMatch_EC;
  fTMVAReader_[k_woG_EC] = readerwoG_EC;
  fTMVAReader_[k_wGwoGSF_EC] = readerwGwoGSF_EC;
  fTMVAReader_[k_wGwGSFwoPFMVA_EC] = readerwGwGSFwoPFMVA_EC;
  fTMVAReader_[k_wGwGSFwPFMVA_EC] = readerwGwGSFwPFMVA_EC;
}

double AntiElectronIDMVA2::MVAValue(Float_t TauEtaAtEcalEntrance,
				    Float_t TauPhi,
				    Float_t TauPt,
				    Float_t TauSignalPFChargedCands,
				    Float_t TauSignalPFGammaCands,
				    Float_t TauLeadPFChargedHadrHoP,
				    Float_t TauLeadPFChargedHadrEoP,
				    Float_t TauHasGsf,
				    Float_t TauVisMass,
				    Float_t TauEmFraction,
				    const std::vector<Float_t>& GammasdEta,
				    const std::vector<Float_t>& GammasdPhi,
				    const std::vector<Float_t>& GammasPt,
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
				    Float_t ElecFbrem,
				    Float_t ElecChi2KF,
				    Float_t ElecChi2GSF,
				    Float_t ElecNumHits,
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

  Float_t GammaEnFrac = sumPt/TauPt;

  if ( sumPt > 0. ) {
    dEta  /= sumPt;
    dPhi  /= sumPt;
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }

  Float_t GammaEtaMom = TMath::Sqrt(dEta2)*TMath::Sqrt(GammaEnFrac)*TauPt;
  Float_t GammaPhiMom = TMath::Sqrt(dPhi2)*TMath::Sqrt(GammaEnFrac)*TauPt;

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPhi,
		  TauPt,
		  TauSignalPFChargedCands,
		  TauSignalPFGammaCands,
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauHasGsf,
		  TauVisMass,
		  TauEmFraction,
		  GammaEtaMom,
		  GammaPhiMom,
		  GammaEnFrac,
		  ElecEta,
		  ElecPhi,
		  ElecPt,
		  ElecPFMvaOutput,
		  ElecEe,
		  ElecEgamma,
		  ElecPin,
		  ElecPout,
		  ElecEarlyBrem,
		  ElecLateBrem,
		  ElecFbrem,
		  ElecChi2KF,
		  ElecChi2GSF,
		  ElecNumHits,
		  ElecGSFTrackResol,
		  ElecGSFTracklnPt,
		  ElecGSFTrackEta);
}

double AntiElectronIDMVA2::MVAValue(Float_t TauEtaAtEcalEntrance,
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
				    Float_t ElecFbrem,
				    Float_t ElecChi2KF,
				    Float_t ElecChi2GSF,
				    Float_t ElecNumHits,
				    Float_t ElecGSFTrackResol,
				    Float_t ElecGSFTracklnPt,
				    Float_t ElecGSFTrackEta)
{

  if ( !isInitialized_ ) {
    std::cout << "Error: AntiElectronMVA not properly initialized.\n";
    return -99.;
  }

  Tau_EtaAtEcalEntrance_ = TauEtaAtEcalEntrance;
  Tau_Pt_ = TauPt;
  Tau_HasGsf_ = TauHasGsf;
  Tau_EmFraction_ = TMath::Max(TauEmFraction,float(0.0));
  Tau_NumChargedCands_ = TauSignalPFChargedCands;
  Tau_NumGammaCands_ = TauSignalPFGammaCands;
  Tau_HadrHoP_ = TauLeadPFChargedHadrHoP;
  Tau_HadrEoP_ = TauLeadPFChargedHadrEoP;
  Tau_VisMass_ = TauVisMass;
  Tau_GammaEtaMom_ = GammaEtaMom;
  Tau_GammaPhiMom_ = GammaPhiMom;
  Tau_GammaEnFrac_ = GammaEnFrac;

  Elec_Eta_ = ElecEta;
  Elec_Pt_ = ElecPt;
  Elec_EtotOverPin_ = (ElecEe + ElecEgamma)/ElecPin;
  Elec_EgammaOverPdif_ = ElecEgamma/(ElecPin - ElecPout);
  Elec_EarlyBrem_ = ElecEarlyBrem;
  Elec_LateBrem_ = ElecLateBrem;
  Elec_Fbrem_ = ElecFbrem;
  Elec_Chi2KF_ = ElecChi2KF;
  Elec_Chi2GSF_ = ElecChi2GSF;
  Elec_NumHits_ = ElecNumHits;
  Elec_GSFTrackResol_ = ElecGSFTrackResol;
  Elec_GSFTracklnPt_ = ElecGSFTracklnPt;
  Elec_GSFTrackEta_ = ElecGSFTrackEta;

  double mva = -99.;
  if ( Tau_NumChargedCands_ == 3 ) mva = 1.0;
  else if ( deltaR(TauEtaAtEcalEntrance, TauPhi, ElecEta, ElecPhi) > 0.3 ) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_NoEleMatch_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_NoEleMatch_EC]->EvaluateMVA(methodName_);
  } else if ( TauSignalPFGammaCands == 0 ) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_woG_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_woG_EC]->EvaluateMVA(methodName_);
  } else if ( TauSignalPFGammaCands > 0 && TauHasGsf < 0.5 ) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_wGwoGSF_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_wGwoGSF_EC]->EvaluateMVA(methodName_);
  } else if ( TauSignalPFGammaCands > 0 && TauHasGsf > 0.5 && ElecPFMvaOutput < -0.1 ) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_wGwGSFwoPFMVA_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_wGwGSFwoPFMVA_EC]->EvaluateMVA(methodName_);
  } else if ( TauSignalPFGammaCands > 0 && TauHasGsf > 0.5 && ElecPFMvaOutput > -0.1 ) {
    if ( TMath::Abs(TauEtaAtEcalEntrance) < 1.5 ) mva = fTMVAReader_[k_wGwGSFwPFMVA_BL]->EvaluateMVA(methodName_);
    else mva = fTMVAReader_[k_wGwGSFwPFMVA_EC]->EvaluateMVA(methodName_);
  }

  return mva;
}


double AntiElectronIDMVA2::MVAValue(const reco::PFTau& thePFTau,
				    const reco::GsfElectron& theGsfEle)

{
  float sumEtaTimesEnergy = 0;
  float sumEnergy = 0;
  Float_t TauEtaAtEcalEntrance = 0;
  for(unsigned int j = 0 ; j < (thePFTau.signalPFCands()).size() ; j++){
       reco::PFCandidateRef pfcandidate = (thePFTau.signalPFCands()).at(j);
       sumEtaTimesEnergy += pfcandidate->positionAtECALEntrance().eta()*pfcandidate->energy();
       sumEnergy += pfcandidate->energy();
  }
  if(sumEnergy>0)TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;

  Float_t TauPhi = thePFTau.phi();
  Float_t TauPt = thePFTau.pt();
  Float_t TauSignalPFChargedCands = thePFTau.signalPFChargedHadrCands().size();
  Float_t TauSignalPFGammaCands = thePFTau.signalPFGammaCands().size();
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
  }
  Float_t TauHasGsf = thePFTau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
  Float_t TauVisMass = thePFTau.mass();
  Float_t TauEmFraction = TMath::Max(thePFTau.emFraction(), (Float_t)0.);
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

  Float_t ElecEta = theGsfEle.eta();
  Float_t ElecPhi = theGsfEle.phi();
  Float_t ElecPt = theGsfEle.pt();
  Float_t ElecPFMvaOutput = theGsfEle.mvaOutput().mva;
  //Variables related to the electron Cluster
  Float_t ElecEe = 0.;
  Float_t ElecEgamma = 0.;
  reco::SuperClusterRef pfSuperCluster = theGsfEle.pflowSuperCluster();
  if ( pfSuperCluster.isNonnull() && pfSuperCluster.isAvailable() ) {
    for ( reco::CaloCluster_iterator pfCluster = pfSuperCluster->clustersBegin();
	  pfCluster != pfSuperCluster->clustersEnd(); ++pfCluster ) {
      double pfClusterEn = (*pfCluster)->energy();
      if ( pfCluster == pfSuperCluster->clustersBegin() ) Elec_Ee_ += pfClusterEn;
      else Elec_Egamma_ += pfClusterEn;
    }
  }
  Float_t ElecPin = TMath::Sqrt(theGsfEle.trackMomentumAtVtx().Mag2());
  Float_t ElecPout = TMath::Sqrt(theGsfEle.trackMomentumOut().Mag2());
  Float_t ElecEarlyBrem = theGsfEle.mvaInput().earlyBrem;
  Float_t ElecLateBrem = theGsfEle.mvaInput().lateBrem;
  Float_t ElecFbrem = theGsfEle.fbrem();
  //Variables related to the CtfTrack
  Float_t ElecChi2KF = -99.;
  Float_t ElecNumHits = -99.;
  if ( theGsfEle.closestCtfTrackRef().isNonnull() ) {
    ElecChi2KF = theGsfEle.closestCtfTrackRef()->normalizedChi2();
    ElecNumHits = theGsfEle.closestCtfTrackRef()->numberOfValidHits();
  }
  //Variables related to the GsfTrack
  Float_t ElecChi2GSF = -99.;
  Float_t ElecGSFTrackResol = -99.;
  Float_t ElecGSFTracklnPt = -99.;
  Float_t ElecGSFTrackEta = -99.;
  if ( theGsfEle.gsfTrack().isNonnull() ) {
    ElecChi2GSF = (theGsfEle).gsfTrack()->normalizedChi2();
    if ( theGsfEle.gsfTrack()->pt() > 0. ) {
      ElecGSFTrackResol = theGsfEle.gsfTrack()->ptError()/theGsfEle.gsfTrack()->pt();
      ElecGSFTracklnPt = log(theGsfEle.gsfTrack()->pt())*TMath::Ln10();
    }
    ElecGSFTrackEta = theGsfEle.gsfTrack()->eta();
  }

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPhi,
		  TauPt,
		  TauSignalPFChargedCands,
		  TauSignalPFGammaCands,
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauHasGsf,
		  TauVisMass,
		  TauEmFraction,
		  GammasdEta,
		  GammasdPhi,
		  GammasPt,
		  ElecEta,
		  ElecPhi,
		  ElecPt,
		  ElecPFMvaOutput,
		  ElecEe,
		  ElecEgamma,
		  ElecPin,
		  ElecPout,
		  ElecEarlyBrem,
		  ElecLateBrem,
		  ElecFbrem,
		  ElecChi2KF,
		  ElecChi2GSF,
		  ElecNumHits,
		  ElecGSFTrackResol,
		  ElecGSFTracklnPt,
		  ElecGSFTrackEta);
}

double AntiElectronIDMVA2::MVAValue(const reco::PFTau& thePFTau)
{
  float sumEtaTimesEnergy = 0;
  float sumEnergy = 0;
  Float_t TauEtaAtEcalEntrance =0;
  for(unsigned int j = 0 ; j < (thePFTau.signalPFCands()).size() ; j++){
       reco::PFCandidateRef pfcandidate = (thePFTau.signalPFCands()).at(j);
       sumEtaTimesEnergy += pfcandidate->positionAtECALEntrance().eta()*pfcandidate->energy();
       sumEnergy += pfcandidate->energy();
  }
  if(sumEnergy>0)TauEtaAtEcalEntrance = sumEtaTimesEnergy/sumEnergy;

  Float_t TauPhi = thePFTau.phi();
  Float_t TauPt = thePFTau.pt();
  Float_t TauSignalPFChargedCands = thePFTau.signalPFChargedHadrCands().size();
  Float_t TauSignalPFGammaCands = thePFTau.signalPFGammaCands().size();
  Float_t TauLeadPFChargedHadrHoP = 0.;
  Float_t TauLeadPFChargedHadrEoP = 0.;
  if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
    TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
  }
  Float_t TauHasGsf = thePFTau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull();
  Float_t TauVisMass = thePFTau.mass();
  Float_t TauEmFraction = TMath::Max(thePFTau.emFraction(), (Float_t)0.);
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

  Float_t dummyElecEta = 9.9;

  return MVAValue(TauEtaAtEcalEntrance,
		  TauPhi,
		  TauPt,
		  TauSignalPFChargedCands,
		  TauSignalPFGammaCands,
		  TauLeadPFChargedHadrHoP,
		  TauLeadPFChargedHadrEoP,
		  TauHasGsf,
		  TauVisMass,
		  TauEmFraction,
		  GammasdEta,
		  GammasdPhi,
		  GammasPt,
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
		  0.,
		  0.,
		  0.,
		  0.,
		  0.);
}
