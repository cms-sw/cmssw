//--------------------------------------------------------------------------------------------------
// $Id $
//
// AntiElectronIDMVA3
//
// Helper Class for applying MVA anti-electron discrimination
//
// Authors: I.Naranjo
//--------------------------------------------------------------------------------------------------

#ifndef RECOTAUTAG_RECOTAU_AntiElectronIDMVA3_H
#define RECOTAUTAG_RECOTAU_AntiElectronIDMVA3_H

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

class AntiElectronIDMVA3 
{
  public:

    AntiElectronIDMVA3();
    ~AntiElectronIDMVA3(); 

    void Initialize_from_file(const std::string& methodName,
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
			      const std::string& oneProng1Pi0wGSF_EC);

    void Initialize_from_string(const std::string& methodName,
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
				const std::string& oneProng1Pi0wGSF_EC);

    double MVAValue(Float_t TauEtaAtEcalEntrance,
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
		    Float_t ElecGSFTrackEta);

    double MVAValue(Float_t TauEtaAtEcalEntrance,
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
		    Float_t ElecGSFTrackEta);

    // CV: this function can be called for all categories
    double MVAValue(const reco::PFTau& thePFTau, 
		    const reco::GsfElectron& theGsfEle);
    // CV: this function can be called for category 1 only !!
    double MVAValue(const reco::PFTau& thePFTau);

 private:

    void bookMVAs();
    double dCrackEta(double eta);
    double minimum(double a,double b);
    double dCrackPhi(double phi, double eta);
    Bool_t isInitialized_;
    std::string methodName_;
    TMVA::Reader* fTMVAReader_[16];

    Float_t GammadEta_;
    Float_t GammadPhi_;
    Float_t GammadPt_;

    Float_t Tau_EtaAtEcalEntrance_;
    Float_t Tau_Pt_;
    Float_t Tau_dCrackEta_;
    Float_t Tau_dCrackPhi_;
    Float_t Tau_EmFraction_; 
    Float_t Tau_NumGammaCands_; 
    Float_t Tau_HadrHoP_; 
    Float_t Tau_HadrEoP_; 
    Float_t Tau_VisMass_; 
    Float_t Tau_HadrMva_; 
    Float_t Tau_GammaEtaMom_;
    Float_t Tau_GammaPhiMom_;
    Float_t Tau_GammaEnFrac_;
    Float_t Tau_GSFChi2_; 
    Float_t Tau_NumHitsVariable_; 
    Float_t Tau_GSFTrackResol_;
    Float_t Tau_GSFTracklnPt_;
    Float_t Tau_GSFTrackEta_;

    Float_t Elec_EtotOverPin_;
    Float_t Elec_EgammaOverPdif_;
    Float_t Elec_Fbrem_;
    Float_t Elec_Chi2GSF_;
    Float_t Elec_GSFNumHits_;
    Float_t Elec_GSFTrackResol_;
    Float_t Elec_GSFTracklnPt_;
    Float_t Elec_GSFTrackEta_;

    int verbosity_;
};

#endif
