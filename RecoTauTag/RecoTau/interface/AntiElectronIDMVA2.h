//--------------------------------------------------------------------------------------------------
// $Id $
//
// AntiElectronIDMVA2
//
// Helper Class for applying MVA anti-electron discrimination
//
// Authors: I.Naranjo
//--------------------------------------------------------------------------------------------------

#ifndef RECOTAUTAG_RECOTAU_AntiElectronIDMVA2_H
#define RECOTAUTAG_RECOTAU_AntiElectronIDMVA2_H

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

class AntiElectronIDMVA2 
{
  public:

    AntiElectronIDMVA2();
    ~AntiElectronIDMVA2(); 

    void Initialize(std::string methodName,
		    std::string oneProngNoEleMatch_BL,
		    std::string oneProng0Pi0_BL,
		    std::string oneProng1pi0woGSF_BL,
		    std::string oneProng1pi0wGSFwoPfEleMva_BL,
		    std::string oneProng1pi0wGSFwPfEleMva_BL,
		    std::string oneProngNoEleMatch_EC,
		    std::string oneProng0Pi0_EC,
		    std::string oneProng1pi0woGSF_EC,
		    std::string oneProng1pi0wGSFwoPfEleMva_EC,
		    std::string oneProng1pi0wGSFwPfEleMva_EC);

    // RECOMMENDED:
    double MVAValue(Float_t TauEta,
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
		    Float_t ElecGSFTrackEta);

    double MVAValue(Float_t TauEta,
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
		    Float_t ElecGSFTrackEta);

    // CV: this function can be called for all categories
    double MVAValue(const reco::PFTau& thePFTau, 
		    const reco::GsfElectron& theGsfEle);
    // CV: this function can be called for category 1 only !!
    double MVAValue(const reco::PFTau& thePFTau);

 private:

    Bool_t isInitialized_;
    std::string methodName_;
    TMVA::Reader* fTMVAReader_[10];

    Float_t GammadEta_;
    Float_t GammadPhi_;
    Float_t GammadPt_;

    Float_t Tau_Eta_;
    Float_t Tau_Pt_;
    Float_t Tau_HasGsf_; 
    Float_t Tau_EmFraction_; 
    Float_t Tau_NumChargedCands_;
    Float_t Tau_NumGammaCands_; 
    Float_t Tau_HadrHoP_; 
    Float_t Tau_HadrEoP_; 
    Float_t Tau_VisMass_; 
    Float_t Tau_GammaEtaMom_;
    Float_t Tau_GammaPhiMom_;
    Float_t Tau_GammaEnFrac_;
    Float_t Tau_HadrMva_; 

    Float_t Elec_AbsEta_;
    Float_t Elec_Pt_;
    Float_t Elec_PFMvaOutput_;
    Float_t Elec_Ee_;
    Float_t Elec_Egamma_;
    Float_t Elec_Pin_;
    Float_t Elec_Pout_;
    Float_t Elec_EtotOverPin_;
    Float_t Elec_EeOverPout_;
    Float_t Elec_EgammaOverPdif_;
    Float_t Elec_EarlyBrem_;
    Float_t Elec_LateBrem_;
    Float_t Elec_Fbrem_;
    Float_t Elec_Chi2KF_;
    Float_t Elec_Chi2GSF_;
    Float_t Elec_NumHits_;
    Float_t Elec_GSFTrackResol_;
    Float_t Elec_GSFTracklnPt_;
    Float_t Elec_GSFTrackEta_;
};

#endif
