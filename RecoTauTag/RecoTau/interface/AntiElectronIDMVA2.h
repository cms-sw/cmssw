//--------------------------------------------------------------------------------------------------
// $Id $
//
// AntiElectronIDMVA2
//
// Helper Class for applying MVA anti-electron discrimination
//
// Authors: I.Naranjo
//--------------------------------------------------------------------------------------------------

/*
  proposed WP:    epsilonB ~ 18%  epsilonS ~ 91% wrt signal taus passing discr. ag. electrons Medium. 
  bool pass = 
  (abs(TauEta)<1.5 && TauSignalPFGammaCands==0 && MVAValue(...)>0.054) ||
  (abs(TauEta)<1.5 && TauSignalPFGammaCands>0  && TauHasGsf>0.5 && MVAValue(...)>0.060) ||
  (abs(TauEta)<1.5 && TauSignalPFGammaCands>0  && TauHasGsf<0.5 && MVAValue(...)>0.054) ||
  (abs(TauEta)>1.5 && TauSignalPFGammaCands==0 && MVAValue(...)>0.060) ||
  (abs(TauEta)>1.5 && TauSignalPFGammaCands>0  && TauHasGsf>0.5 && MVAValue(...)>0.053) ||
  (abs(TauEta)>1.5 && TauSignalPFGammaCands>0  && TauHasGsf<0.5 && MVAValue(...)>0.049);
*/


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

using namespace std;

class AntiElectronIDMVA2 {
  public:

    AntiElectronIDMVA2();
    ~AntiElectronIDMVA2(); 

    void   Initialize(std::string methodName,
		      std::string oneProng0Pi0_BL,
		      std::string oneProng1pi0woGSF_BL,
		      std::string oneProng1pi0wGSFwoPfEleMva_BL,
		      std::string oneProng1pi0wGSFwPfEleMva_BL,
		      std::string oneProng0Pi0_EC,
		      std::string oneProng1pi0woGSF_EC,
		      std::string oneProng1pi0wGSFwoPfEleMva_EC,
		      std::string oneProng1pi0wGSFwPfEleMva_EC
		      );

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
		    );

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
		    );

    double MVAValue(const reco::PFTau& thePFTau, 
		    const reco::GsfElectron& theGsfEle
		    );

 private:

    Bool_t isInitialized_;
    std::string methodName_;
    TMVA::Reader* fTMVAReader_[8];

    Float_t GammadEta_;
    Float_t GammadPhi_;
    Float_t GammadPt_;

    float Tau_AbsEta_;
    float Tau_Pt_;
    float Tau_HasGsf_; 
    float Tau_EmFraction_; 
    float Tau_NumChargedCands_;
    float Tau_NumGammaCands_; 
    float Tau_HadrHoP_; 
    float Tau_HadrEoP_; 
    float Tau_VisMass_; 
    float Tau_GammaEtaMom_;
    float Tau_GammaPhiMom_;
    float Tau_GammaEnFrac_;
    float Tau_HadrMva_; 

    float Elec_AbsEta_;
    float Elec_Pt_;
    float Elec_PFMvaOutput_;
    float Elec_Ee_;
    float Elec_Egamma_;
    float Elec_Pin_;
    float Elec_Pout_;
    float Elec_EtotOverPin_;
    float Elec_EeOverPout_;
    float Elec_EgammaOverPdif_;
    float Elec_EarlyBrem_;//
    float Elec_LateBrem_;//
    float Elec_Logsihih_;
    float Elec_DeltaEta_;
    float Elec_HoHplusE_;
    float Elec_Fbrem_;
    float Elec_Chi2KF_;
    float Elec_Chi2GSF_;
    float Elec_NumHits_;
    float Elec_GSFTrackResol_;
    float Elec_GSFTracklnPt_;
    float Elec_GSFTrackEta_;
};

#endif
