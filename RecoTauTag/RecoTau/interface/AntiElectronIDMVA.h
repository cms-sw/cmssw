//--------------------------------------------------------------------------------------------------
// $Id $
//
// AntiElectronIDMVA
//
// Helper Class for applying MVA anti-electron discrimination
//
// Authors: L.Bianchini
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


#ifndef RECOTAUTAG_RECOTAU_AntiElectronIDMVA_H
#define RECOTAUTAG_RECOTAU_AntiElectronIDMVA_H

#include "DataFormats/TauReco/interface/PFTau.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include <vector>

class AntiElectronIDMVA {
  public:

    AntiElectronIDMVA();
    ~AntiElectronIDMVA(); 

    void   Initialize(std::string methodName,
                      std::string oneProng0Pi0_BL,
                      std::string oneProng1pi0wGSF_BL,
                      std::string oneProng1pi0woGSF_BL,
		      std::string oneProng0Pi0_EC,
                      std::string oneProng1pi0wGSF_EC,
                      std::string oneProng1pi0woGSF_EC
                      );

    // RECOMMENDED:
    double MVAValue(Float_t TauEta, Float_t TauPt,
		    Float_t TauSignalPFChargedCands, Float_t TauSignalPFGammaCands, 
		    Float_t TauLeadPFChargedHadrMva, 
		    Float_t TauLeadPFChargedHadrHoP, Float_t TauLeadPFChargedHadrEoP, 
		    Float_t TauHasGsf, Float_t TauVisMass,  Float_t TauEmFraction,
		    std::vector<Float_t>* GammasdEta, std::vector<Float_t>* GammasdPhi, std::vector<Float_t>* GammasPt
		    );

    /* 
    where:

    TauEta                  = myTau->eta();
    TauPt                   = myTau->pt();
    TauSignalPFChargedCands = myTau->signalPFChargedHadrCands().size();
    TauSignalPFGammaCands   = myTau->signalPFGammaCands().size();
    TauLeadPFChargedHadrMva = myTau->electronPreIDOutput();
    TauLeadPFChargedHadrHoP = myTau->leadPFChargedHadrCand()->hcalEnergy()/myTau->leadPFChargedHadrCand()->p();
    TauLeadPFChargedHadrEoP = myTau->leadPFChargedHadrCand()->ecalEnergy()/myTau->leadPFChargedHadrCand()->p();
    TauHasGsf               = (myTau->leadPFChargedHadrCand()->gsfTrackRef()).isNonnull();
    TauVisMass              = myTau->mass();
    TauEmFraction           = myTau->emFraction();
  
    GammasdEta     = new std::vector< float >();
    GammasdPhi     = new std::vector< float >();
    GammasPt       = new std::vector< float >();
    
    for(unsigned int k = 0 ; k < (myTau->signalPFGammaCands()).size() ; k++){
    reco::PFCandidateRef gamma = (myTau->signalPFGammaCands()).at(k);
    if( (myTau->leadPFChargedHadrCand()).isNonnull() ){
        GammasdEta->push_back( gamma->eta() - myTau->leadPFChargedHadrCand()->eta() );
        GammasdPhi->push_back( gamma->phi() - myTau->leadPFChargedHadrCand()->phi() );
    }
    else{
        GammasdEta->push_back( gamma->eta() - myTau->eta() );
        GammasdPhi->push_back( gamma->phi() - myTau->phi() );
    }
     GammasPt->push_back(  gamma->pt() );
    }
    */

    double MVAValue(Float_t TauEta,  Float_t TauPt,
		    Float_t TauSignalPFChargedCands, Float_t TauSignalPFGammaCands, 
		    Float_t TauLeadPFChargedHadrMva, 
		    Float_t TauLeadPFChargedHadrHoP , Float_t TauLeadPFChargedHadrEoP, 
		    Float_t TauHasGsf, Float_t TauVisMass,  Float_t TauEmFraction,
		    Float_t GammaEtaMom, Float_t GammaPhiMom, Float_t GammaEnFrac
		    );
    /*
      see AntiElectronIDMVA.cc for GammaEtaMom,GammaPhiMom,GammaEnFrac
    */


    double MVAValue(const reco::PFTauRef& thePFTauRef);

 private:

    Bool_t isInitialized_;
    std::string methodName_;
    TMVA::Reader* fTMVAReader_[6];
    Float_t TauSignalPFGammaCands_; 
    Float_t TauVisMass_; 
    Float_t GammadEta_; 
    Float_t GammadPhi_; 
    Float_t GammadPt_;
    Float_t TauLeadPFChargedHadrMva_;
    Float_t TauLeadPFChargedHadrHoP_;
    Float_t TauLeadPFChargedHadrEoP_;
    Float_t TauEmFraction_;
    
};

#endif
