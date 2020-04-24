//--------------------------------------------------------------------------------------------------
// AntiElectronIDMVA6
//
// Helper Class for applying MVA anti-electron discrimination
//
// Authors: F.Colombo, C.Veelken
//--------------------------------------------------------------------------------------------------

#ifndef RECOTAUTAG_RECOTAU_AntiElectronIDMVA6_H
#define RECOTAUTAG_RECOTAU_AntiElectronIDMVA6_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

class AntiElectronIDMVA6 
{
  public:

   AntiElectronIDMVA6(const edm::ParameterSet&);
   ~AntiElectronIDMVA6(); 

   void beginEvent(const edm::Event&, const edm::EventSetup&);
   
   double MVAValue(Float_t TauPt,
                   Float_t TauEtaAtEcalEntrance,
                   Float_t TauPhi,
                   Float_t TauLeadChargedPFCandPt,
                   Float_t TauLeadChargedPFCandEtaAtEcalEntrance,
                   Float_t TauEmFraction,
                   Float_t TauLeadPFChargedHadrHoP,
                   Float_t TauLeadPFChargedHadrEoP,
                   Float_t TauVisMassIn,
                   Float_t TaudCrackEta,
                   Float_t TaudCrackPhi,
                   Float_t TauHasGsf,
                   Int_t TauSignalPFGammaCandsIn,
                   Int_t TauSignalPFGammaCandsOut,
                   const std::vector<Float_t>& GammasdEtaInSigCone,
                   const std::vector<Float_t>& GammasdPhiInSigCone,
                   const std::vector<Float_t>& GammasPtInSigCone,
                   const std::vector<Float_t>& GammasdEtaOutSigCone,
                   const std::vector<Float_t>& GammasdPhiOutSigCone,
                   const std::vector<Float_t>& GammasPtOutSigCone,
                   Float_t ElecEta,
                   Float_t ElecPhi,
                   Float_t ElecEtotOverPin,
                   Float_t ElecChi2NormGSF,
                   Float_t ElecChi2NormKF,
                   Float_t ElecGSFNumHits,
                   Float_t ElecKFNumHits,
                   Float_t ElecGSFTrackResol,
                   Float_t ElecGSFTracklnPt,
                   Float_t ElecPin,
                   Float_t ElecPout,
                   Float_t ElecEecal,
                   Float_t ElecDeltaEta,
                   Float_t ElecDeltaPhi,
                   Float_t ElecMvaInSigmaEtaEta,
                   Float_t ElecMvaInHadEnergy,
                   Float_t ElecMvaInDeltaEta
                  );

   double MVAValue(Float_t TauPt,
                   Float_t TauEtaAtEcalEntrance,
                   Float_t TauPhi,
                   Float_t TauLeadChargedPFCandPt,
                   Float_t TauLeadChargedPFCandEtaAtEcalEntrance,
                   Float_t TauEmFraction,
                   Float_t TauLeadPFChargedHadrHoP,
                   Float_t TauLeadPFChargedHadrEoP,
                   Float_t TauVisMassIn,
                   Float_t TaudCrackEta,
                   Float_t TaudCrackPhi,
                   Float_t TauHasGsf,
                   Int_t TauSignalPFGammaCandsIn,
                   Int_t TauSignalPFGammaCandsOut,
                   Float_t TauGammaEtaMomIn,
                   Float_t TauGammaEtaMomOut,
                   Float_t TauGammaPhiMomIn,
                   Float_t TauGammaPhiMomOut,
                   Float_t TauGammaEnFracIn,
                   Float_t TauGammaEnFracOut,
                   Float_t ElecEta,
                   Float_t ElecPhi,
                   Float_t ElecEtotOverPin,
                   Float_t ElecChi2NormGSF,
                   Float_t ElecChi2NormKF,
                   Float_t ElecGSFNumHits,
                   Float_t ElecKFNumHits,
                   Float_t ElecGSFTrackResol,
                   Float_t ElecGSFTracklnPt,
                   Float_t ElecPin,
                   Float_t ElecPout,
                   Float_t ElecEecal,
                   Float_t ElecDeltaEta,
                   Float_t ElecDeltaPhi,
                   Float_t ElecMvaInSigmaEtaEta,
                   Float_t ElecMvaInHadEnergy,
                   Float_t ElecMvaInDeltaEta
                  );

   // this function can be called for all categories
   double MVAValue(const reco::PFTau& thePFTau, 
		   const reco::GsfElectron& theGsfEle);
   // this function can be called for category 1 only !!
   double MVAValue(const reco::PFTau& thePFTau);

   // this function can be called for all categories
   double MVAValue(const pat::Tau& theTau, 
		   const pat::Electron& theEle);
   // this function can be called for category 1 only !!
   double MVAValue(const pat::Tau& theTau);
   // track extrapolation to ECAL entrance (used to re-calculate varibales that might not be available on miniAOD)
   bool atECalEntrance(const reco::Candidate* part, math::XYZPoint &pos);    

 private:   

   double dCrackEta(double eta);
   double minimum(double a,double b);
   double dCrackPhi(double phi, double eta);

   bool isInitialized_;
   bool loadMVAfromDB_;
   edm::FileInPath inputFileName_;
   
   std::string mvaName_NoEleMatch_woGwoGSF_BL_;
   std::string mvaName_NoEleMatch_wGwoGSF_BL_;
   std::string mvaName_woGwGSF_BL_;
   std::string mvaName_wGwGSF_BL_;
   std::string mvaName_NoEleMatch_woGwoGSF_EC_;
   std::string mvaName_NoEleMatch_wGwoGSF_EC_;
   std::string mvaName_woGwGSF_EC_;
   std::string mvaName_wGwGSF_EC_;

   bool usePhiAtEcalEntranceExtrapolation_;

   Float_t* Var_NoEleMatch_woGwoGSF_Barrel_;
   Float_t* Var_NoEleMatch_wGwoGSF_Barrel_;
   Float_t* Var_woGwGSF_Barrel_;
   Float_t* Var_wGwGSF_Barrel_;
   Float_t* Var_NoEleMatch_woGwoGSF_Endcap_;
   Float_t* Var_NoEleMatch_wGwoGSF_Endcap_;
   Float_t* Var_woGwGSF_Endcap_;
   Float_t* Var_wGwGSF_Endcap_;
   
   const GBRForest* mva_NoEleMatch_woGwoGSF_BL_;
   const GBRForest* mva_NoEleMatch_wGwoGSF_BL_;
   const GBRForest* mva_woGwGSF_BL_;
   const GBRForest* mva_wGwGSF_BL_;
   const GBRForest* mva_NoEleMatch_woGwoGSF_EC_;
   const GBRForest* mva_NoEleMatch_wGwoGSF_EC_;
   const GBRForest* mva_woGwGSF_EC_;
   const GBRForest* mva_wGwGSF_EC_;

   std::vector<TFile*> inputFilesToDelete_;

   double bField_;
   int verbosity_;
};

#endif
