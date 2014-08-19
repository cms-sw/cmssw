//--------------------------------------------------------------------------------------------------
// AntiElectronIDMVA5
//
// Helper Class for applying MVA anti-electron discrimination
//
// Authors: I.Naranjo, C.Veelken
//--------------------------------------------------------------------------------------------------

#ifndef RECOTAUTAG_RECOTAU_AntiElectronIDMVA5_H
#define RECOTAUTAG_RECOTAU_AntiElectronIDMVA5_H

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

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

class AntiElectronIDMVA5 
{
  public:

   AntiElectronIDMVA5(const edm::ParameterSet&);
   ~AntiElectronIDMVA5(); 

   void beginEvent(const edm::Event&, const edm::EventSetup&);

   double MVAValue(Float_t TauEtaAtEcalEntrance,		    
		   Float_t TauPt,
		   Float_t TauLeadChargedPFCandEtaAtEcalEntrance,
		   Float_t TauLeadChargedPFCandPt,
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
		   Float_t TauLeadChargedPFCandEtaAtEcalEntrance,
		   Float_t TauLeadChargedPFCandPt,
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

   double dCrackEta(double eta);
   double minimum(double a,double b);
   double dCrackPhi(double phi, double eta);

   bool isInitialized_;
   bool loadMVAfromDB_;
   edm::FileInPath inputFileName_;
   std::string mvaName_NoEleMatch_woGwoGSF_BL_;
   std::string mvaName_NoEleMatch_woGwGSF_BL_;
   std::string mvaName_NoEleMatch_wGwoGSF_BL_;
   std::string mvaName_NoEleMatch_wGwGSF_BL_;
   std::string mvaName_woGwoGSF_BL_;
   std::string mvaName_woGwGSF_BL_;
   std::string mvaName_wGwoGSF_BL_;
   std::string mvaName_wGwGSF_BL_;
   std::string mvaName_NoEleMatch_woGwoGSF_EC_;
   std::string mvaName_NoEleMatch_woGwGSF_EC_;
   std::string mvaName_NoEleMatch_wGwoGSF_EC_;
   std::string mvaName_NoEleMatch_wGwGSF_EC_;
   std::string mvaName_woGwoGSF_EC_;
   std::string mvaName_woGwGSF_EC_;
   std::string mvaName_wGwoGSF_EC_;
   std::string mvaName_wGwGSF_EC_;

   Float_t* Var_NoEleMatch_woGwoGSF_Barrel_;
   Float_t* Var_NoEleMatch_woGwGSF_Barrel_;
   Float_t* Var_NoEleMatch_wGwoGSF_Barrel_;
   Float_t* Var_NoEleMatch_wGwGSF_Barrel_;
   Float_t* Var_woGwoGSF_Barrel_;
   Float_t* Var_woGwGSF_Barrel_;
   Float_t* Var_wGwoGSF_Barrel_;
   Float_t* Var_wGwGSF_Barrel_;
   Float_t* Var_NoEleMatch_woGwoGSF_Endcap_;
   Float_t* Var_NoEleMatch_woGwGSF_Endcap_;
   Float_t* Var_NoEleMatch_wGwoGSF_Endcap_;
   Float_t* Var_NoEleMatch_wGwGSF_Endcap_;
   Float_t* Var_woGwoGSF_Endcap_;
   Float_t* Var_woGwGSF_Endcap_;
   Float_t* Var_wGwoGSF_Endcap_;
   Float_t* Var_wGwGSF_Endcap_;
   
   const GBRForest* mva_NoEleMatch_woGwoGSF_BL_;
   const GBRForest* mva_NoEleMatch_woGwGSF_BL_ ;
   const GBRForest* mva_NoEleMatch_wGwoGSF_BL_ ;
   const GBRForest* mva_NoEleMatch_wGwGSF_BL_ ;
   const GBRForest* mva_woGwoGSF_BL_;
   const GBRForest* mva_woGwGSF_BL_ ;
   const GBRForest* mva_wGwoGSF_BL_ ;
   const GBRForest* mva_wGwGSF_BL_ ;
   const GBRForest* mva_NoEleMatch_woGwoGSF_EC_ ;
   const GBRForest* mva_NoEleMatch_woGwGSF_EC_ ;
   const GBRForest* mva_NoEleMatch_wGwoGSF_EC_ ;
   const GBRForest* mva_NoEleMatch_wGwGSF_EC_ ;
   const GBRForest* mva_woGwoGSF_EC_ ;
   const GBRForest* mva_woGwGSF_EC_ ;
   const GBRForest* mva_wGwoGSF_EC_ ;
   const GBRForest* mva_wGwGSF_EC_ ;

   std::vector<TFile*> inputFilesToDelete_;

   int verbosity_;
};

#endif
