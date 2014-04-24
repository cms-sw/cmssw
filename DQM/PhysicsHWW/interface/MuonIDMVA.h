//--------------------------------------------------------------------------------------------------
// $Id $
//
// MuonIDMVA
//
// Helper Class for Muon Identification MVA
//
// Authors: S.Xie
// Original based on MitPhysics/Utils/interface/MuonIDMVA.h?view=markup
// Modified by DLE
//--------------------------------------------------------------------------------------------------

#ifndef MUONIDMVA_H
#define MUONIDMVA_H

#include "TString.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class TRandom3;
namespace TMVA {
    class Reader;
}

class MuonIDMVA {
    public:
        MuonIDMVA();
        ~MuonIDMVA(); 

        void    Initialize(TString methodName, unsigned int version,
			   TString Subdet0Pt10To14p5Weights , 
			   TString Subdet1Pt10To14p5Weights , 
			   TString Subdet0Pt14p5To20Weights,
			   TString Subdet1Pt14p5To20Weights, 
			   TString Subdet0Pt20ToInfWeights, 
			   TString Subdet1Pt20ToInfWeights);

        Bool_t   IsInitialized() const { return fIsInitialized; }
        Double_t MVAValue(HWW&, const unsigned int mu, const unsigned int vertex);
	Double_t MVAValue( Double_t MuPt , Double_t MuEta,
			   Double_t                   MuTkNchi2, 
			   Double_t                   MuGlobalNchi2, 
			   Double_t                   MuNValidHits, 
			   Double_t                   MuNTrackerHits, 
			   Double_t                   MuNPixelHits, 
			   Double_t                   MuNMatches, 
			   Double_t                   MuD0, 
			   Double_t                   MuIP3d, 
			   Double_t                   MuIP3dSig, 
			   Double_t                   MuTrkKink, 
			   Double_t                   MuSegmentCompatibility, 
			   Double_t                   MuCaloCompatibility, 
			   Double_t                   MuHadEnergyOverPt, 
			   Double_t                   MuHoEnergyOverPt, 
			   Double_t                   MuEmEnergyOverPt, 
			   Double_t                   MuHadS9EnergyOverPt, 
			   Double_t                   MuHoS9EnergyOverPt, 
			   Double_t                   MuEmS9EnergyOverPt, 
			   Double_t                   MuTrkIso03OverPt,
			   Double_t                   MuEMIso03OverPt,
			   Double_t                   MuHadIso03OverPt,
			   Double_t                   MuTrkIso05OverPt,
			   Double_t                   MuEMIso05OverPt,
			   Double_t                   MuHadIso05OverPt,
			   Bool_t                     printDebug = kFALSE
			   );


	enum EMuonEffectiveAreaType {
	  kMuChargedIso03, 
	  kMuNeutralIso03, 
	  kMuChargedIso04, 
	  kMuNeutralIso04, 
	  kMuHadEnergy, 
	  kMuHoEnergy, 
	  kMuEmEnergy, 
	  kMuHadS9Energy, 
	  kMuHoS9Energy, 
	  kMuEmS9Energy,
	  kMuTrkIso03, 
	  kMuEMIso03, 
	  kMuHadIso03, 
	  kMuTrkIso05, 
	  kMuEMIso05, 
	  kMuHadIso05 
	};

  static const Double_t MuonEffectiveArea(EMuonEffectiveAreaType type, Double_t Eta) {

    Double_t EffectiveArea = 0;
    if (fabs(Eta) < 1.0) {
      if (type == kMuChargedIso03) EffectiveArea = 0.000;
      if (type == kMuNeutralIso03) EffectiveArea = 0.080;
      if (type == kMuChargedIso04) EffectiveArea = 0.000;
      if (type == kMuNeutralIso04) EffectiveArea = 0.163;
      if (type == kMuHadEnergy)    EffectiveArea = 0.000;
      if (type == kMuHoEnergy)     EffectiveArea = 0.000;
      if (type == kMuEmEnergy)     EffectiveArea = 0.000;
      if (type == kMuHadS9Energy)  EffectiveArea = 0.016;
      if (type == kMuHoS9Energy)   EffectiveArea = 0.000;
      if (type == kMuEmS9Energy)   EffectiveArea = 0.000;
      if (type == kMuTrkIso03)     EffectiveArea = 0.000;
      if (type == kMuEMIso03)      EffectiveArea = 0.080;
      if (type == kMuHadIso03)     EffectiveArea = 0.025;
      if (type == kMuTrkIso05)     EffectiveArea = 0.000;
      if (type == kMuEMIso05)      EffectiveArea = 0.290;
      if (type == kMuHadIso05)     EffectiveArea = 0.091;
    } else if (fabs(Eta) >= 1.0 && fabs(Eta) < 1.479 ) {
      if (type == kMuChargedIso03) EffectiveArea = 0.000;
      if (type == kMuNeutralIso03) EffectiveArea = 0.083;
      if (type == kMuChargedIso04) EffectiveArea = 0.000;
      if (type == kMuNeutralIso04) EffectiveArea = 0.168;
      if (type == kMuHadEnergy)    EffectiveArea = 0.005;
      if (type == kMuHoEnergy)     EffectiveArea = 0.000;
      if (type == kMuEmEnergy)     EffectiveArea = 0.000;
      if (type == kMuHadS9Energy)  EffectiveArea = 0.041;
      if (type == kMuHoS9Energy)   EffectiveArea = 0.000;
      if (type == kMuEmS9Energy)   EffectiveArea = 0.000;
      if (type == kMuTrkIso03)     EffectiveArea = 0.000;
      if (type == kMuEMIso03)      EffectiveArea = 0.043;
      if (type == kMuHadIso03)     EffectiveArea = 0.028;
      if (type == kMuTrkIso05)     EffectiveArea = 0.000;
      if (type == kMuEMIso05)      EffectiveArea = 0.184;
      if (type == kMuHadIso05)     EffectiveArea = 0.106;
    } else if (fabs(Eta) >= 1.479 && fabs(Eta) < 2.0 ) {
      if (type == kMuChargedIso03) EffectiveArea = 0.000;
      if (type == kMuNeutralIso03) EffectiveArea = 0.060;
      if (type == kMuChargedIso04) EffectiveArea = 0.000;
      if (type == kMuNeutralIso04) EffectiveArea = 0.131;
      if (type == kMuHadEnergy)    EffectiveArea = 0.020;
      if (type == kMuHoEnergy)     EffectiveArea = 0.000;
      if (type == kMuEmEnergy)     EffectiveArea = 0.000;
      if (type == kMuHadS9Energy)  EffectiveArea = 0.072;
      if (type == kMuHoS9Energy)   EffectiveArea = 0.000;
      if (type == kMuEmS9Energy)   EffectiveArea = 0.000;
      if (type == kMuTrkIso03)     EffectiveArea = 0.000;
      if (type == kMuEMIso03)      EffectiveArea = 0.025;
      if (type == kMuHadIso03)     EffectiveArea = 0.036;
      if (type == kMuTrkIso05)     EffectiveArea = 0.000;
      if (type == kMuEMIso05)      EffectiveArea = 0.124;
      if (type == kMuHadIso05)     EffectiveArea = 0.140;
    } else if (fabs(Eta) >= 2.0 && fabs(Eta) < 2.25 ) {
      if (type == kMuChargedIso03) EffectiveArea = 0.000;
      if (type == kMuNeutralIso03) EffectiveArea = 0.066;
      if (type == kMuChargedIso04) EffectiveArea = 0.000;
      if (type == kMuNeutralIso04) EffectiveArea = 0.149;
      if (type == kMuHadEnergy)    EffectiveArea = 0.056;
      if (type == kMuHoEnergy)     EffectiveArea = 0.000;
      if (type == kMuEmEnergy)     EffectiveArea = 0.000;
      if (type == kMuHadS9Energy)  EffectiveArea = 0.148;
      if (type == kMuHoS9Energy)   EffectiveArea = 0.000;
      if (type == kMuEmS9Energy)   EffectiveArea = 0.000;
      if (type == kMuTrkIso03)     EffectiveArea = 0.000;
      if (type == kMuEMIso03)      EffectiveArea = 0.025;
      if (type == kMuHadIso03)     EffectiveArea = 0.050;
      if (type == kMuTrkIso05)     EffectiveArea = 0.000;
      if (type == kMuEMIso05)      EffectiveArea = 0.120;
      if (type == kMuHadIso05)     EffectiveArea = 0.186;
    } else if (fabs(Eta) >= 2.25 && fabs(Eta) < 2.4 ) {
      if (type == kMuChargedIso03) EffectiveArea = 0.000;
      if (type == kMuNeutralIso03) EffectiveArea = 0.098;
      if (type == kMuChargedIso04) EffectiveArea = 0.000;
      if (type == kMuNeutralIso04) EffectiveArea = 0.200;
      if (type == kMuHadEnergy)    EffectiveArea = 0.093;
      if (type == kMuHoEnergy)     EffectiveArea = 0.000;
      if (type == kMuEmEnergy)     EffectiveArea = 0.000;
      if (type == kMuHadS9Energy)  EffectiveArea = 0.260;
      if (type == kMuHoS9Energy)   EffectiveArea = 0.000;
      if (type == kMuEmS9Energy)   EffectiveArea = 0.000;
      if (type == kMuTrkIso03)     EffectiveArea = 0.000;
      if (type == kMuEMIso03)      EffectiveArea = 0.027;
      if (type == kMuHadIso03)     EffectiveArea = 0.060;
      if (type == kMuTrkIso05)     EffectiveArea = 0.000;
      if (type == kMuEMIso05)      EffectiveArea = 0.139;
      if (type == kMuHadIso05)     EffectiveArea = 0.228;
    }
    return EffectiveArea;
  }


    protected:      
        TMVA::Reader            *fTMVAReader[6];
        TString                  fMethodname;
        Bool_t                    fIsInitialized;

	Float_t                   fMVAVar_MuTkNchi2; 
	Float_t                   fMVAVar_MuGlobalNchi2; 
	Float_t                   fMVAVar_MuNValidHits; 
	Float_t                   fMVAVar_MuNTrackerHits; 
	Float_t                   fMVAVar_MuNPixelHits; 
	Float_t                   fMVAVar_MuNMatches; 
	Float_t                   fMVAVar_MuD0; 
	Float_t                   fMVAVar_MuIP3d; 
	Float_t                   fMVAVar_MuIP3dSig; 
	Float_t                   fMVAVar_MuTrkKink; 
	Float_t                   fMVAVar_MuSegmentCompatibility; 
	Float_t                   fMVAVar_MuCaloCompatibility; 
	Float_t                   fMVAVar_MuHadEnergyOverPt; 
	Float_t                   fMVAVar_MuHoEnergyOverPt; 
	Float_t                   fMVAVar_MuEmEnergyOverPt; 
	Float_t                   fMVAVar_MuHadS9EnergyOverPt; 
	Float_t                   fMVAVar_MuHoS9EnergyOverPt; 
	Float_t                   fMVAVar_MuEmS9EnergyOverPt; 
	Float_t                   fMVAVar_MuChargedIso03OverPt;
	Float_t                   fMVAVar_MuNeutralIso03OverPt;
	Float_t                   fMVAVar_MuChargedIso04OverPt;
	Float_t                   fMVAVar_MuNeutralIso04OverPt;
	Float_t                   fMVAVar_MuTrkIso03OverPt;
	Float_t                   fMVAVar_MuEMIso03OverPt;
	Float_t                   fMVAVar_MuHadIso03OverPt;
	Float_t                   fMVAVar_MuTrkIso05OverPt;
	Float_t                   fMVAVar_MuEMIso05OverPt;
	Float_t                   fMVAVar_MuHadIso05OverPt;
	
};

#endif

