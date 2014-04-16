//--------------------------------------------------------------------------------------------------
// $Id $
//
// ElectronIDMVA
//
// Helper Class for Electron Identification MVA
//
// Authors: S.Xie
// Original based on MitPhysics/Utils/interface/ElectronIDMVA.h?view=markup
// Modified by DLE
//--------------------------------------------------------------------------------------------------

#ifndef ELECTRONIDMVA_H
#define ELECTRONIDMVA_H

#include "TString.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class TRandom3;
namespace TMVA {
    class Reader;
}

class ElectronIDMVA {
    public:
        ElectronIDMVA();
        ~ElectronIDMVA(); 

        void    Initialize(TString methodName, unsigned int version,
                TString Subdet0Pt10To20Weights , 
                TString Subdet1Pt10To20Weights , 
                TString Subdet2Pt10To20Weights,
                TString Subdet0Pt20ToInfWeights, 
                TString Subdet1Pt20ToInfWeights, 
                TString Subdet2Pt20ToInfWeights);

        Bool_t   IsInitialized() const { return fIsInitialized; }
        Double_t MVAValue(HWW&, const unsigned int ele, const unsigned int vertex);
        Double_t MVAValue(Double_t ElePt , Double_t EleSCEta,
                Double_t EleSigmaIEtaIEta,
                Double_t EleDEtaIn,
                Double_t EleDPhiIn,
                Double_t EleD0,
                Double_t EleDZ,
                Double_t EleFBrem,
                Double_t EleEOverP,
                Double_t EleESeedClusterOverPout,
                Double_t EleSigmaIPhiIPhi,
                Double_t EleNBrem,
                Double_t EleOneOverEMinusOneOverP,
                Double_t EleESeedClusterOverPIn,
                Double_t EleIP3d,
                Double_t EleIP3dSig);
	Double_t MVAValue(Double_t ElePt , Double_t EleEta,
			  Double_t PileupEnergyDensity,
			  Double_t EleSigmaIEtaIEta,
			  Double_t EleDEtaIn,
			  Double_t EleDPhiIn,
			  Double_t EleHoverE,
			  Double_t EleD0,
			  Double_t EleDZ,
			  Double_t EleFBrem,
			  Double_t EleEOverP,
			  Double_t EleESeedClusterOverPout,
			  Double_t EleSigmaIPhiIPhi,
			  Double_t EleNBrem,
			  Double_t EleOneOverEMinusOneOverP,
			  Double_t EleESeedClusterOverPIn,
			  Double_t EleIP3d,
			  Double_t EleIP3dSig,
			  Double_t EleGsfTrackChi2OverNdof,
			  Double_t EledEtaCalo,
			  Double_t EledPhiCalo,
			  Double_t EleR9,
			  Double_t EleSCEtaWidth,
			  Double_t EleSCPhiWidth,
			  Double_t EleCovIEtaIPhi,
			  Double_t ElePreShowerOverRaw,
			  Double_t EleChargedIso03,
			  Double_t EleNeutralHadronIso03,
			  Double_t EleGammaIso03,
			  Double_t EleChargedIso04,
			  Double_t EleNeutralHadronIso04,
			  Double_t EleGammaIso04,
			  Bool_t printDebug = kFALSE );

	enum EElectronEffectiveAreaType {
	  kEleChargedIso03, 
	  kEleNeutralHadronIso03, 
	  kEleGammaIso03, 
	  kEleGammaIsoVetoEtaStrip03, 
	  kEleChargedIso04, 
	  kEleNeutralHadronIso04, 
	  kEleGammaIso04, 
	  kEleGammaIsoVetoEtaStrip04, 
	  kEleNeutralHadronIso007, 
	  kEleHoverE, 
	  kEleHcalDepth1OverEcal, 
	  kEleHcalDepth2OverEcal    
	};
	Double_t     ElectronEffectiveArea(EElectronEffectiveAreaType type, Double_t Eta);


    protected:      
        TMVA::Reader            *fTMVAReader[6];
        TString                  fMethodname;
        Bool_t                    fIsInitialized;

        Float_t                   fMVAVar_EleSigmaIEtaIEta; 
        Float_t                   fMVAVar_EleDEtaIn; 
        Float_t                   fMVAVar_EleDPhiIn; 
	Float_t                   fMVAVar_EleHoverE; 
        Float_t                   fMVAVar_EleD0; 
        Float_t                   fMVAVar_EleDZ; 
        Float_t                   fMVAVar_EleFBrem; 
        Float_t                   fMVAVar_EleEOverP; 
        Float_t                   fMVAVar_EleESeedClusterOverPout; 
        Float_t                   fMVAVar_EleSigmaIPhiIPhi; 
        Float_t                   fMVAVar_EleNBrem; 
        Float_t                   fMVAVar_EleOneOverEMinusOneOverP; 
        Float_t                   fMVAVar_EleESeedClusterOverPIn; 
        Float_t                   fMVAVar_EleIP3d; 
        Float_t                   fMVAVar_EleIP3dSig; 
	Float_t                   fMVAVar_EleGsfTrackChi2OverNdof;
	Float_t                   fMVAVar_EledEtaCalo;
	Float_t                   fMVAVar_EledPhiCalo;
	Float_t                   fMVAVar_EleR9;
	Float_t                   fMVAVar_EleSCEtaWidth;
	Float_t                   fMVAVar_EleSCPhiWidth;
	Float_t                   fMVAVar_EleCovIEtaIPhi;
	Float_t                   fMVAVar_ElePreShowerOverRaw;
	Float_t                   fMVAVar_EleChargedIso03OverPt;
	Float_t                   fMVAVar_EleNeutralHadronIso03OverPt;
	Float_t                   fMVAVar_EleGammaIso03OverPt;
	Float_t                   fMVAVar_EleChargedIso04OverPt;
	Float_t                   fMVAVar_EleNeutralHadronIso04OverPt;
	Float_t                   fMVAVar_EleGammaIso04OverPt;

	unsigned int version_;
};

#endif

