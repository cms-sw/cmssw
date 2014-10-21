//--------------------------------------------------------------------------------------------------
// $Id $
//
// EGammaMvaEleEstimator
//
// Helper Class for applying MVA electron ID selection
//
// Authors: D.Benedetti, E.DiMaro, S.Xie
//--------------------------------------------------------------------------------------------------

/// --> NOTE if you want to use this class as standalone without the CMSSW part 
///  you need to uncomment the below line and compile normally with scramv1 b 
///  Then you need just to load it in your root macro the lib with the correct path, eg:
///  gSystem->Load("/data/benedet/CMSSW_5_2_2/lib/slc5_amd64_gcc462/pluginEGammaEGammaAnalysisTools.so");

#ifndef EGammaMvaEleEstimator_H
#define EGammaMvaEleEstimator_H

#include <vector>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "DQM/PhysicsHWW/interface/HWW.h"

class EGammaMvaEleEstimator{
  public:
    EGammaMvaEleEstimator();
    ~EGammaMvaEleEstimator(); 
  
    enum MVAType {
      kTrig = 0,      // MVA for non-triggering electrons          
      kNonTrig,       // MVA for triggering electrons
      kIsoRings
    };
  
    void     initialize( std::string methodName,
                         std::string weightsfile,
                         EGammaMvaEleEstimator::MVAType type);
    void     initialize( std::string methodName,
                         EGammaMvaEleEstimator::MVAType type,
                         Bool_t useBinnedVersion,
                         std::vector<std::string> weightsfiles );
    
    Bool_t   isInitialized() const { return fisInitialized; }
    UInt_t   GetMVABin(double eta,double pt ) const;
    
    void SetPrintMVADebug(bool b) { fPrintMVADebug = b; }
    
    void bindVariables();
    
    Double_t mvaValue(Double_t fbrem, 
                      Double_t kfchi2,
                      Int_t    kfhits,
                      Double_t gsfchi2,
                      Double_t deta,
                      Double_t dphi,
                      Double_t detacalo,
                      //Double_t dphicalo,
                      Double_t see,
                      Double_t spp,
                      Double_t etawidth,
                      Double_t phiwidth,
                      Double_t e1x5e5x5,
                      Double_t R9,
                      //Int_t    nbrems,
                      Double_t HoE,
                      Double_t EoP,
                      Double_t IoEmIoP,
                      Double_t eleEoPout,
                      Double_t PreShowerOverRaw,
                      //Double_t EoPout,
                      Double_t d0,
                      Double_t ip3d,
                      Double_t eta,
                      Double_t pt,
                      Bool_t printDebug = kFALSE );
 
    Double_t mvaValue(Double_t fbrem, 
                      Double_t kfchi2,
                      Int_t    kfhits,
                      Double_t gsfchi2,
                      Double_t deta,
                      Double_t dphi,
                      Double_t detacalo,
                      //Double_t dphicalo,
                      Double_t see,
                      Double_t spp,
                      Double_t etawidth,
                      Double_t phiwidth,
                      Double_t e1x5e5x5,
                      Double_t R9,
                      //Int_t    nbrems,
                      Double_t HoE,
                      Double_t EoP,
                      Double_t IoEmIoP,
                      Double_t eleEoPout,
                      Double_t PreShowerOverRaw,
                      //Double_t EoPout,
                      Double_t eta,
                      Double_t pt,
                      Bool_t printDebug = kFALSE );

	Double_t mvaValue(HWW&, Int_t  ele, 
                      Bool_t printDebug = kFALSE );
 

 
  private:

    std::vector<TMVA::Reader*> fTMVAReader;
    std::string                fMethodname;
    Bool_t                     fisInitialized;
    Bool_t                     fPrintMVADebug;
    MVAType                    fMVAType;
    Bool_t                     fUseBinnedVersion;
    UInt_t                     fNMVABins;

    Float_t                    fMVAVar_fbrem;
    Float_t                    fMVAVar_kfchi2;
    Float_t                    fMVAVar_kfhits;
    Float_t                    fMVAVar_kfhitsall; //added for BAMBU
    Float_t                    fMVAVar_gsfchi2;

    Float_t                   fMVAVar_deta;
    Float_t                   fMVAVar_dphi;
    Float_t                   fMVAVar_detacalo;
    //Float_t                   fMVAVar_dphicalo;

    Float_t                   fMVAVar_see;
    Float_t                   fMVAVar_spp;
    Float_t                   fMVAVar_etawidth;
    Float_t                   fMVAVar_phiwidth;
    Float_t                   fMVAVar_e1x5e5x5;
    Float_t                   fMVAVar_R9;
    //Float_t                   fMVAVar_nbrems;

    Float_t                   fMVAVar_HoE;
    Float_t                   fMVAVar_EoP;
    Float_t                   fMVAVar_IoEmIoP;
    Float_t                   fMVAVar_eleEoPout;
    Float_t                   fMVAVar_EoPout; //added for BAMBU
    Float_t                   fMVAVar_PreShowerOverRaw;

    Float_t                   fMVAVar_d0;
    Float_t                   fMVAVar_ip3d;

    Float_t                   fMVAVar_eta;
    Float_t                   fMVAVar_pt;
  
    //isolation
    Float_t                   fMVAVar_ChargedIso_DR0p0To0p1;
    Float_t                   fMVAVar_ChargedIso_DR0p1To0p2;
    Float_t                   fMVAVar_ChargedIso_DR0p2To0p3;
    Float_t                   fMVAVar_ChargedIso_DR0p3To0p4;
    Float_t                   fMVAVar_ChargedIso_DR0p4To0p5;
    Float_t                   fMVAVar_GammaIso_DR0p0To0p1;
    Float_t                   fMVAVar_GammaIso_DR0p1To0p2;
    Float_t                   fMVAVar_GammaIso_DR0p2To0p3;
    Float_t                   fMVAVar_GammaIso_DR0p3To0p4;
    Float_t                   fMVAVar_GammaIso_DR0p4To0p5;
    Float_t                   fMVAVar_NeutralHadronIso_DR0p0To0p1;
    Float_t                   fMVAVar_NeutralHadronIso_DR0p1To0p2;
    Float_t                   fMVAVar_NeutralHadronIso_DR0p2To0p3;
    Float_t                   fMVAVar_NeutralHadronIso_DR0p3To0p4;
    Float_t                   fMVAVar_NeutralHadronIso_DR0p4To0p5;


};

#endif
