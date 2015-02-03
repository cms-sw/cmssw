//--------------------------------------------------------------------------------------------------
// $Id $
//
// MvaSoftEleEstimator
//
// Helper Class for applying MVA electron ID selection
//
// Authors: S. de Visscher
//--------------------------------------------------------------------------------------------------


/// --> NOTE if you want to use this class as standalone without the CMSSW part 
///  you need to uncomment the below line and compile normally with scramv1 b 
///  Then you need just to load it in your root macro the lib with the correct path, eg:
///  gSystem->Load("/data/benedet/CMSSW_5_2_2/lib/slc5_amd64_gcc462/pluginEGammaEGammaAnalysisTools.so");

//#define STANDALONE   // <---- this line

#ifndef MvaSoftEleEstimator_H
#define MvaSoftEleEstimator_H

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#include <vector>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class MvaSoftEleEstimator{
  public:
    MvaSoftEleEstimator(std::string);
    ~MvaSoftEleEstimator(); 
    
    Double_t mvaValue(Float_t, Float_t, Float_t,Float_t,Float_t,Float_t);

  private:
    TMVA::Reader* TMVAReader;
    float mva_sip3d, mva_sip2d, mva_ptRel, mva_deltaR, mva_ratio, mva_e_pi;
};

#endif
