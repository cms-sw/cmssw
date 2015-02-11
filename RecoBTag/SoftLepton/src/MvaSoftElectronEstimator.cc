#include <TFile.h>
#include "RecoBTag/SoftLepton/interface/MvaSoftElectronEstimator.h"
#include <cmath>
#include <vector>
using namespace std;

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
//#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "CommonTools/Utils/interface/TMVAZipReader.h"
using namespace reco;

//--------------------------------------------------------------------------------------------------
MvaSoftEleEstimator::MvaSoftEleEstimator(std::string weightFile)
{
	TMVAReader = new TMVA::Reader("Color:Silent:Error");
  	TMVAReader->SetVerbose(false);
  	TMVAReader->AddVariable("sip3d", 	&mva_sip3d);
  	TMVAReader->AddVariable("sip2d", 	&mva_sip2d);
  	TMVAReader->AddVariable("ptRel", 	&mva_ptRel);
  	TMVAReader->AddVariable("deltaR",	&mva_deltaR);
  	TMVAReader->AddVariable("ratio", 	&mva_ratio);
	TMVAReader->AddVariable("mva_e_pi", 	&mva_e_pi);
  	reco::details::loadTMVAWeights(TMVAReader, "BDT", weightFile.c_str()); 
        
}
//--------------------------------------------------------------------------------------------------
MvaSoftEleEstimator::~MvaSoftEleEstimator()
{
 delete TMVAReader;
}

//--------------------------------------------------------------------------------------------------

Double_t MvaSoftEleEstimator::mvaValue(Float_t sip2d, Float_t sip3d, Float_t ptRel, float deltaR, Float_t ratio, Float_t mva_e_pi) {
  
  mva_sip3d = sip3d;
  mva_sip2d = sip2d;
  mva_ptRel = ptRel;
  mva_deltaR = deltaR;
  mva_ratio = ratio;
  mva_e_pi = mva_e_pi;

  float tag = TMVAReader->EvaluateMVA("BDT");
  // Transform output between 0 and 1
  tag = (tag+1.0)/2.0;

  return tag;
}
//--------------------------------------------------------------------------------------------------------
