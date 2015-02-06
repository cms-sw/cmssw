// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#include <limits>

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MvaSoftMuonEstimator.h"
#include "CommonTools/Utils/interface/TMVAZipReader.h"

MvaSoftMuonEstimator::MvaSoftMuonEstimator(std::string weightFile) {
  TMVAReader = new TMVA::Reader("Color:Silent:Error");
  TMVAReader->SetVerbose(false);
  TMVAReader->AddVariable("TagInfo1.sip3d", &mva_sip3d);
  TMVAReader->AddVariable("TagInfo1.sip2d", &mva_sip2d);
  TMVAReader->AddVariable("TagInfo1.ptRel", &mva_ptRel);
  TMVAReader->AddVariable("TagInfo1.deltaR", &mva_deltaR);
  TMVAReader->AddVariable("TagInfo1.ratio", &mva_ratio);
  reco::details::loadTMVAWeights(TMVAReader, "BDT", weightFile.c_str()); 
}

MvaSoftMuonEstimator::~MvaSoftMuonEstimator() {
  delete TMVAReader;
}


// b-tag a jet based on track-to-jet parameters in the extened info collection
float MvaSoftMuonEstimator::mvaValue(float sip3d, float sip2d, float ptRel, float deltaR, float ratio) {
  mva_sip3d = sip3d;
  mva_sip2d = sip2d;
  mva_ptRel = ptRel;
  mva_deltaR = deltaR;
  mva_ratio = ratio;
  // Evaluate tagger
  float tag = TMVAReader->EvaluateMVA("BDT");
  // Transform output between approximately 0 and 1
  tag = (tag+1.)/2.;
  
  return tag;
}

