#ifndef DataFormats_BTauReco_ChargedCandidateFeatures_h
#define DataFormats_BTauReco_ChargedCandidateFeatures_h

namespace btagbtvdeep {

  class ChargedCandidateFeatures {
  public:
    float ptrel;
    float ptrel_noclip;
    float erel;
    float etarel;
    float puppiw;
    float vtx_ass;

    float btagPf_trackEtaRel;
    float btagPf_trackPtRel;
    float btagPf_trackPtRatio;
    float btagPf_trackPPar;
    float btagPf_trackDeltaR;
    float btagPf_trackPParRatio;
    float btagPf_trackSip3dVal;
    float btagPf_trackSip3dSig;
    float btagPf_trackSip2dVal;
    float btagPf_trackSip2dSig;

    float btagPf_trackJetDistVal;

    float drsubjet1;
    float drsubjet2;
    float dxy;
    float dxysig;
    float dz;
    float dzsig;
    float deltaR;

    float chi2;
    float quality;

    float drminsv;
    float distminsv;

    float pt;
    float px;
    float py;
    float pz;
    float eta;
    float phi;
    float e;
  };

}  // namespace btagbtvdeep

#endif  //DataFormats_BTauReco_ChargedCandidateFeatures_h
