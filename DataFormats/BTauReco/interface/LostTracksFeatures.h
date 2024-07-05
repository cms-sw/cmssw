#ifndef DataFormats_BTauReco_LostTracksFeatures_h
#define DataFormats_BTauReco_LostTracksFeatures_h

namespace btagbtvdeep {

  class LostTracksFeatures {
  public:
    float btagPf_trackEtaRel;
    float btagPf_trackPtRel;
    float btagPf_trackPPar;
    float btagPf_trackDeltaR;
    float btagPf_trackPParRatio;
    float btagPf_trackSip2dVal;
    float btagPf_trackSip2dSig;
    float btagPf_trackSip3dVal;
    float btagPf_trackSip3dSig;
    float btagPf_trackJetDistVal;
    float drminsv;
    float charge;
    float puppiw;
    float chi2;
    float quality;
    float lostInnerHits;
    float numberOfPixelHits;
    float numberOfStripHits;
    float pt;
    float eta;
    float phi;
    float e;
  };

}  // namespace btagbtvdeep

#endif  //DataFormats_BTauReco_LostTracksFeatures_h
