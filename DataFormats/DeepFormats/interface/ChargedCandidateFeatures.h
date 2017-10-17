#ifndef DataFormats_DeepFormats_ChargedCandidateFeatures_h
#define DataFormats_DeepFormats_ChargedCandidateFeatures_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace btagbtvdeep {

class ChargedCandidateFeatures {

  public:

    float ptrel;
    float puppiw;
    float VTX_ass;

    float btagPf_trackEtaRel;
    float btagPf_trackPtRel;
    float btagPf_trackPPar;
    float btagPf_trackDeltaR;
    float btagPf_trackPParRatio;
    float btagPf_trackSip3dVal;
    float btagPf_trackSip3dSig;
    float btagPf_trackSip2dVal;
    float btagPf_trackSip2dSig;


    float btagPf_trackJetDistVal;

    float chi2;
    float quality;

    float drminsv;

    // for ROOT schema evolution
    CMS_CLASS_VERSION(10)

};

}

#endif //DataFormats_DeepFormats_ChargedCandidateFeatures_h
