#ifndef DataFormats_DeepFormats_NeutralCandidateFeatures_h
#define DataFormats_DeepFormats_NeutralCandidateFeatures_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace btagbtvdeep {

class NeutralCandidateFeatures {

  public:

    float ptrel;

    float puppiw;
    float deltaR;
    float isGamma;

    float HadFrac;
    float drminsv;

    // for ROOT schema evolution
    CMS_CLASS_VERSION(10)

};

}

#endif //DataFormats_DeepFormats_NeutralCandidateFeatures_h
