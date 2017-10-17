#ifndef DataFormats_DeepFormats_JetFeatures_h
#define DataFormats_DeepFormats_JetFeatures_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace btagbtvdeep {

class JetFeatures {

  public:

    float pt;
    float eta;
    float mass;
    float energy;

    // for ROOT schema evolution
    CMS_CLASS_VERSION(10)

};

}

#endif //DataFormats_DeepFormats_JetFeatures_h
