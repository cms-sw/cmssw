#ifndef DataFormats_DeepFormats_SecondaryVertexFeatures_h
#define DataFormats_DeepFormats_SecondaryVertexFeatures_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace btagbtvdeep {

class SecondaryVertexFeatures {

  public:

    float pt;
    float mass;

    float deltaR;

    float ntracks;
    float chi2;
    float normchi2;
    float dxy;
    float dxysig;
    float d3d;
    float d3dsig;
    
    float costhetasvpv;
    float enratio;

    // for ROOT schema evolution
    CMS_CLASS_VERSION(10)

};

}

#endif //DataFormats_DeepFormats_SecondaryVertexFeatures_h
