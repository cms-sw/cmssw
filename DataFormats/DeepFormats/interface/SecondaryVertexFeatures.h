#ifndef DataFormats_DeepFormats_SecondaryVertexFeatures_h
#define DataFormats_DeepFormats_SecondaryVertexFeatures_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace btagbtvdeep {

class SecondaryVertexFeatures {

  public:

    float pt;
    float eta;
    float phi;
    float mass;

    float etarel;
    float phirel;
    float deltaR;

    float ntracks;
    float chi2;
    float ndf;
    float normchi2;
    float dxy;
    float dxyerr;
    float dxysig;
    float d3d;
    float d3derr;
    float d3dsig;
    
    float costhetasvpv;
    float enratio;

    // for ROOT schema evolution
    CMS_CLASS_VERSION(10)

};

}

#endif //DataFormats_DeepFormats_SecondaryVertexFeatures_h
