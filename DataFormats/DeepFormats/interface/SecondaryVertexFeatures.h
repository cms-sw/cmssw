#ifndef DataFormats_DeepFormats_SecondaryVertexFeatures_h
#define DataFormats_DeepFormats_SecondaryVertexFeatures_h

namespace deep {

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

};

}

#endif //DataFormats_DeepFormats_SecondaryVertexFeatures_h
