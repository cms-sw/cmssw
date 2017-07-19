#ifndef DataFormats_DeepFormats_NeutralCandidateFeatures_h
#define DataFormats_DeepFormats_NeutralCandidateFeatures_h

namespace deep {

class NeutralCandidateFeatures {

  public:

    float pt;
    float eta;
    float phi;

    float ptrel;
    float erel;

    float puppiw;
    float phirel;
    float etarel;
    float deltaR;
    float isGamma;

    float HadFrac;
    float drminsv;

};

}

#endif //DataFormats_DeepFormats_NeutralCandidateFeatures_h
