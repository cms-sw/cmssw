#ifndef DataFormats_BTauReco_NeutralCandidateFeatures_h
#define DataFormats_BTauReco_NeutralCandidateFeatures_h

namespace btagbtvdeep {

  class NeutralCandidateFeatures {
  public:
    float ptrel;
    float ptrel_noclip;
    float erel;

    float drsubjet1;
    float drsubjet2;

    float puppiw;
    float deltaR;
    float deltaR_noclip;
    float isGamma;

    float hadFrac;
    float drminsv;

    float etarel;
    float phirel;

    float pt;
    float px;
    float py;
    float pz;
    float eta;
    float phi;
    float e;
  };

}  // namespace btagbtvdeep

#endif  //DataFormats_BTauReco_NeutralCandidateFeatures_h
