#ifndef DataFormats_DeepFormats_ShallowTagInfoFeatures_h
#define DataFormats_DeepFormats_ShallowTagInfoFeatures_h

namespace deep {

class ShallowTagInfoFeatures {

  public:

    // jet general
    float trackJetPt;              // track-based jet transverse momentum
    float jetNTracks;              // tracks associated to jet
    float jetNSecondaryVertices;   // number of secondary vertices associated to the jet
    float trackSumJetEtRatio;      // ratio of track sum transverse energy over jet energy
    float trackSumJetDeltaR;       // pseudoangular distance between jet axis and track fourvector sum
    float trackSip2dValAboveCharm; // track 2D signed impact parameter of first track lifting mass above charm
    float trackSip2dSigAboveCharm; // track 2D signed impact parameter significance of first track lifting mass above charm
    float trackSip3dValAboveCharm; // track 3D signed impact parameter of first track lifting mass above charm
    float trackSip3dSigAboveCharm; // track 3D signed impact parameter significance of first track lifting mass above charm
    float vertexCategory;          // category of secondary vertex (Reco, Pseudo, No)
    // track info
    float jetNTracksEtaRel; // tracks associated to jet for which trackEtaRel is calculated
    float jetNSelectedTracks;    

};

}

#endif //DataFormats_DeepFormats_ShallowTagInfoFeatures_h
