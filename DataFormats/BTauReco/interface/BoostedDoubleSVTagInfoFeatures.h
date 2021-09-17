#ifndef DataFormats_BTauReco_BoostedDoubleSVTagInfoFeatures_h
#define DataFormats_BTauReco_BoostedDoubleSVTagInfoFeatures_h

namespace btagbtvdeep {

  class BoostedDoubleSVTagInfoFeatures {
    // Note: these variables are intended to match the variables defined in DataFormats/BTauReco/interface/TaggingVariable.h

  public:
    float jetNTracks;             // tracks associated to jet
    float jetNSecondaryVertices;  // number of reconstructed possible secondary vertices in jet
    float trackSip3dSig_0;        // 1st largest track 3D signed impact parameter significance
    float trackSip3dSig_1;        // 2nd largest track 3D signed impact parameter significance
    float trackSip3dSig_2;        // 3rd largest track 3D signed impact parameter significance
    float trackSip3dSig_3;        // 4th largest track 3D signed impact parameter significance
    float tau1_trackSip3dSig_0;  // 1st largest track 3D signed impact parameter significance associated to the 1st N-subjettiness axis
    float tau1_trackSip3dSig_1;  // 2nd largest track 3D signed impact parameter significance associated to the 1st N-subjettiness axis
    float tau2_trackSip3dSig_0;  // 1st largest track 3D signed impact parameter significance associated to the 2nd N-subjettiness axis
    float tau2_trackSip3dSig_1;  // 2nd largest track 3D signed impact parameter significance associated to the 2nd N-subjettiness axis
    float trackSip2dSigAboveBottom_0;  // track 2D signed impact parameter significance of 1st track lifting mass above bottom
    float trackSip2dSigAboveBottom_1;  // track 2D signed impact parameter significance of 2nd track lifting mass above bottom
    float trackSip2dSigAboveCharm;  // track 2D signed impact parameter significance of first track lifting mass above charm
    float tau1_trackEtaRel_0;  // 1st smallest track pseudorapidity, relative to the jet axis, associated to the 1st N-subjettiness axis
    float tau1_trackEtaRel_1;  // 2nd smallest track pseudorapidity, relative to the jet axis, associated to the 1st N-subjettiness axis
    float tau1_trackEtaRel_2;  // 3rd smallest track pseudorapidity, relative to the jet axis, associated to the 1st N-subjettiness axis
    float tau2_trackEtaRel_0;  // 1st smallest track pseudorapidity, relative to the jet axis, associated to the 2nd N-subjettiness axis
    float tau2_trackEtaRel_1;  // 2nd smallest track pseudorapidity, relative to the jet axis, associated to the 2nd N-subjettiness axis
    float tau2_trackEtaRel_2;  // 3rd smallest track pseudorapidity, relative to the jet axis, associated to the 2nd N-subjettiness axis
    float tau1_vertexMass;  // mass of track sum at secondary vertex associated to the 1st N-subjettiness axis
    float tau1_vertexEnergyRatio;  // ratio of energy at secondary vertex over total energy associated to the 1st N-subjettiness axis
    float tau1_flightDistance2dSig;  // transverse distance significance between primary and secondary vertex associated to the 1st N-subjettiness axis
    float tau1_vertexDeltaR;  // pseudoangular distance between the 1st N-subjettiness axis and secondary vertex direction
    float tau2_vertexMass;    // mass of track sum at secondary vertex associated to the 2nd N-subjettiness axis
    float tau2_vertexEnergyRatio;  // ratio of energy at secondary vertex over total energy associated to the 2nd N-subjettiness axis
    float tau2_flightDistance2dSig;  // transverse distance significance between primary and secondary vertex associated to the 2nd N-subjettiness axis
    float tau2_vertexDeltaR;  // pseudoangular distance between the 2nd N-subjettiness axis and secondary vertex direction (NOT USED!)
    float z_ratio;  // z ratio
  };

}  // namespace btagbtvdeep

#endif  //DataFormats_BTauReco_BoostedDoubleSVTagInfoFeatures_h
