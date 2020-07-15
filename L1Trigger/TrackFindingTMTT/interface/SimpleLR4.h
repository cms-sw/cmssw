#ifndef L1Trigger_TrackFindingTMTT_SimpleLR4_h
#define L1Trigger_TrackFindingTMTT_SimpleLR4_h

///=== This is the simple linear regression with 4 helix parameters (qOverPt, phiT, z0, tanLambda) track fit algorithm.

///=== Written by: Davide Cieri (davide.cieri@stfc.ac.uk)

#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"

#include <vector>
#include <sstream>
#include <string>

namespace tmtt {

  class SimpleLR4 : public TrackFitGeneric {
  public:
    SimpleLR4(const Settings* settings);

    ~SimpleLR4() override = default;

    L1fittedTrack fit(const L1track3D& l1track3D) override;

  protected:
    float phiSectorWidth_;
    float phiSectorCentre_;
    float phiNonantWidth_;

    float phiMult_;
    float rTMult_;
    float zMult_;
    float qOverPtMult_;
    float phiTMult_;
    float z0Mult_;
    float tanLambdaMult_;
    float numeratorPtMult_;
    float numeratorZ0Mult_;
    float numeratorLambdaMult_;
    float numeratorPhiMult_;
    float denominatorMult_;
    float chi2Mult_;
    float resMult_;
    float chi2cut_;
    float invPtToDPhi_;
    float chosenRofPhi_;
    unsigned int minStubLayersRed_;

    unsigned int dividerBitsHelix_;
    unsigned int dividerBitsHelixZ_;
    unsigned int dividerBitsChi2_;
    unsigned int shiftingBitsPhi_;
    unsigned int shiftingBitsDenRPhi_;
    unsigned int shiftingBitsDenRZ_;
    unsigned int shiftingBitsPt_;
    unsigned int shiftingBitsz0_;
    unsigned int shiftingBitsLambda_;
    bool digitize_;

    bool debug_;
  };

}  // namespace tmtt

#endif
