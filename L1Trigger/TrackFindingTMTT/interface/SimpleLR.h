///=== This is the simple linear regression with 4 helix parameters (qOverPt, phiT, z0, tanLambda) track fit algorithm.

///=== Written by: Davide Cieri (davide.cieri@stfc.ac.uk)

#ifndef __SIMPLELR__
#define __SIMPLELR__

#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"

#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include <vector>
#include <sstream>
#include <string>

namespace TMTT {

class SimpleLR : public TrackFitGeneric {

public:
	SimpleLR(const Settings* settings) : TrackFitGeneric(settings), settings_(settings) {};

	virtual ~SimpleLR() {};

	virtual void initRun();

        L1fittedTrack fit(const L1track3D& l1track3D);

protected:

	const Settings* settings_;

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

	bool                 digitize_;
	unsigned int         dividerBitsHelix_;
	unsigned int         dividerBitsHelixZ_;
	unsigned int         dividerBitsChi2_;
	unsigned int         shiftingBitsPhi_;
	unsigned int         shiftingBitsDenRPhi_;
	unsigned int         shiftingBitsDenRZ_;
	unsigned int 	     shiftingBitsPt_;
	unsigned int 	     shiftingBitsz0_;
	unsigned int         shiftingBitsLambda_;

	
};

}

#endif
