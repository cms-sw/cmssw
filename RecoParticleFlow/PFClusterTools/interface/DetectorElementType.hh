#ifndef DETECTORELEMENTTYPE_HH_
#define DETECTORELEMENTTYPE_HH_


/*
 * Possible detector elements:
 * 		ECAL
 * 		HCAL
 * 		PRESHOWER
 * 		OFFSET
 */

namespace minimiser {
enum DetectorElementType {
	ECAL = 0, HCAL = 1, PRESHOWER = 2, OFFSET = 3
};

const char* const DetElNames[] = { "ECAL", "HCAL", "PRESHOWER", "OFFSET" };

}
#endif /* DETECTORELEMENTTYPE_HH_ */
