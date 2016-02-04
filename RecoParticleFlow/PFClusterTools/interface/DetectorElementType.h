#ifndef DETECTORELEMENTTYPE_HH_
#define DETECTORELEMENTTYPE_HH_


namespace pftools {
/**
 * \class DetectorElementType
 * \brief Enumerates possible DetectorElement objects.
 * 
 * Possible detector elements:
 * 		ECAL
 * 		HCAL
 * 		PRESHOWER
 * 		OFFSET
 * 
 * \author Jamie Ballin
 * \date April 2008
 */
enum DetectorElementType {
	ECAL = 0, HCAL = 1, PRESHOWER = 2, OFFSET = 3, ECAL2 = 4, HCAL2 = 5
};

const char* const DetElNames[] = { "ECAL", "HCAL", "PRESHOWER", "OFFSET", "ECAL2", "HCAL2" };

}
#endif /* DETECTORELEMENTTYPE_HH_ */
