#ifndef REGION_HH_
#define REGION_HH_


namespace pftools {

enum Region {
	NOREGION = 0, BARREL_POS = 1, TRANSITION_POS = 2, ENDCAP_POS = 3
};

const char* const RegionNames[] = { "NOREGION", "BARREL_POS", "TRANSITION_POS", "ENDCAP_POS" };


}
#endif //REGION_HH_
