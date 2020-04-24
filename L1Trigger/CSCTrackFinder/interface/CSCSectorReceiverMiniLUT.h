#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>

#ifndef L1Trigger_CSCSectorReceiverMiniLUT_h
#define L1Trigger_CSCSectorReceiverMiniLUT_h

/**
 * \class CSCSectorReceiverMiniLUT
 * \author Brett Jackson
 *
 * Provides a new way of defining the Lookup tables used by the core.
 * Defines the lookup tables as parameterized functions in order to save
 * on memory usage when compared to the current standard definitions of LUTs
 */

class CSCSectorReceiverMiniLUT
{
public:
  static lclphidat calcLocalPhiMini(unsigned theadd, const bool gangedME1a);
  static global_eta_data calcGlobalEtaMEMini(unsigned short endcap, unsigned short sector, unsigned short station, unsigned short subsector, unsigned theadd, const bool gangedME1a);
  static global_phi_data calcGlobalPhiMEMini(unsigned short endcap, unsigned short sector, unsigned short station, unsigned short subsector, unsigned theadd, const bool gangedME1a);
  static global_phi_data calcGlobalPhiMBMini(unsigned short endcap, unsigned short sector, unsigned short subsector, unsigned theadd, const bool gangedME1a);
  
private:
  static const float lcl_phi_param0[1<<4];
  static const float lcl_phi_param1; 

  static const float gbl_eta_params[2][6][4][2][4][9][3]; // [endcap][sector][station][subsector][localPhi][cscID][param 0, 1, or 2]
  static const unsigned short int gbl_eta_bounds[2][6][4][2][4][9][2]; // [endcap][sector][station][subsector][localPhi][cscID][0=min, 1=max]
  
  static const float gbl_phi_me_params[2][6][4][2][9][2]; // [endcap][sector][station][subsector][cscID][param 0 or 1]
  
  static const float gbl_phi_mb_params[2][6][2][9][2]; // [endcap][sector][subsector][cscID][param 0 or 1]
};

#endif
