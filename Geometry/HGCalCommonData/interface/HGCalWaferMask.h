#ifndef HGCalCommonData_HGCalWaferMask_h
#define HGCalCommonData_HGCalWaferMask_h

/** \class HGCalWaferMask
 *
 * this class determines the masking of wafers to mimic partial wafers
 *
 *  $Date: 2019/01/15 00:06:50 $
 * \author Sunanda Banerjee, Fermilab <sunanda.banerjee@cern.ch>
 *
 */

#include <cmath>
#include <vector>

class HGCalWaferMask {
public:
  HGCalWaferMask() = default;

  static bool maskCell(int u, int v, int N, int ncor, int fcor, int corners);
  static bool goodCell(int u, int v, int N, int type, int rotn);
  static std::pair<int, int> getTypeMode(const double& xpos, const double& ypos, 
					 const double& r1, const double& R1, 
					 const double& rin, const double& rout,
					 const int& NW, const int& mode);
};

#endif
