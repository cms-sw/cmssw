#ifndef PhysicsTools_Heppy_Apc_h
#define PhysicsTools_Heppy_Apc_h

#include <cmath>
#include <numeric>
#include <vector>


/*  Apc
 *
 *  Calculates the Apc event variable for a given input jet collection
 *
 */

namespace heppy {

struct Apc {

  static double getApcJetMetMin( const std::vector<double>& et,
                                 const std::vector<double>& px,
                                 const std::vector<double>& py,
                                 const double metx, const double mety);
   
};

};

#endif // Apc_h
