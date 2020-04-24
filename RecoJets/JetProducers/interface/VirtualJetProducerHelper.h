#ifndef RecoJets_JetProducers_interface_VirtualJetProducerHelper_h
#define RecoJets_JetProducers_interface_VirtualJetProducerHelper_h

#include <cmath>
#include <algorithm>

namespace reco {

  namespace helper {

    namespace VirtualJetProducerHelper {

// Area of intersection of two unit-radius disks with centers separated by r12.
inline  double intersection(double r12)
{
  if (r12 == 0)         return M_PI;
  if (r12 >= 2)         return 0;
  return 2 * std::acos(0.5*r12) - 0.5*r12*sqrt(std::max(0.0 , 4 - r12*r12));
}

// Area of intersection of three unit-radius disks with centers separated by r12, r23, r13.
inline double intersection(double r12, double r23, double r13) 
{
  if (r12 >= 2 || r23 >= 2 || r13 >= 2) return 0;
  const double          r12_2   = r12*r12;
  const double          r13_2   = r13*r13;
  const double          temp    = (r12_2 + r13_2 - r23*r23);
  const double          T2      = std::max(0.0 , 4*r12_2*r13_2 - temp*temp);
  const double          common  = 0.5*( intersection(r12) + intersection(r13) + intersection(r23) - M_PI + 0.5*sqrt(T2) );
  return common;
}
inline double intersection(double r12, double r23, double r13, double a12, double a23, double a13)
{
  if (r12 >= 2 || r23 >= 2 || r13 >= 2) return 0;
  const double          r12_2   = r12*r12;
  const double          r13_2   = r13*r13;
  const double          temp    = (r12_2 + r13_2 - r23*r23);
  const double          T2	= std::max(0.0 , 4*r12_2*r13_2 - temp*temp);
  const double          common  = 0.5*( a12 + a13 + a23 - M_PI + 0.5*sqrt(T2) );
  return common;
}

    }
  }
}
#endif
