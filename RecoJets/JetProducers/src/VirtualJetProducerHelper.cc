#include "RecoJets/JetProducers/interface/VirtualJetProducerHelper.h"

#include <cmath>
#include <algorithm>

double reco::helper::VirtualJetProducerHelper::intersection(double r12) 
{
  if (r12 == 0)         return M_PI;
  if (r12 >= 2)         return 0;
  return 2 * acos(r12/2) - 0.5*r12*sqrt(std::max(0.0 , 4 - r12*r12));
}

double reco::helper::VirtualJetProducerHelper::intersection(double r12, double r23, double r13) 
{
  if (r12 >= 2 || r23 >= 2 || r13 >= 2) return 0;
  const double          r12_2   = r12*r12;
  const double          r13_2   = r13*r13;
  const double          temp    = (r12_2 + r13_2 - r23*r23);
  const double          T2      = std::max(0.0 , 4*r12_2*r13_2 - temp*temp);
  const double          common  = 0.5*( intersection(r12) + intersection(r13) + intersection(r23) - M_PI + sqrt(T2)/2 );
  return common;
}

double reco::helper::VirtualJetProducerHelper::intersection(double r12, double r23, double r13, double a12, double a23, double a13)
{
  if (r12 >= 2 || r23 >= 2 || r13 >= 2) return 0;
  const double          r12_2   = r12*r12;
  const double          r13_2   = r13*r13;
  const double          temp    = (r12_2 + r13_2 - r23*r23);
  const double          T2	= std::max(0.0 , 4*r12_2*r13_2 - temp*temp);
  const double          common  = 0.5*( a12 + a13 + a23 - M_PI + sqrt(T2)/2 );
  return common;
}

