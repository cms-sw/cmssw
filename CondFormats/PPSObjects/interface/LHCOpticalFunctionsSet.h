// Original Author:  Jan Kašpar

#ifndef CondFormats_PPSObjects_LHCOpticalFunctionsSet_h
#define CondFormats_PPSObjects_LHCOpticalFunctionsSet_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>
#include <memory>

/// Set of optical functions corresponding to one scoring plane along LHC.
class LHCOpticalFunctionsSet {
public:
  /// indices for m_fcn_values and m_splines data members
  enum { evx, eLx, e14, exd, evpx, eLpx, e24, expd, e32, evy, eLy, eyd, e42, evpy, eLpy, eypd, nFunctions };

  LHCOpticalFunctionsSet() = default;

  /// fills m_*_values fields from a ROOT file
  LHCOpticalFunctionsSet(const std::string& fileName, const std::string& directoryName, double z);

  ~LHCOpticalFunctionsSet() = default;

  /// returns the position of the scoring plane (LHC/TOTEM convention)
  double getScoringPlaneZ() const { return m_z; }

  const std::vector<double>& getXiValues() const { return m_xi_values; }
  const std::vector<std::vector<double>>& getFcnValues() const { return m_fcn_values; }

protected:
  /// position of the scoring plane, in LHC/TOTEM convention, cm
  double m_z;

  std::vector<double> m_xi_values;
  std::vector<std::vector<double>> m_fcn_values;  ///< length unit cm

  COND_SERIALIZABLE;
};

#endif
