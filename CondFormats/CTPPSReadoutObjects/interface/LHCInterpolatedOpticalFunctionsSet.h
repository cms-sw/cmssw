// Original Author:  Jan Kašpar

#ifndef CondFormats_CTPPSReadoutObjects_LHCInterpolatedOpticalFunctionsSet_h
#define CondFormats_CTPPSReadoutObjects_LHCInterpolatedOpticalFunctionsSet_h

#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSet.h"

#include "TSpline.h"

class CTPPSInterpolatedOpticalFunctionsESSource;
class CTPPSModifiedOpticalFunctionsESSource;

/// Set of optical functions corresponding to one scoring plane along LHC, including splines for interpolation performance.
class LHCInterpolatedOpticalFunctionsSet : public LHCOpticalFunctionsSet {
public:
  LHCInterpolatedOpticalFunctionsSet() = default;

  LHCInterpolatedOpticalFunctionsSet(const LHCOpticalFunctionsSet &src) : LHCOpticalFunctionsSet(src) {}

  ~LHCInterpolatedOpticalFunctionsSet() = default;

  const std::vector<std::shared_ptr<const TSpline3>> &splines() const { return m_splines; }

  /// builds splines from m_*_values fields
  void initializeSplines();

  /// proton kinematics description
  struct Kinematics {
    double x;     // physics vertex position (beam offset subtracted), cm
    double th_x;  // physics scattering angle (crossing angle subtracted), rad
    double y;     // physics vertex position, cm
    double th_y;  // physics scattering angle, rad
    double xi;    // relative momentum loss (positive for diffractive protons)
  };

  /// transports proton according to the splines
  void transport(const Kinematics &input, Kinematics &output, bool calculateAngles = false) const;

protected:
  friend CTPPSInterpolatedOpticalFunctionsESSource;
  friend CTPPSModifiedOpticalFunctionsESSource;

  std::vector<std::shared_ptr<const TSpline3>> m_splines;
};

#endif
