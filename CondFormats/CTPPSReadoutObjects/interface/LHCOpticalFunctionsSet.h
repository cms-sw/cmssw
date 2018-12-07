#ifndef CondFormats_CTPPSReadoutObjects_LHCOpticalFunctionsSet_h
#define CondFormats_CTPPSReadoutObjects_LHCOpticalFunctionsSet_h

// Original Author:  Jan Kašpar

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>
#include <memory>

#include "TSpline.h"

/**
 \brief Set of optical functions corresponding to one scoring plane along LHC.
**/
class LHCOpticalFunctionsSet
{
  public:
    /// indeces for m_fcn_values and m_splines data members
    enum { evx, eLx, e14, exd, evpx, eLpx, e24, expd, e32, evy, eLy, eyd, e42, evpy, eLpy, eypd };

    LHCOpticalFunctionsSet() {}

    /// fills m_*_values fields from a ROOT file
    LHCOpticalFunctionsSet(const std::string &fileName, const std::string &directoryName, const double &z);

    ~LHCOpticalFunctionsSet() {}

    /// returns the position of the scoring plane (LHC/TOTEM convention)
    double getScoringPlaneZ() const { return m_z; }

    const std::vector<double>& getXiValues() const { return m_xi_values; }

    const std::array<std::vector<double>, 16>& getFcnValues() const { return m_fcn_values; }

    const std::array<std::shared_ptr<TSpline3>, 16>& getSplines() const { return m_splines; }
  
    /// builds splines from m_*_values fields
    void initializeSplines();

    /// proton kinematics description
    struct Kinematics
    {
      double x;     // physics vertex position (beam offset subtracted), m
      double th_x;  // physics scattering angle (crossing angle subtracted), rad
      double y;     // physics vertex position, m
      double th_y;  // physics scattering angle, rad
      double xi;    // relative momentum loss (positive for diffractive protons)
    };
  
    /// transports proton according to the splines
    void transport(const Kinematics &input, Kinematics &output, bool calculateAngles = false) const;

    /// interpolates optical functions from (xangle1, of1) and (xangle2, of2) to xangle
    static LHCOpticalFunctionsSet* interpolate(double xangle1, const LHCOpticalFunctionsSet &of1,
      double xangle2, const LHCOpticalFunctionsSet &of2, double xangle);

  private:
    /// position of the scoring plane, in LHC/TOTEM convention, m
    double m_z;

    std::vector<double> m_xi_values;

    std::array<std::vector<double>, 16> m_fcn_values;

    std::array<std::shared_ptr<TSpline3>, 16> m_splines;
    
    COND_SERIALIZABLE;
};

#endif
