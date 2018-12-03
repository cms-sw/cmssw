#ifndef CondFormats_CTPPSReadoutObjects_LHCOpticalFunctionsSet_h
#define CondFormats_CTPPSReadoutObjects_LHCOpticalFunctionsSet_h

// Original Author:  Jan Kašpar

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>
#include <memory>

#include "TSpline.h"

class LHCOpticalFunctionsSet
{
  public:
    LHCOpticalFunctionsSet() {}

    /// fills m_*_values fields from a ROOT file
    LHCOpticalFunctionsSet(const std::string &fileName, const std::string &directoryName);

    ~LHCOpticalFunctionsSet();
  
    /// builds splines from m_*_values fields
    void InitializeSplines();

    /// proton kinematics description
    struct Kinematics
    {
      double x, th_x, y, th_y, xi;
    };
  
    /// transports proton according to the splines
    void Transport(const Kinematics &input, Kinematics &output) const;

    /// interpolates optical functions from (xangle1, of1) and (xangle2, of2) to xangle
    LHCOpticalFunctionsSet* Interpolate(double xangle1, const LHCOpticalFunctionsSet &of1,
      double xangle2, const LHCOpticalFunctionsSet &of2, double xangle);

  private:
    std::vector<double> m_xi_values;

    std::vector<double> m_x_D_values;
    std::vector<double> m_L_x_values;
    std::vector<double> m_v_x_values;
    std::vector<double> m_E_14_values;

    std::vector<double> m_y_D_values;
    std::vector<double> m_L_y_values;
    std::vector<double> m_v_y_values;
    std::vector<double> m_E_32_values;

    std::shared_ptr<TSpline3> m_s_x_D_vs_xi;
    std::shared_ptr<TSpline3> m_s_L_x_vs_xi;
    std::shared_ptr<TSpline3> m_s_v_x_vs_xi;
    std::shared_ptr<TSpline3> m_s_E_14_vs_xi;

    std::shared_ptr<TSpline3> m_s_y_D_vs_xi;
    std::shared_ptr<TSpline3> m_s_L_y_vs_xi;
    std::shared_ptr<TSpline3> m_s_v_y_vs_xi;
    std::shared_ptr<TSpline3> m_s_E_32_vs_xi;
    
    COND_SERIALIZABLE;
};

#endif
