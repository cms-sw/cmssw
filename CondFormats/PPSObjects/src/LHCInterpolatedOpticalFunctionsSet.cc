// Original Author:  Jan Kašpar

#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSet.h"

//----------------------------------------------------------------------------------------------------

void LHCInterpolatedOpticalFunctionsSet::initializeSplines() {
  const unsigned int num_xi_vals = m_xi_values.size();

  m_splines.resize(m_fcn_values.size());
  for (unsigned int i = 0; i < m_fcn_values.size(); ++i)
    m_splines[i] = std::make_shared<TSpline3>("", m_xi_values.data(), m_fcn_values[i].data(), num_xi_vals);
}

//----------------------------------------------------------------------------------------------------

void LHCInterpolatedOpticalFunctionsSet::transport(const LHCInterpolatedOpticalFunctionsSet::Kinematics &input,
                                                   LHCInterpolatedOpticalFunctionsSet::Kinematics &output,
                                                   bool calculateAngles) const {
  const double xi = input.xi;

  output.x = m_splines[exd]->Eval(xi) + m_splines[evx]->Eval(xi) * input.x + m_splines[eLx]->Eval(xi) * input.th_x +
             m_splines[e14]->Eval(xi) * input.th_y;

  output.th_x = (!calculateAngles) ? 0.
                                   : m_splines[expd]->Eval(xi) + m_splines[evpx]->Eval(xi) * input.x +
                                         m_splines[eLpx]->Eval(xi) * input.th_x + m_splines[e24]->Eval(xi) * input.th_y;

  output.y = m_splines[eyd]->Eval(xi) + m_splines[evy]->Eval(xi) * input.y + m_splines[eLy]->Eval(xi) * input.th_y +
             m_splines[e32]->Eval(xi) * input.th_x;

  output.th_y = (!calculateAngles) ? 0.
                                   : m_splines[eypd]->Eval(xi) + m_splines[evpy]->Eval(xi) * input.y +
                                         m_splines[eLpy]->Eval(xi) * input.th_y + m_splines[e42]->Eval(xi) * input.th_x;

  output.xi = input.xi;
}
