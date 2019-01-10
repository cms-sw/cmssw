// Original Author:  Jan Kašpar

#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "TFile.h"
#include "TGraph.h"

//----------------------------------------------------------------------------------------------------

LHCOpticalFunctionsSet::LHCOpticalFunctionsSet(const std::string &fileName, const std::string &directoryName, double z) :
  m_z(z)
{
  TFile *f_in = TFile::Open(fileName.c_str());
  if (f_in == nullptr)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot open file " << fileName << ".";

  std::vector<TGraph*> graphs(m_fcn_values.size());
  for (unsigned int fi = 0; fi < m_fcn_values.size(); ++fi) {
    std::string tag;
    if (fi == evx) tag = "v_x";
    else if (fi == eLx) tag = "L_x";
    else if (fi == e14) tag = "E_14";
    else if (fi == exd) tag = "x_D";
    else if (fi == evpx) tag = "vp_x";
    else if (fi == eLpx) tag = "Lp_x";
    else if (fi == e24) tag = "E_24";
    else if (fi == expd) tag = "xp_D";
    else if (fi == e32) tag = "E_32";
    else if (fi == evy) tag = "v_y";
    else if (fi == eLy) tag = "L_y";
    else if (fi == eyd) tag = "y_D";
    else if (fi == e42) tag = "E_42";
    else if (fi == evpy) tag = "vp_y";
    else if (fi == eLpy) tag = "Lp_y";
    else if (fi == eypd) tag = "yp_D";
    else
      throw cms::Exception("LHCOpticalFunctionsSet") << "Invalid tag for optical functions: \"" << fi << "\"";

    std::string objPath = directoryName + "/g_" + tag + "_vs_xi";
    auto gr_obj = dynamic_cast<TGraph*>( f_in->Get(objPath.c_str()) );
    if (!gr_obj)
      throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << objPath << " from file " << fileName << ".";

    graphs[fi] = gr_obj;
  }

  const unsigned int num_xi_vals = graphs[0]->GetN();

  m_xi_values.resize(num_xi_vals);

  for (unsigned int fi = 0; fi < m_fcn_values.size(); ++fi)
    m_fcn_values[fi].resize(num_xi_vals);

  for (unsigned int pi = 0; pi < num_xi_vals; ++pi) {
    const double xi = graphs[0]->GetX()[pi];
    m_xi_values[pi] = xi;

    for (unsigned int fi = 0; fi < m_fcn_values.size(); ++fi)
      m_fcn_values[fi][pi] = graphs[fi]->Eval(xi);
  }

  delete f_in;
}

//----------------------------------------------------------------------------------------------------

void LHCOpticalFunctionsSet::initializeSplines()
{
  const unsigned int num_xi_vals = m_xi_values.size();

  for (unsigned int i = 0; i < m_fcn_values.size(); ++i)
    m_splines[i] = std::make_shared<TSpline3>("", m_xi_values.data(), m_fcn_values[i].data(), num_xi_vals);
}

//----------------------------------------------------------------------------------------------------

void LHCOpticalFunctionsSet::transport(const LHCOpticalFunctionsSet::Kinematics &input,
  LHCOpticalFunctionsSet::Kinematics &output, bool calculateAngles) const
{
  const double xi = input.xi;

  output.x = m_splines[exd]->Eval(xi) + m_splines[evx]->Eval(xi) * input.x
    + m_splines[eLx]->Eval(xi) * input.th_x + m_splines[e14]->Eval(xi) * input.th_y;

  output.th_x = (!calculateAngles) ? 0. : m_splines[expd]->Eval(xi) + m_splines[evpx]->Eval(xi) * input.x
    + m_splines[eLpx]->Eval(xi) * input.th_x + m_splines[e24]->Eval(xi) * input.th_y;

  output.y = m_splines[eyd]->Eval(xi) + m_splines[evy]->Eval(xi) * input.y
    + m_splines[eLy]->Eval(xi) * input.th_y + m_splines[e32]->Eval(xi) * input.th_x;

  output.th_y = (!calculateAngles) ? 0. : m_splines[eypd]->Eval(xi) + m_splines[evpy]->Eval(xi) * input.y
    + m_splines[eLpy]->Eval(xi) * input.th_y + m_splines[e42]->Eval(xi) * input.th_x;

  output.xi = input.xi;
}

//----------------------------------------------------------------------------------------------------

LHCOpticalFunctionsSet* LHCOpticalFunctionsSet::interpolate(double xangle1, const LHCOpticalFunctionsSet &of1,
  double xangle2, const LHCOpticalFunctionsSet &of2, double xangle)
{
  // check whether interpolation can be done
  if (std::abs(xangle1 - xangle2) < 1e-6) {
    if (std::abs(xangle - xangle1) < 1e-6)
      return new LHCOpticalFunctionsSet(of1);
    else
      throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot interpolate from angles " << xangle1 <<
        " and " << xangle2 << " to angle " << xangle << ".";
  }

  // do interpolation
  LHCOpticalFunctionsSet *output = new LHCOpticalFunctionsSet();

  output->m_z = of1.m_z;

  const size_t num_xi_vals = of1.m_xi_values.size();

  output->m_xi_values.resize(num_xi_vals);

  for (size_t fi = 0; fi < of1.m_fcn_values.size(); ++fi) {
    output->m_fcn_values[fi].resize(num_xi_vals);

    for (size_t pi = 0; pi < num_xi_vals; ++pi) {
      double xi = of1.m_xi_values[pi];

      output->m_xi_values[pi] = xi;

      double v1 = of1.m_splines[fi]->Eval(xi);
      double v2 = of2.m_splines[fi]->Eval(xi);
      output->m_fcn_values[fi][pi] = v1 + (v2 - v1) / (xangle2 - xangle1) * (xangle - xangle1);
    }
  }

  return output;
}
