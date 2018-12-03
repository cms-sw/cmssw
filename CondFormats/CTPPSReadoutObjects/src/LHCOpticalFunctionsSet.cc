// Original Author:  Jan Kašpar

#include "CondFormats/CTPPSReadoutObjects/interface/LHCOpticalFunctionsSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "TFile.h"
#include "TGraph.h"

//----------------------------------------------------------------------------------------------------

LHCOpticalFunctionsSet::LHCOpticalFunctionsSet(const std::string &fileName, const std::string &directoryName)
{
  TFile *f_in = TFile::Open(fileName.c_str());
  if (f_in == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot open file " << fileName << ".";

  TGraph *g_x_D = (TGraph *) f_in->Get((directoryName + "/g_x_D_vs_xi").c_str());
  if (g_x_D == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << directoryName << "/g_x_D_vs_xi from file " << fileName << "."; 

  TGraph *g_L_x = (TGraph *) f_in->Get((directoryName + "/g_L_x_vs_xi").c_str());
  if (g_L_x == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << directoryName << "/g_L_x_vs_xi from file " << fileName << "."; 

  TGraph *g_v_x = (TGraph *) f_in->Get((directoryName + "/g_v_x_vs_xi").c_str());
  if (g_v_x == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << directoryName << "/g_v_x_vs_xi from file " << fileName << "."; 

  TGraph *g_E_14 = (TGraph *) f_in->Get((directoryName + "/g_E_14_vs_xi").c_str());
  if (g_E_14 == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << directoryName << "/g_E_14_vs_xi from file " << fileName << "."; 

  TGraph *g_y_D = (TGraph *) f_in->Get((directoryName + "/g_y_D_vs_xi").c_str());
  if (g_y_D == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << directoryName << "/g_y_D_vs_xi from file " << fileName << "."; 

  TGraph *g_L_y = (TGraph *) f_in->Get((directoryName + "/g_L_y_vs_xi").c_str());
  if (g_L_y == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << directoryName << "/g_L_y_vs_xi from file " << fileName << "."; 

  TGraph *g_v_y = (TGraph *) f_in->Get((directoryName + "/g_v_y_vs_xi").c_str());
  if (g_v_y == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << directoryName << "/g_v_y_vs_xi from file " << fileName << "."; 

  TGraph *g_E_32 = (TGraph *) f_in->Get((directoryName + "/g_E_32_vs_xi").c_str());
  if (g_E_32 == NULL)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot load object " << directoryName << "/g_E_32_vs_xi from file " << fileName << "."; 

  const unsigned int n = g_x_D->GetN();

  m_xi_values.resize(n);
  m_x_D_values.resize(n);
  m_L_x_values.resize(n);
  m_v_x_values.resize(n);
  m_E_14_values.resize(n);

  m_y_D_values.resize(n);
  m_L_y_values.resize(n);
  m_v_y_values.resize(n);
  m_E_32_values.resize(n);

  for (unsigned int i = 0; i < n; ++i)
  {
    const double xi = g_x_D->GetX()[i];
    m_xi_values[i] = xi;
   
    m_x_D_values[i] = g_x_D->Eval(xi); 
    m_L_x_values[i] = g_L_x->Eval(xi); 
    m_v_x_values[i] = g_v_x->Eval(xi); 
    m_E_14_values[i] = g_E_14->Eval(xi); 
   
    m_y_D_values[i] = g_y_D->Eval(xi); 
    m_L_y_values[i] = g_L_y->Eval(xi); 
    m_v_y_values[i] = g_v_y->Eval(xi); 
    m_E_32_values[i] = g_E_32->Eval(xi); 
  }

  delete f_in;
}

//----------------------------------------------------------------------------------------------------

void LHCOpticalFunctionsSet::InitializeSplines()
{
  const unsigned int n = m_xi_values.size();

  m_s_x_D_vs_xi = std::make_shared<TSpline3>("", m_xi_values.data(), m_x_D_values.data(), n);
  m_s_L_x_vs_xi = std::make_shared<TSpline3>("", m_xi_values.data(), m_L_x_values.data(), n);
  m_s_v_x_vs_xi = std::make_shared<TSpline3>("", m_xi_values.data(), m_v_x_values.data(), n);
  m_s_E_14_vs_xi = std::make_shared<TSpline3>("", m_xi_values.data(), m_E_14_values.data(), n);

  m_s_y_D_vs_xi = std::make_shared<TSpline3>("", m_xi_values.data(), m_y_D_values.data(), n);
  m_s_L_y_vs_xi = std::make_shared<TSpline3>("", m_xi_values.data(), m_L_y_values.data(), n);
  m_s_v_y_vs_xi = std::make_shared<TSpline3>("", m_xi_values.data(), m_v_y_values.data(), n);
  m_s_E_32_vs_xi = std::make_shared<TSpline3>("", m_xi_values.data(), m_E_32_values.data(), n);
}

//----------------------------------------------------------------------------------------------------

void LHCOpticalFunctionsSet::Transport(const LHCOpticalFunctionsSet::Kinematics &input, LHCOpticalFunctionsSet::Kinematics &output) const
{
  output.x = m_s_x_D_vs_xi->Eval(input.xi) + m_s_v_x_vs_xi->Eval(input.xi) * input.x
    + m_s_L_x_vs_xi->Eval(input.xi) * input.th_x + m_s_E_14_vs_xi->Eval(input.xi) * input.th_y;

  output.th_x = 0.;

  output.y = m_s_y_D_vs_xi->Eval(input.xi) + m_s_v_y_vs_xi->Eval(input.xi) * input.y
    + m_s_L_y_vs_xi->Eval(input.xi) * input.th_y + m_s_E_32_vs_xi->Eval(input.xi) * input.th_x;

  output.th_y = 0.;

  output.xi = input.xi;
}

//----------------------------------------------------------------------------------------------------

LHCOpticalFunctionsSet* LHCOpticalFunctionsSet::Interpolate(double xangle1, const LHCOpticalFunctionsSet &of1,
  double xangle2, const LHCOpticalFunctionsSet &of2, double xangle)
{
  LHCOpticalFunctionsSet *output = new LHCOpticalFunctionsSet();

  const unsigned int n = of1.m_xi_values.size();

  output->m_xi_values.resize(n);
  output->m_x_D_values.resize(n);
  output->m_L_x_values.resize(n);
  output->m_v_x_values.resize(n);
  output->m_E_14_values.resize(n);
  output->m_y_D_values.resize(n);
  output->m_L_y_values.resize(n);
  output->m_v_y_values.resize(n);
  output->m_E_32_values.resize(n);

  for (unsigned int i = 0; i < n; ++i)
  {
    double xi = of1.m_xi_values[i];

    output->m_xi_values[i] = xi;

    double x_D_1 = of1.m_s_x_D_vs_xi->Eval(xi), x_D_2 = of2.m_s_x_D_vs_xi->Eval(xi);
    output->m_x_D_values[i] = x_D_2 + (x_D_2 - x_D_1) / (xangle2 - xangle1) * (xangle - xangle2);

    double L_x_1 = of1.m_s_L_x_vs_xi->Eval(xi), L_x_2 = of2.m_s_L_x_vs_xi->Eval(xi);
    output->m_L_x_values[i] = L_x_2 + (L_x_2 - L_x_1) / (xangle2 - xangle1) * (xangle - xangle2);

    double v_x_1 = of1.m_s_v_x_vs_xi->Eval(xi), v_x_2 = of2.m_s_v_x_vs_xi->Eval(xi);
    output->m_v_x_values[i] = v_x_2 + (v_x_2 - v_x_1) / (xangle2 - xangle1) * (xangle - xangle2);

    double E_14_1 = of1.m_s_E_14_vs_xi->Eval(xi), E_14_2 = of2.m_s_E_14_vs_xi->Eval(xi);
    output->m_E_14_values[i] = E_14_2 + (E_14_2 - E_14_1) / (xangle2 - xangle1) * (xangle - xangle2);

    double y_D_1 = of1.m_s_y_D_vs_xi->Eval(xi), y_D_2 = of2.m_s_y_D_vs_xi->Eval(xi);
    output->m_y_D_values[i] = y_D_2 + (y_D_2 - y_D_1) / (xangle2 - xangle1) * (xangle - xangle2);

    double L_y_1 = of1.m_s_L_y_vs_xi->Eval(xi), L_y_2 = of2.m_s_L_y_vs_xi->Eval(xi);
    output->m_L_y_values[i] = L_y_2 + (L_y_2 - L_y_1) / (xangle2 - xangle1) * (xangle - xangle2);

    double v_y_1 = of1.m_s_v_y_vs_xi->Eval(xi), v_y_2 = of2.m_s_v_y_vs_xi->Eval(xi);
    output->m_v_y_values[i] = v_y_2 + (v_y_2 - v_y_1) / (xangle2 - xangle1) * (xangle - xangle2);

    double E_32_1 = of1.m_s_E_32_vs_xi->Eval(xi), E_32_2 = of2.m_s_E_32_vs_xi->Eval(xi);
    output->m_E_32_values[i] = E_32_2 + (E_32_2 - E_32_1) / (xangle2 - xangle1) * (xangle - xangle2);
  }

  return output;
}
