// Original Author:  Jan Kašpar

#include "CondFormats/PPSObjects/interface/LHCOpticalFunctionsSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "TFile.h"
#include "TGraph.h"

//----------------------------------------------------------------------------------------------------

LHCOpticalFunctionsSet::LHCOpticalFunctionsSet(const std::string &fileName, const std::string &directoryName, double z)
    : m_z(z) {
  TFile *f_in = TFile::Open(fileName.c_str());
  if (f_in == nullptr)
    throw cms::Exception("LHCOpticalFunctionsSet") << "Cannot open file " << fileName << ".";

  std::vector<TGraph *> graphs(nFunctions);
  for (unsigned int fi = 0; fi < nFunctions; ++fi) {
    std::string tag;
    if (fi == evx)
      tag = "v_x";
    else if (fi == eLx)
      tag = "L_x";
    else if (fi == e14)
      tag = "E_14";
    else if (fi == exd)
      tag = "x_D";
    else if (fi == evpx)
      tag = "vp_x";
    else if (fi == eLpx)
      tag = "Lp_x";
    else if (fi == e24)
      tag = "E_24";
    else if (fi == expd)
      tag = "xp_D";
    else if (fi == e32)
      tag = "E_32";
    else if (fi == evy)
      tag = "v_y";
    else if (fi == eLy)
      tag = "L_y";
    else if (fi == eyd)
      tag = "y_D";
    else if (fi == e42)
      tag = "E_42";
    else if (fi == evpy)
      tag = "vp_y";
    else if (fi == eLpy)
      tag = "Lp_y";
    else if (fi == eypd)
      tag = "yp_D";
    else
      throw cms::Exception("LHCOpticalFunctionsSet") << "Invalid tag for optical functions: \"" << fi << "\"";

    std::string objPath = directoryName + "/g_" + tag + "_vs_xi";
    auto gr_obj = dynamic_cast<TGraph *>(f_in->Get(objPath.c_str()));
    if (!gr_obj)
      throw cms::Exception("LHCOpticalFunctionsSet")
          << "Cannot load object " << objPath << " from file " << fileName << ".";

    graphs[fi] = gr_obj;
  }

  const unsigned int num_xi_vals = graphs[0]->GetN();
  m_xi_values.resize(num_xi_vals);

  m_fcn_values.resize(nFunctions);

  for (unsigned int fi = 0; fi < nFunctions; ++fi)
    m_fcn_values[fi].resize(num_xi_vals);

  for (unsigned int pi = 0; pi < num_xi_vals; ++pi) {
    const double xi = graphs[0]->GetX()[pi];
    m_xi_values[pi] = xi;

    for (unsigned int fi = 0; fi < m_fcn_values.size(); ++fi)
      m_fcn_values[fi][pi] = graphs[fi]->Eval(xi);
  }

  delete f_in;
}
