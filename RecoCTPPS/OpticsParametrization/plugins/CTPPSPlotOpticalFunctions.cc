/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoCTPPS/OpticsParametrization/interface/LHCOpticsApproximator.h"

#include "TGraph.h"

//----------------------------------------------------------------------------------------------------

class CTPPSPlotOpticalFunctions : public edm::one::EDAnalyzer<>
{
  public:
    explicit CTPPSPlotOpticalFunctions(const edm::ParameterSet&);
    ~CTPPSPlotOpticalFunctions();

  private: 
    std::string opticsFile;
    std::vector<std::string> opticsObjects;

    double vtx0_y_45, vtx0_y_56;
    double half_crossing_angle_45, half_crossing_angle_56;

    std::string outputFile;

    virtual void beginJob() override;

    virtual void analyze(edm::Event const&, edm::EventSetup const&) {}
};

using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------

CTPPSPlotOpticalFunctions::CTPPSPlotOpticalFunctions(const edm::ParameterSet& ps) :
  opticsFile(ps.getParameter<string>("opticsFile")),
  opticsObjects(ps.getParameter<vector<string>>("opticsObjects")),

  vtx0_y_45(ps.getParameter<double>("vtx0_y_45")),
  vtx0_y_56(ps.getParameter<double>("vtx0_y_56")),

  half_crossing_angle_45(ps.getParameter<double>("half_crossing_angle_45")),
  half_crossing_angle_56(ps.getParameter<double>("half_crossing_angle_56")),

  outputFile(ps.getParameter<string>("outputFile"))
{
}

//----------------------------------------------------------------------------------------------------

CTPPSPlotOpticalFunctions::~CTPPSPlotOpticalFunctions()
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSPlotOpticalFunctions::beginJob()
{
  printf(">> CTPPSPlotOpticalFunctions::beginJob\n");

  // open input file
  TFile *f_in = TFile::Open(opticsFile.c_str());
  if (!f_in)
    throw cms::Exception("CTPPSPlotOpticalFunctions") << "Cannot open file '" << opticsFile << "'.";

  // prepare output file
  TFile *f_out = TFile::Open(outputFile.c_str(), "recreate");

  // go through all optics objects
  for (const auto &objName : opticsObjects)
  {
    LHCOpticsApproximator *optApp = (LHCOpticsApproximator *) f_in->Get(objName.c_str());
    if (!optApp)
      throw cms::Exception("CTPPSPlotOpticalFunctions") << "Cannot load object '" << objName << "'.";

    printf("* %s --> %s\n", objName.c_str(), optApp->GetName());

    // make output directory
    gDirectory = f_out->mkdir(objName.c_str());

    // determine crossing angle, vertex offset
    double crossing_angle = 0.;
    double vtx0_y = 0.;

    if (optApp->GetBeamType() == LHCOpticsApproximator::lhcb2)
    {
      crossing_angle = half_crossing_angle_45;
      vtx0_y = vtx0_y_45;
    }

    if (optApp->GetBeamType() == LHCOpticsApproximator::lhcb1)
    {
      crossing_angle = half_crossing_angle_56;
      vtx0_y = vtx0_y_56;
    }

    // book graphs
    TGraph *g_x0_vs_xi = new TGraph();
    TGraph *g_y0_vs_xi = new TGraph();
    TGraph *g_y0_vs_x0 = new TGraph();
    TGraph *g_y0_vs_x0so = new TGraph();
    TGraph *g_y0so_vs_x0so = new TGraph();

    TGraph *g_v_x_vs_xi = new TGraph();
    TGraph *g_L_x_vs_xi = new TGraph();

    TGraph *g_v_y_vs_xi = new TGraph();
    TGraph *g_L_y_vs_xi = new TGraph();

    TGraph *g_xi_vs_x = new TGraph();
    TGraph *g_xi_vs_xso = new TGraph();
     
    const bool check_appertures = false;
    const bool invert_beam_coord_systems = true;

    // input: all zero
    double kin_in_zero[5] = { 0., crossing_angle, vtx0_y, 0., 0. };
    double kin_out_zero[5] = { 0., 0., 0., 0., 0. };
    optApp->Transport(kin_in_zero, kin_out_zero, check_appertures, invert_beam_coord_systems);

    // sample curves
    for (double xi = 0.; xi <= 0.151; xi += 0.001)
    {
      // input: only xi
      double kin_in_xi[5] = { 0., crossing_angle * (1. - xi), vtx0_y, 0., -xi };
      double kin_out_xi[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi, kin_out_xi, check_appertures, invert_beam_coord_systems);
  
      // input: xi and vtx_x
      const double vtx_x = 10E-6;  // m
      double kin_in_xi_vtx_x[5] = { vtx_x, crossing_angle * (1. - xi), vtx0_y, 0., -xi };
      double kin_out_xi_vtx_x[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi_vtx_x, kin_out_xi_vtx_x, check_appertures, invert_beam_coord_systems);
  
      // input: xi and th_x
      const double th_x = 20E-6;  // rad
      double kin_in_xi_th_x[5] = { 0., (crossing_angle + th_x) * (1. - xi), vtx0_y, 0., -xi };
      double kin_out_xi_th_x[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi_th_x, kin_out_xi_th_x, check_appertures, invert_beam_coord_systems);
  
      // input: xi and vtx_y
      const double vtx_y = 10E-6;  // m
      double kin_in_xi_vtx_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y + vtx_y, 0., -xi };
      double kin_out_xi_vtx_y[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi_vtx_y, kin_out_xi_vtx_y, check_appertures, invert_beam_coord_systems);
  
      // input: xi and th_y
      const double th_y = 20E-6;  // rad
      double kin_in_xi_th_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y, th_y * (1. - xi), -xi };
      double kin_out_xi_th_y[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi_th_y, kin_out_xi_th_y, check_appertures, invert_beam_coord_systems);
  
      // fill graphs
      int idx = g_xi_vs_x->GetN();
      g_x0_vs_xi->SetPoint(idx, xi, kin_out_xi[0]);
      g_y0_vs_xi->SetPoint(idx, xi, kin_out_xi[2]);
      g_y0_vs_x0->SetPoint(idx, kin_out_xi[0], kin_out_xi[2]);
      g_y0_vs_x0so->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], kin_out_xi[2]);
      g_y0so_vs_x0so->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], kin_out_xi[2] - kin_out_zero[2]);

      g_v_x_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_x[0] - kin_out_xi[0]) / vtx_x);
      g_L_x_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_x[0] - kin_out_xi[0]) / th_x);

      g_v_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_y[2] - kin_out_xi[2]) / vtx_y);
      g_L_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_y[2] - kin_out_xi[2]) / th_y);

      g_xi_vs_x->SetPoint(idx, kin_out_xi[0], xi);
      g_xi_vs_xso->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], xi);
    }

    // write graphs
    g_x0_vs_xi->Write("g_x0_vs_xi");
    g_y0_vs_xi->Write("g_y0_vs_xi");
    g_y0_vs_x0->Write("g_y0_vs_x0");
    g_y0_vs_x0so->Write("g_y0_vs_x0so");
    g_y0so_vs_x0so->Write("g_y0so_vs_x0so");

    g_v_x_vs_xi->Write("g_v_x_vs_xi");
    g_L_x_vs_xi->Write("g_L_x_vs_xi");

    g_v_y_vs_xi->Write("g_v_y_vs_xi");
    g_L_y_vs_xi->Write("g_L_y_vs_xi");

    g_xi_vs_x->Write("g_xi_vs_x");
    g_xi_vs_xso->Write("g_xi_vs_xso");
  }
  
  // clean up
  delete f_out;
  delete f_in;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSPlotOpticalFunctions);

