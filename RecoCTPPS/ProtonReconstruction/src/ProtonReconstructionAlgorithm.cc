/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "RecoCTPPS/ProtonReconstruction/interface/ProtonReconstructionAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"


// TODO: needed?
//#include <cmath>

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

ProtonReconstructionAlgorithm::ProtonReconstructionAlgorithm(const std::string &optics_file_beam1, const std::string &optics_file_beam2,
    const edm::ParameterSet &beam_conditions) :

  beamConditions_(beam_conditions),
  halfCrossingAngleSector45_(beamConditions_.getParameter<double>("halfCrossingAngleSector45" )),
  halfCrossingAngleSector56_(beamConditions_.getParameter<double>("halfCrossingAngleSector56" )),
  yOffsetSector45_(beamConditions_.getParameter<double>("yOffsetSector45")),
  yOffsetSector56_(beamConditions_.getParameter<double>("yOffsetSector56")),

  fitter_(std::make_unique<ROOT::Fit::Fitter>()),
  chiSquareCalculator_(std::make_unique<ChiSquareCalculator>(beamConditions_))
{
  // open files with optics
  TFile *f_in_optics_beam1 = TFile::Open(optics_file_beam1.c_str());
  if (f_in_optics_beam1 == NULL)
    throw cms::Exception("ProtonReconstructionAlgorithm") << "Can't open file '" << optics_file_beam1 << "'.";

  TFile *f_in_optics_beam2 = TFile::Open(optics_file_beam2.c_str());
  if (f_in_optics_beam2 == NULL)
    throw cms::Exception("ProtonReconstructionAlgorithm") << "Can't open file '" << optics_file_beam2 << "'.";

  // build RP id, optics object name association
  std::map<unsigned int, std::string> idNameMap = {
    { TotemRPDetId(0, 0, 2), "ip5_to_station_150_h_1_lhcb2" },
    { TotemRPDetId(0, 0, 3), "ip5_to_station_150_h_2_lhcb2" },
    { TotemRPDetId(1, 0, 2), "ip5_to_station_150_h_1_lhcb1" },
    { TotemRPDetId(1, 0, 3), "ip5_to_station_150_h_2_lhcb1" }
  };

  // build optics data for each object
  for (const auto &it : idNameMap)
  {
    const CTPPSDetId rpId(it.first);
    const std::string &ofName = it.second;

    // load optics approximation
    TFile *f_in_optics = (rpId.arm() == 1) ? f_in_optics_beam1 : f_in_optics_beam2;
    LHCOpticsApproximator *of_orig = (LHCOpticsApproximator *) f_in_optics->Get(ofName.c_str());
    if (of_orig == NULL)
      throw cms::Exception("ProtonReconstructionAlgorithm") << "Can't load object '" << ofName << "'.";

    // copy approximator object not to loose it when the input ROOT file is closed
    RPOpticsData rpod;
    rpod.optics = make_shared<LHCOpticsApproximator>(LHCOpticsApproximator(* of_orig));

    // build auxiliary optical functions
    double crossing_angle = 0.;
    double vtx0_y = 0.;

    if (rpId.arm() == 0)
    {
      crossing_angle = halfCrossingAngleSector45_;
      vtx0_y = yOffsetSector45_;
    } else {
      crossing_angle = halfCrossingAngleSector56_;
      vtx0_y = yOffsetSector56_;
    }

    const bool check_appertures = false;
    const bool invert_beam_coord_sytems = true;

    TGraph *g_xi_vs_x = new TGraph();
    TGraph *g_y0_vs_xi = new TGraph();
    TGraph *g_v_y_vs_xi = new TGraph();
    TGraph *g_L_y_vs_xi = new TGraph();

    for (double xi = 0.; xi <= 0.201; xi += 0.005)
    {
      // input: only xi
      double kin_in_xi[5] = { 0., crossing_angle * (1. - xi), vtx0_y, 0., -xi };
      double kin_out_xi[5];
        rpod.optics->Transport(kin_in_xi, kin_out_xi, check_appertures, invert_beam_coord_sytems);

      // input: xi and vtx_y
      const double vtx_y = 10E-6;  // m
      double kin_in_xi_vtx_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y + vtx_y, 0., -xi };
      double kin_out_xi_vtx_y[5];
        rpod.optics->Transport(kin_in_xi_vtx_y, kin_out_xi_vtx_y, check_appertures, invert_beam_coord_sytems);

      // input: xi and th_y
      const double th_y = 20E-6;  // rad
      double kin_in_xi_th_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y, th_y * (1. - xi), -xi };
      double kin_out_xi_th_y[5];
        rpod.optics->Transport(kin_in_xi_th_y, kin_out_xi_th_y, check_appertures, invert_beam_coord_sytems);

      // fill graphs
      int idx = g_xi_vs_x->GetN();
      g_xi_vs_x->SetPoint(idx, kin_out_xi[0], xi);
      g_y0_vs_xi->SetPoint(idx, xi, kin_out_xi[2]);
      g_v_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_y[2] - kin_out_xi[2]) / vtx_y);
      g_L_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_y[2] - kin_out_xi[2]) / th_y);
    }

    rpod.s_xi_vs_x = make_shared<TSpline3>("", g_xi_vs_x->GetX(), g_xi_vs_x->GetY(), g_xi_vs_x->GetN());
    delete g_xi_vs_x;

    rpod.s_y0_vs_xi = make_shared<TSpline3>("", g_y0_vs_xi->GetX(), g_y0_vs_xi->GetY(), g_y0_vs_xi->GetN());
    delete g_y0_vs_xi;

    rpod.s_v_y_vs_xi = make_shared<TSpline3>("", g_v_y_vs_xi->GetX(), g_v_y_vs_xi->GetY(), g_v_y_vs_xi->GetN());
    delete g_v_y_vs_xi;

    rpod.s_L_y_vs_xi = make_shared<TSpline3>("", g_L_y_vs_xi->GetX(), g_L_y_vs_xi->GetY(), g_L_y_vs_xi->GetN());
    delete g_L_y_vs_xi;

    // insert optics data
    m_rp_optics_[rpId] = rpod;
  }

  // initialise fitter
  double pStart[] = { 0, 0, 0, 0 };
  fitter_->SetFCN( 4, *chiSquareCalculator_, pStart, 0, true );
}

//----------------------------------------------------------------------------------------------------

ProtonReconstructionAlgorithm::~ProtonReconstructionAlgorithm()
{
}

//----------------------------------------------------------------------------------------------------

double ProtonReconstructionAlgorithm::ChiSquareCalculator::operator() (const double* parameters) const
{
  // TODO: make use of check_apertures

  // extract proton parameters
  const double& xi = parameters[0];
  const double& th_x = parameters[1];
  const double& th_y = parameters[2];
  const double vtx_x = 0;
  const double& vtx_y = parameters[3];

  // calculate chi^2 by looping over hits
  double S2 = 0.0;

  for (const auto &track : *tracks)
  {
    const CTPPSDetId rpId(track->getRPId());

    double crossing_angle = 0., vtx0_y = 0.;

    if (rpId.arm() == 0)
    {
      crossing_angle = halfCrossingAngleSector45_;
      vtx0_y = yOffsetSector45_;
    } else {
      crossing_angle = halfCrossingAngleSector56_;
      vtx0_y = yOffsetSector56_;
    }

    // transport proton to the RP
    auto oit = m_rp_optics->find(rpId);

    double kin_in[5] = { vtx_x, (th_x + crossing_angle) * (1. - xi), vtx0_y + vtx_y, th_y * (1. - xi), -xi };
    double kin_out[5];
    const bool invert_beam_coord_sytems = true;
    oit->second.optics->Transport(kin_in, kin_out, check_apertures, invert_beam_coord_sytems);

    const double& x = kin_out[0];
    const double& y = kin_out[2];

    // calculate chi^2 contributions
    const double x_diff_norm = (x - track->getX()) / track->getXUnc();
    const double y_diff_norm = (y - track->getY()) / track->getYUnc();

    // increase chi^2
    S2 += x_diff_norm*x_diff_norm + y_diff_norm*y_diff_norm;
  }

  edm::LogInfo("ChiSquareCalculator")
    << "xi = " << xi << ", "
    << "th_x = " << th_x << ", "
    << "th_y = " << th_y << ", "
    << "vtx_y = " << vtx_y << " | S2 = " << S2 << "\n";

  return S2;
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::reconstruct(const vector<const CTPPSLocalTrackLite*> &tracks,
  vector<reco::ProtonTrack> &out, bool check_apertures) const
{
  // need at least two tracks
  if (tracks.size() < 2)
    return;

  // make sure optics is available for all tracks
  for (const auto &it : tracks)
  {
    auto oit = m_rp_optics_.find(it->getRPId());
    if (oit == m_rp_optics_.end())
      throw cms::Exception("") << "Optics data not available for RP " << it->getRPId() << ".";
  }

  // rough (initial) estimate of xi
  double S_xi0 = 0., S_1 = 0.;
  for (const auto &track : tracks)
  {
    auto oit = m_rp_optics_.find(track->getRPId());
    double xi = oit->second.s_xi_vs_x->Eval(track->getX());

    S_1 += 1.;
    S_xi0 += xi;
  }

  const double xi_0 = S_xi0 / S_1;

  // rough (initial) estimate of th_y and vtx_y
  double y[2], v_y[2], L_y[2];
  unsigned int y_idx = 0;
  for (const auto &track : tracks)
  {
    if (y_idx >= 2)
      continue;

    auto oit = m_rp_optics_.find(track->getRPId());

    y[y_idx] = track->getY() - oit->second.s_y0_vs_xi->Eval(xi_0);
    v_y[y_idx] = oit->second.s_v_y_vs_xi->Eval(xi_0);
    L_y[y_idx] = oit->second.s_L_y_vs_xi->Eval(xi_0);

    y_idx++;
  }

  const double det_y = v_y[0] * L_y[1] - L_y[0] * v_y[1];
  const double vtx_y_0 = (L_y[1] * y[0] - L_y[0] * y[1]) / det_y;
  const double th_y_0 = (v_y[0] * y[1] - v_y[1] * y[0]) / det_y;

  // minimisation
  fitter_->Config().ParSettings(0).Set("xi", xi_0, 0.005);
  fitter_->Config().ParSettings(1).Set("th_x", 0., 20E-6);
  fitter_->Config().ParSettings(2).Set("th_y", th_y_0, 1E-6);
  fitter_->Config().ParSettings(3).Set("vtx_y", vtx_y_0, 1E-6);

  chiSquareCalculator_->tracks = &tracks;
  chiSquareCalculator_->m_rp_optics = &m_rp_optics_;
  chiSquareCalculator_->check_apertures = check_apertures;

  fitter_->FitFCN();

  // extract proton parameters
  const ROOT::Fit::FitResult& result = fitter_->Result();
  const double *params = result.GetParams();

  edm::LogInfo("ProtonReconstructionAlgorithm")
    << "at reconstructed level: "
    << "xi=" << params[0] << ", "
    << "theta_x=" << params[1] << ", "
    << "theta_y=" << params[2] << ", "
    << "vertex_y=" << params[3] << "\n";

  reco::ProtonTrack pt;
  pt.setValid(result.IsValid());
  pt.setVertex(Local3DPoint(0., params[3], 0.));  // TODO: apply the CMS coordinate convention
  pt.setDirection(Local3DVector(params[1], params[2], 1.)); // TODO: make this correct, apply the CMS coordinate convention
  pt.setXi(params[0]);
  
  pt.fitChiSq = result.Chi2();
  pt.method = reco::ProtonTrack::rmMultipleRP;
  pt.lhcSector = (CTPPSDetId(tracks[0]->getRPId()).arm() == 0) ? reco::ProtonTrack::sector45 : reco::ProtonTrack::sector56;

  for (const auto &track : tracks)
    pt.contributingRPIds.insert(track->getRPId());

  out.push_back(move(pt));
}
