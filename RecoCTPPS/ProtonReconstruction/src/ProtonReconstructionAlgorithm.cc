/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#include "RecoCTPPS/ProtonReconstruction/interface/ProtonReconstructionAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"

#include "TMinuitMinimizer.h"

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

ProtonReconstructionAlgorithm::ProtonReconstructionAlgorithm(bool fit_vtx_y, bool improved_estimate, unsigned int verbosity) :
  verbosity_(verbosity),
  fitVtxY_(fit_vtx_y),
  useImprovedInitialEstimate_(improved_estimate),
  initialized_(false),
  fitter_(new ROOT::Fit::Fitter), chiSquareCalculator_(new ChiSquareCalculator)
{
  // needed for thread safety
  TMinuitMinimizer::UseStaticMinuit(false);

  // initialise fitter
  double pStart[] = { 0, 0, 0, 0 };
  fitter_->SetFCN(4, *chiSquareCalculator_, pStart, 0, true);
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::init(const std::unordered_map<unsigned int, LHCOpticalFunctionsSet> &opticalFunctions)
{
  // reset cache
  release();

  // build optics data for each object
  for (const auto &p : opticalFunctions)
  {
    const LHCOpticalFunctionsSet &ofs = p.second;

    // make record
    RPOpticsData rpod;
    rpod.optics = &p.second;
    rpod.s_y_d_vs_xi = ofs.getSplines()[LHCOpticalFunctionsSet::eyd];
    rpod.s_v_y_vs_xi = ofs.getSplines()[LHCOpticalFunctionsSet::evy];
    rpod.s_L_y_vs_xi = ofs.getSplines()[LHCOpticalFunctionsSet::eLy];

    vector<double> xiValues = ofs.getXiValues();  // local copy made since the TSpline constructor needs non-const parameters
    vector<double> xDValues = ofs.getFcnValues()[LHCOpticalFunctionsSet::exd];
    rpod.s_xi_vs_x_d = make_shared<TSpline3>("", xDValues.data(), xiValues.data(), xiValues.size());

    // calculate auxiliary data
    LHCOpticalFunctionsSet::Kinematics k_in = { 0., 0., 0., 0., 0. };
    LHCOpticalFunctionsSet::Kinematics k_out;
    rpod.optics->transport(k_in, k_out);
    rpod.x0 = k_out.x;
    rpod.y0 = k_out.y;

    doLinearFit(ofs.getXiValues(), ofs.getFcnValues()[LHCOpticalFunctionsSet::exd], rpod.ch0, rpod.ch1);
    rpod.ch0 -= rpod.x0;

    doLinearFit(ofs.getXiValues(), ofs.getFcnValues()[LHCOpticalFunctionsSet::eLx], rpod.la0, rpod.la1);

    // insert record
    const CTPPSDetId rpId(p.first);
    m_rp_optics_.emplace(rpId, std::move(rpod));
  }

  // update settings
  initialized_ = true;
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::doLinearFit(const std::vector<double> &vx, const std::vector<double> &vy, double &b, double &a)
{
  double s_1=0., s_x=0., s_xx=0., s_y=0., s_xy=0.;
  for (unsigned int i = 0; i < vx.size(); ++i)
  {
    s_1 += 1.;
    s_x += vx[i];
    s_xx += vx[i] * vx[i];
    s_y += vy[i];
    s_xy += vx[i] * vy[i];
  }

  const double d = s_xx * s_1 - s_x * s_x;
  a = ( s_1 * s_xy -  s_x * s_y) / d;
  b = (-s_x * s_xy + s_xx * s_y) / d;
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::release()
{
  initialized_ = false;

  m_rp_optics_.clear();
}

//----------------------------------------------------------------------------------------------------

double ProtonReconstructionAlgorithm::ChiSquareCalculator::operator() (const double* parameters) const
{
  // extract proton parameters
  const LHCOpticalFunctionsSet::Kinematics k_in = { 0., parameters[1], parameters[3], parameters[2], parameters[0] };

  // calculate chi^2 by looping over hits
  double s2 = 0.;

  for (const auto &track : *tracks) {
    const CTPPSDetId rpId(track->getRPId());

    // transport proton to the RP
    auto oit = m_rp_optics->find(rpId);
    LHCOpticalFunctionsSet::Kinematics k_out;
    oit->second.optics->transport(k_in, k_out);

    // proton position wrt. beam
    const double x = k_out.x - oit->second.x0;
    const double y = k_out.y - oit->second.y0;

    // calculate chi^2 contributions, convert track data mm --> cm
    const double x_diff_norm = (x - track->getX()*1E-1) / (track->getXUnc()*1E-1);
    const double y_diff_norm = (y - track->getY()*1E-1) / (track->getYUnc()*1E-1);

    // increase chi^2
    s2 += x_diff_norm*x_diff_norm + y_diff_norm*y_diff_norm;
  }

  return s2;
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::reconstructFromMultiRP(
  const reco::ForwardProton::CTPPSLocalTrackLiteRefVector &tracks,
  std::vector<reco::ForwardProton> &output,
  const LHCInfo &lhcInfo, std::ostream& os) const
{
  if (!initialized_)
    return;

  // need at least two tracks
  if (tracks.size() < 2)
    return;

  // make sure optics is available for all tracks
  for (const auto &it : tracks)
  {
    auto oit = m_rp_optics_.find(it->getRPId());
    if (oit == m_rp_optics_.end())
      throw cms::Exception("ProtonReconstructionAlgorithm") << "Optics data not available for RP " <<
        it->getRPId() << ", i.e. " << CTPPSDetId(it->getRPId()) << ".";
  }

  // initial estimate of xi and th_x
  double xi_init = 0., th_x_init = 0.;

  if (useImprovedInitialEstimate_)
  {
    double x_N = tracks[0]->getX()*1E-1,  // conversion: mm --> cm
      x_F = tracks[1]->getX()*1E-1;

    const RPOpticsData &i_N = m_rp_optics_.find(tracks[0]->getRPId())->second,
      &i_F = m_rp_optics_.find(tracks[1]->getRPId())->second;

    const double a = i_F.ch1*i_N.la1 - i_N.ch1*i_F.la1;
    const double b = i_F.ch0*i_N.la1 - i_N.ch0*i_F.la1 + i_F.ch1*i_N.la0 - i_N.ch1*i_F.la0 + x_N*i_F.la1 - x_F*i_N.la1;
    const double c = x_N*i_F.la0 - x_F*i_N.la0 + i_F.ch0*i_N.la0 - i_N.ch0*i_F.la0;
    const double D = b*b - 4.*a*c;

    xi_init = (-b + sqrt(D)) / 2. / a;
    th_x_init = (x_N - i_N.ch0 - i_N.ch1 * xi_init) / (i_N.la0 + i_N.la1 * xi_init);
  } else {
    double s_xi0 = 0., s_1 = 0.;
    for (const auto &track : tracks)
    {
      auto oit = m_rp_optics_.find(track->getRPId());
      double xi = oit->second.s_xi_vs_x_d->Eval(track->getX()*1E-1 + oit->second.x0);  // conversion: mm --> cm

      s_1 += 1.;
      s_xi0 += xi;
    }

    xi_init = s_xi0 / s_1;
  }

  // initial estimate of th_y and vtx_y
  double y[2], v_y[2], L_y[2];
  unsigned int y_idx = 0;
  for (const auto &track : tracks)
  {
    if (y_idx >= 2)
      continue;

    auto oit = m_rp_optics_.find(track->getRPId());

    y[y_idx] = track->getY()*1E-1 - oit->second.s_y_d_vs_xi->Eval(xi_init); // track y: mm --> cm
    v_y[y_idx] = oit->second.s_v_y_vs_xi->Eval(xi_init);
    L_y[y_idx] = oit->second.s_L_y_vs_xi->Eval(xi_init);

    y_idx++;
  }

  double vtx_y_init = 0.;
  double th_y_init = 0.;

  if (fitVtxY_)
  {
    const double det_y = v_y[0] * L_y[1] - L_y[0] * v_y[1];
    vtx_y_init = (L_y[1] * y[0] - L_y[0] * y[1]) / det_y;
    th_y_init = (v_y[0] * y[1] - v_y[1] * y[0]) / det_y;
  } else {
    vtx_y_init = 0.;
    th_y_init = (y[1]/L_y[1] + y[0]/L_y[0]) / 2.;
  }

  unsigned int armId = CTPPSDetId((*tracks.begin())->getRPId()).arm();

  if (verbosity_)
    os << "\n"
      << "ProtonReconstructionAlgorithm::reconstructFromMultiRP(" << armId << ")" << std::endl
      << "    initial estimate: xi_init = " << xi_init << ", th_x_init = " << th_x_init
      << ", th_y_init = " << th_y_init << ", vtx_y_init = " << vtx_y_init << ".";

  // minimisation
  fitter_->Config().ParSettings(0).Set("xi", xi_init, 0.005);
  fitter_->Config().ParSettings(1).Set("th_x", th_x_init, 2E-6);
  fitter_->Config().ParSettings(2).Set("th_y", th_y_init, 2E-6);
  fitter_->Config().ParSettings(3).Set("vtx_y", vtx_y_init, 10E-6);

  if (!fitVtxY_)
    fitter_->Config().ParSettings(3).Fix();

  chiSquareCalculator_->tracks = &tracks;
  chiSquareCalculator_->m_rp_optics = &m_rp_optics_;

  fitter_->FitFCN();
  fitter_->FitFCN();  // second minimisation in case the first one had troubles

  // extract proton parameters
  const ROOT::Fit::FitResult& result = fitter_->Result();
  const double *params = result.GetParams();

  if (verbosity_)
    os << "\n"
      << "xi=" << params[0] << " +- " << result.Error(0)
      << ", th_x=" << params[1] << " +-" << result.Error(1)
      << ", th_y=" << params[2] << " +-" << result.Error(2)
      << ", vtx_y=" << params[3] << " +-" << result.Error(3)
      << ", chiSq = " << result.Chi2();

  // save reco candidate
  using FP = reco::ForwardProton;

  const double sign_z = (armId == 0) ? +1. : -1.;  // CMS convention
  const FP::Point vertex(0., params[3], 0.);
  const double xi = params[0];
  const double th_x = params[1];
  const double th_y = params[2];
  const double cos_th = sqrt(1. - th_x*th_x - th_y*th_y);
  const double p = lhcInfo.energy() * (1. - xi);
  const FP::Vector momentum(
    - p * th_x,   // the signs reflect change LHC --> CMS convention
    + p * th_y,
    sign_z * p * cos_th
  );
  signed int ndf = 2.*tracks.size() - ((fitVtxY_) ? 4. : 3.);

  map<unsigned int, signed int> index_map = {
    {(unsigned int) FP::Index::xi, 0},
    {(unsigned int) FP::Index::th_x, 1},
    {(unsigned int) FP::Index::th_y, 2},
    {(unsigned int) FP::Index::vtx_y, ((fitVtxY_) ? 3 : -1)},
    {(unsigned int) FP::Index::vtx_x, -1},
  };

  FP::CovarianceMatrix cm;
  for (unsigned int i = 0; i < (unsigned int) FP::Index::num_indices; ++i)
  {
    signed int fit_i = index_map[i];

    for (unsigned int j = 0; j < (unsigned int) FP::Index::num_indices; ++j)
    {
      signed int fit_j = index_map[j];

      cm(i, j) = (fit_i >= 0 && fit_j >= 0) ? result.CovMatrix(fit_i, fit_j) : 0.;
    }
  }

  output.emplace_back(result.Chi2(), ndf, vertex, momentum, xi, cm, FP::ReconstructionMethod::multiRP, tracks, result.IsValid());
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::reconstructFromSingleRP(
  const reco::ForwardProton::CTPPSLocalTrackLiteRefVector &tracks,
  std::vector<reco::ForwardProton> &output,
  const LHCInfo &lhcInfo, std::ostream& os) const
{
  if (!initialized_)
    return;

  // make sure optics is available for all tracks
  for (const auto &it : tracks)
  {
    auto oit = m_rp_optics_.find(it->getRPId());
    if (oit == m_rp_optics_.end())
      throw cms::Exception("ProtonReconstructionAlgorithm") << "Optics data not available for RP " << it->getRPId()
        << ", i.e. " << CTPPSDetId(it->getRPId()) << ".";
  }

  // rough estimate of xi and th_y from each track
  for (const auto &track : tracks)
  {
    CTPPSDetId rpId(track->getRPId());

    if (verbosity_)
      os << "\nreconstructFromSingleRP(" << rpId.arm()*100 + rpId.station()*10 + rpId.rp() << ")";

    auto oit = m_rp_optics_.find(track->getRPId());
    const double x_full = track->getX()*1E-1 + oit->second.x0; // conversion mm --> cm
    const double xi = oit->second.s_xi_vs_x_d->Eval(x_full);
    const double L_y = oit->second.s_L_y_vs_xi->Eval(xi);
    const double th_y = track->getY()*1E-1 / L_y; // conversion mm --> cm

    const double ep_x = 1E-6;
    const double dxi_dx = (oit->second.s_xi_vs_x_d->Eval(x_full + ep_x) - xi) / ep_x;
    const double xi_unc = abs(dxi_dx) * track->getXUnc() * 1E-1; // conversion mm --> cm

    const double ep_xi = 1E-4;
    const double dL_y_dxi = ( oit->second.s_L_y_vs_xi->Eval(xi + ep_xi) - L_y ) / ep_xi;
    const double th_y_unc = th_y * sqrt( pow(track->getYUnc() / track->getY(), 2.) + pow(dL_y_dxi * xi_unc / L_y, 2.) );

    if (verbosity_)
      os << "\n    xi = " << xi << " +- " << xi_unc << ", th_y = " << th_y << " +- " << th_y_unc << ".";

    using FP = reco::ForwardProton;

    // save proton candidate

    const double sign_z = (CTPPSDetId(track->getRPId()).arm() == 0) ? +1. : -1.;  // CMS convention
    const FP::Point vertex(0., 0., 0.);
    const double cos_th = sqrt(1. - th_y*th_y);
    const double p = lhcInfo.energy() * (1. - xi);
    const FP::Vector momentum(0., p * th_y, sign_z * p * cos_th);

    FP::CovarianceMatrix cm;
    cm((int)FP::Index::xi, (int)FP::Index::xi) = xi_unc * xi_unc;
    cm((int)FP::Index::th_y, (int)FP::Index::th_y) = th_y_unc * th_y_unc;

    FP::CTPPSLocalTrackLiteRefVector trk;
    trk.push_back( track );
    output.emplace_back(0., 0, vertex, momentum, xi, cm, FP::ReconstructionMethod::singleRP, trk, true);
  }
}

