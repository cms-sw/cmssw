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
#include "DataFormats/ProtonReco/interface/ProtonTrackExtra.h"

#include "TF1.h"

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

ProtonReconstructionAlgorithm::ProtonReconstructionAlgorithm(bool fit_vtx_y, unsigned int verbosity) :
  verbosity_(verbosity),
  fitVtxY_(fit_vtx_y),
  initialized_(false),
  fitter_(std::make_unique<ROOT::Fit::Fitter>()),
  chiSquareCalculator_(std::make_unique<ChiSquareCalculator>())
{
  // initialise fitter
  double pStart[] = { 0, 0, 0, 0 };
  fitter_->SetFCN(4, *chiSquareCalculator_, pStart, 0, true);
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::init(const std::unordered_map<unsigned int, LHCOpticalFunctionsSet> &opticalFunctions)
{
  // reset cache
  release();

  // prepare helper objects
  unique_ptr<TF1> ff(new TF1("ff", "[0] + [1]*x"));

  // build optics data for each object
  for (const auto &p : opticalFunctions)
  {
    const LHCOpticalFunctionsSet &ofs = p.second;

    // make record
    RPOpticsData rpod;
    rpod.optics = &p.second;
    rpod.s_xi_vs_x_d = make_shared<TSpline3>("",
      (double *) ofs.getFcnValues()[LHCOpticalFunctionsSet::exd].data(),
      (double *) ofs.getXiValues().data(), ofs.getXiValues().size());
    rpod.s_y_d_vs_xi = ofs.getSplines()[LHCOpticalFunctionsSet::eyd];
    rpod.s_v_y_vs_xi = ofs.getSplines()[LHCOpticalFunctionsSet::evy];
    rpod.s_L_y_vs_xi = ofs.getSplines()[LHCOpticalFunctionsSet::eLy];

    // calculate auxiliary data
    LHCOpticalFunctionsSet::Kinematics k_in = { 0., 0., 0., 0., 0. };
    LHCOpticalFunctionsSet::Kinematics k_out;
    rpod.optics->transport(k_in, k_out);
    rpod.x0 = k_out.x;
    rpod.y0 = k_out.y;

    unique_ptr<TGraph> g_x_d_vs_xi = make_unique<TGraph>(ofs.getXiValues().size(), ofs.getXiValues().data(),
      ofs.getFcnValues()[LHCOpticalFunctionsSet::exd].data());
    ff->SetParameters(0., 0.);
    g_x_d_vs_xi->Fit(ff.get(), "Q");
    g_x_d_vs_xi->Fit(ff.get(), "Q");
    g_x_d_vs_xi->Fit(ff.get(), "Q");
    rpod.ch0 = ff->GetParameter(0) - rpod.x0;
    rpod.ch1 = ff->GetParameter(1);

    unique_ptr<TGraph> g_L_x_vs_xi = make_unique<TGraph>(ofs.getXiValues().size(), ofs.getXiValues().data(),
      ofs.getFcnValues()[LHCOpticalFunctionsSet::eLx].data());
    ff->SetParameters(0., 0.);
    g_L_x_vs_xi->Fit(ff.get(), "Q");
    g_L_x_vs_xi->Fit(ff.get(), "Q");
    g_L_x_vs_xi->Fit(ff.get(), "Q");
    rpod.la0 = ff->GetParameter(0);
    rpod.la1 = ff->GetParameter(1);

    // insert record
    const CTPPSDetId rpId(p.first);
    m_rp_optics_.emplace(rpId, std::move(rpod));
  }

  // update settings
  initialized_ = true;
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
  double S2 = 0.;

  for (const auto &track : *tracks_)
  {
    const CTPPSDetId rpId(track->getRPId());

    // transport proton to the RP
    auto oit = m_rp_optics_->find(rpId);
    LHCOpticalFunctionsSet::Kinematics k_out;
    oit->second.optics->transport(k_in, k_out);

    // proton position wrt. beam
    const double& x = k_out.x - oit->second.x0;
    const double& y = k_out.y - oit->second.y0;

    // make sure that uncertainties are reasonable
    double x_unc = track->getXUnc();
    if (x_unc < 1E-3)
      x_unc = 40E-3;

    double y_unc = track->getYUnc();
    if (y_unc < 1E-3)
      y_unc = 40E-3;

    // calculate chi^2 contributions, convert track data mm --> m
    const double x_diff_norm = (x - track->getX()*1E-3) / (x_unc*1E-3);
    const double y_diff_norm = (y - track->getY()*1E-3) / (y_unc*1E-3);

    // increase chi^2
    S2 += x_diff_norm*x_diff_norm + y_diff_norm*y_diff_norm;
  }

  return S2;
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::reconstructFromMultiRP(const vector<const CTPPSLocalTrackLite*> &tracks,
  vector<reco::ProtonTrack> &out, const LHCInfo &lhcInfo) const
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

  const bool use_improved_estimate = true;
  if (use_improved_estimate)
  {
    double x_N = 0., x_F = 0.;
    const RPOpticsData *i_N = NULL, *i_F = NULL;
    unsigned int idx = 0;
    for (const auto &track : tracks)
    {
      auto oit = m_rp_optics_.find(track->getRPId());

      if (idx == 0) { x_N = track->getX() * 1E-3; i_N = &oit->second; }
      if (idx == 1) { x_F = track->getX() * 1E-3; i_F = &oit->second; }
      if (idx == 2) break;

      idx++;
    }

    const double a = i_F->ch1*i_N->la1 - i_N->ch1*i_F->la1;
    const double b = i_F->ch0*i_N->la1 - i_N->ch0*i_F->la1 + i_F->ch1*i_N->la0 - i_N->ch1*i_F->la0 + x_N*i_F->la1 - x_F*i_N->la1;
    const double c = x_N*i_F->la0 - x_F*i_N->la0 + i_F->ch0*i_N->la0 - i_N->ch0*i_F->la0;
    const double D = b*b - 4.*a*c;

    xi_init = (-b + sqrt(D)) / 2. / a;
    th_x_init = (x_N - i_N->ch0 - i_N->ch1 * xi_init) / (i_N->la0 + i_N->la1 * xi_init);
  } else {
    double S_xi0 = 0., S_1 = 0.;
    for (const auto &track : tracks)
    {
      auto oit = m_rp_optics_.find(track->getRPId());
      double xi = oit->second.s_xi_vs_x_d->Eval(track->getX() * 1E-3 + oit->second.x0);  // conversion: mm --> m

      S_1 += 1.;
      S_xi0 += xi;
    }

    xi_init = S_xi0 / S_1;
  }

  // initial estimate of th_y and vtx_y
  double y[2], v_y[2], L_y[2];
  unsigned int y_idx = 0;
  for (const auto &track : tracks)
  {
    if (y_idx >= 2)
      continue;

    auto oit = m_rp_optics_.find(track->getRPId());

    y[y_idx] = track->getY()*1E-3 - oit->second.s_y_d_vs_xi->Eval(xi_init); // track y: mm --> m
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

  if (verbosity_)
  {
    unsigned int armId = CTPPSDetId((*tracks.begin())->getRPId()).arm();
    printf("* ProtonReconstructionAlgorithm::reconstructFromMultiRP(%u)\n", armId);
    printf("    initial estimate: xi_init = %f, th_x_init = %E, th_y_init = %E, vtx_y_init = %E\n", xi_init, th_x_init, th_y_init, vtx_y_init);
  }

  // minimisation
  fitter_->Config().ParSettings(0).Set("xi", xi_init, 0.005);
  fitter_->Config().ParSettings(1).Set("th_x", th_x_init, 2E-6);
  fitter_->Config().ParSettings(2).Set("th_y", th_y_init, 2E-6);
  fitter_->Config().ParSettings(3).Set("vtx_y", vtx_y_init, 10E-6);

  if (!fitVtxY_)
    fitter_->Config().ParSettings(3).Fix();

  chiSquareCalculator_->tracks_ = &tracks;
  chiSquareCalculator_->m_rp_optics_ = &m_rp_optics_;

  fitter_->FitFCN();
  fitter_->FitFCN();  // second minimisation in case the first one had troubles

  // extract proton parameters
  const ROOT::Fit::FitResult& result = fitter_->Result();
  const double *params = result.GetParams();

  if (verbosity_)
    edm::LogInfo("ProtonReconstructionAlgorithm")
      << "at reconstructed level: "
      << "xi=" << params[0] << ", "
      << "theta_x=" << params[1] << ", "
      << "theta_y=" << params[2] << ", "
      << "vertex_y=" << params[3] << "\n";

  if (verbosity_)
    printf("    fit: xi = %f, th_x = %E, th_y = %E, vtx_y = %E, chiSq = %.0f\n", params[0], params[1], params[2], params[3], result.Chi2());

  // save reco candidate
  reco::ProtonTrack pt;
  // TODO
  /*
  pt.method = reco::ProtonTrack::rmMultiRP;
  pt.halfCrossingAngleSector45 = halfCrossingAngleSector45_;
  pt.halfCrossingAngleSector56 = halfCrossingAngleSector56_;

  pt.fitChiSq = result.Chi2();
  pt.fitNDF = 2.*tracks.size() - ((fitVtxY_) ? 4. : 3.);
  pt.lhcSector = (CTPPSDetId(tracks[0]->getRPId()).arm() == 0) ? reco::ProtonTrack::sector45 : reco::ProtonTrack::sector56;

  pt.setVertex(Local3DPoint(0., params[3]*1E3, 0.));  // vertext in mm

  const double p_nom = 6500.;  // GeV
  const double p = p_nom *  (1. - params[0]);
  const double th_x = params[1];
  const double th_y = params[2];
  const double cos_th = sqrt(1. - th_x*th_x - th_y*th_y);
  const double sign_z_lhc = (pt.lhcSector == reco::ProtonTrack::sector45) ? -1. : +1.;

  pt.setXi(params[0]);

  pt.setDirection(Local3DVector(
    - p * th_x,   // the signs reflect change LHC --> CMS convention
    + p * th_y,
    - sign_z_lhc * p * cos_th
  ));

  for (const auto &track : tracks)
    pt.contributingRPIds.insert(track->getRPId());

  const double max_chi_sq = 2.;

  const bool valid = result.IsValid() && pt.fitChiSq < max_chi_sq;
  pt.setValid(valid);

  if (verbosity && !valid)
    printf("WARNING: invalid proton-reconstruction fit (result.IsValid = %u, fitChiSq = %.2E).\n", result.IsValid(), pt.fitChiSq);
  */

  out.push_back(move(pt));
}

//----------------------------------------------------------------------------------------------------

void ProtonReconstructionAlgorithm::reconstructFromSingleRP(const vector<const CTPPSLocalTrackLite*> &tracks,
  vector<reco::ProtonTrack> &out, const LHCInfo &lhcInfo) const
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
    unsigned int decRPId = rpId.arm()*100 + rpId.station()*10 + rpId.rp();

    if (verbosity_)
      printf("* reconstructFromSingleRP(%u)\n", decRPId);

    auto oit = m_rp_optics_.find(track->getRPId());
    const double xi = oit->second.s_xi_vs_x_d->Eval(track->getX() * 1E-3 + oit->second.x0); // conversions mm --> m
    const double L_y = oit->second.s_L_y_vs_xi->Eval(xi);
    const double th_y = track->getY() * 1E-3 / L_y;

    if (verbosity_)
      printf("    xi = %f, th_y = %E\n", xi, th_y);

    using ex = reco::ProtonTrackExtra;

    // save proton candidate
    auto lhcSector = (CTPPSDetId(track->getRPId()).arm() == 0) ? ex::LHCSector::sector45 : ex::LHCSector::sector56;
    ex::RPList rpList;
    rpList.insert(track->getRPId());
    reco::ProtonTrackExtra ptExtra(true, ex::ReconstructionMethod::singleRP, lhcSector, rpList);

    const double sign_z_lhc = (lhcSector == ex::LHCSector::sector45) ? -1. : +1.;
    const reco::Track::Point vtx(0., 0., 0.);
    const reco::Track::Vector direction(0., th_y, - sign_z_lhc);
    const double p = lhcInfo.energy() * (1. - xi);
    reco::ProtonTrack pt(0., 0, vtx, p * direction, xi);  // TODO: covariance matrix

    // TODO: fix this
    //pt.setProtonTrackExtra(ptExtra);

    out.push_back(move(pt));
  }
}
