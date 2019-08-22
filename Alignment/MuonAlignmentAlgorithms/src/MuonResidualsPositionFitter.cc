#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsPositionFitter.h"

static TMinuit *MuonResidualsPositionFitter_TMinuit;

void MuonResidualsPositionFitter::inform(TMinuit *tMinuit) { MuonResidualsPositionFitter_TMinuit = tMinuit; }

void MuonResidualsPositionFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) {
  MuonResidualsFitterFitInfo *fitinfo =
      (MuonResidualsFitterFitInfo *)(MuonResidualsPositionFitter_TMinuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double *>::const_iterator resiter = fitter->residuals_begin(); resiter != fitter->residuals_end();
       ++resiter) {
    const double residual = (*resiter)[MuonResidualsPositionFitter::kResidual];
    const double angleerror = (*resiter)[MuonResidualsPositionFitter::kAngleError];
    const double trackangle = (*resiter)[MuonResidualsPositionFitter::kTrackAngle];
    const double trackposition = (*resiter)[MuonResidualsPositionFitter::kTrackPosition];

    double center = 0.;
    center += par[MuonResidualsPositionFitter::kPosition];
    center += par[MuonResidualsPositionFitter::kZpos] * trackangle;
    center += par[MuonResidualsPositionFitter::kPhiz] * trackposition;
    center += par[MuonResidualsPositionFitter::kScattering] * angleerror;

    if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
      fval += -MuonResidualsFitter_logPureGaussian(residual, center, par[MuonResidualsPositionFitter::kSigma]);
    } else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
      fval += -MuonResidualsFitter_logPowerLawTails(
          residual, center, par[MuonResidualsPositionFitter::kSigma], par[MuonResidualsPositionFitter::kGamma]);
    } else if (fitter->residualsModel() == MuonResidualsFitter::kROOTVoigt) {
      fval += -MuonResidualsFitter_logROOTVoigt(
          residual, center, par[MuonResidualsPositionFitter::kSigma], par[MuonResidualsPositionFitter::kGamma]);
    } else if (fitter->residualsModel() == MuonResidualsFitter::kGaussPowerTails) {
      fval += -MuonResidualsFitter_logGaussPowerTails(residual, center, par[MuonResidualsPositionFitter::kSigma]);
    } else {
      assert(false);
    }
  }
}

bool MuonResidualsPositionFitter::fit(Alignable *ali) {
  initialize_table();  // if not already initialized

  double sum_x = 0.;
  double sum_xx = 0.;
  int N = 0;

  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double residual = (*resiter)[kResidual];
    //    const double angleerror = (*resiter)[kAngleError];
    //    const double trackangle = (*resiter)[kTrackAngle];
    //    const double trackposition = (*resiter)[kTrackPosition];

    if (fabs(residual) < 10.) {  // truncate at 100 mm
      sum_x += residual;
      sum_xx += residual * residual;
      N++;
    }
  }

  if (N < m_minHits)
    return false;

  // truncated mean and stdev to seed the fit
  double mean = sum_x / double(N);
  double stdev = sqrt(sum_xx / double(N) - pow(sum_x / double(N), 2));

  // refine the standard deviation calculation
  sum_x = 0.;
  sum_xx = 0.;
  N = 0;
  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double residual = (*resiter)[kResidual];
    if (mean - 1.5 * stdev < residual && residual < mean + 1.5 * stdev) {
      sum_x += residual;
      sum_xx += residual * residual;
      N++;
    }
  }
  mean = sum_x / double(N);
  stdev = sqrt(sum_xx / double(N) - pow(sum_x / double(N), 2));

  sum_x = 0.;
  sum_xx = 0.;
  N = 0;
  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double residual = (*resiter)[kResidual];
    if (mean - 1.5 * stdev < residual && residual < mean + 1.5 * stdev) {
      sum_x += residual;
      sum_xx += residual * residual;
      N++;
    }
  }
  mean = sum_x / double(N);
  stdev = sqrt(sum_xx / double(N) - pow(sum_x / double(N), 2));

  std::vector<int> parNum;
  std::vector<std::string> parName;
  std::vector<double> start;
  std::vector<double> step;
  std::vector<double> low;
  std::vector<double> high;

  parNum.push_back(kPosition);
  parName.push_back(std::string("position"));
  start.push_back(mean);
  step.push_back(0.1);
  low.push_back(0.);
  high.push_back(0.);
  parNum.push_back(kZpos);
  parName.push_back(std::string("zpos"));
  start.push_back(0.);
  step.push_back(0.1);
  low.push_back(0.);
  high.push_back(0.);
  parNum.push_back(kPhiz);
  parName.push_back(std::string("phiz"));
  start.push_back(0.);
  step.push_back(0.1);
  low.push_back(0.);
  high.push_back(0.);
  parNum.push_back(kScattering);
  parName.push_back(std::string("scattering"));
  start.push_back(0.);
  step.push_back(0.1 * 1000.);
  low.push_back(0.);
  high.push_back(0.);
  parNum.push_back(kSigma);
  parName.push_back(std::string("sigma"));
  start.push_back(stdev);
  step.push_back(0.1 * stdev);
  low.push_back(0.);
  high.push_back(0.);
  if (residualsModel() != kPureGaussian && residualsModel() != kGaussPowerTails) {
    parNum.push_back(kGamma);
    parName.push_back(std::string("gamma"));
    start.push_back(stdev);
    step.push_back(0.1 * stdev);
    low.push_back(0.);
    high.push_back(0.);
  }

  return dofit(&MuonResidualsPositionFitter_FCN, parNum, parName, start, step, low, high);
}

double MuonResidualsPositionFitter::plot(std::string name, TFileDirectory *dir, Alignable *ali) {
  std::stringstream raw_name, narrowed_name, angleerror_name, trackangle_name, trackposition_name;
  raw_name << name << "_raw";
  narrowed_name << name << "_narrowed";
  angleerror_name << name << "_angleerror";
  trackangle_name << name << "_trackangle";
  trackposition_name << name << "_trackposition";

  TH1F *raw_hist =
      dir->make<TH1F>(raw_name.str().c_str(), (raw_name.str() + std::string(" (mm)")).c_str(), 100, -100., 100.);
  TH1F *narrowed_hist = dir->make<TH1F>(
      narrowed_name.str().c_str(), (narrowed_name.str() + std::string(" (mm)")).c_str(), 100, -100., 100.);
  TProfile *angleerror_hist = dir->make<TProfile>(
      angleerror_name.str().c_str(), (angleerror_name.str() + std::string(" (mm)")).c_str(), 100, -30., 30.);
  TProfile *trackangle_hist = dir->make<TProfile>(
      trackangle_name.str().c_str(), (trackangle_name.str() + std::string(" (mm)")).c_str(), 100, -0.5, 0.5);
  TProfile *trackposition_hist = dir->make<TProfile>(
      trackposition_name.str().c_str(), (trackposition_name.str() + std::string(" (mm)")).c_str(), 100, -300., 300.);

  angleerror_hist->SetAxisRange(-100., 100., "Y");
  trackangle_hist->SetAxisRange(-10., 10., "Y");
  trackposition_hist->SetAxisRange(-10., 10., "Y");

  narrowed_name << "fit";
  angleerror_name << "fit";
  trackangle_name << "fit";
  trackposition_name << "fit";

  double scale_factor = double(numResiduals()) * (100. - -100.) / 100;  // (max - min)/nbins

  TF1 *narrowed_fit = nullptr;
  if (residualsModel() == kPureGaussian) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, -100., 100., 3);
    narrowed_fit->SetParameters(scale_factor, value(kPosition) * 10., value(kSigma) * 10.);
    narrowed_fit->Write();
  } else if (residualsModel() == kPowerLawTails) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, -100., 100., 4);
    narrowed_fit->SetParameters(scale_factor, value(kPosition) * 10., value(kSigma) * 10., value(kGamma) * 10.);
    narrowed_fit->Write();
  } else if (residualsModel() == kROOTVoigt) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, -100., 100., 4);
    narrowed_fit->SetParameters(scale_factor, value(kPosition) * 10., value(kSigma) * 10., value(kGamma) * 10.);
    narrowed_fit->Write();
  } else if (residualsModel() == kGaussPowerTails) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, -100., 100., 3);
    narrowed_fit->SetParameters(scale_factor, value(kPosition) * 10., value(kSigma) * 10.);
    narrowed_fit->Write();
  }

  TF1 *angleerror_fit = new TF1(angleerror_name.str().c_str(), "[0]+x*[1]", -30., 30.);
  angleerror_fit->SetParameters(value(kPosition) * 10., value(kScattering) * 10. / 1000.);
  angleerror_fit->Write();

  TF1 *trackangle_fit = new TF1(trackangle_name.str().c_str(), "[0]+x*[1]", -0.5, 0.5);
  trackangle_fit->SetParameters(value(kPosition) * 10., value(kZpos) * 10.);
  trackangle_fit->Write();

  TF1 *trackposition_fit = new TF1(trackposition_name.str().c_str(), "[0]+x*[1]", -300., 300.);
  trackposition_fit->SetParameters(value(kPosition) * 10., value(kPhiz) * 10.);
  trackposition_fit->Write();

  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double raw_residual = (*resiter)[kResidual];
    const double angleerror = (*resiter)[kAngleError];
    const double trackangle = (*resiter)[kTrackAngle];
    const double trackposition = (*resiter)[kTrackPosition];

    double angleerror_correction = value(kScattering) * angleerror;
    double trackangle_correction = value(kZpos) * trackangle;
    double trackposition_correction = value(kPhiz) * trackposition;

    double corrected_residual = raw_residual - angleerror_correction - trackangle_correction - trackposition_correction;

    raw_hist->Fill(raw_residual * 10.);
    narrowed_hist->Fill(corrected_residual * 10.);

    angleerror_hist->Fill(angleerror * 1000., (raw_residual - trackangle_correction - trackposition_correction) * 10.);
    trackangle_hist->Fill(trackangle, (raw_residual - angleerror_correction - trackposition_correction) * 10.);
    trackposition_hist->Fill(trackposition, (raw_residual - angleerror_correction - trackangle_correction) * 10.);
  }

  return 0.;
}
