#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsBfieldAngleFitter.h"

static TMinuit *MuonResidualsBfieldAngleFitter_TMinuit;

void MuonResidualsBfieldAngleFitter::inform(TMinuit *tMinuit) { MuonResidualsBfieldAngleFitter_TMinuit = tMinuit; }

void MuonResidualsBfieldAngleFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) {
  MuonResidualsFitterFitInfo *fitinfo =
      (MuonResidualsFitterFitInfo *)(MuonResidualsBfieldAngleFitter_TMinuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double *>::const_iterator resiter = fitter->residuals_begin(); resiter != fitter->residuals_end();
       ++resiter) {
    const double residual = (*resiter)[MuonResidualsBfieldAngleFitter::kResidual];
    const double qoverpt = (*resiter)[MuonResidualsBfieldAngleFitter::kQoverPt];
    const double qoverpz = (*resiter)[MuonResidualsBfieldAngleFitter::kQoverPz];

    double center = 0.;
    center += par[MuonResidualsBfieldAngleFitter::kAngle];
    center += par[MuonResidualsBfieldAngleFitter::kBfrompt] * qoverpt;
    center += par[MuonResidualsBfieldAngleFitter::kBfrompz] * qoverpz;
    center += par[MuonResidualsBfieldAngleFitter::kdEdx] * (1. / qoverpt / qoverpt + 1. / qoverpz / qoverpz) *
              (qoverpt > 0. ? 1. : -1.);

    if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
      fval += -MuonResidualsFitter_logPureGaussian(residual, center, par[MuonResidualsBfieldAngleFitter::kSigma]);
    } else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
      fval += -MuonResidualsFitter_logPowerLawTails(
          residual, center, par[MuonResidualsBfieldAngleFitter::kSigma], par[MuonResidualsBfieldAngleFitter::kGamma]);
    } else if (fitter->residualsModel() == MuonResidualsFitter::kROOTVoigt) {
      fval += -MuonResidualsFitter_logROOTVoigt(
          residual, center, par[MuonResidualsBfieldAngleFitter::kSigma], par[MuonResidualsBfieldAngleFitter::kGamma]);
    } else if (fitter->residualsModel() == MuonResidualsFitter::kGaussPowerTails) {
      fval += -MuonResidualsFitter_logGaussPowerTails(residual, center, par[MuonResidualsBfieldAngleFitter::kSigma]);
    } else {
      assert(false);
    }
  }
}

bool MuonResidualsBfieldAngleFitter::fit(Alignable *ali) {
  initialize_table();  // if not already initialized

  double sum_x = 0.;
  double sum_xx = 0.;
  int N = 0;

  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double residual = (*resiter)[kResidual];
    //     const double qoverpt = (*resiter)[kQoverPt];

    if (fabs(residual) < 0.1) {  // truncate at 100 mrad
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

  parNum.push_back(kAngle);
  parName.push_back(std::string("angle"));
  start.push_back(mean);
  step.push_back(0.1);
  low.push_back(0.);
  high.push_back(0.);
  parNum.push_back(kBfrompt);
  parName.push_back(std::string("bfrompt"));
  start.push_back(0.);
  step.push_back(0.1 * stdev / 0.05);
  low.push_back(0.);
  high.push_back(0.);
  parNum.push_back(kBfrompz);
  parName.push_back(std::string("bfrompz"));
  start.push_back(0.);
  step.push_back(0.1 * stdev / 0.05);
  low.push_back(0.);
  high.push_back(0.);
  parNum.push_back(kdEdx);
  parName.push_back(std::string("dEdx"));
  start.push_back(0.);
  step.push_back(0.1 * stdev / 0.05);
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

  return dofit(&MuonResidualsBfieldAngleFitter_FCN, parNum, parName, start, step, low, high);
}

double MuonResidualsBfieldAngleFitter::plot(std::string name, TFileDirectory *dir, Alignable *ali) {
  std::stringstream raw_name, narrowed_name, qoverpt_name, qoverpz_name, psquared_name;
  raw_name << name << "_raw";
  narrowed_name << name << "_narrowed";
  qoverpt_name << name << "_qoverpt";
  qoverpz_name << name << "_qoverpz";
  psquared_name << name << "_psquared";

  TH1F *raw_hist =
      dir->make<TH1F>(raw_name.str().c_str(), (raw_name.str() + std::string(" (mrad)")).c_str(), 100, -100., 100.);
  TH1F *narrowed_hist = dir->make<TH1F>(
      narrowed_name.str().c_str(), (narrowed_name.str() + std::string(" (mrad)")).c_str(), 100, -100., 100.);
  TProfile *qoverpt_hist = dir->make<TProfile>(
      qoverpt_name.str().c_str(), (qoverpt_name.str() + std::string(" (mrad)")).c_str(), 100, -0.05, 0.05);
  TProfile *qoverpz_hist = dir->make<TProfile>(
      qoverpz_name.str().c_str(), (qoverpz_name.str() + std::string(" (mrad)")).c_str(), 100, -0.05, 0.05);
  TProfile *psquared_hist = dir->make<TProfile>(
      psquared_name.str().c_str(), (psquared_name.str() + std::string(" (mrad)")).c_str(), 100, -0.05, 0.05);

  narrowed_name << "fit";
  qoverpt_name << "fit";
  qoverpz_name << "fit";
  psquared_name << "fit";

  double scale_factor = double(numResiduals()) * (100. - -100.) / 100;  // (max - min)/nbins

  TF1 *narrowed_fit = nullptr;
  if (residualsModel() == kPureGaussian) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, -100., 100., 3);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000.);
    narrowed_fit->Write();
  } else if (residualsModel() == kPowerLawTails) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, -100., 100., 4);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000., value(kGamma) * 1000.);
    narrowed_fit->Write();
  } else if (residualsModel() == kROOTVoigt) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, -100., 100., 4);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000., value(kGamma) * 1000.);
    narrowed_fit->Write();
  } else if (residualsModel() == kGaussPowerTails) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, -100., 100., 3);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000.);
    narrowed_fit->Write();
  }

  TF1 *qoverpt_fit = new TF1(qoverpt_name.str().c_str(), "[0]+x*[1]", -0.05, 0.05);
  qoverpt_fit->SetParameters(value(kAngle) * 1000., value(kBfrompt) * 1000.);
  qoverpt_fit->Write();

  TF1 *qoverpz_fit = new TF1(qoverpz_name.str().c_str(), "[0]+x*[1]", -0.05, 0.05);
  qoverpz_fit->SetParameters(value(kAngle) * 1000., value(kBfrompz) * 1000.);
  qoverpz_fit->Write();

  TF1 *psquared_fit = new TF1(psquared_name.str().c_str(), "[0]+[1]*x**2", -0.05, 0.05);
  psquared_fit->SetParameters(value(kAngle) * 1000., value(kdEdx) * 1000.);
  psquared_fit->Write();

  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double raw_residual = (*resiter)[kResidual];
    const double qoverpt = (*resiter)[kQoverPt];
    const double qoverpz = (*resiter)[kQoverPz];
    const double psquared = (1. / qoverpt / qoverpt + 1. / qoverpz / qoverpz) * (qoverpt > 0. ? 1. : -1.);

    double qoverpt_correction = value(kBfrompt) * qoverpt;
    double qoverpz_correction = value(kBfrompz) * qoverpz;
    double dEdx_correction = value(kdEdx) * psquared;
    double corrected_residual = raw_residual - qoverpt_correction - qoverpz_correction - dEdx_correction;

    raw_hist->Fill(raw_residual * 1000.);
    narrowed_hist->Fill(corrected_residual * 1000.);

    qoverpt_hist->Fill(qoverpt, (raw_residual - qoverpz_correction - dEdx_correction) * 1000.);
    qoverpz_hist->Fill(qoverpz, (raw_residual - qoverpt_correction - dEdx_correction) * 1000.);
    psquared_hist->Fill(psquared, (raw_residual - qoverpt_correction - qoverpz_correction) * 1000.);
  }

  return 0.;
}
