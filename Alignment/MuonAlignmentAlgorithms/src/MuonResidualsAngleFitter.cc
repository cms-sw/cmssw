#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsAngleFitter.h"

static TMinuit *MuonResidualsAngleFitter_TMinuit;

void MuonResidualsAngleFitter::inform(TMinuit *tMinuit) {
  MuonResidualsAngleFitter_TMinuit = tMinuit;
}

void MuonResidualsAngleFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) {
  MuonResidualsFitterFitInfo *fitinfo = (MuonResidualsFitterFitInfo*)(MuonResidualsAngleFitter_TMinuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double*>::const_iterator resiter = fitter->residuals_begin();  resiter != fitter->residuals_end();  ++resiter) {
    const double residual = (*resiter)[MuonResidualsAngleFitter::kResidual];
    const double xangle = (*resiter)[MuonResidualsAngleFitter::kXAngle];
    const double yangle = (*resiter)[MuonResidualsAngleFitter::kYAngle];
    
    double center = 0.;
    center += par[MuonResidualsAngleFitter::kAngle];
    center += par[MuonResidualsAngleFitter::kXControl] * xangle;
    center += par[MuonResidualsAngleFitter::kYControl] * yangle;
    
    if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
      fval += -MuonResidualsFitter_logPureGaussian(residual, center, par[MuonResidualsAngleFitter::kSigma]);
    }
    else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
      fval += -MuonResidualsFitter_logPowerLawTails(residual, center, par[MuonResidualsAngleFitter::kSigma], par[MuonResidualsAngleFitter::kGamma]);
    }
    else if (fitter->residualsModel() == MuonResidualsFitter::kROOTVoigt) {
      fval += -MuonResidualsFitter_logROOTVoigt(residual, center, par[MuonResidualsAngleFitter::kSigma], par[MuonResidualsAngleFitter::kGamma]);
    }
    else if (fitter->residualsModel() == MuonResidualsFitter::kGaussPowerTails) {
      fval += -MuonResidualsFitter_logGaussPowerTails(residual, center, par[MuonResidualsAngleFitter::kSigma]);
    }
    else { assert(false); }
  }
}

bool MuonResidualsAngleFitter::fit(Alignable *ali) {
  initialize_table();  // if not already initialized

  double sum_x = 0.;
  double sum_xx = 0.;
  int N = 0;

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residual = (*resiter)[kResidual];
//     const double xangle = (*resiter)[kXAngle];
//     const double yangle = (*resiter)[kYAngle];

    if (fabs(residual) < 0.1) {  // truncate at 100 mrad
      sum_x += residual;
      sum_xx += residual*residual;
      N++;
    }
  }

  if (N < m_minHits) return false;

  // truncated mean and stdev to seed the fit
  double mean = sum_x/double(N);
  double stdev = sqrt(sum_xx/double(N) - pow(sum_x/double(N), 2));

  // refine the standard deviation calculation
  sum_x = 0.;
  sum_xx = 0.;
  N = 0;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residual = (*resiter)[kResidual];
    if (mean - 1.5*stdev < residual  &&  residual < mean + 1.5*stdev) {
      sum_x += residual;
      sum_xx += residual*residual;
      N++;
    }
  }
  mean = sum_x/double(N);
  stdev = sqrt(sum_xx/double(N) - pow(sum_x/double(N), 2));

  sum_x = 0.;
  sum_xx = 0.;
  N = 0;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residual = (*resiter)[kResidual];
    if (mean - 1.5*stdev < residual  &&  residual < mean + 1.5*stdev) {
      sum_x += residual;
      sum_xx += residual*residual;
      N++;
    }
  }
  mean = sum_x/double(N);
  stdev = sqrt(sum_xx/double(N) - pow(sum_x/double(N), 2));

  std::vector<int> parNum;
  std::vector<std::string> parName;
  std::vector<double> start;
  std::vector<double> step;
  std::vector<double> low;
  std::vector<double> high;

  parNum.push_back(kAngle);     parName.push_back(std::string("angle"));     start.push_back(mean);     step.push_back(0.1);               low.push_back(0.);    high.push_back(0.);
  parNum.push_back(kXControl);  parName.push_back(std::string("xcontrol"));  start.push_back(0.);       step.push_back(0.1);               low.push_back(0.);    high.push_back(0.);
  parNum.push_back(kYControl);  parName.push_back(std::string("ycontrol"));  start.push_back(0.);       step.push_back(0.1);               low.push_back(0.);    high.push_back(0.);
  parNum.push_back(kSigma);     parName.push_back(std::string("sigma"));     start.push_back(stdev);  step.push_back(0.1*stdev);       low.push_back(0.);    high.push_back(0.);
  if (residualsModel() != kPureGaussian && residualsModel() != kGaussPowerTails) {
  parNum.push_back(kGamma);     parName.push_back(std::string("gamma"));     start.push_back(stdev);  step.push_back(0.1*stdev);         low.push_back(0.);    high.push_back(0.);
  }

  return dofit(&MuonResidualsAngleFitter_FCN, parNum, parName, start, step, low, high);
}

double MuonResidualsAngleFitter::plot(std::string name, TFileDirectory *dir, Alignable *ali) {
  std::stringstream raw_name, narrowed_name, xcontrol_name, ycontrol_name;
  raw_name << name << "_raw";
  narrowed_name << name << "_narrowed";
  xcontrol_name << name << "_xcontrol";
  ycontrol_name << name << "_ycontrol";

  TH1F *raw_hist = dir->make<TH1F>(raw_name.str().c_str(), (raw_name.str() + std::string(" (mrad)")).c_str(), 100, -100., 100.);
  TH1F *narrowed_hist = dir->make<TH1F>(narrowed_name.str().c_str(), (narrowed_name.str() + std::string(" (mrad)")).c_str(), 100, -100., 100.);
  TProfile *xcontrol_hist = dir->make<TProfile>(xcontrol_name.str().c_str(), (xcontrol_name.str() + std::string(" (mrad)")).c_str(), 100, -1., 1.);
  TProfile *ycontrol_hist = dir->make<TProfile>(ycontrol_name.str().c_str(), (ycontrol_name.str() + std::string(" (mrad)")).c_str(), 100, -1., 1.);

  narrowed_name << "fit";
  xcontrol_name << "fit";
  ycontrol_name << "fit";

  double scale_factor = double(numResiduals()) * (100. - -100.)/100;   // (max - min)/nbins

  TF1 *narrowed_fit = nullptr;
  if (residualsModel() == kPureGaussian) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, -100., 100., 3);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000.);
    narrowed_fit->Write();
  }
  else if (residualsModel() == kPowerLawTails) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, -100., 100., 4);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000., value(kGamma) * 1000.);
    narrowed_fit->Write();
  }
  else if (residualsModel() == kROOTVoigt) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, -100., 100., 4);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000., value(kGamma) * 1000.);
    narrowed_fit->Write();
  }
  else if (residualsModel() == kGaussPowerTails) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, -100., 100., 3);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000.);
    narrowed_fit->Write();
  }

  TF1 *xcontrol_fit = new TF1(xcontrol_name.str().c_str(), "[0]+x*[1]", -1., 1.);
  xcontrol_fit->SetParameters(value(kAngle) * 1000., value(kXControl) * 1000.);
  xcontrol_fit->Write();

  TF1 *ycontrol_fit = new TF1(ycontrol_name.str().c_str(), "[0]+x*[1]", -1., 1.);
  ycontrol_fit->SetParameters(value(kAngle) * 1000., value(kYControl) * 1000.);
  ycontrol_fit->Write();

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double raw_residual = (*resiter)[kResidual];
    const double xangle = (*resiter)[kXAngle];
    const double yangle = (*resiter)[kYAngle];

    double xangle_correction = value(kXControl) * xangle;
    double yangle_correction = value(kYControl) * yangle;
    double corrected_residual = raw_residual - xangle_correction - yangle_correction;

    raw_hist->Fill(raw_residual * 1000.);
    narrowed_hist->Fill(corrected_residual * 1000.);

    xcontrol_hist->Fill(xangle, (raw_residual - yangle_correction) * 1000.);
    ycontrol_hist->Fill(yangle, (raw_residual - xangle_correction) * 1000.);
  }

  return 0.;
}
