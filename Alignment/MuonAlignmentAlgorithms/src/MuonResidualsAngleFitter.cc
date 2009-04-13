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
    if (fitter->inRange(residual)) {
      const double xangle = (*resiter)[MuonResidualsAngleFitter::kXAngle];
      const double yangle = (*resiter)[MuonResidualsAngleFitter::kYAngle];

      double center = 0.;
      center += par[MuonResidualsAngleFitter::kAngle];
      center += par[MuonResidualsAngleFitter::kXControl] * xangle;
      center += par[MuonResidualsAngleFitter::kYControl] * yangle;

      if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
	fval += -log(MuonResidualsFitter_pureGaussian(residual, center, par[MuonResidualsAngleFitter::kSigma]));
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
	fval += -log(MuonResidualsFitter_powerLawTails(residual, center, par[MuonResidualsAngleFitter::kSigma], par[MuonResidualsAngleFitter::kGamma]));
      }
      else { assert(false); }
    }
  }
}

bool MuonResidualsAngleFitter::fit(double v1) {
  initialize_table();  // if not already initialized
  m_goodfit = false;
  m_minResidual = m_maxResidual = 0.;

  double sum_x = 0.;
  double sum_xx = 0.;
  int N = 0;
  int N_bin[4];
  for (int i = 0;  i < 4;  i++) N_bin[i] = 0;

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residual = (*resiter)[kResidual];
    const double xangle = (*resiter)[kXAngle];
    const double yangle = (*resiter)[kYAngle];

    if (fabs(residual) < 0.1) {  // truncate at 100 mrad
      sum_x += residual;
      sum_xx += residual*residual;
      N++;
    }

    int index = 0;
    if (xangle > 0.) index += 1;
    if (yangle > 0.) index += 2;
    N_bin[index]++;
  }

  bool enough_in_every_bin = true;
  for (int i = 0;  i < 4;  i++) {
    if (N_bin[i] < m_minHitsPerRegion) enough_in_every_bin = false;
  }

  if ((m_minHitsPerRegion > 0  &&  !enough_in_every_bin)  ||  (m_minHitsPerRegion <= 0  &&  N <= 10)) return false;

  // truncated mean and stdev to seed the fit
  m_mean = sum_x/double(N);
  m_stdev = sqrt(sum_xx/double(N) - pow(sum_x/double(N), 2));

  // refine the standard deviation calculation
  sum_x = 0.;
  sum_xx = 0.;
  N = 0;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residual = (*resiter)[kResidual];
    if (m_mean - 1.5*m_stdev < residual  &&  residual < m_mean + 1.5*m_stdev) {
      sum_x += residual;
      sum_xx += residual*residual;
      N++;
    }
  }
  m_mean = sum_x/double(N);
  m_stdev = sqrt(sum_xx/double(N) - pow(sum_x/double(N), 2));

  sum_x = 0.;
  sum_xx = 0.;
  N = 0;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residual = (*resiter)[kResidual];
    if (m_mean - 1.5*m_stdev < residual  &&  residual < m_mean + 1.5*m_stdev) {
      sum_x += residual;
      sum_xx += residual*residual;
      N++;
    }
  }
  m_mean = sum_x/double(N);
  m_stdev = sqrt(sum_xx/double(N) - pow(sum_x/double(N), 2));

  m_minResidual = m_mean - 10.0*m_stdev;
  m_maxResidual = m_mean + 10.0*m_stdev;

  std::vector<int> parNum;
  std::vector<std::string> parName;
  std::vector<double> start;
  std::vector<double> step;
  std::vector<double> low;
  std::vector<double> high;

  parNum.push_back(kAngle);     parName.push_back(std::string("angle"));     start.push_back(m_mean);   step.push_back(0.1);               low.push_back(m_mean-m_stdev);      high.push_back(m_mean+m_stdev);
  parNum.push_back(kXControl);  parName.push_back(std::string("xcontrol"));  start.push_back(0.);       step.push_back(0.1);               low.push_back(-2.*m_stdev);         high.push_back(2.*m_stdev);
  parNum.push_back(kYControl);  parName.push_back(std::string("ycontrol"));  start.push_back(0.);       step.push_back(0.1);               low.push_back(-2.*m_stdev);         high.push_back(2.*m_stdev);
  parNum.push_back(kSigma);     parName.push_back(std::string("sigma"));     start.push_back(m_stdev);  step.push_back(0.1*m_stdev);       low.push_back(0.00001);             high.push_back(3.*m_stdev);
  if (residualsModel() != kPureGaussian) {
  parNum.push_back(kGamma);     parName.push_back(std::string("gamma"));     start.push_back(m_stdev);  step.push_back(0.1*m_stdev);       low.push_back(0.00001);             high.push_back(3.*m_stdev);
  }

  return dofit(&MuonResidualsAngleFitter_FCN, parNum, parName, start, step, low, high);
}

void MuonResidualsAngleFitter::plot(double v1, std::string name, TFileDirectory *dir) {
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

  TF1 *narrowed_fit = NULL;
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

    if (inRange(corrected_residual)) {
      xcontrol_hist->Fill(xangle, (raw_residual - yangle_correction) * 1000.);
      ycontrol_hist->Fill(yangle, (raw_residual - xangle_correction) * 1000.);
    }
  }
}

double MuonResidualsAngleFitter::redchi2(double v1, std::string name, TFileDirectory *dir, bool write, int bins, double low, double high) {
  std::stringstream histname;
  histname << name << "_norm";

  double scale_factor = double(numResiduals()) * (high - low)/double(bins);
  double sigma = value(kSigma);
  double gamma = 0.;
  if (residualsModel() != MuonResidualsFitter::kPureGaussian) {
    gamma = value(kGamma);
  }

  TH1F *hist = NULL;
  if (write) hist = dir->make<TH1F>(histname.str().c_str(), histname.str().c_str(), bins, low, high);
  else hist = new TH1F(histname.str().c_str(), histname.str().c_str(), bins, low, high);

  histname << "fit";
  TF1 *func = NULL;
  if (residualsModel() == kPureGaussian) {
    func = new TF1(histname.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, low, high, 3);
    func->SetParameters(scale_factor, 0., 1.);
  }
  else if (residualsModel() == kPowerLawTails) {
    func = new TF1(histname.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, low, high, 4);
    func->SetParameters(scale_factor, 0., 1., gamma/sigma);
  }
  else { assert(false); }
  if (write) func->Write();

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double raw_residual = (*resiter)[kResidual];
    const double xangle = (*resiter)[kXAngle];
    const double yangle = (*resiter)[kYAngle];

    double correction = value(kAngle);
    double xangle_correction = value(kXControl) * xangle;
    double yangle_correction = value(kYControl) * yangle;
    double scale = value(kSigma);

    hist->Fill((raw_residual - correction - xangle_correction - yangle_correction)/scale);
  } // end loop over residuals

  double chi2 = 0.;
  int ndof = 0;
  for (int i = 1;  i <= bins;  i++) {
    double angle = hist->GetBinCenter(i);
    double histvalue = hist->GetBinContent(i);
    double histerror = hist->GetBinError(i);
    
    if (histvalue != 0.) {
      double fitvalue = 0.;

      if (residualsModel() == kPureGaussian) {
	fitvalue = scale_factor * MuonResidualsFitter_pureGaussian(angle, 0., 1.);
	ndof++;
      }
      else if (residualsModel() == kPowerLawTails) {
	fitvalue = scale_factor * MuonResidualsFitter_powerLawTails(angle, 0., 1., gamma/sigma);
	ndof++;
      }
      else { assert(false); }

      chi2 += pow(histvalue - fitvalue, 2) / histerror;
    }
  } // end loop over histogram bins
  ndof -= npar();

  if (!write) {
    delete hist;
    delete func;
  }

  if (ndof <= 0) return -1.;
  else return chi2/double(ndof);
}
