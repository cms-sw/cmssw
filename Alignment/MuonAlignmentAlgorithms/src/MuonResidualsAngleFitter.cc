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
    const double qoverpt = (*resiter)[MuonResidualsAngleFitter::kQoverPt];

    double center = 0.;
    center += par[MuonResidualsAngleFitter::kAngle];
    center += par[MuonResidualsAngleFitter::kBfield] * qoverpt;

    if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
      fval += -log(MuonResidualsFitter_pureGaussian(residual, center, par[MuonResidualsAngleFitter::kSigma]));
    }
    else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
      fval += -log(MuonResidualsFitter_powerLawTails(residual, center, par[MuonResidualsAngleFitter::kSigma], par[MuonResidualsAngleFitter::kGamma]));
    }
    else { assert(false); }
  }
}

bool MuonResidualsAngleFitter::fit() {
  initialize_table();  // if not already initialized
  m_goodfit = false;
  m_minResidual = m_maxResidual = 0.;

  double sum_x = 0.;
  double sum_xx = 0.;
  int N = 0;
  int N_bin[2];
  for (int i = 0;  i < 2;  i++) N_bin[i] = 0;

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residual = (*resiter)[kResidual];
    const double qoverpt = (*resiter)[kQoverPt];

    if (fabs(residual) < 0.1) {  // truncate at 100 mrad
      sum_x += residual;
      sum_xx += residual*residual;
      N++;
    }

    int index = 0;
    if (qoverpt > 0.) index += 1;
    N_bin[index]++;
  }

  bool enough_in_every_bin = true;
  for (int i = 0;  i < 2;  i++) {
    if (N_bin[i] < m_minHitsPerRegion) enough_in_every_bin = false;
  }

  if ((m_minHitsPerRegion > 0  &&  !enough_in_every_bin)  ||  (m_minHitsPerRegion <= 0  &&  N <= 10)) return false;

  // truncated mean and stdev to seed the fit
  double mean = sum_x/double(N);
  double stdev = sqrt(sum_xx/double(N) - pow(sum_x/double(N), 2));
  m_minResidual = mean - 10.*stdev;
  m_maxResidual = mean + 10.*stdev;

  std::vector<int> parNum;
  std::vector<std::string> parName;
  std::vector<double> start;
  std::vector<double> step;
  std::vector<double> low;
  std::vector<double> high;

  parNum.push_back(kAngle);     parName.push_back(std::string("angle"));     start.push_back(mean);   step.push_back(0.1);             low.push_back(mean-stdev);      high.push_back(mean+stdev);
  parNum.push_back(kBfield);    parName.push_back(std::string("bfield"));    start.push_back(0.);     step.push_back(0.1*stdev/0.05);  low.push_back(-2.*stdev/0.05);  high.push_back(2.*stdev/0.05);
  parNum.push_back(kSigma);     parName.push_back(std::string("sigma"));     start.push_back(stdev);  step.push_back(0.1*stdev);       low.push_back(0.00001);         high.push_back(3.*stdev);
  if (residualsModel() != kPureGaussian) {
  parNum.push_back(kGamma);     parName.push_back(std::string("gamma"));     start.push_back(stdev);  step.push_back(0.1*stdev);       low.push_back(0.00001);         high.push_back(3.*stdev);
  }

  return dofit(&MuonResidualsAngleFitter_FCN, parNum, parName, start, step, low, high);
}

void MuonResidualsAngleFitter::plot(std::string name, TFileDirectory *dir) {
  std::stringstream raw_name, narrowed_name, qoverpt_name;
  raw_name << name << "_raw";
  narrowed_name << name << "_narrowed";
  qoverpt_name << name << "_qoverpt";

  TH1F *raw_hist = dir->make<TH1F>(raw_name.str().c_str(), (raw_name.str() + std::string(" (mm)")).c_str(), 100, -300., 300.);
  TH1F *narrowed_hist = dir->make<TH1F>(narrowed_name.str().c_str(), (narrowed_name.str() + std::string(" (mm)")).c_str(), 100, -300., 300.);
  TProfile *qoverpt_hist = dir->make<TProfile>(qoverpt_name.str().c_str(), (qoverpt_name.str() + std::string(" (mm)")).c_str(), 100, -0.05, 0.05);

  narrowed_name << "fit";
  qoverpt_name << "fit";

  double scale_factor = double(numResiduals()) * (300. - -300.)/100;   // (max - min)/nbins

  TF1 *narrowed_fit = NULL;
  if (residualsModel() == kPureGaussian) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, -300., 300., 3);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000.);
    narrowed_fit->Write();
  }
  else if (residualsModel() == kPowerLawTails) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, -300., 300., 4);
    narrowed_fit->SetParameters(scale_factor, value(kAngle) * 1000., value(kSigma) * 1000., value(kGamma) * 1000.);
    narrowed_fit->Write();
  }

  TF1 *qoverpt_fit = new TF1(qoverpt_name.str().c_str(), "[0]+x*[1]", -0.05, 0.05);
  qoverpt_fit->SetParameters(value(kAngle) * 1000., value(kBfield) * 1000.);
  qoverpt_fit->Write();

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double raw_residual = (*resiter)[kResidual];
    const double qoverpt = (*resiter)[kQoverPt];
    double qoverpt_correction = value(kBfield) * qoverpt;
    double corrected_residual = raw_residual - qoverpt_correction;

    raw_hist->Fill(raw_residual * 1000.);
    narrowed_hist->Fill(corrected_residual * 1000.);
    if (inRange(corrected_residual)) {
      qoverpt_hist->Fill(qoverpt, raw_residual * 1000.);
    }
  }
}

double MuonResidualsAngleFitter::redchi2(std::string name, TFileDirectory *dir, bool write, int bins, double low, double high) {
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
    const double qoverpt = (*resiter)[kQoverPt];

    double correction = value(kAngle);
    double qoverpt_correction = value(kBfield) * qoverpt;
    double scale = value(kSigma);

    hist->Fill((raw_residual - correction - qoverpt_correction)/scale);
  } // end loop over residuals

  double chi2 = 0.;
  int ndof = 0;
  for (int i = 1;  i <= bins;  i++) {
    double position = hist->GetBinCenter(i);
    double histvalue = hist->GetBinContent(i);
    double histerror = hist->GetBinError(i);
    
    if (histvalue != 0.) {
      double fitvalue = 0.;

      if (residualsModel() == kPureGaussian) {
	fitvalue = scale_factor * MuonResidualsFitter_pureGaussian(position, 0., 1.);
	ndof++;
      }
      else if (residualsModel() == kPowerLawTails) {
	fitvalue = scale_factor * MuonResidualsFitter_powerLawTails(position, 0., 1., gamma/sigma);
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
