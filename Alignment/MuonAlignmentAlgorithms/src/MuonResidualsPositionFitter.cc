#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsPositionFitter.h"

static TMinuit *MuonResidualsPositionFitter_TMinuit;

void MuonResidualsPositionFitter::inform(TMinuit *tMinuit) {
  MuonResidualsPositionFitter_TMinuit = tMinuit;
}

void MuonResidualsPositionFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) {
  MuonResidualsFitterFitInfo *fitinfo = (MuonResidualsFitterFitInfo*)(MuonResidualsPositionFitter_TMinuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double*>::const_iterator resiter = fitter->residuals_begin();  resiter != fitter->residuals_end();  ++resiter) {
    const double residual = (*resiter)[MuonResidualsPositionFitter::kResidual];
    if (fitter->inRange(residual)) {
      const double angleerror = (*resiter)[MuonResidualsPositionFitter::kAngleError];
      const double trackangle = (*resiter)[MuonResidualsPositionFitter::kTrackAngle];
      const double trackposition = (*resiter)[MuonResidualsPositionFitter::kTrackPosition];

      double center = 0.;
      center += par[MuonResidualsPositionFitter::kPosition];
      center += par[MuonResidualsPositionFitter::kZpos] * trackangle;
      center += par[MuonResidualsPositionFitter::kPhiz] * trackposition;
      center += par[MuonResidualsPositionFitter::kScattering] * angleerror;

      if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
	fval += -log(MuonResidualsFitter_pureGaussian(residual, center, par[MuonResidualsPositionFitter::kSigma]));
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
	fval += -log(MuonResidualsFitter_powerLawTails(residual, center, par[MuonResidualsPositionFitter::kSigma], par[MuonResidualsPositionFitter::kGamma]));
      }
      else { assert(false); }
    }
  }
}

bool MuonResidualsPositionFitter::fit() {
  initialize_table();  // if not already initialized
  m_goodfit = false;
  m_minResidual = m_maxResidual = 0.;

  double sum_x = 0.;
  double sum_xx = 0.;
  int N = 0;
  int N_bin[8];
  for (int i = 0;  i < 8;  i++) N_bin[i] = 0;

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residual = (*resiter)[kResidual];
    const double angleerror = (*resiter)[kAngleError];
    const double trackangle = (*resiter)[kTrackAngle];
    const double trackposition = (*resiter)[kTrackPosition];

    if (fabs(residual) < 10.) {  // truncate at 100 mm
      sum_x += residual;
      sum_xx += residual*residual;
      N++;
    }

    int index = 0;
    if (angleerror > 0.) index += 1;
    if (trackangle > 0.) index += 2;
    if (trackposition > 0.) index += 4;
    N_bin[index]++;
  }

  bool enough_in_every_bin = true;
  for (int i = 0;  i < 8;  i++) {
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

  parNum.push_back(kPosition);    parName.push_back(std::string("position"));  	 start.push_back(m_mean);   step.push_back(0.1);             low.push_back(m_mean-m_stdev);    high.push_back(m_mean+m_stdev);
  parNum.push_back(kZpos);        parName.push_back(std::string("zpos"));      	 start.push_back(0.);       step.push_back(0.1);             low.push_back(-2.*m_stdev);       high.push_back(2.*m_stdev);
  parNum.push_back(kPhiz);        parName.push_back(std::string("phiz"));      	 start.push_back(0.);       step.push_back(0.1);             low.push_back(-2.*m_stdev);       high.push_back(2.*m_stdev);
  parNum.push_back(kScattering);  parName.push_back(std::string("scattering"));  start.push_back(0.);       step.push_back(0.1*1000.);       low.push_back(-2.*m_stdev*1000.); high.push_back(2.*m_stdev*1000.);
  parNum.push_back(kSigma);       parName.push_back(std::string("sigma"));       start.push_back(m_stdev);  step.push_back(0.1*m_stdev);     low.push_back(0.001);             high.push_back(3.*m_stdev);
  if (residualsModel() != kPureGaussian) {
  parNum.push_back(kGamma);       parName.push_back(std::string("gamma"));       start.push_back(m_stdev);  step.push_back(0.1*m_stdev);     low.push_back(0.001);             high.push_back(3.*m_stdev);
  }

  return dofit(&MuonResidualsPositionFitter_FCN, parNum, parName, start, step, low, high);
}

void MuonResidualsPositionFitter::plot(std::string name, TFileDirectory *dir) {
  std::stringstream raw_name, narrowed_name, angleerror_name, trackangle_name, trackposition_name;
  raw_name << name << "_raw";
  narrowed_name << name << "_narrowed";
  angleerror_name << name << "_angleerror";
  trackangle_name << name << "_trackangle";
  trackposition_name << name << "_trackposition";

  TH1F *raw_hist = dir->make<TH1F>(raw_name.str().c_str(), (raw_name.str() + std::string(" (mm)")).c_str(), 100, -100., 100.);
  TH1F *narrowed_hist = dir->make<TH1F>(narrowed_name.str().c_str(), (narrowed_name.str() + std::string(" (mm)")).c_str(), 100, -100., 100.);
  TProfile *angleerror_hist = dir->make<TProfile>(angleerror_name.str().c_str(), (angleerror_name.str() + std::string(" (mm)")).c_str(), 100, -20., 20.);
  TProfile *trackangle_hist = dir->make<TProfile>(trackangle_name.str().c_str(), (trackangle_name.str() + std::string(" (mm)")).c_str(), 100, -0.5, 0.5);
  TProfile *trackposition_hist = dir->make<TProfile>(trackposition_name.str().c_str(), (trackposition_name.str() + std::string(" (mm)")).c_str(), 100, -300., 300.);

  narrowed_name << "fit";
  angleerror_name << "fit";
  trackangle_name << "fit";
  trackposition_name << "fit";

  double scale_factor = double(numResiduals()) * (100. - -100.)/100;   // (max - min)/nbins

  TF1 *narrowed_fit = NULL;
  if (residualsModel() == kPureGaussian) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, -100., 100., 3);
    narrowed_fit->SetParameters(scale_factor, value(kPosition) * 10., value(kSigma) * 10.);
    narrowed_fit->Write();
  }
  else if (residualsModel() == kPowerLawTails) {
    narrowed_fit = new TF1(narrowed_name.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, -100., 100., 4);
    narrowed_fit->SetParameters(scale_factor, value(kPosition) * 10., value(kSigma) * 10., value(kGamma) * 10.);
    narrowed_fit->Write();
  }

  TF1 *angleerror_fit = new TF1(angleerror_name.str().c_str(), "[0]+x*[1]", -20., 20.);
  angleerror_fit->SetParameters(value(kPosition) * 10., value(kScattering) * 10./1000.);
  angleerror_fit->Write();

  TF1 *trackangle_fit = new TF1(trackangle_name.str().c_str(), "[0]+x*[1]", -0.5, 0.5);
  trackangle_fit->SetParameters(value(kPosition) * 10., value(kZpos) * 10.);
  trackangle_fit->Write();

  TF1 *trackposition_fit = new TF1(trackposition_name.str().c_str(), "[0]+x*[1]", -300., 300.);
  trackposition_fit->SetParameters(value(kPosition) * 10., value(kPhiz) * 10.);
  trackposition_fit->Write();

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
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

    if (inRange(corrected_residual)) {
      angleerror_hist->Fill(angleerror*1000., (raw_residual - trackangle_correction - trackposition_correction) * 10.);
      trackangle_hist->Fill(trackangle, (raw_residual - angleerror_correction - trackposition_correction) * 10.);
      trackposition_hist->Fill(trackposition, (raw_residual - angleerror_correction - trackangle_correction) * 10.);
    }
  }
}

double MuonResidualsPositionFitter::redchi2(std::string name, TFileDirectory *dir, bool write, int bins, double low, double high) {
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
    const double angleerror = (*resiter)[kAngleError];
    const double trackangle = (*resiter)[kTrackAngle];
    const double trackposition = (*resiter)[kTrackPosition];

    double correction = value(kPosition);
    double angleerror_correction = value(kScattering) * angleerror;
    double trackangle_correction = value(kZpos) * trackangle;
    double trackposition_correction = value(kPhiz) * trackposition;
    double scale = value(kSigma);

    hist->Fill((raw_residual - correction - angleerror_correction - trackangle_correction - trackposition_correction)/scale);
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
