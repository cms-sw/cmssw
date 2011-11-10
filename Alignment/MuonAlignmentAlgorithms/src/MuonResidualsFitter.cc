#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"
#include <fstream>
#include "TMath.h"

// all global variables begin with "MuonResidualsFitter_" to avoid
// namespace clashes (that is, they do what would ordinarily be done
// with a class structure, but Minuit requires them to be global)

const double MuonResidualsFitter_gsbinsize = 0.01;
const double MuonResidualsFitter_tsbinsize = 0.1;
const int MuonResidualsFitter_numgsbins = 500;
const int MuonResidualsFitter_numtsbins = 500;

bool MuonResidualsFitter_table_initialized = false;
double MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins][MuonResidualsFitter_numtsbins];

static TMinuit* MuonResidualsFitter_TMinuit;

// fit function
double MuonResidualsFitter_logPureGaussian(double residual, double center, double sigma) {
  sigma = fabs(sigma);
  return (-pow(residual - center, 2) / 2. / sigma / sigma) + log(1. / sqrt(2.*M_PI) / sigma);
}

// TF1 interface version
Double_t MuonResidualsFitter_pureGaussian_TF1(Double_t *xvec, Double_t *par) {
  return par[0] * exp(MuonResidualsFitter_logPureGaussian(xvec[0], par[1], par[2]));
}

double MuonResidualsFitter_compute_log_convolution(double toversigma, double gammaoversigma, double max, double step, double power) {
  if (gammaoversigma == 0.) return (-toversigma*toversigma/2.) - log(sqrt(2*M_PI));

  double sum = 0.;
  double uplus = 0.;
  double integrandplus_last = 0.;
  double integrandminus_last = 0.;
  for (double inc = 0.;  uplus < max;  inc += step) {
    double uplus_last = uplus;
    uplus = pow(inc, power);

    double integrandplus = exp(-pow(uplus - toversigma, 2)/2.) / (uplus*uplus/gammaoversigma + gammaoversigma) / M_PI / sqrt(2.*M_PI);
    double integrandminus = exp(-pow(-uplus - toversigma, 2)/2.) / (uplus*uplus/gammaoversigma + gammaoversigma) / M_PI / sqrt(2.*M_PI);

    sum += integrandplus * (uplus - uplus_last);
    sum += 0.5 * fabs(integrandplus_last - integrandplus) * (uplus - uplus_last);

    sum += integrandminus * (uplus - uplus_last);
    sum += 0.5 * fabs(integrandminus_last - integrandminus) * (uplus - uplus_last);

    integrandplus_last = integrandplus;
    integrandminus_last = integrandminus;
  }
  return log(sum);
}

// fit function
double MuonResidualsFitter_logPowerLawTails(double residual, double center, double sigma, double gamma) {
  sigma = fabs(sigma);
  double gammaoversigma = fabs(gamma / sigma);
  double toversigma = fabs((residual - center) / sigma);

  int gsbin0 = int(floor(gammaoversigma / MuonResidualsFitter_gsbinsize));
  int gsbin1 = int(ceil(gammaoversigma / MuonResidualsFitter_gsbinsize));
  int tsbin0 = int(floor(toversigma / MuonResidualsFitter_tsbinsize));
  int tsbin1 = int(ceil(toversigma / MuonResidualsFitter_tsbinsize));

  bool gsisbad = (gsbin0 >= MuonResidualsFitter_numgsbins  ||  gsbin1 >= MuonResidualsFitter_numgsbins);
  bool tsisbad = (tsbin0 >= MuonResidualsFitter_numtsbins  ||  tsbin1 >= MuonResidualsFitter_numtsbins);

  if (gsisbad  ||  tsisbad) {
    return log(gammaoversigma/M_PI) - log(toversigma*toversigma + gammaoversigma*gammaoversigma) - log(sigma);
  }
  else {
    double val00 = MuonResidualsFitter_lookup_table[gsbin0][tsbin0];
    double val01 = MuonResidualsFitter_lookup_table[gsbin0][tsbin1];
    double val10 = MuonResidualsFitter_lookup_table[gsbin1][tsbin0];
    double val11 = MuonResidualsFitter_lookup_table[gsbin1][tsbin1];

    double val0 = val00 + ((toversigma / MuonResidualsFitter_tsbinsize) - tsbin0) * (val01 - val00);
    double val1 = val10 + ((toversigma / MuonResidualsFitter_tsbinsize) - tsbin0) * (val11 - val10);
    
    double val = val0 + ((gammaoversigma / MuonResidualsFitter_gsbinsize) - gsbin0) * (val1 - val0);
    
    return val - log(sigma);
  }
}

// TF1 interface version
Double_t MuonResidualsFitter_powerLawTails_TF1(Double_t *xvec, Double_t *par) {
  return par[0] * exp(MuonResidualsFitter_logPowerLawTails(xvec[0], par[1], par[2], par[3]));
}

double MuonResidualsFitter_logROOTVoigt(double residual, double center, double sigma, double gamma) {
  return log(TMath::Voigt(residual - center, fabs(sigma), fabs(gamma)*2.));
}

Double_t MuonResidualsFitter_ROOTVoigt_TF1(Double_t *xvec, Double_t *par) {
  return par[0] * TMath::Voigt(xvec[0] - par[1], fabs(par[2]), fabs(par[3])*2.);
}

double MuonResidualsFitter_logGaussPowerTails(double residual, double center, double sigma) {
  double x = residual-center;
  double s = fabs(sigma);
  double m = 2*s;
  double a = pow(m,4)*exp(-2);
  double n = sqrt(2*M_PI)*s*erf(sqrt(2))+2*a*pow(m,-3)/3;

  if (fabs(x)<m) return -x*x/(2*s*s) - log(n);
  else return log(a) -4*log(fabs(x)) - log(n);
}

Double_t MuonResidualsFitter_GaussPowerTails_TF1(Double_t *xvec, Double_t *par) {
  return par[0] * exp(MuonResidualsFitter_logGaussPowerTails(xvec[0], par[1], par[2]));
}

double MuonResidualsFitter_integrate_pureGaussian(double low, double high, double center, double sigma) {
  return (erf((high + center) / sqrt(2.) / sigma) - erf((low + center) / sqrt(2.) / sigma)) * exp(0.5/sigma/sigma) / 2.;
}

void MuonResidualsFitter::initialize_table() {
  if (MuonResidualsFitter_table_initialized  ||  residualsModel() != kPowerLawTails) return;
  MuonResidualsFitter_table_initialized = true;

  std::ifstream convolution_table("convolution_table.txt");
  if (convolution_table.is_open()) {
    int numgsbins = 0;
    int numtsbins = 0;
    double tsbinsize = 0.;
    double gsbinsize = 0.;

    convolution_table >> numgsbins >> numtsbins >> tsbinsize >> gsbinsize;
    if (numgsbins != MuonResidualsFitter_numgsbins  ||  numtsbins != MuonResidualsFitter_numtsbins  ||  
	tsbinsize != MuonResidualsFitter_tsbinsize  ||  gsbinsize != MuonResidualsFitter_gsbinsize) {
      throw cms::Exception("MuonResidualsFitter") << "convolution_table.txt has the wrong bin width/bin size.  Throw it away and let the fitter re-create the file." << std::endl;
    }

    for (int gsbin = 0;  gsbin < MuonResidualsFitter_numgsbins;  gsbin++) {
      for (int tsbin = 0;  tsbin < MuonResidualsFitter_numtsbins;  tsbin++) {
	int read_gsbin = 0;
	int read_tsbin = 0;
	double value = 0.;

	convolution_table >> read_gsbin >> read_tsbin >> value;
	if (read_gsbin != gsbin  ||  read_tsbin != tsbin) {
	  throw cms::Exception("MuonResidualsFitter") << "convolution_table.txt is out of order.  Throw it away and let the fitter re-create the file." << std::endl;
	}

	MuonResidualsFitter_lookup_table[gsbin][tsbin] = value;
      }
    }

    convolution_table.close();
  }

  else {
    std::ofstream convolution_table2("convolution_table.txt");

    if (!convolution_table2.is_open()) {
      throw cms::Exception("MuonResidualsFitter") << "Couldn't write to file convolution_table.txt" << std::endl;
    }

    convolution_table2 << MuonResidualsFitter_numgsbins << " " << MuonResidualsFitter_numtsbins << " " << MuonResidualsFitter_tsbinsize << " " << MuonResidualsFitter_gsbinsize << std::endl;

    edm::LogWarning("MuonResidualsFitter") << "Initializing convolution look-up table (takes a few minutes)..." << std::endl;
    std::cout << "Initializing convolution look-up table (takes a few minutes)..." << std::endl;

    for (int gsbin = 0;  gsbin < MuonResidualsFitter_numgsbins;  gsbin++) {
      double gammaoversigma = double(gsbin) * MuonResidualsFitter_gsbinsize;

      std::cout << "    gsbin " << gsbin << "/" << MuonResidualsFitter_numgsbins << std::endl;

      for (int tsbin = 0;  tsbin < MuonResidualsFitter_numtsbins;  tsbin++) {
	double toversigma = double(tsbin) * MuonResidualsFitter_tsbinsize;

	// 1e-6 errors (out of a value of ~0.01) with max=100, step=0.001, power=4 (max=1000 does a little better with the tails)
	MuonResidualsFitter_lookup_table[gsbin][tsbin] = MuonResidualsFitter_compute_log_convolution(toversigma, gammaoversigma);

	// <10% errors with max=20, step=0.005, power=4 (faster computation for testing)
	// MuonResidualsFitter_lookup_table[gsbin][tsbin] = MuonResidualsFitter_compute_log_convolution(toversigma, gammaoversigma, 100., 0.005, 4.);

	convolution_table2 << gsbin << " " << tsbin << " " << MuonResidualsFitter_lookup_table[gsbin][tsbin] << std::endl;
      }
    }

    convolution_table2.close();

    edm::LogWarning("MuonResidualsFitter") << "Initialization done!" << std::endl;
    std::cout << "Initialization done!" << std::endl;
  }
}

bool MuonResidualsFitter::dofit(void (*fcn)(int&,double*,double&,double*,int), std::vector<int> &parNum, std::vector<std::string> &parName, std::vector<double> &start, std::vector<double> &step, std::vector<double> &low, std::vector<double> &high) {
  MuonResidualsFitterFitInfo *fitinfo = new MuonResidualsFitterFitInfo(this);

  MuonResidualsFitter_TMinuit = new TMinuit(npar());
  MuonResidualsFitter_TMinuit->SetPrintLevel(m_printLevel);
  MuonResidualsFitter_TMinuit->SetObjectFit(fitinfo);
  MuonResidualsFitter_TMinuit->SetFCN(fcn);
  inform(MuonResidualsFitter_TMinuit);

  std::vector<int>::const_iterator iNum = parNum.begin();
  std::vector<std::string>::const_iterator iName = parName.begin();
  std::vector<double>::const_iterator istart = start.begin();
  std::vector<double>::const_iterator istep = step.begin();
  std::vector<double>::const_iterator ilow = low.begin();
  std::vector<double>::const_iterator ihigh = high.begin();

  for (; iNum != parNum.end();  ++iNum, ++iName, ++istart, ++istep, ++ilow, ++ihigh) {
    MuonResidualsFitter_TMinuit->DefineParameter(*iNum, iName->c_str(), *istart, *istep, *ilow, *ihigh);
    if (fixed(*iNum)) MuonResidualsFitter_TMinuit->FixParameter(*iNum);
  }

  double arglist[10];
  int ierflg;
  int smierflg; //second MIGRAD ierflg

  // chi^2 errors should be 1.0, log-likelihood should be 0.5
  for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
  arglist[0] = 0.5;
  ierflg = 0;
  smierflg = 0;
  MuonResidualsFitter_TMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
  if (ierflg != 0) { delete MuonResidualsFitter_TMinuit; delete fitinfo; return false; }

  // set strategy = 2 (more refined fits)
  for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
  arglist[0] = m_strategy;
  ierflg = 0;
  MuonResidualsFitter_TMinuit->mnexcm("SET STR", arglist, 1, ierflg);
  if (ierflg != 0) { delete MuonResidualsFitter_TMinuit; delete fitinfo; return false; }

  bool try_again = false;

  // minimize
  for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
  arglist[0] = 50000;
  ierflg = 0;
  MuonResidualsFitter_TMinuit->mnexcm("MIGRAD", arglist, 1, ierflg);
  if (ierflg != 0) try_again = true;

  // just once more, if needed (using the final Minuit parameters from the failed fit; often works)
  if (try_again) {
    // minimize
    for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
    arglist[0] = 50000;
    MuonResidualsFitter_TMinuit->mnexcm("MIGRAD", arglist, 1, smierflg);
  }

  Double_t fmin, fedm, errdef;
  Int_t npari, nparx, istat;
  MuonResidualsFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);

  if (istat != 3) {
    for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
    ierflg = 0;
    MuonResidualsFitter_TMinuit->mnexcm("HESSE", arglist, 0, ierflg);
  }

  // read-out the results
  m_loglikelihood = -fmin;

  m_value.clear();
  m_error.clear();
  for (int i = 0;  i < npar();  i++) {
    double v, e;
    MuonResidualsFitter_TMinuit->GetParameter(i, v, e);
    m_value.push_back(v);
    m_error.push_back(e);
  }

  delete MuonResidualsFitter_TMinuit;
  delete fitinfo;
  if (smierflg != 0) return false;
  return true;
}

void MuonResidualsFitter::write(FILE *file, int which) {
  long rows = numResiduals();
  int cols = ndata();
  int whichcopy = which;

  fwrite(&rows, sizeof(long), 1, file);
  fwrite(&cols, sizeof(int), 1, file);
  fwrite(&whichcopy, sizeof(int), 1, file);

  double *likeAChecksum = new double[cols];
  double *likeAChecksum2 = new double[cols];
  for (int i = 0;  i < cols;  i++) {
    likeAChecksum[i] = 0.;
    likeAChecksum2[i] = 0.;
  }

  for (std::vector<double*>::const_iterator residual = residuals_begin();  residual != residuals_end();  ++residual) {
    fwrite((*residual), sizeof(double), cols, file);

    for (int i = 0;  i < cols;  i++) {
      if (fabs((*residual)[i]) > likeAChecksum[i]) likeAChecksum[i] = fabs((*residual)[i]);
      if (fabs((*residual)[i]) < likeAChecksum2[i]) likeAChecksum2[i] = fabs((*residual)[i]);
    }
  } // end loop over residuals

  // the idea is that mal-formed doubles are likely to be huge values (or tiny values)
  // because the exponent gets screwed up; we want to check for that
  fwrite(likeAChecksum, sizeof(double), cols, file);
  fwrite(likeAChecksum2, sizeof(double), cols, file);

  delete [] likeAChecksum;
  delete [] likeAChecksum2;
}

void MuonResidualsFitter::read(FILE *file, int which) {
  long rows = -100;
  int cols = -100;
  int readwhich = -100;

  fread(&rows, sizeof(long), 1, file);
  fread(&cols, sizeof(int), 1, file);
  fread(&readwhich, sizeof(int), 1, file);

  if (cols != ndata()  ||  rows < 0  ||  readwhich != which) throw cms::Exception("MuonResidualsFitter") << "temporary file is corrupted (which = " << which << " readwhich = " << readwhich << " rows = " << rows << " cols = " << cols << ")" << std::endl;

  double *likeAChecksum = new double[cols];
  double *likeAChecksum2 = new double[cols];
  for (int i = 0;  i < cols;  i++) {
    likeAChecksum[i] = 0.;
    likeAChecksum2[i] = 0.;
  }

  for (long row = 0;  row < rows;  row++) {
    double *residual = new double[cols];
    fread(residual, sizeof(double), cols, file);
    fill(residual);

    for (int i = 0;  i < cols;  i++) {
      if (fabs(residual[i]) > likeAChecksum[i]) likeAChecksum[i] = fabs(residual[i]);
      if (fabs(residual[i]) < likeAChecksum2[i]) likeAChecksum2[i] = fabs(residual[i]);
    }
  } // end loop over records in file

  double *readChecksum = new double[cols];
  double *readChecksum2 = new double[cols];
  fread(readChecksum, sizeof(double), cols, file);
  fread(readChecksum2, sizeof(double), cols, file);

  for (int i = 0;  i < cols;  i++) {
    if (fabs(likeAChecksum[i] - readChecksum[i]) > 1e-10  ||  fabs(1./likeAChecksum2[i] - 1./readChecksum2[i]) > 1e10) {
      throw cms::Exception("MuonResidualsFitter") << "temporary file is corrupted (which = " << which << " rows = " << rows << " likeAChecksum " << likeAChecksum[i] << " != readChecksum " << readChecksum[i] << " " << " likeAChecksum2 " << likeAChecksum2[i] << " != readChecksum2 " << readChecksum2[i] << ")" << std::endl;
    }
  }

  delete [] likeAChecksum;
  delete [] likeAChecksum2;
}

void MuonResidualsFitter::plotsimple(std::string name, TFileDirectory *dir, int which, double multiplier) {
   double window = 100.;
   if (which == 0) window = 2.*30.;
   else if (which == 1) window = 2.*30.;
   else if (which == 2) window = 2.*20.;
   else if (which == 3) window = 2.*50.;

   TH1F *hist = dir->make<TH1F>(name.c_str(), "", 200, -window, window);

   for (std::vector<double*>::const_iterator r = residuals_begin();  r != residuals_end();  ++r) {
      hist->Fill(multiplier * (*r)[which]);
   }
}

void MuonResidualsFitter::plotweighted(std::string name, TFileDirectory *dir, int which, int whichredchi2, double multiplier) {
   double window = 100.;
   if (which == 0) window = 2.*30.;
   else if (which == 1) window = 2.*30.;
   else if (which == 2) window = 2.*20.;
   else if (which == 3) window = 2.*50.;

   TH1F *hist = dir->make<TH1F>(name.c_str(), "", 200, -window, window);

   for (std::vector<double*>::const_iterator r = residuals_begin();  r != residuals_end();  ++r) {
      double weight = 1./(*r)[whichredchi2];
      if (TMath::Prob(1./weight*12, 12) < 0.99) {
	 hist->Fill(multiplier * (*r)[which], weight);
      }
   }
}
