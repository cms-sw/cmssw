// $Id: MuonResidualsFitter.cc,v 1.12 2011/10/12 23:44:11 khotilov Exp $

#ifdef STANDALONE_FITTER
#include "MuonResidualsFitter.h"
#else
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"
#endif

#include <fstream>
#include <set>
#include "TMath.h"
#include "TH1.h"
#include "TF1.h"
#include "TRobustEstimator.h"

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
double MuonResidualsFitter_logPureGaussian(double residual, double center, double sigma)
{
  sigma = fabs(sigma);
  static const double cgaus = 0.5 * log( 2.*M_PI );
  return (-pow(residual - center, 2) *0.5 / sigma / sigma) - cgaus - log(sigma);
}

// TF1 interface version
Double_t MuonResidualsFitter_pureGaussian_TF1(Double_t *xvec, Double_t *par)
{
  return par[0] * exp(MuonResidualsFitter_logPureGaussian(xvec[0], par[1], par[2]));
}


// 2D Gauss fit function
double MuonResidualsFitter_logPureGaussian2D(double x, double y, double x0, double y0, double sx, double sy, double r)
{
  sx = fabs(sx);  sy = fabs(sy);
  static const double c2gaus = log( 2.*M_PI );
  double one_r2 = 1. - r*r;
  double dx = x - x0;
  double dy = y - y0;
  return -0.5/one_r2 * ( pow(dx/sx, 2) + pow(dy/sy, 2) - 2.*r*dx/sx*dy/sy ) - c2gaus - log(sx*sy*sqrt(one_r2));
}


double MuonResidualsFitter_compute_log_convolution(double toversigma, double gammaoversigma, double max, double step, double power)
{
  if (gammaoversigma == 0.) return (-toversigma*toversigma/2.) - log(sqrt(2*M_PI));

  double sum = 0.;
  double uplus = 0.;
  double integrandplus_last = 0.;
  double integrandminus_last = 0.;
  for (double inc = 0.;  uplus < max;  inc += step)
  {
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
double MuonResidualsFitter_logPowerLawTails(double residual, double center, double sigma, double gamma)
{
  sigma = fabs(sigma);
  double gammaoversigma = fabs(gamma / sigma);
  double toversigma = fabs((residual - center) / sigma);

  int gsbin0 = int(floor(gammaoversigma / MuonResidualsFitter_gsbinsize));
  int gsbin1 = int(ceil(gammaoversigma / MuonResidualsFitter_gsbinsize));
  int tsbin0 = int(floor(toversigma / MuonResidualsFitter_tsbinsize));
  int tsbin1 = int(ceil(toversigma / MuonResidualsFitter_tsbinsize));

  bool gsisbad = (gsbin0 >= MuonResidualsFitter_numgsbins  ||  gsbin1 >= MuonResidualsFitter_numgsbins);
  bool tsisbad = (tsbin0 >= MuonResidualsFitter_numtsbins  ||  tsbin1 >= MuonResidualsFitter_numtsbins);

  if (gsisbad  ||  tsisbad)
  {
    return log(gammaoversigma/M_PI) - log(toversigma*toversigma + gammaoversigma*gammaoversigma) - log(sigma);
  }
  else
  {
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
Double_t MuonResidualsFitter_powerLawTails_TF1(Double_t *xvec, Double_t *par)
{
  return par[0] * exp(MuonResidualsFitter_logPowerLawTails(xvec[0], par[1], par[2], par[3]));
}


double MuonResidualsFitter_logROOTVoigt(double residual, double center, double sigma, double gamma)
{
  return log(TMath::Voigt(residual - center, fabs(sigma), fabs(gamma)*2.));
}


Double_t MuonResidualsFitter_ROOTVoigt_TF1(Double_t *xvec, Double_t *par)
{
  return par[0] * TMath::Voigt(xvec[0] - par[1], fabs(par[2]), fabs(par[3])*2.);
}


double MuonResidualsFitter_logGaussPowerTails(double residual, double center, double sigma) {
  double x = residual-center;
  double s = fabs(sigma);
  double m = 2.*s;
  static const double e2 = exp(-2.), sqerf = sqrt(2.*M_PI)*erf(sqrt(2.));
  double a = pow(m,4)*e2;
  double n = sqerf*s + 2.*a*pow(m,-3)/3.;

  if (fabs(x)<m) return -x*x/(2.*s*s) - log(n);
  else return log(a) -4.*log(fabs(x)) - log(n);
}


Double_t MuonResidualsFitter_GaussPowerTails_TF1(Double_t *xvec, Double_t *par)
{
  return par[0] * exp(MuonResidualsFitter_logGaussPowerTails(xvec[0], par[1], par[2]));
}


double MuonResidualsFitter_integrate_pureGaussian(double low, double high, double center, double sigma)
{
  static const double isqr2 = 1./sqrt(2.);
  return (erf((high + center) * isqr2 / sigma) - erf((low + center) * isqr2 / sigma)) * exp(0.5/sigma/sigma) * 0.5;
}


void MuonResidualsFitter::initialize_table()
{
  if (MuonResidualsFitter_table_initialized  ||  residualsModel() != kPowerLawTails) return;
  MuonResidualsFitter_table_initialized = true;

  std::ifstream convolution_table("convolution_table.txt");
  if (convolution_table.is_open())
  {
    int numgsbins = 0;
    int numtsbins = 0;
    double tsbinsize = 0.;
    double gsbinsize = 0.;

    convolution_table >> numgsbins >> numtsbins >> tsbinsize >> gsbinsize;
    if (numgsbins != MuonResidualsFitter_numgsbins  ||  numtsbins != MuonResidualsFitter_numtsbins  ||  
	tsbinsize != MuonResidualsFitter_tsbinsize  ||  gsbinsize != MuonResidualsFitter_gsbinsize)
    {
      throw cms::Exception("MuonResidualsFitter") << "convolution_table.txt has the wrong bin width/bin size.  Throw it away and let the fitter re-create the file.\n";
    }

    for (int gsbin = 0;  gsbin < MuonResidualsFitter_numgsbins;  gsbin++)
    {
      for (int tsbin = 0;  tsbin < MuonResidualsFitter_numtsbins;  tsbin++)
      {
	int read_gsbin = 0;
	int read_tsbin = 0;
	double val = 0.;

	convolution_table >> read_gsbin >> read_tsbin >> val;
	if (read_gsbin != gsbin  ||  read_tsbin != tsbin)
	{
	  throw cms::Exception("MuonResidualsFitter") << "convolution_table.txt is out of order.  Throw it away and let the fitter re-create the file.\n";
	}
	MuonResidualsFitter_lookup_table[gsbin][tsbin] = val;
      }
    }
    convolution_table.close();
  }
  else
  {
    std::ofstream convolution_table2("convolution_table.txt");

    if (!convolution_table2.is_open()) throw cms::Exception("MuonResidualsFitter") << "Couldn't write to file convolution_table.txt\n";

    convolution_table2 << MuonResidualsFitter_numgsbins << " " << MuonResidualsFitter_numtsbins << " " << MuonResidualsFitter_tsbinsize << " " << MuonResidualsFitter_gsbinsize << std::endl;

    std::cout << "Initializing convolution look-up table (takes a few minutes)..." << std::endl;

    for (int gsbin = 0;  gsbin < MuonResidualsFitter_numgsbins;  gsbin++)
    {
      double gammaoversigma = double(gsbin) * MuonResidualsFitter_gsbinsize;
      std::cout << "    gsbin " << gsbin << "/" << MuonResidualsFitter_numgsbins << std::endl;
      for (int tsbin = 0;  tsbin < MuonResidualsFitter_numtsbins;  tsbin++)
      {
	double toversigma = double(tsbin) * MuonResidualsFitter_tsbinsize;

	// 1e-6 errors (out of a value of ~0.01) with max=100, step=0.001, power=4 (max=1000 does a little better with the tails)
	MuonResidualsFitter_lookup_table[gsbin][tsbin] = MuonResidualsFitter_compute_log_convolution(toversigma, gammaoversigma);

	// <10% errors with max=20, step=0.005, power=4 (faster computation for testing)
	// MuonResidualsFitter_lookup_table[gsbin][tsbin] = MuonResidualsFitter_compute_log_convolution(toversigma, gammaoversigma, 100., 0.005, 4.);

	convolution_table2 << gsbin << " " << tsbin << " " << MuonResidualsFitter_lookup_table[gsbin][tsbin] << std::endl;
      }
    }
    convolution_table2.close();
    std::cout << "Initialization done!" << std::endl;
  }
}


bool MuonResidualsFitter::dofit(void (*fcn)(int&,double*,double&,double*,int), std::vector<int> &parNum, std::vector<std::string> &parName,
                                std::vector<double> &start, std::vector<double> &step, std::vector<double> &low, std::vector<double> &high)
{
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
  
  //MuonResidualsFitter_TMinuit->SetPrintLevel(-1);
  
  for (; iNum != parNum.end();  ++iNum, ++iName, ++istart, ++istep, ++ilow, ++ihigh)
  {
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
  if (try_again)
  {
    for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
    arglist[0] = 50000;
    MuonResidualsFitter_TMinuit->mnexcm("MIGRAD", arglist, 1, smierflg);
  }

  Double_t fmin, fedm, errdef;
  Int_t npari, nparx, istat;
  MuonResidualsFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);

  if (istat != 3)
  {
    for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
    ierflg = 0;
    MuonResidualsFitter_TMinuit->mnexcm("HESSE", arglist, 0, ierflg);
  }

  // read-out the results
  m_loglikelihood = -fmin;

  m_value.clear();
  m_error.clear();
  for (int i = 0;  i < npar();  i++)
  {
    double v, e;
    MuonResidualsFitter_TMinuit->GetParameter(i, v, e);
    m_value.push_back(v);
    m_error.push_back(e);
  }
  m_cov.ResizeTo(npar(),npar());
  MuonResidualsFitter_TMinuit->mnemat( m_cov.GetMatrixArray(), npar());

  delete MuonResidualsFitter_TMinuit;
  delete fitinfo;
  if (smierflg != 0) return false;
  return true;
}


void MuonResidualsFitter::write(FILE *file, int which)
{
  long rows = numResiduals();
  int cols = ndata();
  int whichcopy = which;

  fwrite(&rows, sizeof(long), 1, file);
  fwrite(&cols, sizeof(int), 1, file);
  fwrite(&whichcopy, sizeof(int), 1, file);

  double *likeAChecksum = new double[cols];
  double *likeAChecksum2 = new double[cols];
  for (int i = 0;  i < cols;  i++)
  {
    likeAChecksum[i] = 0.;
    likeAChecksum2[i] = 0.;
  }

  for (std::vector<double*>::const_iterator residual = residuals_begin();  residual != residuals_end();  ++residual)
  {
    fwrite((*residual), sizeof(double), cols, file);
    for (int i = 0;  i < cols;  i++)
    {
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

void MuonResidualsFitter::read(FILE *file, int which)
{
  long rows = -100;
  int cols = -100;
  int readwhich = -100;

  fread(&rows, sizeof(long), 1, file);
  fread(&cols, sizeof(int), 1, file);
  fread(&readwhich, sizeof(int), 1, file);

  if (cols != ndata()  ||  rows < 0  ||  readwhich != which)
  {
    throw cms::Exception("MuonResidualsFitter") << "temporary file is corrupted (which = " << which << " readwhich = " << readwhich << " rows = " << rows << " cols = " << cols << ")\n";
  }

  double *likeAChecksum = new double[cols];
  double *likeAChecksum2 = new double[cols];
  for (int i = 0;  i < cols;  i++)
  {
    likeAChecksum[i] = 0.;
    likeAChecksum2[i] = 0.;
  }

  m_residuals.reserve(rows);
  for (long row = 0;  row < rows;  row++)
  {
    double *residual = new double[cols];
    fread(residual, sizeof(double), cols, file);
    fill(residual);
    for (int i = 0;  i < cols;  i++)
    {
      if (fabs(residual[i]) > likeAChecksum[i]) likeAChecksum[i] = fabs(residual[i]);
      if (fabs(residual[i]) < likeAChecksum2[i]) likeAChecksum2[i] = fabs(residual[i]);
    }
  } // end loop over records in file

  double *readChecksum = new double[cols];
  double *readChecksum2 = new double[cols];
  fread(readChecksum, sizeof(double), cols, file);
  fread(readChecksum2, sizeof(double), cols, file);

  for (int i = 0;  i < cols;  i++)
  {
    if (fabs(likeAChecksum[i] - readChecksum[i]) > 1e-10  ||  fabs(1./likeAChecksum2[i] - 1./readChecksum2[i]) > 1e10)
    {
      throw cms::Exception("MuonResidualsFitter") << "temporary file is corrupted (which = " << which << " rows = " << rows << " likeAChecksum " << likeAChecksum[i] << " != readChecksum " << readChecksum[i] << " " << " likeAChecksum2 " << likeAChecksum2[i] << " != readChecksum2 " << readChecksum2[i] << ")\n";
    }
  }

  m_residuals_ok.resize(numResiduals(), true);

  delete [] likeAChecksum;
  delete [] likeAChecksum2;
}


void MuonResidualsFitter::plotsimple(std::string name, TFileDirectory *dir, int which, double multiplier)
{
  double window = 100.;
  if (which == 0) window = 2.*30.;
  else if (which == 1) window = 2.*30.;
  else if (which == 2) window = 2.*20.;
  else if (which == 3) window = 2.*50.;

  TH1F *hist = dir->make<TH1F>(name.c_str(), "", 200, -window, window);
  for (std::vector<double*>::const_iterator r = residuals_begin();  r != residuals_end();  ++r) hist->Fill(multiplier * (*r)[which]);
}


void MuonResidualsFitter::plotweighted(std::string name, TFileDirectory *dir, int which, int whichredchi2, double multiplier)
{
  double window = 100.;
  if (which == 0) window = 2.*30.;
  else if (which == 1) window = 2.*30.;
  else if (which == 2) window = 2.*20.;
  else if (which == 3) window = 2.*50.;

  TH1F *hist = dir->make<TH1F>(name.c_str(), "", 200, -window, window);
  for (std::vector<double*>::const_iterator r = residuals_begin();  r != residuals_end();  ++r)
  {
    double weight = 1./(*r)[whichredchi2];
    if (TMath::Prob(1./weight*12, 12) < 0.99) hist->Fill(multiplier * (*r)[which], weight);
  }
}


void MuonResidualsFitter::computeHistogramRangeAndBinning(int which, int &nbins, double &a, double &b) 
{
  // first, make a numeric array while discarding some crazy outliers
  double *data = new double[numResiduals()];
  int n = 0;
  for (std::vector<double*>::const_iterator r = m_residuals.begin();  r != m_residuals.end();  r++)  
    if (fabs((*r)[which])<50.) 
    {
      data[n] = (*r)[which];
      n++;
    }
  
  // compute "3 normal sigma" and regular interquantile ranges
  const int n_quantiles = 7;
  double probabilities[n_quantiles] = {0.00135, 0.02275, 0.25, 0.5, 0.75, 0.97725, 0.99865}; // "3 normal sigma"
  //double probabilities[n_quantiles] = {0.02275, 0.25, 0.75, 0.97725}; // "2 normal sigma"
  double quantiles[n_quantiles];
  std::sort(data, data + n);
  TMath::Quantiles(n, n_quantiles, data, quantiles, probabilities, true, NULL, 7);
  delete [] data;
  double iqr = quantiles[4] - quantiles[2];
  
  // estimate optimal bin size according to Freedman-Diaconis rule
  double hbin = 2 * iqr / pow( n, 1./3);

  a = quantiles[1];
  b = quantiles[5];
  nbins = (int) ( (b - a) / hbin + 3. ); // add extra safety margin of 3

  std::cout<<"   quantiles: ";  for (int i=0;i<n_quantiles;i++) std::cout<<quantiles[i]<<" "; std::cout<<std::endl;
  //cout<<"n="<<select_count<<" quantiles ["<<quantiles[1]<<", "<<quantiles[2]<<"]  IQR="<<iqr
  //  <<"  full range=["<<minx<<","<<maxx<<"]"<<"  2 normal sigma quantile range = ["<<quantiles[0]<<", "<<quantiles[3]<<"]"<<endl;
  std::cout<<"   optimal h="<<hbin<<" nbins="<<nbins<<std::endl;
}


void MuonResidualsFitter::histogramChi2GaussianFit(int which, double &fit_mean, double &fit_sigma)
{
  int nbins;
  double a, b;
  computeHistogramRangeAndBinning(which, nbins, a, b); 
  if (a==b || a > b) { fit_mean = a; fit_sigma = 0; return; }

  TH1D *hist = new TH1D("htmp", "", nbins, a, b);
  for (std::vector<double*>::const_iterator r = m_residuals.begin();  r != m_residuals.end();  ++r)   hist->Fill( (*r)[which] );
  
  // do simple chi2 gaussian fit
  TF1 *f1= new TF1("f1","gaus", a, b);
  f1->SetParameter(0, hist->GetEntries());
  f1->SetParameter(1, 0);
  f1->SetParameter(2, hist->GetRMS());
  hist->Fit("f1","RQ");
  
  fit_mean  = f1->GetParameter(1);
  fit_sigma = f1->GetParameter(2);
  std::cout<<" h("<<nbins<<","<<a<<","<<b<<") mu="<<fit_mean<<" sig="<<fit_sigma<<std::endl;
  
  delete f1;
  delete hist;
}


// simple non-turned ellipsoid selection
void MuonResidualsFitter::selectPeakResiduals_simple(double nsigma, int nvar, int *vars)
{
  // does not make sense for small statistics
  if (numResiduals()<25) return;

  int nbefore = numResiduals();

  //just to be sure (can't see why it might ever be more then 10)
  assert(nvar<=10);

  // estimate nvar-D ellipsoid center & axes
  for (int v = 0; v<nvar; v++)
  {
    int which = vars[v];
    histogramChi2GaussianFit(which, m_center[which], m_radii[which]);
    m_radii[which] = nsigma * m_radii[which];
  }

  // filter out residuals that don't fit into the ellipsoid
  std::vector<double*>::iterator r = m_residuals.begin();
  while (r != m_residuals.end())
  {
    double ellipsoid_sum = 0;
    for (int v = 0; v<nvar; v++)
    {
      int which = vars[v];
      if (m_radii[which] == 0.) continue;
      ellipsoid_sum += pow( ( (*r)[which] - m_center[which]) / m_radii[which] , 2);
    }
    if (ellipsoid_sum <= 1.)  ++r;
    else
    {
      delete [] (*r);
      r = m_residuals.erase(r);
    }
  }
  std::cout<<" N residuals "<<nbefore<<" -> "<<numResiduals()<<std::endl;
}


// pre-selection using robust covariance estimator
void MuonResidualsFitter::selectPeakResiduals(double nsigma, int nvar, int *vars)
{
  //std::cout<<"doing selectpeakresiduals: nsig="<<nsigma<<" nvar="<<nvar<<" vars=";
  for (int i=0; i<nvar; ++i) std::cout<<vars[i]<<" ";
  std::cout<<std::endl;

  // does not make sense for small statistics
  if (numResiduals()<50) return;
  
  size_t nbefore = numResiduals();
  std::cout<<" N residuals "<<nbefore<<" ~ "<<(size_t) std::count(m_residuals_ok.begin(), m_residuals_ok.end(), true)<<std::endl;
  //just to be sure (can't see why it might ever be more then 10)
  assert(nvar<=10 && nvar>0);

  std::vector<double*>::iterator r = m_residuals.begin();

  // it's awkward, but the 1D case has to be handled separately
  if (nvar==1)
  {
    // get robust estimates for the peak and sigma
    double *data = new double[nbefore];
    for (size_t i = 0; i < nbefore; i++) data[i] = m_residuals[i][ vars[0] ];
    double peak, sigma;
    TRobustEstimator re;
    re.EvaluateUni(nbefore, data, peak, sigma);

    // filter out residuals that are more then nsigma away from the peak
    while (r != m_residuals.end())
    {
      double distance = fabs( ((*r)[ vars[0] ] - peak)/sigma );
      if (distance <= nsigma)  ++r;
      else
      {
        m_residuals_ok[r - m_residuals.begin()] = false;
        ++r;
        //delete [] (*r);
        //r = m_residuals.erase(r);
      }
    }
    std::cout<<" N residuals "<<nbefore<<" -> "<<numResiduals()<<std::endl;
    return;
  } // end 1D case
  
  // initialize and run the robust estimator for D>1
  TRobustEstimator re(nbefore, nvar);
  r = m_residuals.begin();
  while (r != m_residuals.end())
  {
    double *row = new double[nvar];
    for (int v = 0; v<nvar; v++)  row[v] = (*r)[ vars[v] ];
    re.AddRow(row);
    delete[] row;
    ++r;
  }
  re.Evaluate();
  
  // get nvar-dimensional ellipsoid center & covariance
  TVectorD M(nvar);
  re.GetMean(M);
  TMatrixDSym Cov(nvar);
  re.GetCovariance(Cov);
  Cov.Invert();

  // calculate the normalized radius for this nvar-dimensional ellipsoid from a 1D-Gaussian nsigma equivalent distance
  double conf_1d = TMath::Erf(nsigma/sqrt(2));
  double surf_radius = sqrt(TMath::ChisquareQuantile(conf_1d, nvar));

  // filter out residuals that are outside of the covariance ellipsoid with the above normalized radius
  r = m_residuals.begin();
  while (r != m_residuals.end())
  {
    TVectorD res(nvar);
    for (int v = 0; v<nvar; v++)  res[v] = (*r)[ vars[v] ];
    double distance = sqrt( Cov.Similarity(res - M) );
    if (distance <= surf_radius)  ++r;
    else
    {
      m_residuals_ok[r - m_residuals.begin()] = false;
      ++r;
      //delete [] (*r);
      //r = m_residuals.erase(r);
    }
  }
  std::cout<<" N residuals "<<nbefore<<" -> "<<(size_t) std::count(m_residuals_ok.begin(), m_residuals_ok.end(), true)<<std::endl;
}


void MuonResidualsFitter::correctBField(int idx_momentum, int idx_q)
{
  const int Nbin = 17;
  // find max 1/pt and bin width
  double min_pt = 9999.;
  for (std::vector<double*>::const_iterator r = residuals_begin();  r != residuals_end();  ++r)
  {
    double pt = fabs((*r)[idx_momentum]);
    if (pt < min_pt) min_pt = pt;
  }
  min_pt -= 0.01; // to prevent bin # overflow
  const double bin_width = 1./min_pt/Nbin;

  // fill indices of positive and negative charge residuals in each bin
  std::vector<size_t> pos[Nbin], neg[Nbin], to_erase;
  for (size_t i = 0; i < m_residuals.size(); i++)
  {
    if (!m_residuals_ok[i]) continue;
    int bin = (int)floor(1./fabs(m_residuals[i][idx_momentum])/bin_width);
    if (m_residuals[i][idx_q] > 0) pos[bin].push_back(i);
    else                           neg[bin].push_back(i);
  }

  // equalize pos and neg in each bin
  for (int j = 0; j < Nbin; j++)
  {
    size_t psize = pos[j].size();
    size_t nsize = neg[j].size();
    if (psize == nsize) continue;

    std::set<int> idx_set; // use a set to collect certain number of unique random indices to erase
    if (psize > nsize)
    {
      while (idx_set.size() < psize - nsize)  idx_set.insert( gRandom->Integer(psize) );
      for (std::set<int>::iterator it = idx_set.begin() ; it != idx_set.end(); it++ )  to_erase.push_back(pos[j][*it]);
    }
    else
    {
      while (idx_set.size() < nsize - psize)  idx_set.insert( gRandom->Integer(nsize) );
      for (std::set<int>::iterator it = idx_set.begin() ; it != idx_set.end(); it++ )  to_erase.push_back(neg[j][*it]);
    }
  }
  // sort in descending order, so we safely go from higher to lower indices:
  std::sort(to_erase.begin(), to_erase.end(), std::greater<int>() );
  for (std::vector<size_t>::const_iterator e = to_erase.begin();  e != to_erase.end();  ++e)
  {
    m_residuals_ok[*e] = false;
    //delete[] *(m_residuals.begin() + *e);
    //m_residuals.erase(m_residuals.begin() + *e);
  }

  std::vector<size_t> apos[Nbin], aneg[Nbin];
  for (size_t i = 0; i < m_residuals.size(); i++)
  {
    if (!m_residuals_ok[i]) continue;
    int bin = (int)floor(1./fabs(m_residuals[i][idx_momentum])/bin_width);
    if (m_residuals[i][idx_q] > 0) apos[bin].push_back(i);
    else                           aneg[bin].push_back(i);
  }
  for (int j = 0; j < Nbin; j++) std::cout << "bin " << j << ": [pos,neg] sizes before & after: ["
    << pos[j].size() <<","<< neg[j].size()<<"] -> [" << apos[j].size() <<","<< aneg[j].size() << "]" << std::endl;
  std::cout<<" N residuals "<<m_residuals.size()<<" -> "<<(size_t) std::count(m_residuals_ok.begin(), m_residuals_ok.end(), true)<<std::endl;
}


void MuonResidualsFitter::eraseNotSelectedResiduals()
{
  // it should probably be faster then doing erase
  size_t n_ok = (size_t) std::count(m_residuals_ok.begin(), m_residuals_ok.end(), true);
  std::vector<double*> tmp(n_ok, 0);
  std::cout << "residuals sizes: all=" << m_residuals.size()<<" good="<<n_ok<<std::endl;
  int iok=0;
  for (size_t i = 0; i < m_residuals.size(); i++)
  {
    if (!m_residuals_ok[i]) continue;
    tmp[iok++] = m_residuals[i];
  }
  m_residuals.swap(tmp);

  std::vector<bool> tmp_ok(n_ok, true);
  m_residuals_ok.swap(tmp_ok);

  std::cout << "residuals size after eraseNotSelectedResiduals =" << m_residuals.size()<<"  ok size="<<m_residuals_ok.size()<<std::endl;
}
