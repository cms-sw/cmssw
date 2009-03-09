#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"

// all global variables begin with "MuonResidualsFitter_" to avoid
// namespace clashes (that is, they do what would ordinarily be done
// with a class structure, but Minuit requires them to be global)

const double MuonResidualsFitter_gsbinsize = 0.1;
const double MuonResidualsFitter_tsbinsize = 0.1;
const int MuonResidualsFitter_numgsbins = 200;
const int MuonResidualsFitter_numtsbins = 500;

bool MuonResidualsFitter_table_initialized = false;
double MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins][MuonResidualsFitter_numtsbins];

static TMinuit* MuonResidualsFitter_TMinuit;
bool MuonResidualsFitter_inbadregion;

// fit function
double MuonResidualsFitter_pureGaussian(double residual, double center, double sigma) {
  sigma = fabs(sigma);
  return exp(-pow(residual - center, 2) / 2. / sigma / sigma) / sqrt(2.*M_PI) / sigma;
}

// TF1 interface version
Double_t MuonResidualsFitter_pureGaussian_TF1(Double_t *xvec, Double_t *par) {
  return par[0] * MuonResidualsFitter_pureGaussian(xvec[0], par[1], par[2]);
}

// fit function
double MuonResidualsFitter_powerLawTails(double residual, double center, double sigma, double gamma) {
  sigma = fabs(sigma);
  double gammaoversigma = fabs(gamma / sigma);
  double toversigma = fabs((residual - center) / sigma);

  int gsbin0 = int(floor(gammaoversigma / MuonResidualsFitter_gsbinsize));
  int gsbin1 = int(ceil(gammaoversigma / MuonResidualsFitter_gsbinsize));
  int tsbin0 = int(floor(toversigma / MuonResidualsFitter_tsbinsize));
  int tsbin1 = int(ceil(toversigma / MuonResidualsFitter_tsbinsize));

  bool gsisbad = (gsbin0 >= MuonResidualsFitter_numgsbins  ||  gsbin1 >= MuonResidualsFitter_numgsbins);
  bool tsisbad = (tsbin0 >= MuonResidualsFitter_numtsbins  ||  tsbin1 >= MuonResidualsFitter_numtsbins);

  double val00, val01, val10, val11;
  if (gsisbad  &&  !tsisbad) {
    MuonResidualsFitter_inbadregion = true;
    val00 = MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins-1][tsbin0] * exp(-gammaoversigma);
    val01 = MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins-1][tsbin1] * exp(-gammaoversigma);
    val10 = MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins-1][tsbin0] * exp(-gammaoversigma);
    val11 = MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins-1][tsbin1] * exp(-gammaoversigma);
  }
  else if (!gsisbad  &&  tsisbad) {
    MuonResidualsFitter_inbadregion = true;
    val00 = MuonResidualsFitter_lookup_table[gsbin0][MuonResidualsFitter_numtsbins-1] * exp(-toversigma);
    val01 = MuonResidualsFitter_lookup_table[gsbin0][MuonResidualsFitter_numtsbins-1] * exp(-toversigma);
    val10 = MuonResidualsFitter_lookup_table[gsbin1][MuonResidualsFitter_numtsbins-1] * exp(-toversigma);
    val11 = MuonResidualsFitter_lookup_table[gsbin1][MuonResidualsFitter_numtsbins-1] * exp(-toversigma);
  }
  else if (gsisbad  &&  tsisbad) {
    MuonResidualsFitter_inbadregion = true;
    val00 = MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins-1][MuonResidualsFitter_numtsbins-1] * exp(-gammaoversigma) * exp(-toversigma);
    val01 = MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins-1][MuonResidualsFitter_numtsbins-1] * exp(-gammaoversigma) * exp(-toversigma);
    val10 = MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins-1][MuonResidualsFitter_numtsbins-1] * exp(-gammaoversigma) * exp(-toversigma);
    val11 = MuonResidualsFitter_lookup_table[MuonResidualsFitter_numgsbins-1][MuonResidualsFitter_numtsbins-1] * exp(-gammaoversigma) * exp(-toversigma);
  }
  else {
    MuonResidualsFitter_inbadregion = false;
    val00 = MuonResidualsFitter_lookup_table[gsbin0][tsbin0];
    val01 = MuonResidualsFitter_lookup_table[gsbin0][tsbin1];
    val10 = MuonResidualsFitter_lookup_table[gsbin1][tsbin0];
    val11 = MuonResidualsFitter_lookup_table[gsbin1][tsbin1];
  }

  double val0 = val00 + ((toversigma / MuonResidualsFitter_tsbinsize) - tsbin0) * (val01 - val00);
  double val1 = val10 + ((toversigma / MuonResidualsFitter_tsbinsize) - tsbin0) * (val11 - val10);

  double val = val0 + ((gammaoversigma / MuonResidualsFitter_gsbinsize) - gsbin0) * (val1 - val0);

  return val / sigma;
}

// TF1 interface version
Double_t MuonResidualsFitter_powerLawTails_TF1(Double_t *xvec, Double_t *par) {
  return par[0] * MuonResidualsFitter_powerLawTails(xvec[0], par[1], par[2], par[3]);
}

void MuonResidualsFitter::initialize_table() {
  if (MuonResidualsFitter_table_initialized  ||  residualsModel() == kPureGaussian) return;
  MuonResidualsFitter_table_initialized = true;

  edm::LogWarning("MuonResidualsFitter") << "Initializing convolution look-up table (takes a few minutes)..." << std::endl;
  std::cout << "Initializing convolution look-up table (takes a few minutes)..." << std::endl;

  for (int gsbin = 0;  gsbin < MuonResidualsFitter_numgsbins;  gsbin++) {
    double gammaoversigma = double(gsbin) * MuonResidualsFitter_gsbinsize;

    std::cout << "    gsbin " << gsbin << "/" << MuonResidualsFitter_numgsbins << std::endl;

    for (int tsbin = 0;  tsbin < MuonResidualsFitter_numtsbins;  tsbin++) {
      double toversigma = double(tsbin) * MuonResidualsFitter_tsbinsize;

      // 1e-6 errors (out of a value of ~0.01) with max=100, step=0.001, power=4 (max=1000 does a little better with the tails)
      MuonResidualsFitter_lookup_table[gsbin][tsbin] = compute_convolution(toversigma, gammaoversigma);

      // <10% errors with max=20, step=0.005, power=4 (faster computation for testing)
      // MuonResidualsFitter_lookup_table[gsbin][tsbin] = compute_convolution(toversigma, gammaoversigma, 100., 0.005, 4.);
    }
  }

  edm::LogWarning("MuonResidualsFitter") << "Initialization done!" << std::endl;
  std::cout << "Initialization done!" << std::endl;
}

double MuonResidualsFitter::compute_convolution(double toversigma, double gammaoversigma, double max, double step, double power) {
  if (gammaoversigma == 0.) return exp(-toversigma*toversigma/2.) / sqrt(2*M_PI);

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
  return sum;
}

bool MuonResidualsFitter::dofit(void (*fcn)(int&,double*,double&,double*,int), std::vector<int> &parNum, std::vector<std::string> &parName, std::vector<double> &start, std::vector<double> &step, std::vector<double> &low, std::vector<double> &high) {
  MuonResidualsFitterFitInfo *fitinfo = new MuonResidualsFitterFitInfo(this);
  MuonResidualsFitter_inbadregion = true;

  MuonResidualsFitter_TMinuit = new TMinuit(npar());
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

  // chi^2 errors should be 1.0, log-likelihood should be 0.5
  for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
  arglist[0] = 0.5;
  ierflg = 0;
  MuonResidualsFitter_TMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
  if (ierflg != 0) { delete MuonResidualsFitter_TMinuit; delete fitinfo; return false; }

  // set strategy = 2 (more refined fits)
  for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
  arglist[0] = 2;
  ierflg = 0;
  MuonResidualsFitter_TMinuit->mnexcm("SET STR", arglist, 1, ierflg);
  if (ierflg != 0) { delete MuonResidualsFitter_TMinuit; delete fitinfo; return false; }

  // minimize
  for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
  ierflg = 0;
  MuonResidualsFitter_TMinuit->mnexcm("MIGRAD", arglist, 0, ierflg);
  if (ierflg != 0) { delete MuonResidualsFitter_TMinuit; delete fitinfo; return false; }

  // uncertainty in parameters
  for (int i = 0;  i < 10;  i++) arglist[i] = 0.;
  ierflg = 0;
  MuonResidualsFitter_TMinuit->mnexcm("MINOS", arglist, 0, ierflg);
  if (ierflg != 0) { delete MuonResidualsFitter_TMinuit; delete fitinfo; return false; }

  if (MuonResidualsFitter_inbadregion) { delete MuonResidualsFitter_TMinuit; delete fitinfo; return false; }
  
  // read-out the results
  m_goodfit = true;
  m_value.clear();
  m_error.clear();
  m_uperr.clear();
  m_downerr.clear();
  for (int i = 0;  i < npar();  i++) {
    double v, e, u, d, dummy1, dummy2;
    MuonResidualsFitter_TMinuit->GetParameter(i, v, e);
    MuonResidualsFitter_TMinuit->mnerrs(i, u, d, dummy1, dummy2);
    m_value.push_back(v);
    m_error.push_back(e);
    m_uperr.push_back(u);
    m_downerr.push_back(d);
  }

  delete MuonResidualsFitter_TMinuit;
  delete fitinfo;
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
  for (int i = 0;  i < cols;  i++) {
    likeAChecksum[i] = 0.;
  }
  
  for (std::vector<double*>::const_iterator residual = residuals_begin();  residual != residuals_end();  ++residual) {
    fwrite((*residual), sizeof(double), cols, file);

    for (int i = 0;  i < cols;  i++) {
      if (fabs((*residual)[i]) > likeAChecksum[i]) likeAChecksum[i] = fabs((*residual)[i]);
    }
  } // end loop over residuals

  // the idea is that mal-formed doubles are likely to be huge values
  // because the exponent gets screwed up; we want to check for that
  fwrite(likeAChecksum, sizeof(double), cols, file);
  delete [] likeAChecksum;
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
  for (int i = 0;  i < cols;  i++) {
    likeAChecksum[i] = 0.;
  }

  for (long row = 0;  row < rows;  row++) {
    double *residual = new double[cols];
    fread(residual, sizeof(double), cols, file);
    fill(residual);

    for (int i = 0;  i < cols;  i++) {
      if (fabs(residual[i]) > likeAChecksum[i]) likeAChecksum[i] = fabs(residual[i]);
    }
    
    // delete [] residual;
  } // end loop over records in file

  double *readChecksum = new double[cols];
  fread(readChecksum, sizeof(double), cols, file);

  for (int i = 0;  i < cols;  i++) {
    if (fabs(likeAChecksum[i] - readChecksum[i]) > 1e-10) {
      throw cms::Exception("MuonResidualsFitter") << "temporary file is corrupted (which = " << which << " rows = " << rows << " likeAChecksum " << likeAChecksum[i] << " != readChecksum " << readChecksum[i] << ")" << std::endl;
    }
  }
  delete [] likeAChecksum;
}
