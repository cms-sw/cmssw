#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsFitter_H

/** \class MuonResidualsFitter
 *  $Date: 2010/02/11 19:11:56 $
 *  $Revision: 1.14 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "TMinuit.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TF1.h"
#include "TMath.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>

class MuonResidualsFitter {
public:
  enum {
    kPureGaussian,
    kPowerLawTails,
    kROOTVoigt,
    kGaussPowerTails
  };

  enum {
    k1DOF,
    k5DOF,
    k6DOF,
    k6DOFrphi,
    kPositionFitter,
    kAngleFitter,
    kAngleBfieldFitter
  };

  MuonResidualsFitter(int residualsModel, int minHits, bool weightAlignment=true)
    : m_residualsModel(residualsModel), m_minHits(minHits), m_weightAlignment(weightAlignment), m_printLevel(0), m_strategy(2), m_loglikelihood(0.) {
    if (m_residualsModel != kPureGaussian  &&  m_residualsModel != kPowerLawTails  &&  m_residualsModel != kROOTVoigt  &&  m_residualsModel != kGaussPowerTails) throw cms::Exception("MuonResidualsFitter") << "unrecognized residualsModel";
  };

  virtual ~MuonResidualsFitter() {
    for (std::vector<double*>::const_iterator residual = residuals_begin();  residual != residuals_end();  ++residual) {
      delete [] (*residual);
    }
  };

  virtual int type() const = 0;

  int residualsModel() const { return m_residualsModel; };
  long numResiduals() const { return m_residuals.size(); };
  virtual int npar() = 0;
  virtual int ndata() = 0;

  void fix(int parNum, bool value=true) {
    assert(0 <= parNum  &&  parNum < npar());
    if (m_fixed.size() == 0) {
      for (int i = 0;  i < npar();  i++) {
	m_fixed.push_back(false);
      }
    }
    m_fixed[parNum] = value;
  };

  bool fixed(int parNum) {
    assert(0 <= parNum  &&  parNum < npar());
    if (m_fixed.size() == 0) return false;
    else return m_fixed[parNum];
  };

  void setPrintLevel(int printLevel) { m_printLevel = printLevel; };
  void setStrategy(int strategy) { m_strategy = strategy; };

  // an array of the actual residual and associated baggage (qoverpt, trackangle, trackposition)
  // arrays passed to fill() are "owned" by MuonResidualsFitter: MuonResidualsFitter will delete them, don't do it yourself!
  void fill(double *residual) {
    m_residuals.push_back(residual);
  };

  // this block of results is only valid if fit() returns true
  // also gamma is only valid if the model is kPowerLawTails or kROOTVoigt
  virtual bool fit(Alignable *ali) = 0;
  double value(int parNum) { assert(0 <= parNum  &&  parNum < npar());  return m_value[parNum]; };
  double errorerror(int parNum) { assert(0 <= parNum  &&  parNum < npar());  return m_error[parNum]; };
  double loglikelihood() { return m_loglikelihood; };
  long numsegments() {
    long num = 0;
    for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
      num++;
    }
    return num;
  };
  virtual double sumofweights() = 0;

  // demonstration plots; return reduced chi**2
  virtual double plot(std::string name, TFileDirectory *dir, Alignable *ali) = 0;

  // I/O of temporary files for collect mode
  void write(FILE *file, int which=0);
  void read(FILE *file, int which=0);

  // these are for the FCN to access what it needs to
  std::vector<double*>::const_iterator residuals_begin() const { return m_residuals.begin(); };
  std::vector<double*>::const_iterator residuals_end() const { return m_residuals.end(); };

  void plotsimple(std::string name, TFileDirectory *dir, int which, double multiplier);
  void plotweighted(std::string name, TFileDirectory *dir, int which, int whichredchi2, double multiplier);

protected:
  void initialize_table();
  bool dofit(void (*fcn)(int&,double*,double&,double*,int), std::vector<int> &parNum, std::vector<std::string> &parName, std::vector<double> &start, std::vector<double> &step, std::vector<double> &low, std::vector<double> &high);
  virtual void inform(TMinuit *tMinuit) = 0;

  int m_residualsModel;
  int m_minHits;
  bool m_weightAlignment;
  std::vector<bool> m_fixed;
  int m_printLevel, m_strategy;

  std::vector<double*> m_residuals;

  std::vector<double> m_value;
  std::vector<double> m_error;
  double m_loglikelihood;
};

// A ROOT-sponsored hack to get information into the fit function
class MuonResidualsFitterFitInfo: public TObject {
public:
  MuonResidualsFitterFitInfo(MuonResidualsFitter *fitter)
    : m_fitter(fitter) {};
  MuonResidualsFitter *fitter() { return m_fitter; };
private:
  MuonResidualsFitter *m_fitter;
};

// fit functions (these can't be put in the class; "MuonResidualsFitter_" prefix avoids namespace clashes)
double MuonResidualsFitter_integrate_pureGaussian(double low, double high, double center, double sigma);
double MuonResidualsFitter_logPureGaussian(double residual, double center, double sigma);
Double_t MuonResidualsFitter_pureGaussian_TF1(Double_t *xvec, Double_t *par);
double MuonResidualsFitter_compute_log_convolution(double toversigma, double gammaoversigma, double max=1000., double step=0.001, double power=4.);
double MuonResidualsFitter_logPowerLawTails(double residual, double center, double sigma, double gamma);
Double_t MuonResidualsFitter_powerLawTails_TF1(Double_t *xvec, Double_t *par);
double MuonResidualsFitter_logROOTVoigt(double residual, double center, double sigma, double gamma);
Double_t MuonResidualsFitter_ROOTVoigt_TF1(Double_t *xvec, Double_t *par);
double MuonResidualsFitter_logGaussPowerTails(double residual, double center, double sigma);
Double_t MuonResidualsFitter_GaussPowerTails_TF1(Double_t *xvec, Double_t *par);
void MuonResidualsPositionFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag);
void MuonResidualsAngleFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag);

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsFitter_H
