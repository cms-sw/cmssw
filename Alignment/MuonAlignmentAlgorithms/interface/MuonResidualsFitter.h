#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsFitter_H

/** \class MuonResidualsFitter
 *  $Date: 2009/03/15 19:54:23 $
 *  $Revision: 1.3 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "TMinuit.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TF1.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>

class MuonResidualsFitter {
public:
  enum {
    kPureGaussian,
    kPowerLawTails
  };

  MuonResidualsFitter(int residualsModel, int minHitsPerRegion)
    : m_residualsModel(residualsModel), m_minHitsPerRegion(minHitsPerRegion), m_minResidual(0.), m_maxResidual(0.), m_goodfit(false) {
    if (m_residualsModel != kPureGaussian  &&  m_residualsModel != kPowerLawTails) throw cms::Exception("MuonResidualsFitter") << "unrecognized residualsModel";
  };

  virtual ~MuonResidualsFitter() {
    for (std::vector<double*>::const_iterator residual = residuals_begin();  residual != residuals_end();  ++residual) {
      delete [] (*residual);
    }
  };

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

  // an array of the actual residual and associated baggage (qoverpt, trackangle, trackposition)
  // arrays passed to fill() are "owned" by MuonResidualsFitter: MuonResidualsFitter will delete them, don't do it yourself!
  void fill(double *residual) {
    m_residuals.push_back(residual);
  };

  // this block of results is only valid if fit() returns true
  // also gamma is only valid if the model is kPowerLawTails
  virtual bool fit(double v1) = 0;
  double value(int parNum) { assert(m_goodfit  &&  0 <= parNum  &&  parNum < npar());  return m_value[parNum]; };
  double error(int parNum) { assert(m_goodfit  &&  0 <= parNum  &&  parNum < npar());  return m_error[parNum]; };
  double uperr(int parNum) { assert(m_goodfit  &&  0 <= parNum  &&  parNum < npar());  return m_uperr[parNum]; };
  double downerr(int parNum) { assert(m_goodfit  &&  0 <= parNum  &&  parNum < npar());  return m_downerr[parNum]; };
  double minoserr(int parNum) { return (fabs(uperr(parNum)) + fabs(downerr(parNum))) / 2.; };

  // demonstration plots
  virtual void plot(double v1, std::string name, TFileDirectory *dir) = 0;
  virtual double redchi2(double v1, std::string name, TFileDirectory *dir=NULL, bool write=false, int bins=100, double low=-5., double high=5.) = 0;

  // I/O of temporary files for collect mode
  void write(FILE *file, int which=0);
  void read(FILE *file, int which=0);

  // these are for the FCN to access what it needs to
  bool inRange(double residual) const { return (m_minResidual < residual  &&  residual < m_maxResidual); };
  std::vector<double*>::const_iterator residuals_begin() const { return m_residuals.begin(); };
  std::vector<double*>::const_iterator residuals_end() const { return m_residuals.end(); };

protected:
  void initialize_table();
  double compute_convolution(double toversigma, double gammaoversigma, double max=1000., double step=0.001, double power=4.);
  bool dofit(void (*fcn)(int&,double*,double&,double*,int), std::vector<int> &parNum, std::vector<std::string> &parName, std::vector<double> &start, std::vector<double> &step, std::vector<double> &low, std::vector<double> &high);
  virtual void inform(TMinuit *tMinuit) = 0;

  int m_residualsModel;
  int m_minHitsPerRegion;
  std::vector<bool> m_fixed;

  std::vector<double*> m_residuals;
  double m_minResidual, m_maxResidual, m_mean, m_stdev;

  bool m_goodfit;
  std::vector<double> m_value;
  std::vector<double> m_error;
  std::vector<double> m_uperr;
  std::vector<double> m_downerr;
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
double MuonResidualsFitter_pureGaussian(double residual, double center, double sigma);
Double_t MuonResidualsFitter_pureGaussian_TF1(Double_t *xvec, Double_t *par);
double MuonResidualsFitter_powerLawTails(double residual, double center, double sigma, double gamma);
Double_t MuonResidualsFitter_powerLawTails_TF1(Double_t *xvec, Double_t *par);
void MuonResidualsPositionFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag);
void MuonResidualsAngleFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag);

#endif // Alignment_MuonAlignmentAlgorithms_MuonResidualsFitter_H
