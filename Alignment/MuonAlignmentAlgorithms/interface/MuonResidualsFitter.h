#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsFitter_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsFitter_H

/** \class MuonResidualsFitter
 *  $Date: 2011/10/12 23:44:10 $
 *  $Revision: 1.17 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 *
 *  $Id: MuonResidualsFitter.h,v 1.17 2011/10/12 23:44:10 khotilov Exp $
 */

#ifndef STANDALONE_FITTER
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#endif

#include "TMinuit.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TF1.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TMatrixDSym.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <map>

// some mock classes for the case if we want to compile fitters with pure root outside of CMSSW
#ifdef STANDALONE_FITTER

#include <cassert>

class Alignable
{
public:
  struct Surface
  {
    double width() {return 300.;}
    double length() {return 300.;}
  };
  Surface s;
  Surface & surface() {return s;}
};

class TFileDirectory
{
public:
  template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5>
  T * make(const A1 &a1, const A2 &a2, const A3 &a3, const A4 &a4, const A5 &a5 ) const  {  T * t =  new T(a1, a2, a3, a4, a5); return t; }
  template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
  T * make(const A1 &a1, const A2 &a2, const A3 &a3, const A4 &a4, const A5 &a5, const A6 &a6, const A7 &a7) const  {  T * t =  new T(a1, a2, a3, a4, a5, a6, a7); return t; }
  template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8>
  T * make(const A1 &a1, const A2 &a2, const A3 &a3, const A4 &a4, const A5 &a5, const A6 &a6, const A7 &a7, const A8 &a8) const  {  T * t =  new T(a1, a2, a3, a4, a5, a6, a7, a8); return t; }
};

#include <exception>
namespace cms{
class Exception : public std::exception
{
public:
  Exception(const std::string & s): name(s) {}
  ~Exception () throw () {}
  std::string name;
  template <class T> Exception& operator<<(const T& what)
  {
    std::cout<<name<<" exception: "<<what<<std::endl;
    return *this;
  }
};
}// namespace cms

#endif // STANDALONE_FITTER



class MuonResidualsFitter
{
public:
  enum {
    kPureGaussian,
    kPowerLawTails,
    kROOTVoigt,
    kGaussPowerTails,
    kPureGaussian2D
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

  enum {
    k1111,
    k1110,
    k1100,
    k1010,
    k0010
  };

  struct MuonAlignmentTreeRow
  {
    Bool_t is_plus; // for +- endcap
    Bool_t is_dt; // DT or CSC
    UChar_t station; // 8bit uint
    Char_t ring_wheel;
    UChar_t sector;
    Float_t res_x;
    Float_t res_y;
    Float_t res_slope_x;
    Float_t res_slope_y;
    Float_t pos_x;
    Float_t pos_y;
    Float_t angle_x;
    Float_t angle_y;
    Float_t pz;
    Float_t pt;
    Char_t q;
    Bool_t select;
  };

  MuonResidualsFitter(int residualsModel, int minHits, int useResiduals, bool weightAlignment=true)
    : m_residualsModel(residualsModel), m_minHits(minHits), m_useResiduals(useResiduals), m_weightAlignment(weightAlignment), m_printLevel(0), m_strategy(1), m_cov(1), m_loglikelihood(0.)
  {
    if (m_residualsModel != kPureGaussian  &&  m_residualsModel != kPowerLawTails  &&  
        m_residualsModel != kROOTVoigt     &&  m_residualsModel != kGaussPowerTails && m_residualsModel != kPureGaussian2D)
      throw cms::Exception("MuonResidualsFitter") << "unrecognized residualsModel";
  };

  virtual ~MuonResidualsFitter()
  {
    for (std::vector<double*>::const_iterator residual = residuals_begin();  residual != residuals_end();  ++residual) {
      delete [] (*residual);
    }
  }

  virtual int type() const = 0;
  virtual int npar() = 0;
  virtual int ndata() = 0;

  int useRes() const { return m_useResiduals; }
  int residualsModel() const { return m_residualsModel; }
  long numResiduals() const { return m_residuals.size(); }

  void fix(int parNum, bool val=true)
  {
    assert(0 <= parNum  &&  parNum < npar());
    if (m_fixed.size() == 0) m_fixed.resize(npar(), false);
    m_fixed[parNum] = val;
  }

  bool fixed(int parNum)
  {
    assert(0 <= parNum  &&  parNum < npar());
    if (m_fixed.size() == 0) return false;
    else return m_fixed[parNum];
  }
  int nfixed() { return std::count(m_fixed.begin(), m_fixed.end(), true); }

  void setPrintLevel(int printLevel) { m_printLevel = printLevel; }
  void setStrategy(int strategy) { m_strategy = strategy; }

  // an array of the actual residual and associated baggage (qoverpt, trackangle, trackposition)
  // arrays passed to fill() are "owned" by MuonResidualsFitter: MuonResidualsFitter will delete them, don't do it yourself!
  void fill(double *residual)
  {
    m_residuals.push_back(residual);
    m_residuals_ok.push_back(true);
  }

  // this block of results is only valid if fit() returns true
  // also gamma is only valid if the model is kPowerLawTails or kROOTVoigt
  virtual bool fit(Alignable *ali) = 0;

  double value(int parNum) { assert(0 <= parNum  &&  parNum < npar());  return m_value[parNum]; }
  double errorerror(int parNum) { assert(0 <= parNum  &&  parNum < npar());  return m_error[parNum]; }

  // parNum corresponds to the parameter number defined by enums in specific fitters
  // parIdx is a continuous index in a set of parameters
  int parNum2parIdx(int parNum) { return m_parNum2parIdx[parNum];}

  TMatrixDSym covarianceMatrix() {return m_cov;}
  double covarianceElement(int parNum1, int parNum2)
  {
    assert(0 <= parNum1  &&  parNum1 < npar());
    assert(0 <= parNum2  &&  parNum2 < npar());
    assert(m_cov.GetNcols() == npar()); // m_cov might have not yet been resized to account for proper #parameters
    return m_cov(parNum2parIdx(parNum1),  parNum2parIdx(parNum2));
  }
  TMatrixDSym correlationMatrix();

  double loglikelihood() { return m_loglikelihood; }

  long numsegments()
  {
    long num = 0;
    for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) num++;
    return num;
  }

  virtual double sumofweights() = 0;

  // demonstration plots; return reduced chi**2
  virtual double plot(std::string name, TFileDirectory *dir, Alignable *ali) = 0;

  void plotsimple(std::string name, TFileDirectory *dir, int which, double multiplier);
  void plotweighted(std::string name, TFileDirectory *dir, int which, int whichredchi2, double multiplier);

#ifdef STANDALONE_FITTER
  Alignable m_ali;
  TFileDirectory m_dir;
  bool fit() { return fit(&m_ali); }
  virtual double plot(std::string &name) { return plot(name, &m_dir, &m_ali); }
  void plotsimple(std::string &name, int which, double multiplier) { plotsimple(name, &m_dir, which, multiplier); }
  void plotweighted(std::string &name, int which, int whichredchi2, double multiplier) { plotweighted(name, &m_dir, which, whichredchi2, multiplier); }
#endif

  // I/O of temporary files for collect mode
  void write(FILE *file, int which=0);
  void read(FILE *file, int which=0);

  // these are for the FCN to access what it needs to
  std::vector<double*>::const_iterator residuals_begin() const { return m_residuals.begin(); }
  std::vector<double*>::const_iterator residuals_end() const { return m_residuals.end(); }

  void computeHistogramRangeAndBinning(int which, int &nbins, double &a, double &b); 
  void histogramChi2GaussianFit(int which, double &fit_mean, double &fit_sigma);
  void selectPeakResiduals_simple(double nsigma, int nvar, int *vars);
  void selectPeakResiduals(double nsigma, int nvar, int *vars);

  virtual void correctBField() = 0;
  virtual void correctBField(int idx_momentum, int idx_q);

  std::vector<bool> & selectedResidualsFlags() {return m_residuals_ok;}

  void eraseNotSelectedResiduals();

protected:
  void initialize_table();
  bool dofit(void (*fcn)(int&,double*,double&,double*,int), std::vector<int> &parNum, std::vector<std::string> &parName, std::vector<double> &start, std::vector<double> &step, std::vector<double> &low, std::vector<double> &high);
  virtual void inform(TMinuit *tMinuit) = 0;

  int m_residualsModel;
  int m_minHits;
  int m_useResiduals;
  bool m_weightAlignment;
  std::vector<bool> m_fixed;
  int m_printLevel, m_strategy;

  std::vector<double*> m_residuals;
  std::vector<bool> m_residuals_ok;

  std::vector<double> m_value;
  std::vector<double> m_error;
  TMatrixDSym m_cov;
  double m_loglikelihood;

  std::map<int,int> m_parNum2parIdx;

  // center and radii of ellipsoid for peak selection
  // NOTE: 10 is a hardcoded maximum of ellipsoid variables!!!
  // but I can't imagine it ever becoming larger
  double m_center[20];
  double m_radii[20];
};

// A ROOT-sponsored hack to get information into the fit function
class MuonResidualsFitterFitInfo: public TObject
{
public:
  MuonResidualsFitterFitInfo(MuonResidualsFitter *fitter) : m_fitter(fitter) {}
  MuonResidualsFitter *fitter() { return m_fitter; }
private:
  MuonResidualsFitter *m_fitter;
#ifdef STANDALONE_FITTER
  ClassDef(MuonResidualsFitterFitInfo,1);
#endif
};
#ifdef STANDALONE_FITTER
ClassImp(MuonResidualsFitterFitInfo);
#endif

// fit functions (these can't be put in the class; "MuonResidualsFitter_" prefix avoids namespace clashes)
double MuonResidualsFitter_integrate_pureGaussian(double low, double high, double center, double sigma);
double MuonResidualsFitter_logPureGaussian(double residual, double center, double sigma);
Double_t MuonResidualsFitter_pureGaussian_TF1(Double_t *xvec, Double_t *par);
double MuonResidualsFitter_logPureGaussian2D(double x, double y, double x0, double y0, double sx, double sy, double r);
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
