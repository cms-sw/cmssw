#ifndef PhisycsTools_Utilities_RootMinuit_h
#define PhisycsTools_Utilities_RootMinuit_h
/** \class fit::RootMinuit
 *
 */
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/ParameterMap.h"
#include "PhysicsTools/Utilities/interface/RootMinuitResultPrinter.h"
#include "PhysicsTools/Utilities/interface/RootMinuitFuncEvaluator.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "TMinuit.h"
#include "Math/SMatrix.h"

#include <vector>
#include <string>
#include <memory>

namespace fit {

  template <class Function>
  class RootMinuit {
  public:
    RootMinuit(const Function &f, bool verbose = false) : initialized_(false), minValue_(0), verbose_(verbose) {
      f_ = f;
    }
    void addParameter(const std::string &name, std::shared_ptr<double> val, double err, double min, double max) {
      if (initialized_)
        throw edm::Exception(edm::errors::Configuration)
            << "RootMinuit: can't add parameter " << name << " after minuit initialization\n";
      pars_.push_back(val);
      parameter_t par;
      par.val = *val;
      par.err = err;
      par.min = min;
      par.max = max;
      par.fixed = false;
      parMap_.push_back(std::make_pair(name, par));
      size_t s = parIndices_.size();
      parIndices_[name] = s;
    }
    void addParameter(const funct::Parameter &par, double err, double min, double max) {
      return addParameter(par.name(), par, err, min, max);
    }
    double getParameter(const std::string &name, double &err) {
      double val;
      init();
      minuit_->GetParameter(parameterIndex(name), val, err);
      return val;
    }
    double getParameter(const std::string &name) {
      double val, err;
      init();
      minuit_->GetParameter(parameterIndex(name), val, err);
      return val;
    }
    double getParameterError(const std::string &name, double &val) {
      double err;
      init();
      minuit_->GetParameter(parameterIndex(name), val, err);
      return err;
    }
    double getParameterError(const std::string &name) {
      double val, err;
      init();
      minuit_->GetParameter(parameterIndex(name), val, err);
      return err;
    }
    template <unsigned int N>
    void getErrorMatrix(ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepSym<double, N> > &err) {
      init();
      if (N != numberOfParameters())
        throw edm::Exception(edm::errors::Configuration)
            << "RootMinuit: can't call getErrorMatrix passing an SMatrix of dimension " << N
            << " while the number of parameters is " << numberOfParameters() << "\n";
      double *e = new double[N * N];
      minuit_->mnemat(e, numberOfParameters());
      for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
          err(i, j) = e[i + N * j];
        }
      }
      delete[] e;
      setParameters();
    }
    void fixParameter(const std::string &name) {
      size_t i = parameterIndex(name);
      parMap_[i].second.fixed = true;
      if (initialized_) {
        minuit_->FixParameter(i);
      }
    }
    void releaseParameter(const std::string &name) {
      size_t i = parameterIndex(name);
      parMap_[i].second.fixed = false;
      if (initialized_) {
        minuit_->Release(i);
      }
    }
    void setParameter(const std::string &name, double val) {
      size_t i = parameterIndex(name);
      parameter_t &par = parMap_[i].second;
      par.val = val;
      if (initialized_) {
        int ierflg = 0;
        minuit_->mnparm(i, name, par.val, par.err, par.min, par.max, ierflg);
        if (ierflg != 0)
          throw edm::Exception(edm::errors::Configuration)
              << "RootMinuit: error in setting parameter " << i << " value = " << par.val << " error = " << par.err
              << " range = [" << par.min << ", " << par.max << "]\n";
      }
    }
    void setParameters() {
      std::map<std::string, size_t>::const_iterator i = parIndices_.begin(), end = parIndices_.end();
      double val, err;
      for (; i != end; ++i) {
        size_t index = i->second;
        minuit_->GetParameter(index, val, err);
        *pars_[index] = val;
      }
    }
    int numberOfParameters() {
      init();
      return minuit_->GetNumPars();
    }
    int numberOfFreeParameters() {
      init();
      return minuit_->GetNumFreePars();
    }
    double minimize() {
      init();
      double arglist[10];
      arglist[0] = 5000;
      arglist[1] = 0.1;
      int ierflag;
      minuit_->mnexcm("MINIMIZE", arglist, 2, ierflag);
      if (ierflag != 0)
        std::cerr << "ERROR in minimize!!" << std::endl;
      if (verbose_)
        minuit_->mnmatu(1);  //Prints the covariance matrix
      double m = minValue();
      if (verbose_)
        minuit_->mnprin(3, m);
      setParameters();
      return m;
    }
    double migrad() {
      init();
      double arglist[10];
      arglist[0] = 5000;
      arglist[1] = 0.1;
      int ierflag;
      minuit_->mnexcm("MIGRAD", arglist, 2, ierflag);
      if (ierflag != 0)
        std::cerr << "ERROR in migrad!!" << std::endl;
      if (verbose_)
        minuit_->mnmatu(1);  //Prints the covariance matrix
      double m = minValue();
      if (verbose_)
        minuit_->mnprin(3, m);
      setParameters();
      return m;
    }
    double minValue() {
      init();
      int ierflag;
      double edm, errdef;
      int nvpar, nparx;
      minuit_->mnstat(minValue_, edm, errdef, nvpar, nparx, ierflag);
      return minValue_;
    }
    void printParameters(std::ostream &cout = std::cout) {
      std::map<std::string, size_t>::const_iterator i = parIndices_.begin(), end = parIndices_.end();
      for (; i != end; ++i) {
        cout << i->first << " = " << *pars_[i->second] << " +/- " << getParameterError(i->first) << std::endl;
      }
    }
    void printFitResults(std::ostream &cout = std::cout) {
      RootMinuitResultPrinter<Function>::print(minValue(), numberOfFreeParameters(), f_);
      printParameters(cout);
    }

  private:
    parameterVector_t parMap_;
    std::map<std::string, size_t> parIndices_;
    bool initialized_;
    double minValue_;
    std::unique_ptr<TMinuit> minuit_;
    std::vector<std::shared_ptr<double> > pars_;
    static std::vector<std::shared_ptr<double> > *fPars_;
    bool verbose_;
    static Function f_;
    static void fcn_(int &, double *, double &f, double *par, int) {
      size_t size = fPars_->size();
      for (size_t i = 0; i < size; ++i)
        *((*fPars_)[i]) = par[i];
      f = RootMinuitFuncEvaluator<Function>::evaluate(f_);
    }
    size_t parameterIndex(const std::string &name) const {
      typename std::map<std::string, size_t>::const_iterator p = parIndices_.find(name);
      if (p == parIndices_.end())
        throw edm::Exception(edm::errors::Configuration) << "RootMinuit: can't find parameter " << name << "\n";
      return p->second;
    }
    void init() {
      if (initialized_)
        return;
      minuit_.reset(new TMinuit(parMap_.size()));
      double arglist[10];
      int ierflg = 0;
      if (!verbose_) {
        arglist[0] = -1;
        minuit_->mnexcm("SET PRINT", arglist, 1, ierflg);
        if (ierflg != 0)
          throw edm::Exception(edm::errors::Configuration) << "RootMinuit: error in calling SET PRINT\n";
      }
      arglist[0] = 1;
      minuit_->mnexcm("SET ERR", arglist, 1, ierflg);
      if (ierflg != 0)
        throw edm::Exception(edm::errors::Configuration) << "RootMinuit: error in calling SET ERR\n";

      size_t i = 0;
      typename parameterVector_t::const_iterator p = parMap_.begin(), end = parMap_.end();
      for (; p != end; ++p, ++i) {
        const std::string &name = p->first;
        const parameter_t &par = p->second;
        minuit_->mnparm(i, name, par.val, par.err, par.min, par.max, ierflg);
        if (ierflg != 0)
          throw edm::Exception(edm::errors::Configuration)
              << "RootMinuit: error in setting parameter " << i << " value = " << par.val << " error = " << par.err
              << " range = [" << par.min << ", " << par.max << "]\n";
      }
      initialized_ = true;
      for (i = 0, p = parMap_.begin(); p != end; ++p, ++i)
        if (p->second.fixed)
          minuit_->FixParameter(i);
      fPars_ = &pars_;
      minuit_->SetFCN(fcn_);
    }
  };

  template <class Function>
  Function RootMinuit<Function>::f_;

  template <class Function>
  std::vector<std::shared_ptr<double> > *RootMinuit<Function>::fPars_ = 0;
}  // namespace fit

#endif
