#ifndef PhisycsTools_Utilities_RootMinuit_h
#define PhisycsTools_Utilities_RootMinuit_h
/** \class fit::RootMinuit
 *
 */
#include "TMinuit.h"
#include <iostream>
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "PhysicsTools/Utilities/interface/Parameter.h"

namespace fit {
  
  template<class Function>
  class RootMinuit {
  public:
    RootMinuit(int pars, Function f, bool verbose = false) : 
      minuit_(pars), pars_(pars), verbose_(verbose) { 
      f_ = f;
      double arglist[10];
      int ierflg = 0;      
      if (! verbose_) {
	arglist[0] = -1;
	minuit_.mnexcm("SET PRINT", arglist, 1, ierflg); 
	if ( ierflg != 0 ) std::cerr << "ERROR in set print!" << std::endl;
      }     
      arglist[0] = 1;
      minuit_.mnexcm("SET ERR", arglist, 1, ierflg);
      if ( ierflg != 0 ) std::cerr << "ERROR in set err!" << std::endl;
    }
    bool setParameter(int i, const std::string & name, boost::shared_ptr<double> val, double err, double min, double max) {
      pars_[i] = val;
      int ierflag;
      minuit_.mnparm(i, name.c_str(), *val, err, min, max, ierflag);
      return ierflag == 0;
    }
    bool setParameter(int i, const function::Parameter & par, double err, double min, double max) {
      return setParameter(i, par.name(), par, err, min, max);
    }
    double getParameter(int i, double & err) const {
      double val;
      minuit_.GetParameter(i, val, err);
      return val;
    }
    double getParameter(int i) const {
      double val, err;
      minuit_.GetParameter(i, val, err);
      return val;
    }
    double getParameterError(int i, double & val) const {
      double err;
      minuit_.GetParameter(i, val, err);
      return err;
    }
    double getParameterError(int i) const {
      double val, err;
      minuit_.GetParameter(i, val, err);
      return err;
    }
    void fixParameter(int i) {
      minuit_.FixParameter(i);
    }
    int getNumberOfFreeParameters() const { 
      return minuit_.GetNumFreePars();
    }
    double minimize() {
      std::cout << ">>> running fit" << std::endl;
      double arglist[10];
      arglist[0] = 5000;
      arglist[1] = 0.1;
      int ierflag;
      fPars_= & pars_; 
      minuit_.SetFCN(fcn_);
      minuit_.mnexcm("MIGRAD", arglist, 2, ierflag);
      if ( ierflag != 0 ) std::cerr << "ERROR in migrad!!" << std::endl;
      if(verbose_) minuit_.mnmatu(1); //Prints the covariance matrix
      std::cout << ">>> fit completed" << std::endl;
      double amin, edm, errdef;
      int nvpar, nparx;
      minuit_.mnstat(amin, edm, errdef, nvpar, nparx, ierflag);
      if(verbose_) minuit_.mnprin(3, amin);
      return amin;
    }
  private:
    TMinuit minuit_;
    std::vector<boost::shared_ptr<double> > pars_;
    static std::vector<boost::shared_ptr<double> > *fPars_;
    bool verbose_;
    static Function f_;
    static void fcn_(int &, double *, double &f, double *par, int) {
      size_t size = fPars_->size();
      for(size_t i = 0; i < size; ++i) 
	*((*fPars_)[i]) = par[i];
      f = f_();
    }
  };
  
  template<class Function>
  Function RootMinuit<Function>::f_;

  template<class Function>
  std::vector<boost::shared_ptr<double> > * RootMinuit<Function>::fPars_ = 0;
}

#endif
