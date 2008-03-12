#ifndef PhisycsTools_Utilities_RootMinuit_h
#define PhisycsTools_Utilities_RootMinuit_h
/** \class fit::RootMinuit
 *
 */
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/ParameterMap.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "TMinuit.h"
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

namespace fit {
  
  template<class Function>
  class RootMinuit {
  public:
    RootMinuit(Function f, bool verbose = false) : 
      initialized_(false), verbose_(verbose) { 
      f_ = f;
    }
    void init() { 
      minuit_.reset(new TMinuit(parMap_.size()));
      double arglist[10];
      int ierflg = 0;      
      if (! verbose_) {
	arglist[0] = -1;
	minuit_->mnexcm("SET PRINT", arglist, 1, ierflg); 
	if (ierflg != 0) 
	  throw edm::Exception(edm::errors::Configuration)
	    << "RootMinuit: error in calling SET PRINT\n";
      }     
      arglist[0] = 1;
      minuit_->mnexcm("SET ERR", arglist, 1, ierflg);
      if (ierflg != 0) 
	throw edm::Exception(edm::errors::Configuration)
	  << "RootMinuit: error in calling SET ERR\n";

      size_t i = 0;
      typename parameterVector_t::const_iterator p = parMap_.begin(), end = parMap_.end();
      for(; p != end; ++p, ++i) {
	const std::string & name = p->first;
	const parameter_t & par = p->second;
	minuit_->mnparm(i, name, par.val, par.err, par.min, par.max, ierflg);
	if(ierflg != 0)
	  throw edm::Exception(edm::errors::Configuration)
	    << "RootMinuit: error in setting parameter " << i 
	    << " value = " << par.val << " error = " << par.err
	    << " range = [" << par.min << ", " << par.max << "]\n";
	if(par.fixed)
	  minuit_->FixParameter(i);
      }
      initialized_ = true;
    }
    void addParameter(const std::string & name, boost::shared_ptr<double> val, double err, double min, double max) {
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
    void addParameter(const function::Parameter & par, double err, double min, double max) {
      return addParameter(par.name(), par, err, min, max);
    }
    double getParameter(const std::string & name, double & err) {
      double val;
      if(!initialized_) init();
      minuit_->GetParameter(parameterIndex(name), val, err);
      return val;
    }
    double getParameter(const std::string & name) {
      double val, err;
      if(!initialized_) init();
      minuit_->GetParameter(parameterIndex(name), val, err);
      return val;
    }
    double getParameterError(const std::string & name, double & val) {
      double err;
      if(!initialized_) init();
      minuit_->GetParameter(parameterIndex(name), val, err);
      return err;
    }
    double getParameterError(const std::string & name) {
      double val, err;
      if(!initialized_) init();
      minuit_->GetParameter(parameterIndex(name), val, err);
      return err;
    }
    void fixParameter(const std::string & name) {
      size_t i = parameterIndex(name);
      if(initialized_) {
	minuit_->FixParameter(i);
      }
      parMap_[i].second.fixed = true;
    }
    int getNumberOfFreeParameters() { 
      if(!initialized_) init();
      return minuit_->GetNumFreePars();
    }
    double minimize() {
      if(!initialized_) init();
      std::cout << ">>> running fit" << std::endl;
      double arglist[10];
      arglist[0] = 5000;
      arglist[1] = 0.1;
      int ierflag;
      fPars_= & pars_; 
      minuit_->SetFCN(fcn_);
      minuit_->mnexcm("MIGRAD", arglist, 2, ierflag);
      if ( ierflag != 0 ) std::cerr << "ERROR in migrad!!" << std::endl;
      if(verbose_) minuit_->mnmatu(1); //Prints the covariance matrix
      std::cout << ">>> fit completed" << std::endl;
      double amin, edm, errdef;
      int nvpar, nparx;
      minuit_->mnstat(amin, edm, errdef, nvpar, nparx, ierflag);
      if(verbose_) minuit_->mnprin(3, amin);
      return amin;
    }

  private:
    parameterVector_t parMap_;
    std::map<std::string, size_t> parIndices_;
    bool initialized_;
    std::auto_ptr<TMinuit> minuit_;
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
    size_t parameterIndex(const std::string &name) const {
      typename std::map<std::string, size_t>::const_iterator p = parIndices_.find(name);
      if(p == parIndices_.end())
	throw edm::Exception(edm::errors::Configuration)
	  << "RootMinuit: can't find parameter " << name << "\n";
      return p->second;
    }
  };
  
  template<class Function>
  Function RootMinuit<Function>::f_;

  template<class Function>
  std::vector<boost::shared_ptr<double> > * RootMinuit<Function>::fPars_ = 0;
}

#endif
