#ifndef PhysicsTools_Utilities_RooFitFunction_h
#define PhysicsTools_Utilities_RooFitFunction_h
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include <vector>
#include <boost/type_traits.hpp>
#include <iostream>
namespace root {
  struct no_args;
  
  template<typename X, typename Expr>
  class RooFitFunction : public RooAbsReal {
  public:
    RooFitFunction(const RooFitFunction<X, Expr> & other, const char* name=0) :
      RooAbsReal(other, name), e_(other.e_), x_(X::name(), this, other.x_) {
      std::cout << ">>> making new RooFitFunction" << std::endl;
      std::vector<std::pair<boost::shared_ptr<double>, RooRealProxy> >::const_iterator 
	i = other.pars_.begin(), end = other.pars_.end();
      for(; i != end; ++i) {
	std::cout << ">>> adding par to RooFitFunction" << std::endl;
	pars_.push_back(std::make_pair(i->first, RooRealProxy(i->second.GetName(), this, i->second)));
      }
    }
    RooFitFunction(const char * name, const char * title,
		   const Expr & e, RooAbsReal & x) : 
      RooAbsReal(name, title), e_(e), x_(X::name(), X::name(), this, x) {
    } 
    RooFitFunction(const char * name, const char * title,
		   const Expr & e, RooAbsReal & x, 
		   RooAbsReal & rA, funct::Parameter & a) : 
      RooAbsReal(name, title), e_(e), x_(X::name(), X::name(), this, x) {
      pars_.push_back(std::make_pair(a.ptr(), RooRealProxy(a.name().c_str(), a.name().c_str(), this, rA)));
    } 
    RooFitFunction(const char * name, const char * title,
		   const Expr & e, RooAbsReal & x, 
		   RooAbsReal & rA, funct::Parameter & a,
		   RooAbsReal & rB, funct::Parameter & b) : 
      RooAbsReal(name, title), e_(e), x_(X::name(), X::name(), this, x) {
      pars_.push_back(std::make_pair(a.ptr(), RooRealProxy(a.name().c_str(), a.name().c_str(), this, rA)));
      pars_.push_back(std::make_pair(b.ptr(), RooRealProxy(b.name().c_str(), b.name().c_str(), this, rB)));
    } 
    RooFitFunction(const char * name, const char * title,
		   const Expr & e, RooAbsReal & x, 
		   RooAbsReal & rA, funct::Parameter & a,
		   RooAbsReal & rB, funct::Parameter & b,
		   RooAbsReal & rC, funct::Parameter & c) : 
      RooAbsReal(name, title), e_(e), x_(X::name(), X::name(), this, x) {
      pars_.push_back(std::make_pair(a.ptr(), RooRealProxy(a.name().c_str(), a.name().c_str(), this, rA)));
      pars_.push_back(std::make_pair(b.ptr(), RooRealProxy(b.name().c_str(), b.name().c_str(), this, rB)));
      pars_.push_back(std::make_pair(c.ptr(), RooRealProxy(c.name().c_str(), c.name().c_str(), this, rC)));
    } 
    virtual ~RooFitFunction() { }
    void add(RooAbsReal & rA, funct::Parameter & a) {
      pars_.push_back(std::make_pair(a.ptr(), RooRealProxy(a.name().c_str(), a.name().c_str(), this, rA)));      
    }
    virtual TObject* clone(const char* newName) const { 
      return new RooFitFunction<X, Expr>(* this, newName);
    }
  private:
    Expr e_;
    RooRealProxy x_;
    std::vector<std::pair<boost::shared_ptr<double>, RooRealProxy> > pars_;
    Double_t evaluate() const {
      X::set(x_);
      std::vector<std::pair<boost::shared_ptr<double>, RooRealProxy> >::const_iterator 
	i = pars_.begin(), end = pars_.end();
      for(; i != end; ++i) *(i->first) = i->second;
      return  e_();
    }

  };

}

#endif
