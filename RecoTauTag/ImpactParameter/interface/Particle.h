#ifndef RecoTauTag_ImpactParameter_Particle_h
#define RecoTauTag_ImpactParameter_Particle_h

#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TVectorT.h"

// Notes
// Store B in units of 1/GeV

namespace tauImpactParameter {

class Particle {
 public:
  Particle(const TVectorT<double>& par, const TMatrixTSym<double>& cov, int pdgid, double charge, double b)
    : par_(par),
      cov_(cov),
      b_(b),
      charge_(charge),
      pdgid_(pdgid)
  {}
  virtual ~Particle(){};

  virtual double parameter(int i) const {if(0<=i && i<par_.GetNrows()) return par_(i); return 0;}
  virtual double covariance(int i,int j) const {if(0<=i && i<cov_.GetNrows() && 0<=j && j<cov_.GetNrows()) return cov_(i,j); return 0;}
  virtual double bField() const {return b_;}
  virtual double mass() const =0;
  virtual int    pdgId() const {return pdgid_;}
  virtual double charge() const {return charge_;}
  virtual double qB() const {return b_*charge_;}
  virtual int    nParameters() const =0;
  virtual const TVectorT<double>& parameter() const {return par_;}
  virtual const TMatrixTSym<double>& covariance() const {return cov_;}

 private:
  TVectorT<double> par_;
  TMatrixTSym<double> cov_;
  double b_;
  double charge_;
  int pdgid_;
};

}
#endif


