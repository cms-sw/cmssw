#ifndef Particle_h
#define Particle_h

#include "TMatrixT.h"
#include "TMatrixTSym.h"

// Notes
// Store B in units of 1/GeV

class Particle {
 public:
  Particle(TMatrixT<double> par_, TMatrixTSym<double> cov_, int pdgid_, double charge_, double b_);
  virtual ~Particle(){};

  virtual double Parameter(int i){if(0<=i && i<par.GetNrows()) return par(i,0); return 0;}
  virtual double Covariance(int i,int j){if(0<=i && i<cov.GetNrows() && 0<=j && j<cov.GetNrows()) return cov(i,j); return 0;}
  virtual double BField(){return b;}
  virtual double Mass()=0;
  virtual int    PDGID(){return pdgid;}
  virtual double Charge(){return charge;}
  virtual double qB(){return b*charge;}
  virtual int    NParameters()=0;
  virtual TMatrixT<double> Parameter(){return par;}
  virtual TMatrixTSym<double> Covariance(){return cov;}

 private:
  TMatrixT<double> par;
  TMatrixTSym<double> cov;
  double b;
  double charge;
  int pdgid;
};
#endif


