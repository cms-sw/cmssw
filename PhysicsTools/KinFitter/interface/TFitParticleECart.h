#ifndef TFitParticleECart_hh
#define TFitParticleECart_hh


#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"


class TFitParticleECart: public TAbsFitParticle {

public :

  TFitParticleECart();
  TFitParticleECart( const TFitParticleECart& fitParticle );
  TFitParticleECart(TLorentzVector* pini, const TMatrixD* theCovMatrix);
  TFitParticleECart(const TString &name, const TString &title, 
	       TLorentzVector* pini,
	       const TMatrixD* theCovMatrix);
  virtual ~TFitParticleECart();
  virtual TAbsFitParticle* clone( TString newname = "" ) const;

  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi, ...) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, ...).
  virtual TMatrixD* getDerivative();
  virtual TMatrixD* transform(const TLorentzVector& vec);
  virtual void setIni4Vec(const TLorentzVector* pini);
  virtual TLorentzVector* calc4Vec( const TMatrixD* params );

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);


private:
  
};

#endif
