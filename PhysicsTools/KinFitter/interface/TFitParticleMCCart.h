#ifndef TFitParticleMCCart_hh
#define TFitParticleMCCart_hh

#include "TMatrixD.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TVector3.h"

class TFitParticleMCCart: public TAbsFitParticle {

public :

  TFitParticleMCCart();
  TFitParticleMCCart( const TFitParticleMCCart& fitParticle );
  TFitParticleMCCart(TVector3* p, Double_t M, const TMatrixD* theCovMatrix);
  TFitParticleMCCart(const TString &name, const TString &title, 
		 TVector3* p, Double_t M, const TMatrixD* theCovMatrix);
  virtual ~TFitParticleMCCart();
  virtual TAbsFitParticle* clone( TString newname = "" ) const;

  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi, ...) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, ...).
  virtual TMatrixD* getDerivative();
  virtual TMatrixD* transform(const TLorentzVector& vec);
  virtual void setIni4Vec(const TLorentzVector* pini);
  void setIni4Vec(const TVector3* p, Double_t M);
  virtual TLorentzVector* calc4Vec( const TMatrixD* params );

protected :

  void init(TVector3* p, Double_t M, const TMatrixD* theCovMatrix);

  
private:
};

#endif
