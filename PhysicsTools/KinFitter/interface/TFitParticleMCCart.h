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
  ~TFitParticleMCCart() override;
  TAbsFitParticle* clone( const TString& newname = TString("") ) const override;

  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi, ...) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, ...).
  TMatrixD* getDerivative() override;
  TMatrixD* transform(const TLorentzVector& vec) override;
  void setIni4Vec(const TLorentzVector* pini) override;
  void setIni4Vec(const TVector3* p, Double_t M);
  TLorentzVector* calc4Vec( const TMatrixD* params ) override;

protected :

  void init(TVector3* p, Double_t M, const TMatrixD* theCovMatrix);


private:

  ClassDefOverride(TFitParticleMCCart, 0)
};

#endif
