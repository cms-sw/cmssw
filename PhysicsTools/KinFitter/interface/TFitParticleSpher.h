#ifndef TFitParticleSpher_hh
#define TFitParticleSpher_hh


#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"


class TFitParticleSpher: public TAbsFitParticle {

public :

  TFitParticleSpher();
  TFitParticleSpher( const TFitParticleSpher& fitParticle );
  TFitParticleSpher(TLorentzVector* pini, const TMatrixD* theCovMatrix);
  TFitParticleSpher(const TString &name, const TString &title, 
	       TLorentzVector* pini,
	       const TMatrixD* theCovMatrix);
  ~TFitParticleSpher() override;
  TAbsFitParticle* clone( const TString& newname = TString("") ) const override;

  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi, ...) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, ...).
  TMatrixD* getDerivative() override;
  TMatrixD* transform(const TLorentzVector& vec) override;
  void setIni4Vec(const TLorentzVector* pini) override;
  TLorentzVector* calc4Vec( const TMatrixD* params ) override;

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);


private:

  ClassDefOverride(TFitParticleSpher, 0)
};

#endif
