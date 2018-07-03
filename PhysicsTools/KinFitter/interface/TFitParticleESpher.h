#ifndef TFitParticleESpher_hh
#define TFitParticleESpher_hh


#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"


class TFitParticleESpher: public TAbsFitParticle {

public :

  TFitParticleESpher();
  TFitParticleESpher( const TFitParticleESpher& fitParticle );
  TFitParticleESpher(TLorentzVector* pini, const TMatrixD* theCovMatrix);
  TFitParticleESpher(const TString &name, const TString &title, 
	       TLorentzVector* pini,
	       const TMatrixD* theCovMatrix);
  ~TFitParticleESpher() override;
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

  ClassDefOverride(TFitParticleESpher, 0)
};

#endif
