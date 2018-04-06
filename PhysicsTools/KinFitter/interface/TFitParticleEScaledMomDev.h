#ifndef TFitParticleEScaledMomDev_hh
#define TFitParticleEScaledMomDev_hh

#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"


class TFitParticleEScaledMomDev: public TAbsFitParticle {

public :

  TFitParticleEScaledMomDev();
  TFitParticleEScaledMomDev( const TFitParticleEScaledMomDev& fitParticle );
  TFitParticleEScaledMomDev(TLorentzVector* pini, const TMatrixD* theCovMatrix);
  TFitParticleEScaledMomDev(const TString &name, const TString &title,
		      TLorentzVector* pini, const TMatrixD* theCovMatrix);
  ~TFitParticleEScaledMomDev() override;
  TAbsFitParticle* clone( const TString& newname = "" ) const override;

  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi, ...) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, ...).
  TMatrixD* getDerivative() override;
  TMatrixD* transform(const TLorentzVector& vec) override;
  void setIni4Vec(const TLorentzVector* pini) override;
  TLorentzVector* calc4Vec( const TMatrixD* params ) override;

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);

private :

  ClassDefOverride(TFitParticleEScaledMomDev, 0)
};

#endif
