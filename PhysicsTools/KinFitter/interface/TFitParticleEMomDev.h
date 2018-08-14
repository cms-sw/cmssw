

#ifndef TFitParticleEMomDev_hh
#define TFitParticleEMomDev_hh


#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"


class TFitParticleEMomDev: public TAbsFitParticle {

public :

  TFitParticleEMomDev();
  TFitParticleEMomDev( const TFitParticleEMomDev& fitParticle );
  TFitParticleEMomDev(TLorentzVector* pini, const TMatrixD* theCovMatrix);
  TFitParticleEMomDev(const TString &name, const TString &title, 
	       TLorentzVector* pini,
	       const TMatrixD* theCovMatrix);
  ~TFitParticleEMomDev() override;
  TAbsFitParticle* clone( const TString& newname = "" ) const override;

  // returns derivative dP/dy with P=(p,E) and y=(par1, par2, par3, ...) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dpar1, dP/dpar2, ...).
  TMatrixD* getDerivative() override;
  TMatrixD* transform(const TLorentzVector& vec) override;
  void setIni4Vec(const TLorentzVector* pini) override;
  TLorentzVector* calc4Vec( const TMatrixD* params ) override;

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);


private:

  ClassDefOverride(TFitParticleEMomDev, 0)
};

#endif
