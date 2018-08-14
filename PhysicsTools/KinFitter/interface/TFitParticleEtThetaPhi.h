#ifndef TFitParticleEtThetaPhi_hh
#define TFitParticleEtThetaPhi_hh


#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"


class TFitParticleEtThetaPhi: public TAbsFitParticle {

public :

  TFitParticleEtThetaPhi();
  TFitParticleEtThetaPhi( const TFitParticleEtThetaPhi& fitParticle );
  TFitParticleEtThetaPhi(TLorentzVector* pini, const TMatrixD* theCovMatrix);
  TFitParticleEtThetaPhi(const TString &name, const TString &title, 
	       TLorentzVector* pini,
	       const TMatrixD* theCovMatrix);
  ~TFitParticleEtThetaPhi() override;
  TAbsFitParticle* clone( const TString& newname = "" ) const override;

  // returns derivative dP/dy with P=(p,E) and y=(et, theta, phi) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/d(et), dP/d(theta), dP/d(phi)).
  TMatrixD* getDerivative() override;
  TMatrixD* transform(const TLorentzVector& vec) override;
  void setIni4Vec(const TLorentzVector* pini) override;
  TLorentzVector* calc4Vec( const TMatrixD* params ) override;

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);


private:

  ClassDefOverride(TFitParticleEtThetaPhi, 0)
};

#endif
