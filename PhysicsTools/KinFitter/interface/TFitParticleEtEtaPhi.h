#ifndef TFitParticleEtEtaPhi_hh
#define TFitParticleEtEtaPhi_hh


#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"


class TFitParticleEtEtaPhi: public TAbsFitParticle {

public :

  TFitParticleEtEtaPhi();
  TFitParticleEtEtaPhi( const TFitParticleEtEtaPhi& fitParticle );
  TFitParticleEtEtaPhi(TLorentzVector* pini, const TMatrixD* theCovMatrix);
  TFitParticleEtEtaPhi(const TString &name, const TString &title, 
	       TLorentzVector* pini,
	       const TMatrixD* theCovMatrix);
  ~TFitParticleEtEtaPhi() override;
  TAbsFitParticle* clone( const TString& newname = "" ) const override;

  // returns derivative dP/dy with P=(p,E) and y=(et, eta, phi) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/d(et), dP/d(eta), dP/d(phi)).
  TMatrixD* getDerivative() override;
  TMatrixD* transform(const TLorentzVector& vec) override;
  void setIni4Vec(const TLorentzVector* pini) override;
  TLorentzVector* calc4Vec( const TMatrixD* params ) override;

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);


private:

  ClassDefOverride(TFitParticleEtEtaPhi, 0)
};


#endif
