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
  virtual ~TFitParticleEtEtaPhi();
  virtual TAbsFitParticle* clone( TString newname = "" ) const;

  // returns derivative dP/dy with P=(p,E) and y=(et, eta, phi) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/d(et), dP/d(eta), dP/d(phi)).
  virtual TMatrixD* getDerivative();
  virtual TMatrixD* transform(const TLorentzVector& vec);
  virtual void setIni4Vec(const TLorentzVector* pini);
  virtual TLorentzVector* calc4Vec( const TMatrixD* params );

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);


private:
  
};


#endif
