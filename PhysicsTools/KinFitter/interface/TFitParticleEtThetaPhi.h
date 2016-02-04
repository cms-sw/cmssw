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
  virtual ~TFitParticleEtThetaPhi();
  virtual TAbsFitParticle* clone( TString newname = "" ) const;

  // returns derivative dP/dy with P=(p,E) and y=(et, theta, phi) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/d(et), dP/d(theta), dP/d(phi)).
  virtual TMatrixD* getDerivative();
  virtual TMatrixD* transform(const TLorentzVector& vec);
  virtual void setIni4Vec(const TLorentzVector* pini);
  virtual TLorentzVector* calc4Vec( const TMatrixD* params );

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);


private:
  
};

#endif
