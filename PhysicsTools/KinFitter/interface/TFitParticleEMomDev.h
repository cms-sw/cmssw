

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
  virtual ~TFitParticleEMomDev();
  virtual TAbsFitParticle* clone( TString newname = "" ) const;

  // returns derivative dP/dy with P=(p,E) and y=(par1, par2, par3, ...) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dpar1, dP/dpar2, ...).
  virtual TMatrixD* getDerivative();
  virtual TMatrixD* transform(const TLorentzVector& vec);
  virtual void setIni4Vec(const TLorentzVector* pini);
  virtual TLorentzVector* calc4Vec( const TMatrixD* params );

protected :

  void init(TLorentzVector* pini, const TMatrixD* theCovMatrix);


private:
  
};

#endif
