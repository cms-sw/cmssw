

#ifndef TFitParticleMomDev_hh
#define TFitParticleMomDev_hh


#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"


class TFitParticleMomDev: public TAbsFitParticle {

public :

  TFitParticleMomDev();
  TFitParticleMomDev( const TFitParticleMomDev& fitParticle );
  TFitParticleMomDev(TLorentzVector* pini, const TMatrixD* theCovMatrix);
  TFitParticleMomDev(const TString &name, const TString &title, 
	       TLorentzVector* pini,
	       const TMatrixD* theCovMatrix);
  ~TFitParticleMomDev() override;
  TAbsFitParticle* clone( const TString& newname = TString("") ) const override;

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

  ClassDefOverride(TFitParticleMomDev, 0)
};

#endif
