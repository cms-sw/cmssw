
#ifndef TFitParticleMCMomDev_hh
#define TFitParticleMCMomDev_hh

#include "TMatrixD.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TLorentzVector.h"
#include "TVector3.h"

class TFitParticleMCMomDev: public TAbsFitParticle {

public :

  TFitParticleMCMomDev();
  TFitParticleMCMomDev( const TFitParticleMCMomDev& fitParticle );
  TFitParticleMCMomDev(TVector3* p, Double_t M, const TMatrixD* theCovMatrix);
  TFitParticleMCMomDev(const TString &name, const TString &title, 
		 TVector3* p, Double_t M, const TMatrixD* theCovMatrix);
  ~TFitParticleMCMomDev() override;
  TAbsFitParticle* clone( const TString& newname = TString("") ) const override;

  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi, ...) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, ...).
  TMatrixD* getDerivative() override;
  TMatrixD* transform(const TLorentzVector& vec) override;
  void setIni4Vec(const TLorentzVector* pini) override;
  void setIni4Vec(const TVector3* p, Double_t M);
  TLorentzVector* calc4Vec( const TMatrixD* params ) override;

protected :

  void init(TVector3* p, Double_t M, const TMatrixD* theCovMatrix);


private:

  ClassDefOverride(TFitParticleMCMomDev, 0)
};

#endif
