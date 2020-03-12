// Classname: TFitParticleSpher
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)

//________________________________________________________________
//
// TFitParticleSpher::
// --------------------
//
// Particle with spherical  parametrization of the momentum 4vector and
// free mass (4 free parameters). The parametrization is chosen as
// follows:
//
// p = (r, theta, phi)
// E(fit) =  Sqrt( |p|^2 + d^2*m^2 )
//

#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleSpher.h"
#include "TMath.h"

//----------------
// Constructor --
//----------------
TFitParticleSpher::TFitParticleSpher() : TAbsFitParticle() { init(nullptr, nullptr); }

TFitParticleSpher::TFitParticleSpher(const TFitParticleSpher& fitParticle)
    : TAbsFitParticle(fitParticle.GetName(), fitParticle.GetTitle()) {
  _nPar = fitParticle._nPar;
  _u1 = fitParticle._u1;
  _u2 = fitParticle._u2;
  _u3 = fitParticle._u3;
  _covMatrix.ResizeTo(fitParticle._covMatrix);
  _covMatrix = fitParticle._covMatrix;
  _iniparameters.ResizeTo(fitParticle._iniparameters);
  _iniparameters = fitParticle._iniparameters;
  _parameters.ResizeTo(fitParticle._parameters);
  _parameters = fitParticle._parameters;
  _pini = fitParticle._pini;
  _pcurr = fitParticle._pcurr;
}

TFitParticleSpher::TFitParticleSpher(TLorentzVector* pini, const TMatrixD* theCovMatrix) : TAbsFitParticle() {
  init(pini, theCovMatrix);
}

TFitParticleSpher::TFitParticleSpher(const TString& name,
                                     const TString& title,
                                     TLorentzVector* pini,
                                     const TMatrixD* theCovMatrix)
    : TAbsFitParticle(name, title) {
  init(pini, theCovMatrix);
}

TAbsFitParticle* TFitParticleSpher::clone(const TString& newname) const {
  // Returns a copy of itself

  TAbsFitParticle* myclone = new TFitParticleSpher(*this);
  if (newname.Length() > 0)
    myclone->SetName(newname);
  return myclone;
}

//--------------
// Destructor --
//--------------
TFitParticleSpher::~TFitParticleSpher() {}

//--------------
// Operations --
//--------------
void TFitParticleSpher::init(TLorentzVector* pini, const TMatrixD* theCovMatrix) {
  _nPar = 4;
  setIni4Vec(pini);
  setCovMatrix(theCovMatrix);
}

TLorentzVector* TFitParticleSpher::calc4Vec(const TMatrixD* params) {
  // Calculates a 4vector corresponding to the given
  // parameter values

  if (params == nullptr) {
    return nullptr;
  }

  if (params->GetNcols() != 1 || params->GetNrows() != _nPar) {
    edm::LogError("WrongMatrixSize") << GetName() << "::calc4Vec - Parameter matrix has wrong size.";
    return nullptr;
  }

  Double_t r = (*params)(0, 0);
  Double_t theta = (*params)(1, 0);
  Double_t phi = (*params)(2, 0);
  Double_t d = (*params)(3, 0);

  Double_t X = r * TMath::Cos(phi) * TMath::Sin(theta);
  Double_t Y = r * TMath::Sin(phi) * TMath::Sin(theta);
  Double_t Z = r * TMath::Cos(theta);
  Double_t E = TMath::Sqrt(X * X + Y * Y + Z * Z + d * d * _pini.M2());

  TLorentzVector* vec = new TLorentzVector(X, Y, Z, E);
  return vec;
}

void TFitParticleSpher::setIni4Vec(const TLorentzVector* pini) {
  // Set the initial 4vector. Will also set the
  // inital parameter values

  if (pini == nullptr) {
    _u1.SetXYZ(0., 0., 0.);
    _u3.SetXYZ(0., 0., 0.);
    _u2.SetXYZ(0., 0., 0.);
    _pini.SetXYZT(0., 0., 0., 0.);
    _pcurr = _pini;

    _iniparameters.ResizeTo(_nPar, 1);
    _iniparameters(0, 0) = 0.;
    _iniparameters(1, 0) = 0.;
    _iniparameters(2, 0) = 0.;
    _iniparameters(3, 0) = 1.;

    _parameters.ResizeTo(_nPar, 1);
    _parameters(0, 0) = 0.;
    _parameters(1, 0) = 0.;
    _parameters(2, 0) = 0.;
    _parameters(3, 0) = 1.;

  } else {
    Double_t r = pini->P();
    Double_t theta = pini->Theta();
    Double_t phi = pini->Phi();

    _pini = (*pini);
    _pcurr = _pini;

    _iniparameters.ResizeTo(_nPar, 1);
    _iniparameters(0, 0) = r;
    _iniparameters(1, 0) = theta;
    _iniparameters(2, 0) = phi;
    _iniparameters(3, 0) = 1.;

    _parameters.ResizeTo(_nPar, 1);
    _parameters = _iniparameters;

    _u1.SetXYZ(TMath::Cos(phi) * TMath::Sin(theta), TMath::Sin(phi) * TMath::Sin(theta), TMath::Cos(theta));
    _u2.SetXYZ(TMath::Cos(phi) * TMath::Cos(theta), TMath::Sin(phi) * TMath::Cos(theta), -1. * TMath::Sin(theta));
    _u3.SetXYZ(-1. * TMath::Sin(phi), TMath::Cos(phi), 0.);
  }
}

TMatrixD* TFitParticleSpher::getDerivative() {
  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi, d)
  // the free parameters of the fit. The columns of the matrix contain
  // (dP/dr, dP/dtheta, ...).

  TMatrixD* DerivativeMatrix = new TMatrixD(4, 4);
  (*DerivativeMatrix) *= 0.;

  Double_t r = _parameters(0, 0);
  Double_t theta = _parameters(1, 0);
  Double_t phi = _parameters(2, 0);
  Double_t d = _parameters(3, 0);

  //1st column: dP/dr
  (*DerivativeMatrix)(0, 0) = TMath::Cos(phi) * TMath::Sin(theta);
  (*DerivativeMatrix)(1, 0) = TMath::Sin(phi) * TMath::Sin(theta);
  (*DerivativeMatrix)(2, 0) = TMath::Cos(theta);
  (*DerivativeMatrix)(3, 0) = 0.;

  //2nd column: dP/dtheta
  (*DerivativeMatrix)(0, 1) = r * TMath::Cos(phi) * TMath::Cos(theta);
  (*DerivativeMatrix)(1, 1) = r * TMath::Sin(phi) * TMath::Cos(theta);
  (*DerivativeMatrix)(2, 1) = -1. * r * TMath::Sin(theta);
  (*DerivativeMatrix)(3, 1) = 0.;

  //3rd column: dP/dphi
  (*DerivativeMatrix)(0, 2) = -1. * r * TMath::Sin(phi) * TMath::Sin(theta);
  (*DerivativeMatrix)(1, 2) = r * TMath::Cos(phi) * TMath::Sin(theta);
  ;
  (*DerivativeMatrix)(2, 2) = 0.;
  (*DerivativeMatrix)(3, 2) = 0.;

  //4th column: dP/dm
  (*DerivativeMatrix)(0, 3) = 0.;
  (*DerivativeMatrix)(1, 3) = 0.;
  (*DerivativeMatrix)(2, 3) = 0.;
  (*DerivativeMatrix)(3, 3) = _pini.M() * _pini.M() * d / _pcurr.E();

  return DerivativeMatrix;
}

TMatrixD* TFitParticleSpher::transform(const TLorentzVector& vec) {
  // Returns the parameters corresponding to the given
  // 4vector

  // retrieve parameters
  TMatrixD* tparams = new TMatrixD(_nPar, 1);
  (*tparams)(0, 0) = vec.P();
  (*tparams)(1, 0) = vec.Theta();
  (*tparams)(2, 0) = vec.Phi();
  (*tparams)(3, 0) = vec.M() / _pini.M();

  return tparams;
}
