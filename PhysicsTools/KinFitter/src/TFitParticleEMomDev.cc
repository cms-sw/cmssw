// Classname: TFitParticleEMomDev
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)

//________________________________________________________________
//
// TFitParticleEMomDev
// --------------------
//
// Particle with special parametrization of the momentum 4vector and
// free mass (4 free parameters). The parametrization is chosen as
// follows:
//
// p = r*|p|*u_r + theta*u_theta + phi*u_phi
// E(fit) = E_meas * d
//
// with u_r = p/|p|
//      u_phi = (u_z x u_r)/|u_z x u_r|
//      u_theta = (u_r x u_phi)/|u_r x u_phi|
//
// The initial parameters values are chosen like (r, theta, phi, d) = (1., 0., 0., 1.)
// corresponding to the measured momentum and mass.
//

#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"

//----------------
// Constructor --
//----------------
TFitParticleEMomDev::TFitParticleEMomDev() : TAbsFitParticle() { init(nullptr, nullptr); }

TFitParticleEMomDev::TFitParticleEMomDev(const TFitParticleEMomDev& fitParticle)
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

TFitParticleEMomDev::TFitParticleEMomDev(TLorentzVector* pini, const TMatrixD* theCovMatrix) : TAbsFitParticle() {
  init(pini, theCovMatrix);
}

TFitParticleEMomDev::TFitParticleEMomDev(const TString& name,
                                         const TString& title,
                                         TLorentzVector* pini,
                                         const TMatrixD* theCovMatrix)
    : TAbsFitParticle(name, title) {
  init(pini, theCovMatrix);
}

TAbsFitParticle* TFitParticleEMomDev::clone(const TString& newname) const {
  // Returns a copy of itself

  TAbsFitParticle* myclone = new TFitParticleEMomDev(*this);
  if (newname.Length() > 0)
    myclone->SetName(newname);
  return myclone;
}

//--------------
// Destructor --
//--------------
TFitParticleEMomDev::~TFitParticleEMomDev() {}

//--------------
// Operations --
//--------------
void TFitParticleEMomDev::init(TLorentzVector* pini, const TMatrixD* theCovMatrix) {
  _nPar = 4;
  setIni4Vec(pini);
  setCovMatrix(theCovMatrix);
}

void TFitParticleEMomDev::setIni4Vec(const TLorentzVector* pini) {
  // Set the initial 4vector. Will also set the
  // inital parameter values
  _iniparameters.ResizeTo(_nPar, 1);
  _parameters.ResizeTo(_nPar, 1);

  if (pini == nullptr) {
    _iniparameters(0, 0) = 0.;
    _iniparameters(1, 0) = 0.;
    _iniparameters(2, 0) = 0.;
    _iniparameters(3, 0) = 0.;

    _u1.SetXYZ(0., 0., 0.);
    _u3.SetXYZ(0., 0., 0.);
    _u2.SetXYZ(0., 0., 0.);
    _pini.SetXYZT(0., 0., 0., 0.);
    _pcurr = _pini;

  } else {
    _iniparameters(0, 0) = 1.;
    _iniparameters(1, 0) = 0.;
    _iniparameters(2, 0) = 0.;
    _iniparameters(3, 0) = 1.;

    _pini = (*pini);
    _pcurr = _pini;

    _u1 = pini->Vect();
    _u1 *= 1. / _u1.Mag();

    TVector3 uz(0., 0., 1.);
    _u3 = uz.Cross(_u1);
    _u3 *= 1. / _u3.Mag();

    _u2 = _u3.Cross(_u1);
    _u2 *= 1. / _u2.Mag();
  }

  // reset parameters
  _parameters = _iniparameters;
}

TLorentzVector* TFitParticleEMomDev::calc4Vec(const TMatrixD* params) {
  // Calculates a 4vector corresponding to the given
  // parameter values

  if (params == nullptr) {
    return nullptr;
  }

  if (params->GetNcols() != 1 || params->GetNrows() != _nPar) {
    edm::LogError("WrongMatrixSize") << GetName() << "::calc4Vec - Parameter matrix has wrong size.";
    return nullptr;
  }

  Double_t X = _pini.P() * (*params)(0, 0) * _u1.X() + (*params)(1, 0) * _u2.X() + (*params)(2, 0) * _u3.X();
  Double_t Y = _pini.P() * (*params)(0, 0) * _u1.Y() + (*params)(1, 0) * _u2.Y() + (*params)(2, 0) * _u3.Y();
  Double_t Z = _pini.P() * (*params)(0, 0) * _u1.Z() + (*params)(1, 0) * _u2.Z() + (*params)(2, 0) * _u3.Z();
  Double_t E = _pini.E() * (*params)(3, 0);

  TLorentzVector* vec = new TLorentzVector(X, Y, Z, E);
  return vec;
}

TMatrixD* TFitParticleEMomDev::getDerivative() {
  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi, d)
  // the free parameters of the fit. The columns of the matrix contain
  // (dP/dr, dP/dtheta, dP/dphi, dP/dd).

  TMatrixD* DerivativeMatrix = new TMatrixD(4, 4);
  (*DerivativeMatrix) *= 0.;

  //1st column: dP/dr
  (*DerivativeMatrix)(0, 0) = _pini.P() * _u1.X();
  (*DerivativeMatrix)(1, 0) = _pini.P() * _u1.Y();
  (*DerivativeMatrix)(2, 0) = _pini.P() * _u1.Z();
  (*DerivativeMatrix)(3, 0) = 0.;

  //2nd column: dP/dtheta
  (*DerivativeMatrix)(0, 1) = _u2.X();
  (*DerivativeMatrix)(1, 1) = _u2.Y();
  (*DerivativeMatrix)(2, 1) = _u2.Z();
  (*DerivativeMatrix)(3, 1) = 0.;

  //3rd column: dP/dphi
  (*DerivativeMatrix)(0, 2) = _u3.X();
  (*DerivativeMatrix)(1, 2) = _u3.Y();
  (*DerivativeMatrix)(2, 2) = _u3.Z();
  (*DerivativeMatrix)(3, 2) = 0.;

  //4th column: dP/dm
  (*DerivativeMatrix)(0, 3) = 0.;
  (*DerivativeMatrix)(1, 3) = 0.;
  (*DerivativeMatrix)(2, 3) = 0.;
  (*DerivativeMatrix)(3, 3) = _pini.E();

  return DerivativeMatrix;
}

TMatrixD* TFitParticleEMomDev::transform(const TLorentzVector& vec) {
  // Returns the parameters corresponding to the given
  // 4vector wrt. to the current base vectors u_r, u_theta, and u_phi

  // construct rotation matrix
  TRotation rot;
  rot.RotateAxes(_u1, _u2, _u3);
  rot.Invert();

  // rotate vector
  TVector3 vec3(vec.Vect());
  vec3.Transform(rot);

  // retrieve parameters
  TMatrixD* tparams = new TMatrixD(_nPar, 1);
  (*tparams)(0, 0) = vec3.x() / _pini.P();
  (*tparams)(1, 0) = vec3.y();
  (*tparams)(2, 0) = vec3.z();
  (*tparams)(3, 0) = vec.E() / _pini.E();

  return tparams;
}
