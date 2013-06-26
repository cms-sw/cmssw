// Classname: TFitParticleEScaledMomDev
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TFitParticleEScaledMomDev::
// --------------------
//
// Particle with special parametrization of the momentum 4vector and
// constant E/p (3 free parameters). The parametrization is chosen as
// follows:
//
// p = r*|p|*u_r + theta*u_theta + phi*u_phi
// E(fit) = E(ini)/P(ini)*p(fit)
//
// with u_r = p/|p|
//      u_phi = (u_z x u_r)/|u_z x u_r|
//      u_theta = (u_r x u_phi)/|u_r x u_phi|
//
// The initial parameters values are chosen like (r, theta, phi) = (1., 0., 0.)
// corresponding to the measured momentum.
//

#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"
#include "TMath.h"


//----------------
// Constructor --
//----------------
TFitParticleEScaledMomDev::TFitParticleEScaledMomDev()
  :TAbsFitParticle()
{
  init(0, 0);
}

TFitParticleEScaledMomDev::TFitParticleEScaledMomDev( const TFitParticleEScaledMomDev& fitParticle )
  :TAbsFitParticle( fitParticle.GetName(), fitParticle.GetTitle() )
{

  _nPar = fitParticle._nPar;
  _u1 = fitParticle._u1;
  _u2 = fitParticle._u2;
  _u3 = fitParticle._u3;
  _covMatrix.ResizeTo(  fitParticle._covMatrix );
  _covMatrix = fitParticle._covMatrix;
  _iniparameters.ResizeTo( fitParticle._iniparameters );
  _iniparameters = fitParticle._iniparameters;
  _parameters.ResizeTo( fitParticle._parameters );
  _parameters = fitParticle._parameters;
  _pini = fitParticle._pini;
  _pcurr = fitParticle._pcurr;

}

TFitParticleEScaledMomDev::TFitParticleEScaledMomDev(TLorentzVector* pini, const TMatrixD* theCovMatrix)
  :TAbsFitParticle()
{
  init(pini, theCovMatrix);
}

TFitParticleEScaledMomDev::TFitParticleEScaledMomDev(const TString &name, const TString &title,
					 TLorentzVector* pini, const TMatrixD* theCovMatrix)
  :TAbsFitParticle(name, title)
{
  init(pini, theCovMatrix);
}  

TAbsFitParticle* TFitParticleEScaledMomDev::clone( TString newname ) const {
  // Returns a copy of itself
  
  TAbsFitParticle* myclone = new TFitParticleEScaledMomDev( *this );
  if ( newname.Length() > 0 ) myclone->SetName(newname);
  return myclone;

}

//--------------
// Destructor --
//--------------
TFitParticleEScaledMomDev::~TFitParticleEScaledMomDev() {

}

//--------------
// Operations --
//--------------
void TFitParticleEScaledMomDev::init(TLorentzVector* pini, const TMatrixD* theCovMatrix) {

  _nPar = 3;
  setIni4Vec(pini);
  _iniparameters.ResizeTo(_nPar,1);
  _iniparameters(0,0)=1.;
  _iniparameters(1,0)=0.;
  _iniparameters(2,0)=0.;
  _parameters.ResizeTo(_nPar,1);
  _parameters = _iniparameters;
  setCovMatrix(theCovMatrix);

}

TLorentzVector* TFitParticleEScaledMomDev::calc4Vec( const TMatrixD* params ) {
  // Calculates a 4vector corresponding to the given
  // parameter values

  if (params == 0) {
    return 0;
  }

  if ( params->GetNcols() != 1 || params->GetNrows() !=_nPar ) {
    edm::LogError ("WrongMatrixSize")
      << GetName() << "::calc4Vec - Parameter matrix has wrong size.";
    return 0;
  }
  
  Double_t X = _pini.P() * (*params)(0,0) *_u1.X() +
    (*params)(1,0) * _u2.X()+
    (*params)(2,0) * _u3.X();
  Double_t Y =  _pini.P() * (*params)(0,0) * _u1.Y() +
    (*params)(1,0) * _u2.Y()+
    (*params)(2,0) * _u3.Y();
  Double_t Z =  _pini.P()*(*params)(0,0)*_u1.Z() +
    (*params)(1,0) * _u2.Z()+
    (*params)(2,0) * _u3.Z();
  Double_t pcurr = TMath::Sqrt( X*X + Y*Y + Z*Z );
  Double_t E =  _pini.E()*pcurr/_pini.P();

  TLorentzVector* vec = new TLorentzVector( X, Y, Z, E );
  return vec;

}

void TFitParticleEScaledMomDev::setIni4Vec(const TLorentzVector* pini) {
  // Set the initial 4vector. Will also set the 
  // inital parameter values

  if (pini == 0) {

    _u1.SetXYZ(0., 0., 0.);
    _u3.SetXYZ(0., 0., 0.);
    _u2.SetXYZ(0., 0., 0.);
    _pini.SetXYZT(0., 0., 0., 0.);

  } else {

    _pini = (*pini);
    _pcurr = _pini;
    
    _u1 = pini->Vect();
    _u1 *= 1./_u1.Mag();
    
    TVector3 uz(0., 0., 1.);
    _u3 = uz.Cross(_u1);
    _u3 *= 1./_u3.Mag();
  
    _u2 = _u3.Cross(_u1);
    _u2 *= 1./_u2.Mag();
  }  

  _parameters = _iniparameters;
  
}

TMatrixD*  TFitParticleEScaledMomDev::getDerivative() {
  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, dP/dphi).

  TMatrixD* DerivativeMatrix = new TMatrixD(4,3);
  (*DerivativeMatrix) *= 0.;

   //1st column: dP/dr
  (*DerivativeMatrix)(0,0)=_pini.P()*_u1.X();
  (*DerivativeMatrix)(1,0)=_pini.P()*_u1.Y();
  (*DerivativeMatrix)(2,0)=_pini.P()*_u1.Z();
  (*DerivativeMatrix)(3,0)=_pini.P()*_pini.E()*_parameters(0,0)/_pcurr.P();
  //2nd column: dP/dtheta
  (*DerivativeMatrix)(0,1)=_u2.X();
  (*DerivativeMatrix)(1,1)=_u2.Y();
  (*DerivativeMatrix)(2,1)=_u2.Z();
  (*DerivativeMatrix)(3,1)=_pini.E()/_pini.P()/_pcurr.P()*_parameters(1,0);
  //3rd column: dP/dphi
  (*DerivativeMatrix)(0,2)=_u3.X();
  (*DerivativeMatrix)(1,2)=_u3.Y();
  (*DerivativeMatrix)(2,2)=_u3.Z();;
  (*DerivativeMatrix)(3,2)=_pini.E()/_pini.P()/_pcurr.P()*_parameters(2,0);

  return DerivativeMatrix;  

}

TMatrixD* TFitParticleEScaledMomDev::transform(const TLorentzVector& vec) {
  // Returns the parameters corresponding to the given 
  // 4vector wrt. to the current base vectors u_r, u_theta, and u_phi

  // construct rotation matrix
  TRotation rot;
  rot.RotateAxes( _u1, _u2, _u3 );
  rot.Invert();

  // rotate vector
  TVector3 vec3( vec.Vect() );
  vec3.Transform( rot );

  // retrieve parameters
  TMatrixD* tparams = new TMatrixD( _nPar, 1 );
  (*tparams)(0,0) = vec3(0)/_pini.P();
  (*tparams)(1,0) = vec3(1);
  (*tparams)(2,0) = vec3(2);

  return tparams;

}
