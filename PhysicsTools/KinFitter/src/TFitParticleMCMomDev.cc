// Classname: TFitParticleMCMomDev
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TFitParticleMCMomDev
// --------------------
//
// Particle with special parametrization of the momentum 4vector and
// constant mass (3 free parameters). The parametrization is chosen as
// follows:
//
// p = r*|p|*u_r + theta*u_theta + phi*u_phi
// E(fit) =  Sqrt( |p|^2 + m^2 )
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
#include "PhysicsTools/KinFitter/interface/TFitParticleMCMomDev.h"
#include "TMath.h"


//----------------
// Constructor --
//----------------
TFitParticleMCMomDev::TFitParticleMCMomDev()
  :TAbsFitParticle()
{ 
  init( 0, 0., 0);
}

TFitParticleMCMomDev::TFitParticleMCMomDev( const TFitParticleMCMomDev& fitParticle )
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

TFitParticleMCMomDev::TFitParticleMCMomDev(TVector3* p, Double_t M, const TMatrixD* theCovMatrix)
  :TAbsFitParticle()
{ 
  init(p, M, theCovMatrix);
}

TFitParticleMCMomDev::TFitParticleMCMomDev(const TString &name, const TString &title,
			       TVector3* p, Double_t M, const TMatrixD* theCovMatrix)
  :TAbsFitParticle(name, title)
{ 
  init(p, M, theCovMatrix);
}

TAbsFitParticle* TFitParticleMCMomDev::clone( TString newname ) const {
  // Returns a copy of itself
  
  TAbsFitParticle* myclone = new TFitParticleMCMomDev( *this );
  if ( newname.Length() > 0 ) myclone->SetName(newname);
  return myclone;

}

//--------------
// Destructor --
//--------------
TFitParticleMCMomDev::~TFitParticleMCMomDev() {

}


//--------------
// Operations --
//--------------
void TFitParticleMCMomDev::init(TVector3* p, Double_t M, const TMatrixD* theCovMatrix) {

  _nPar = 3;
  setIni4Vec(p, M);
  _iniparameters.ResizeTo(_nPar,1);
  _iniparameters(0,0)=1.;
  _iniparameters(1,0)=0.;
  _iniparameters(2,0)=0.;
  _parameters.ResizeTo(_nPar, 1);
  _parameters = _iniparameters;
  setCovMatrix(theCovMatrix);
  
}

TLorentzVector* TFitParticleMCMomDev::calc4Vec( const TMatrixD* params ) {
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

  Double_t X =  _pini.P() * (*params)(0,0) *_u1.X()+
    (*params)(1,0) * _u2.X()+
    (*params)(2,0) * _u3.X() ;
  Double_t Y = _pini.P() * (*params)(0,0) *_u1.Y()+
    (*params)(1,0) * _u2.Y()+
    (*params)(2,0) * _u3.Y() ;
  Double_t Z = _pini.P() * (*params)(0,0) *_u1.Z()+
    (*params)(1,0) * _u2.Z()+
    (*params)(2,0) * _u3.Z() ;
  Double_t E =  TMath::Sqrt(  X*X + Y*Y + Z*Z + _pini.M2() );

  TLorentzVector* vec = new TLorentzVector( X, Y, Z, E );
  return vec;

}
 
void TFitParticleMCMomDev::setIni4Vec(const TLorentzVector* pini) {
  // Set the initial 4vector. Will also set the 
  // inital parameter values

  TVector3 vec( pini->Vect() );
  setIni4Vec( &vec, pini->M() );

}

void TFitParticleMCMomDev::setIni4Vec(const TVector3* p, Double_t M) {
  // Set the initial 4vector. Will also set the 
  // inital parameter values

  if ( p == 0 ) {

    _pini.SetXYZM( 0., 0., 0., M);
    _pcurr = _pini;

  } else {

    _pini.SetXYZM( p->x(), p->y(), p->z(), M);
    _pcurr = _pini;

    _u1 = (*p);
    _u1 *= 1./_u1.Mag();

    TVector3 uz(0., 0., 1.);
    _u3 = uz.Cross(_u1);
    _u3 *= 1./_u3.Mag();
  
    _u2 = _u3.Cross(_u1);
    _u2 *= 1./_u2.Mag();
  
  }

  _parameters = _iniparameters;

}

TMatrixD* TFitParticleMCMomDev::getDerivative() {
  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, dP/dphi).

  TMatrixD* DerivativeMatrix = new TMatrixD(4,3);
  (*DerivativeMatrix) *= 0.;
  //1st column: dP/dr
  (*DerivativeMatrix)(0,0)=_pini.P()*_u1.X();
  (*DerivativeMatrix)(1,0)=_pini.P()*_u1.Y();
  (*DerivativeMatrix)(2,0)=_pini.P()*_u1.Z();
  (*DerivativeMatrix)(3,0)=_pini.P()*_pini.P()*_parameters(0,0)/_pcurr.E();

//  (*DerivativeMatrix)(3,0)=0.;

  //2nd column: dP/dtheta
  (*DerivativeMatrix)(0,1)=_u2.X();
  (*DerivativeMatrix)(1,1)=_u2.Y();
  (*DerivativeMatrix)(2,1)=_u2.Z();
  (*DerivativeMatrix)(3,1)=_parameters(1,0)/_pcurr.E();

  //(*DerivativeMatrix)(3,1)=0.;

  //3rd column: dP/dphi
  (*DerivativeMatrix)(0,2)=_u3.X();
  (*DerivativeMatrix)(1,2)=_u3.Y();
  (*DerivativeMatrix)(2,2)=_u3.Z();
  (*DerivativeMatrix)(3,2)=_parameters(2,0)/_pcurr.E();

  //(*DerivativeMatrix)(3,2)=0.;

  return DerivativeMatrix;  

}

TMatrixD* TFitParticleMCMomDev::transform(const TLorentzVector& vec) {
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
