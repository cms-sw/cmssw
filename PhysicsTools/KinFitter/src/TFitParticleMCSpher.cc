// Classname: TFitParticleMCSpher
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TFitParticleMCSpher::
// --------------------
//
// Particle with spherical  parametrization of the momentum 4vector and
// constant mass (3 free parameters). The parametrization is chosen as
// follows:
//
// p = (r, theta, phi)
// E(fit) =  Sqrt( |p|^2 + m^2 )
//

#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCSpher.h"
#include "TMath.h"


//----------------
// Constructor --
//----------------
TFitParticleMCSpher::TFitParticleMCSpher()
  :TAbsFitParticle()
{ 
  init( 0, 0., 0);
}

TFitParticleMCSpher::TFitParticleMCSpher( const TFitParticleMCSpher& fitParticle )
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

TFitParticleMCSpher::TFitParticleMCSpher(TVector3* p, Double_t M, const TMatrixD* theCovMatrix)
  :TAbsFitParticle()
{ 
  init(p, M, theCovMatrix);
}

TFitParticleMCSpher::TFitParticleMCSpher(const TString &name, const TString &title,
			       TVector3* p, Double_t M, const TMatrixD* theCovMatrix)
  :TAbsFitParticle(name, title)
{ 
  init(p, M, theCovMatrix);
}

TAbsFitParticle* TFitParticleMCSpher::clone( TString newname ) const {
  // Returns a copy of itself
  
  TAbsFitParticle* myclone = new TFitParticleMCSpher( *this );
  if ( newname.Length() > 0 ) myclone->SetName(newname);
  return myclone;

}

//--------------
// Destructor --
//--------------
TFitParticleMCSpher::~TFitParticleMCSpher() {

}


//--------------
// Operations --
//--------------
void TFitParticleMCSpher::init(TVector3* p, Double_t M, const TMatrixD* theCovMatrix) {

  _nPar = 3;
  setIni4Vec(p, M);
  setCovMatrix(theCovMatrix);
  
}
 
TLorentzVector* TFitParticleMCSpher::calc4Vec( const TMatrixD* params ) {
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
  
  Double_t r = (*params)(0,0);
  Double_t theta = (*params)(1,0);
  Double_t phi = (*params)(2,0);

  Double_t X = r*TMath::Cos(phi)*TMath::Sin(theta);
  Double_t Y = r*TMath::Sin(phi)*TMath::Sin(theta);
  Double_t Z = r*TMath::Cos(theta);
  Double_t E = TMath::Sqrt(  X*X + Y*Y + Z*Z + _pini.M2() );

  TLorentzVector* vec = new TLorentzVector( X, Y, Z, E );
  return vec;

}

void TFitParticleMCSpher::setIni4Vec(const TLorentzVector* pini) {
  // Set the initial 4vector. Will also set the 
  // inital parameter values 

  TVector3 vec( pini->Vect() );
  setIni4Vec( &vec, pini->M() );

}

void TFitParticleMCSpher::setIni4Vec(const TVector3* p, Double_t M) {
  // Set the initial 4vector. Will also set the 
  // inital parameter values 

  if ( p == 0 ) {

    _u1.SetXYZ(0., 0., 0.);
    _u3.SetXYZ(0., 0., 0.);
    _u2.SetXYZ(0., 0., 0.);
    _pini.SetXYZM(0., 0., 0., M);
    _pcurr = _pini;

    _iniparameters.ResizeTo(_nPar,1);
    _iniparameters(0,0) = 0.;
    _parameters.ResizeTo(_nPar,1);
    _parameters = _iniparameters;

  } else {

    _pini.SetXYZM( p->x(), p->y(), p->z(), M);
    _pcurr = _pini;

    Double_t r = _pini.P();
    Double_t theta = _pini.Theta();
    Double_t phi = _pini.Phi();

    _iniparameters.ResizeTo(_nPar,1);
    _iniparameters(0,0) = r;
    _iniparameters(1,0) = theta;
    _iniparameters(2,0) = phi;
    _parameters.ResizeTo(_nPar,1);
    _parameters = _iniparameters;

    _u1.SetXYZ( TMath::Cos(phi)*TMath::Sin(theta), TMath::Sin(phi)*TMath::Sin(theta), TMath::Cos(theta) );
    _u2.SetXYZ( TMath::Cos(phi)*TMath::Cos(theta), TMath::Sin(phi)*TMath::Cos(theta), -1.*TMath::Sin(theta) );
    _u3.SetXYZ( -1.*TMath::Sin(phi), TMath::Cos(phi), 0. );
  
  }

}

TMatrixD* TFitParticleMCSpher::getDerivative() {
  // returns derivative dP/dy with P=(p,E) and y=(r, theta, phi) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dr, dP/dtheta, ...).

  TMatrixD* DerivativeMatrix = new TMatrixD(4,3);
  (*DerivativeMatrix) *= 0.;

  Double_t r = _parameters(0,0);
  Double_t theta = _parameters(1,0);
  Double_t phi = _parameters(2,0);

  //1st column: dP/dr
  (*DerivativeMatrix)(0,0) = TMath::Cos(phi)*TMath::Sin(theta);
  (*DerivativeMatrix)(1,0) = TMath::Sin(phi)*TMath::Sin(theta);
  (*DerivativeMatrix)(2,0) = TMath::Cos(theta);
  (*DerivativeMatrix)(3,0) = r/_pcurr.E();

  //2nd column: dP/dtheta
  (*DerivativeMatrix)(0,1) = r*TMath::Cos(phi)*TMath::Cos(theta);
  (*DerivativeMatrix)(1,1) = r*TMath::Sin(phi)*TMath::Cos(theta);
  (*DerivativeMatrix)(2,1) = -1.*r*TMath::Sin(theta);
  (*DerivativeMatrix)(3,1) = 0.;

   //3rd column: dP/dphi
  (*DerivativeMatrix)(0,2) = -1.*r*TMath::Sin(phi)*TMath::Sin(theta);
  (*DerivativeMatrix)(1,2) = r*TMath::Cos(phi)*TMath::Sin(theta);;
  (*DerivativeMatrix)(2,2) = 0.;
  (*DerivativeMatrix)(3,2) = 0.;

  return DerivativeMatrix;  

}

TMatrixD* TFitParticleMCSpher::transform(const TLorentzVector& vec) {
  // Returns the parameters corresponding to the given 
  // 4vector

  // retrieve parameters
  TMatrixD* tparams = new TMatrixD( _nPar, 1 );
  (*tparams)(0,0) = vec.P();
  (*tparams)(1,0) = vec.Theta();
  (*tparams)(2,0) = vec.Phi();

  return tparams;

}
