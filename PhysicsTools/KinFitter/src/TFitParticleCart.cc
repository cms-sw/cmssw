// Classname: TFitParticleCart
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TFitParticleCart::
// --------------------
//
// Particle with cartesian 4vector parametrization and free mass
// [four free parameters (px, py, pz, d) with initial values
//  (px, py, pz, 1)]
//
// p = px*u1 + py*u2 + pz*u3
// E = Sqrt( |p|^2 + d^2*m^2 )
//

#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleCart.h"
#include "TMath.h"


//----------------
// Constructor --
//----------------
TFitParticleCart::TFitParticleCart()
  :TAbsFitParticle()  
{
  init(0, 0);
}

TFitParticleCart::TFitParticleCart( const TFitParticleCart& fitParticle )
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

TFitParticleCart::TFitParticleCart(TLorentzVector* pini, const TMatrixD* theCovMatrix)
  :TAbsFitParticle()  
{
  init(pini, theCovMatrix);
}

TFitParticleCart::TFitParticleCart(const TString &name, const TString &title, 
				   TLorentzVector* pini, const TMatrixD* theCovMatrix)
  :TAbsFitParticle(name, title)  
{
  init(pini, theCovMatrix);
}

TAbsFitParticle* TFitParticleCart::clone( TString newname ) const {
  // Returns a copy of itself

  TAbsFitParticle* myclone = new TFitParticleCart( *this );
  if ( newname.Length() > 0 ) myclone->SetName(newname);
  return myclone;

}

//--------------
// Destructor --
//--------------
TFitParticleCart::~TFitParticleCart() {

}

//--------------
// Operations --
//--------------
void TFitParticleCart::init(TLorentzVector* pini, const TMatrixD* theCovMatrix ) {

  _nPar = 4;
  setIni4Vec(pini);
  setCovMatrix(theCovMatrix);

}

TLorentzVector* TFitParticleCart::calc4Vec( const TMatrixD* params ) {
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
  
  Double_t X = (*params)(0,0);
  Double_t Y = (*params)(1,0);
  Double_t Z = (*params)(2,0);
  Double_t E =  TMath::Sqrt( X*X + Y*Y + Z*Z + (*params)(3,0)*(*params)(3,0)*_pini.M2() );

  TLorentzVector* vec = new TLorentzVector( X, Y, Z, E );
  return vec;

}

void TFitParticleCart::setIni4Vec(const TLorentzVector* pini) {
  // Set the initial 4vector. Will also set the 
  // inital parameter values

  if (pini == 0) {

    _u1.SetXYZ(0., 0., 0.);
    _u3.SetXYZ(0., 0., 0.);
    _u2.SetXYZ(0., 0., 0.);
    _pini.SetXYZT(0., 0., 0., 0.);
    _pcurr = _pini;

    _iniparameters.ResizeTo(_nPar,1);
    _iniparameters(0,0)=0;
    _iniparameters(1,0)=0;
    _iniparameters(2,0)=0;
    _iniparameters(3,0)=0.;

    _parameters.ResizeTo(_nPar,1);
    _parameters(0,0)=0.;
    _parameters(1,0)=0.;
    _parameters(2,0)=0.;   
    _parameters(3,0)=0.;   
    
  } else {
    
    _pini = (*pini);
    _pcurr = _pini;

    _u1.SetXYZ( 1., 0., 0. );
    _u2.SetXYZ( 0., 1., 0. );
    _u3.SetXYZ( 0., 0., 1. );
  
    _iniparameters.ResizeTo(_nPar,1);
    _iniparameters(0,0)=_pini.X();
    _iniparameters(1,0)=_pini.Y();
    _iniparameters(2,0)=_pini.Z();
    _iniparameters(3,0)=1.;
    _parameters.ResizeTo(_nPar,1);
    _parameters = _iniparameters;

  }

}

TMatrixD* TFitParticleCart::getDerivative() {
  // returns derivative dP/dy with P=(p,E) and y=(px, py, pz, d) 
  // the free parameters of the fit. The columns of the matrix contain 
  // (dP/dpx, dP/dpy, ...).

  TMatrixD* DerivativeMatrix = new TMatrixD(4,4);
  (*DerivativeMatrix) *= 0.;

  //1st column: dP/dx
  (*DerivativeMatrix)(0,0)=1.;
  (*DerivativeMatrix)(1,0)=0.;
  (*DerivativeMatrix)(2,0)=0.;
  (*DerivativeMatrix)(3,0)=0.;

  //2nd column: dP/dy
  (*DerivativeMatrix)(0,1)=0;
  (*DerivativeMatrix)(1,1)=1;
  (*DerivativeMatrix)(2,1)=0;
  (*DerivativeMatrix)(3,1)=0.;

   //3rd column: dP/dz
  (*DerivativeMatrix)(0,2)=0.;
  (*DerivativeMatrix)(1,2)=0.;
  (*DerivativeMatrix)(2,2)=1.;
  (*DerivativeMatrix)(3,2)=0.;

   //4th column: dP/dm
  (*DerivativeMatrix)(0,3)=0.;
  (*DerivativeMatrix)(1,3)=0.;
  (*DerivativeMatrix)(2,3)=0.;
  (*DerivativeMatrix)(3,3)=_pini.M()*_pini.M()*_parameters(3,0)/_pcurr.E();

  return DerivativeMatrix;  
}

TMatrixD* TFitParticleCart::transform(const TLorentzVector& vec) {
  // Returns the parameters corresponding to the given 
  // 4vector

  TMatrixD* tparams = new TMatrixD( _nPar, 1 );
  (*tparams)(0,0) = vec.X();
  (*tparams)(1,0) = vec.Y();
  (*tparams)(2,0) = vec.Z();
  (*tparams)(3,0) = vec.M()/_pini.M();

  return tparams;

}
