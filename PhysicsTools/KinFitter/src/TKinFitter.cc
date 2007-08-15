// Classname: TKinFitter
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TKinFitter::
// --------------------
//
// Class to perform kinematic fit with non-linear constraints
//


using namespace std;

#include <iostream>
#include <iomanip>
#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitConstraint.h"

ClassImp(TKinFitter)

TKinFitter::TKinFitter():
  TNamed("UnNamed", "UnTitled"),
  _A(1, 1),
  _AT(1, 1),
  _B(1, 1),
  _BT(1, 1),
  _V(1, 1),
  _Vinv(1, 1),
  _VB(1, 1),
  _VBinv(1, 1),
  _VA(1, 1),
  _VAinv(1, 1),
  _c(1, 1),
  _C11(1, 1),
  _C11T(1, 1),
  _C21(1, 1),
  _C21T(1, 1),
  _C22(1, 1),
  _C22T(1, 1),
  _C31(1, 1),
  _C31T(1, 1),
  _C32(1, 1),
  _C32T(1, 1),
  _C33(1, 1),
  _C33T(1, 1),
  _deltaA(1, 1),
  _deltaY(1, 1),
  _lambda(1, 1),
  _lambdaT(1, 1),
  _lambdaVFit(1, 1),
  _yaVFit(1, 1),
  _constraints(0),
  _measParticles(0),
  _unmeasParticles(0)
{

  reset();

}

TKinFitter::TKinFitter(const TString &name, const TString &title):
  TNamed(name, title),
  _A(1, 1),
  _AT(1, 1),
  _B(1, 1),
  _BT(1, 1),
  _V(1, 1),
  _Vinv(1, 1),
  _VB(1, 1),
  _VBinv(1, 1),
  _VA(1, 1),
  _VAinv(1, 1),
  _c(1, 1),
  _C11(1, 1),
  _C11T(1, 1),
  _C21(1, 1),
  _C21T(1, 1),
  _C22(1, 1),
  _C22T(1, 1),
  _C31(1, 1),
  _C31T(1, 1),
  _C32(1, 1),
  _C32T(1, 1),
  _C33(1, 1),
  _C33T(1, 1),
  _deltaA(1, 1),
  _deltaY(1, 1),
  _lambda(1, 1),
  _lambdaT(1, 1),
  _lambdaVFit(1, 1),
  _yaVFit(1, 1),
  _constraints(0),
  _measParticles(0),
  _unmeasParticles(0)
{

  reset();

}

void TKinFitter::reset() {
  // reset all internal parameters of the fitter

  _status = -1;
  _nbIter = 0;
  _nParA = 0;
  _nParB = 0;
  _verbosity = 1;
  _A.ResizeTo(1, 1);
  _AT.ResizeTo(1, 1);
  _B.ResizeTo(1, 1);
  _BT.ResizeTo(1, 1);
  _V.ResizeTo(1, 1);
  _Vinv.ResizeTo(1, 1);
  _VB.ResizeTo(1, 1);
  _VBinv.ResizeTo(1, 1);
  _VA.ResizeTo(1, 1);
  _VAinv.ResizeTo(1, 1);
  _c.ResizeTo(1, 1);
  _C11.ResizeTo(1, 1);
  _C11T.ResizeTo(1, 1);
  _C21.ResizeTo(1, 1);
  _C21T.ResizeTo(1, 1);
  _C22.ResizeTo(1, 1);
  _C22T.ResizeTo(1, 1);
  _C31.ResizeTo(1, 1);
  _C31T.ResizeTo(1, 1);
  _C32.ResizeTo(1, 1);
  _C32T.ResizeTo(1, 1);
  _C33.ResizeTo(1, 1);
  _C33T.ResizeTo(1, 1);
  _deltaA.ResizeTo(1, 1);
  _deltaY.ResizeTo(1, 1);
  _lambda.ResizeTo(1, 1);
  _lambdaT.ResizeTo(1, 1);
  _lambdaVFit.ResizeTo(1, 1);
  _yaVFit.ResizeTo(1, 1);

  _constraints.clear();
  _measParticles.clear();
  _unmeasParticles.clear();

  // Set to default values
  _maxNbIter = 50;
  _maxDeltaS = 5e-3;
  _maxF =  1e-4;

}

void TKinFitter::resetStatus() {
  // reset status of the fit

  _status = -1;
  _nbIter = 0;

}

void TKinFitter::resetParams() {
  // reset all particles to their initial parameters

  for (unsigned int iP = 0; iP < _measParticles.size(); iP++) {
    TAbsFitParticle* particle = _measParticles[iP];
    particle->reset();
  }
  for (unsigned int iP = 0; iP < _unmeasParticles.size(); iP++) {
    TAbsFitParticle* particle = _unmeasParticles[iP];
    particle->reset();
  }
  for (unsigned int iC = 0; iC < _constraints.size(); iC++) {
    TAbsFitConstraint* constraint = _constraints[iC];
    constraint->reset();
  }

}

TKinFitter::~TKinFitter() {

}

void TKinFitter::countMeasParams() {
  // count number of measured parameters

  _nParB = 0;
  for (unsigned int indexParticle = 0; indexParticle < _measParticles.size(); indexParticle++) {
    _nParB += _measParticles[indexParticle]->getNPar();
  }
  for (unsigned int indexConstraint = 0; indexConstraint < _constraints.size(); indexConstraint++) {
    _nParB += _constraints[indexConstraint]->getNPar();
  }

}

void TKinFitter::countUnmeasParams() {
  // count number of unmeasured parameters

  _nParA = 0;
  for (unsigned int indexParticle = 0; indexParticle < _unmeasParticles.size(); indexParticle++) {
    _nParA += _unmeasParticles[indexParticle]->getNPar();
  }

}

void TKinFitter::addMeasParticle( TAbsFitParticle* particle ) {
  // add one measured particle

  resetStatus();

  if ( particle != 0 ) {
    _measParticles.push_back( particle );
  } else {
    cout << "Particle points to NULL." << endl;
  }

  countMeasParams();

}

void TKinFitter::addMeasParticles( TAbsFitParticle* p1, TAbsFitParticle* p2, TAbsFitParticle* p3, 
			      TAbsFitParticle* p4, TAbsFitParticle* p5, TAbsFitParticle* p6,
			      TAbsFitParticle* p7, TAbsFitParticle* p8, TAbsFitParticle* p9) {
  // add many measured particles

  resetStatus();

  if ( p1 != 0 ) _measParticles.push_back( p1 );
  if ( p2 != 0 ) _measParticles.push_back( p2 );
  if ( p3 != 0 ) _measParticles.push_back( p3 );
  if ( p4 != 0 ) _measParticles.push_back( p4 );
  if ( p5 != 0 ) _measParticles.push_back( p5 );
  if ( p6 != 0 ) _measParticles.push_back( p6 );
  if ( p7 != 0 ) _measParticles.push_back( p7 );
  if ( p8 != 0 ) _measParticles.push_back( p8 );
  if ( p9 != 0 ) _measParticles.push_back( p9 );

  countMeasParams();

}

void TKinFitter::addUnmeasParticle( TAbsFitParticle* particle ) {
  // add one unmeasured particle

  resetStatus();

  if ( particle != 0 ) {
    _unmeasParticles.push_back( particle );
  } else {
    cout << "Particle points to NULL." << endl;
  }

  countUnmeasParams();

}

void TKinFitter::addUnmeasParticles( TAbsFitParticle* p1, TAbsFitParticle* p2, TAbsFitParticle* p3, 
				     TAbsFitParticle* p4, TAbsFitParticle* p5, TAbsFitParticle* p6,
				     TAbsFitParticle* p7, TAbsFitParticle* p8, TAbsFitParticle* p9) {
  // add many unmeasured particles

  resetStatus();

  if ( p1 != 0 ) _unmeasParticles.push_back( p1 );
  if ( p2 != 0 ) _unmeasParticles.push_back( p2 );
  if ( p3 != 0 ) _unmeasParticles.push_back( p3 );
  if ( p4 != 0 ) _unmeasParticles.push_back( p4 );
  if ( p5 != 0 ) _unmeasParticles.push_back( p5 );
  if ( p6 != 0 ) _unmeasParticles.push_back( p6 );
  if ( p7 != 0 ) _unmeasParticles.push_back( p7 );
  if ( p8 != 0 ) _unmeasParticles.push_back( p8 );
  if ( p9 != 0 ) _unmeasParticles.push_back( p9 );

  countUnmeasParams();

}

void TKinFitter::addConstraint( TAbsFitConstraint* constraint ) {
  // add a constraint

  resetStatus();

  if ( constraint != 0 ) {
    _constraints.push_back( constraint );
  }

  countMeasParams();

}

void TKinFitter::setVerbosity( Int_t verbosity ) { 
  // Set verbosity of the fitter:
  // 0: quiet
  // 1: print information before and after the fit
  // 2: print output for every iteration of the fit
  // 3: print debug information


  if ( verbosity < 0 ) verbosity = 0;
  if ( verbosity > 3 ) verbosity = 3;
  _verbosity = verbosity;

}


Int_t TKinFitter::fit() {
  // Perform the fit
  // Returns:
  // 0: converged
  // 1: not converged

  resetParams();
  resetStatus();

  _nbIter = 0;
  Bool_t isConverged = false;
  Double_t prevF;
  Double_t currF = getF();
  Double_t prevS;
  Double_t currS = 0.;

  // Calculate covariance matrix V
  calcV();

  // print status
  if ( _verbosity >= 1 ) {
    print();
  }

  do {

    // Reset status to "RUNNING"
    _status = 10;

    // Calculate measured matrices
    calcB();
    calcVB();
    calcC31();
    calcC33();
    calcC();

    // Calculate unmeasured
    if ( _nParA > 0 ) {
      calcA();
      calcVA();
      calcC32();
      calcDeltaA();
    }

    // Calculate corretion for a, y, and lambda
    calcDeltaY();
    calcLambda();
   
    if( _verbosity >= 3 ) {
      cout << endl << "================   A   ================" << endl;
      _A.Print();
      cout << endl << "================   B   ================" << endl;
      _B.Print();
      cout << endl << "================  VBinv   ================" << endl;
      _VBinv.Print();
      cout << endl << "================  VB   ================" << endl;
      _VB.Print();
      cout << endl << "================  V   ================" << endl;
      _V.Print();
      cout << endl << "================  deltaY   ================" << endl;
      _deltaY.Print();
      cout << endl << "================  deltaA   ================" << endl;
      _deltaA.Print();
      cout << endl << "================  C32T   ================" << endl;
      _C32T.Print();
      cout << endl << "================  C   ================" << endl;
      _c.Print();
    }

    if( _verbosity >= 2 ) {
      cout << endl << endl << endl << endl;
      print();   
      cout <<"---------" <<endl ;
      cout << endl << endl << endl << endl;
    }

    // Apply calculated corrections to measured and unmeasured particles
    if ( _nParA > 0 ) {
      applyDeltaA();
    }
    applyDeltaY();

    _nbIter++;
    
    //calculate F and S
    prevF = currF;
    currF = getF();
    prevS = currS;
    currS = getS();

    // If S or F are getting bigger reduce step width
//     Int_t nstep =0;
//     while ( currF >= prevF ) {
//       nstep++;
//       if (nstep < 6) {cout <<nstep <<" : currF: "<< currF << "\t , prevF: " << prevF << endl;}
//       //      cout << "Reducing step width ..." << endl;
//       _deltaA *= (1.-0.001);
//       _deltaY *= (1.-0.001);
// //       _lambda *= 0.00001;
// //       _lambdaT *= 0.00001;
//       applyDeltaA();
//       applyDeltaY();
//       currF = getF();
//       currS = getS();
//     }
    
    // Test convergence
    isConverged = converged(currF, prevS, currS);

 
  } while ( (! isConverged) && (_nbIter < _maxNbIter) );

  // Calculate covariance matrices
  calcB();
  calcVB();

  if ( _nParA > 0 ) {
    calcA();
    calcVA();
    calcC21();
    calcC22();
    calcC32();
  }
  calcC11();
  calcC31();
  calcC33();
  calcVFit();
  applyVFit();

  // Set status information
  if (! isConverged ) {
    _status = 1;
  } else {
    _status = 0;
  }

  // print status
  if ( _verbosity >= 1 ) {
    print();
  }

  return _status;

}

void TKinFitter::setCovMatrix( TMatrixD &V ) {
  // Set the covariance matrix of the measured particles

  if ( (V.GetNrows() != _nParB) || (V.GetNcols() != _nParB) ) {
    cout << "Matrix needs to be a " << _nParB << "x" << _nParB << " matrix." << endl;
  } else {
    _V.ResizeTo( V );
    _V = V;
  }

}

Bool_t TKinFitter::calcV() {
  // Combines the covariance matrices of all measured particles

  _V.ResizeTo( _nParB, _nParB );
  _V.Zero();

  Int_t offsetP = 0;
  for (unsigned int iP = 0; iP < _measParticles.size(); iP++) {
    TAbsFitParticle* particle = _measParticles[iP];
    Int_t nParP = particle->getNPar();
    const TMatrixD* covMatrix =  particle->getCovMatrix();

    for (int iX = offsetP; iX < offsetP + nParP; iX++) {
      for (int iY = offsetP; iY < offsetP + nParP; iY++) {

	_V(iX, iY) = (*covMatrix)(iX-offsetP, iY-offsetP);

      }
    }

    offsetP += nParP;
  }

  for (unsigned int iC = 0; iC < _constraints.size(); iC++) {
    TAbsFitConstraint* constraint = _constraints[iC];
    Int_t nParP = constraint->getNPar();
    const TMatrixD* covMatrix =  constraint->getCovMatrix();

    for (int iX = offsetP; iX < offsetP + nParP; iX++) {
      for (int iY = offsetP; iY < offsetP + nParP; iY++) {

	_V(iX, iY) = (*covMatrix)(iX-offsetP, iY-offsetP);

      }
    }

    offsetP += nParP;
  }

  _Vinv.ResizeTo( _V );
  _Vinv = _V;
  _Vinv.Invert();

  return true;

}

Bool_t TKinFitter::calcA() {
  // Calculate the Jacobi matrix of unmeasured parameters df_i/da_i
  // Row i contains the derivatives of constraint f_i. Column q contains
  // the derivative wrt. a_q.

  _A.ResizeTo( _constraints.size(), _nParA );

  for (unsigned int indexConstr = 0; indexConstr < _constraints.size(); indexConstr++) {

    int offsetParam = 0;
    for (unsigned int indexParticle = 0; indexParticle < _unmeasParticles.size(); indexParticle++) {

      // Calculate matrix product  df/dP * dP/dy = (df/dr, df/dtheta, df/dphi, ...)

      TAbsFitParticle* particle = _unmeasParticles[indexParticle];
      TMatrixD* derivParticle = particle->getDerivative();
      TMatrixD* derivConstr = _constraints[indexConstr]->getDerivative( particle );
      TMatrixD deriv( *derivConstr, TMatrixD::kMult, *derivParticle );

      for (int indexParam = 0; indexParam < deriv.GetNcols(); indexParam++) {
	_A(indexConstr,indexParam+offsetParam) = deriv(0, indexParam);
      }
      offsetParam += deriv.GetNcols();

      delete derivParticle;
      delete derivConstr;

    }
  }

  // Calculate transposed matrix
  TMatrixD AT(TMatrixD::kTransposed, _A);
  _AT.ResizeTo( AT );
  _AT = AT;

  return true;

}

Bool_t TKinFitter::calcB() {
  // Calculate the Jacobi matrix of measured parameters df_i/da_i
  // Row i contains the derivatives of constraint f_i. Column q contains
  // the derivative wrt. a_q.

  _B.ResizeTo( _constraints.size(), _nParB );

  int offsetParam = 0;
  for (unsigned int indexConstr = 0; indexConstr < _constraints.size(); indexConstr++) {

    offsetParam = 0;
    for (unsigned int indexParticle = 0; indexParticle < _measParticles.size(); indexParticle++) {

      // Calculate matrix product  df/dP * dP/dy = (df/dr, df/dtheta, df/dphi, ...)
      TAbsFitParticle* particle = _measParticles[indexParticle];
      TMatrixD* derivParticle = particle->getDerivative();
      TMatrixD* derivConstr = _constraints[indexConstr]->getDerivative( particle );
      TMatrixD deriv( *derivConstr,  TMatrixD::kMult, *derivParticle );
      if (_verbosity >= 3) {
	cout << endl << "===  B deriv: Particle -> " << particle->GetName()<<" Constraint -> " 
	     << _constraints[indexConstr]->GetName()<< "===" << endl;
	derivParticle->Print();
	derivConstr->Print();
      }	
      for (int indexParam = 0; indexParam < deriv.GetNcols(); indexParam++) {
	_B(indexConstr,indexParam+offsetParam) = deriv(0, indexParam);
      }
      offsetParam += deriv.GetNcols();

      delete derivParticle;
      delete derivConstr;

    }
  }

  for (unsigned int iC = 0; iC < _constraints.size(); iC++) {

    TAbsFitConstraint* constraint = _constraints[iC];
    TMatrixD* deriv = constraint->getDerivativeAlpha();

    if (deriv != 0) {

      if (_verbosity >= 3) {
	cout << endl << "===  B deriv alpha: Constraint -> " 
	     << constraint->GetName() << "===" << endl;
	deriv->Print();
      }	
      for (int indexParam = 0; indexParam < deriv->GetNcols(); indexParam++) {
	_B( iC, indexParam+offsetParam ) = (*deriv)(0, indexParam);
      }
      offsetParam += deriv->GetNcols();
      
      delete deriv;
    }
  }

  TMatrixD BT( TMatrixD::kTransposed,  _B );
  _BT.ResizeTo( BT );
  _BT = BT;

  return true;

}

Bool_t TKinFitter::calcVB() {
  // Calculate the matrix V_B = (B*V*B^T)^-1

  TMatrixD BV( _B, TMatrixD::kMult, _V );
  TMatrixD VBinv( BV, TMatrixD::kMult, _BT );
  _VBinv.ResizeTo( VBinv );
  _VBinv = VBinv;

  _VB.ResizeTo( _VBinv );
  _VB = _VBinv;
  _VB.Invert();

  return true;

}

Bool_t TKinFitter::calcVA() {
  // Calculate the matrix VA = (A^T*VB*A)

  TMatrixD ATVB( _AT, TMatrixD::kMult, _VB );
  TMatrixD VA(ATVB, TMatrixD::kMult, _A);
  _VA.ResizeTo( VA );
  _VA = VA;

  _VAinv.ResizeTo( _VA );
  _VAinv = _VA;
  _VAinv.Invert();

  return true;

}

Bool_t TKinFitter::calcC11() {
  // Calculate the matrix C11 = V^(-1) - V^(-1)*BT*VB*B*V^(-1) + V^(-1)*BT*VB*A*VA^(-1)*AT*VB*B*V^(-1)

  TMatrixD VBT( _V, TMatrixD::kMult, _BT );
  TMatrixD VBB( _VB, TMatrixD::kMult, _B );
  TMatrixD VBTVBB( VBT, TMatrixD::kMult, VBB );
  TMatrixD m2( VBTVBB,  TMatrixD::kMult, _V );

  _C11.ResizeTo( _V );
  _C11 = _V;
  _C11 -= m2;

  if ( _nParA > 0 ) {
    TMatrixD VBA( _VB, TMatrixD::kMult, _A );
    TMatrixD VBTVBA( VBT, TMatrixD::kMult, VBA );
    TMatrixD VAinvAT( _VAinv, TMatrixD::kMult, _AT );
    TMatrixD VBTVBAVAinvAT( VBTVBA, TMatrixD::kMult, VAinvAT );
    TMatrixD VBTVBAVAinvATVBB( VBTVBAVAinvAT, TMatrixD::kMult, VBB );
    TMatrixD m3( VBTVBAVAinvATVBB, TMatrixD::kMult, _V );
    _C11 += m3;
  }

  TMatrixD C11T( TMatrixD::kTransposed,  _C11 );
  _C11T.ResizeTo( C11T );
  _C11T = C11T;

  return true;

}

Bool_t TKinFitter::calcC21() {
  // Calculate the matrix  C21 = -VA^(-1)*AT*VB*B*V^(-1)

  TMatrixD VAinvAT( _VAinv, TMatrixD::kMult, _AT );
  TMatrixD VBB( _VB, TMatrixD::kMult, _B );
  TMatrixD VAinvATVBB( VAinvAT, TMatrixD::kMult, VBB );
  TMatrixD C21( VAinvATVBB, TMatrixD::kMult, _V );
  C21 *= -1.;
  _C21.ResizeTo( C21 );
  _C21 = C21;
  
  TMatrixD C21T( TMatrixD::kTransposed,  _C21 );
  _C21T.ResizeTo( C21T );
  _C21T = C21T;

  return true;

}

Bool_t TKinFitter::calcC22() {
  //  Calculate the matrix C22 = VA^(-1)

  _C22.ResizeTo( _VAinv );
  _C22 = _VAinv;

  TMatrixD C22T( TMatrixD::kTransposed,  _C22 );
  _C22T.ResizeTo( C22T );
  _C22T = C22T;

  return true;

}

Bool_t TKinFitter::calcC31() {
  // Calculate the matrix  C31 = VB*B*V^(-1) - VB*A*VA^(-1)*AT*VB*B*V^(-1)

  TMatrixD VbB(_VB, TMatrixD::kMult, _B);
  TMatrixD m1( VbB, TMatrixD::kMult, _V );

  _C31.ResizeTo( m1 );
  _C31 = m1;

  if ( _nParA > 0 ) {
    TMatrixD VbA(_VB, TMatrixD::kMult, _A);
    TMatrixD VAinvAT( _VAinv, TMatrixD::kMult, _AT );
    TMatrixD VbBV( VbB,  TMatrixD::kMult, _V );
    TMatrixD VbAVAinvAT(VbA, TMatrixD::kMult, VAinvAT); 
    TMatrixD m2(VbAVAinvAT, TMatrixD::kMult, VbBV);

    _C31 -= m2;
  }

  TMatrixD C31T( TMatrixD::kTransposed,  _C31 );
  _C31T.ResizeTo( C31T );
  _C31T = C31T;

  return true;

}

Bool_t TKinFitter::calcC32() {
  // Calculate the matrix  C32 = VB*A*VA^(-1)

  TMatrixD VbA( _VB, TMatrixD::kMult, _A );
  TMatrixD C32( VbA, TMatrixD::kMult, _VAinv );
  _C32.ResizeTo( C32 );
  _C32 = C32;

  TMatrixD C32T( TMatrixD::kTransposed,  _C32 );
  _C32T.ResizeTo( C32T );
  _C32T = C32T;

  return true;

}

Bool_t TKinFitter::calcC33() {
  // Calculate the matrix C33 = -VB + VB*A*VA^(-1)*AT*VB

  _C33.ResizeTo( _VB );
  _C33 = _VB;
  _C33 *= -1.;

  if ( _nParA > 0 ) {
    TMatrixD VbA(_VB, TMatrixD::kMult, _A );
    TMatrixD VAinvAT( _VAinv, TMatrixD::kMult, _AT );
    TMatrixD VbAVAinvAT( VbA, TMatrixD::kMult, VAinvAT );
    TMatrixD C33( VbAVAinvAT,  TMatrixD::kMult, _VB );
    _C33 += C33;
  }

  TMatrixD C33T( TMatrixD::kTransposed,  _C33 );
  _C33T.ResizeTo( C33T );
  _C33T = C33T;

  return true;
}

Bool_t TKinFitter::calcC() {
  // Calculate the matrix c = A*deltaAStar + B*deltaYStar - fStar

  int offsetParam = 0;

  // calculate delta(a*), = 0 in the first iteration
  TMatrixD deltaastar( 1, 1 );
  if ( _nParA > 0 ) {

    deltaastar.ResizeTo( _nParA, 1 );
    for (unsigned int indexParticle = 0; indexParticle < _unmeasParticles.size(); indexParticle++) {
    
      TAbsFitParticle* particle = _unmeasParticles[indexParticle];
      const TMatrixD* astar = particle->getParCurr();
      const TMatrixD* a = particle->getParIni();
      TMatrixD deltaastarpart(*astar);
      deltaastarpart -= *a;
      
      for (int indexParam = 0; indexParam < deltaastarpart.GetNrows(); indexParam++) {
	deltaastar(indexParam+offsetParam, 0) = deltaastarpart(indexParam, 0);
      }
      offsetParam += deltaastarpart.GetNrows();
      
    }

    if ( _verbosity >= 3 ) {
      cout << "  ==== deltaastar =====" << endl;
      deltaastar.Print();
      cout << endl;
    }

  }

  // calculate delta(y*), = 0 in the first iteration
  TMatrixD deltaystar( _nParB, 1 );
  offsetParam = 0;
  for (unsigned int indexParticle = 0; indexParticle < _measParticles.size(); indexParticle++) {

    TAbsFitParticle* particle = _measParticles[indexParticle];
    const TMatrixD* ystar = particle->getParCurr();
    const TMatrixD* y = particle->getParIni();
    TMatrixD deltaystarpart(*ystar);
    deltaystarpart -= *y;

    for (int indexParam = 0; indexParam < deltaystarpart.GetNrows(); indexParam++) {
      deltaystar(indexParam+offsetParam, 0) = deltaystarpart(indexParam, 0);
    }
    offsetParam += deltaystarpart.GetNrows();

  } 

  for (unsigned int iC = 0; iC < _constraints.size(); iC++) {

    TAbsFitConstraint* constraint = _constraints[iC];
    
    if ( constraint->getNPar() > 0 ) {

      const TMatrixD* alphastar = constraint->getParCurr();
      const TMatrixD* alpha = constraint->getParIni();

      TMatrixD deltaalphastarpart(*alphastar);
      deltaalphastarpart -= *alpha;

      for (int indexParam = 0; indexParam < deltaalphastarpart.GetNrows(); indexParam++) {
	deltaystar(indexParam+offsetParam, 0) = deltaalphastarpart(indexParam, 0);
      }
      offsetParam += deltaalphastarpart.GetNrows();

    }
  }

  if ( _verbosity >= 3 ) {
    cout << "  ==== deltaystar =====" << endl;
    deltaystar.Print();
    cout << endl;
  }

  // calculate f*
  TMatrixD fstar( _constraints.size(), 1 );
  for (unsigned int indexConstr = 0; indexConstr < _constraints.size(); indexConstr++) {
    fstar( indexConstr, 0 ) = _constraints[indexConstr]->getCurrentValue();
  }

  // calculate c
  _c.ResizeTo( fstar );
  _c = fstar;
  _c *= (-1.);
  TMatrixD Bdeltaystar( _B, TMatrixD::kMult, deltaystar );
  _c += Bdeltaystar;
  if ( _nParA ) {
    TMatrixD Adeltaastar( _A, TMatrixD::kMult, deltaastar );
    _c += Adeltaastar;
  }

  return true;

}

Bool_t TKinFitter::calcDeltaA() {
  // Calculate the matrix deltaA = C32T * c
  // (corrections to unmeasured parameters)

  TMatrixD deltaA( _C32T, TMatrixD::kMult, _c );
  _deltaA.ResizeTo( deltaA );
  _deltaA = deltaA;

  return true;

}

Bool_t TKinFitter::calcDeltaY() {
  // Calculate the matrix deltaY = C31T * c 
  // (corrections to measured parameters)

  TMatrixD deltaY( _C31T, TMatrixD::kMult, _c );
  _deltaY.ResizeTo( deltaY );
  _deltaY = deltaY;

  return true;

}

Bool_t TKinFitter::calcLambda() {
  // Calculate the matrix Lambda = C33 * c 
  // (Lagrange Multipliers)

  TMatrixD lambda( _C33,  TMatrixD::kMult, _c );
  _lambda.ResizeTo( lambda );
  _lambda = lambda;

  TMatrixD lambdaT( TMatrixD::kTransposed,  _lambda );
  _lambdaT.ResizeTo( lambdaT );
  _lambdaT = lambdaT;

  return true;

}

Bool_t TKinFitter::calcVFit() {
  // Calculate the covariance matrix of fitted parameters
  //
  // Vfit(y) = ( C11  C21T )
  //     (a)   ( C21  C22  )
  //
  // Vfit(lambda) = (-C33)
  
  // Calculate covariance matrix of lambda
  _lambdaVFit.ResizeTo( _C33 );
  _lambdaVFit = _C33;
  _lambdaVFit *= -1.;


  // Calculate combined covariance matrix of y and a
  Int_t nbRows = _C11.GetNrows();
  Int_t nbCols = _C11.GetNcols();
  if ( _nParA > 0 ) {
    nbRows += _C21.GetNrows();
    nbCols += _C21T.GetNcols();
  }
  _yaVFit.ResizeTo( nbRows, nbCols );

  for (int iRow = 0; iRow < nbRows; iRow++) {
    for (int iCol = 0; iCol < nbCols; iCol++) {

      if (iRow >= _C11.GetNrows()) {
	if (iCol >= _C11.GetNcols()) {
	  _yaVFit(iRow, iCol) = _C22(iRow-_C11.GetNrows(), iCol-_C11.GetNcols());
	} else {
	  _yaVFit(iRow, iCol) = _C21(iRow-_C11.GetNrows(), iCol);
	}
      } else {
	if (iCol >= _C11.GetNcols()) {
	  _yaVFit(iRow, iCol) = _C21T(iRow, iCol-_C11.GetNcols());
	} else {
	  _yaVFit(iRow, iCol) = _C11(iRow, iCol);
	}
      }

    }
  }

  return true;

}

void TKinFitter::applyVFit() {
  // apply fit covariance matrix to measured and unmeasured  particles

  int offsetParam = 0;
  for (unsigned int indexParticle = 0; indexParticle < _measParticles.size(); indexParticle++) {
    TAbsFitParticle* particle = _measParticles[indexParticle];
    Int_t nbParams = particle->getNPar();
    TMatrixD vfit( nbParams, nbParams );
    for (Int_t c = 0; c < nbParams; c++) {
      for (Int_t r = 0; r < nbParams; r++) {
	vfit(r, c) = _yaVFit(r + offsetParam, c + offsetParam);
      }
    }
    particle->setCovMatrixFit( &vfit );
    offsetParam += nbParams;
  }

  for (unsigned int indexConstraint = 0; indexConstraint < _constraints.size(); indexConstraint++) {
    TAbsFitConstraint* constraint = _constraints[indexConstraint];
    Int_t nbParams = constraint->getNPar();
    if (nbParams > 0) {
      TMatrixD vfit( nbParams, nbParams );
      for (Int_t c = 0; c < nbParams; c++) {
	for (Int_t r = 0; r < nbParams; r++) {
	  vfit(r, c) = _yaVFit(r + offsetParam, c + offsetParam);
	}
      }
      constraint->setCovMatrixFit( &vfit );
      offsetParam += nbParams;
    }
  }

  for (unsigned int indexParticle = 0; indexParticle < _unmeasParticles.size(); indexParticle++) {
    TAbsFitParticle* particle = _unmeasParticles[indexParticle];
    Int_t nbParams = particle->getNPar();
    TMatrixD vfit( nbParams, nbParams );
    for (Int_t c = 0; c < nbParams; c++) {
      for (Int_t r = 0; r < nbParams; r++) {
	vfit(r, c) = _yaVFit(r + offsetParam, c + offsetParam);
      }
    }
    particle->setCovMatrixFit( &vfit );
    offsetParam += nbParams;
  }

}

Bool_t TKinFitter::applyDeltaA() {
  //apply corrections to unmeasured particles

  int offsetParam = 0;
  for (unsigned int indexParticle = 0; indexParticle < _unmeasParticles.size(); indexParticle++) {

    TAbsFitParticle* particle = _unmeasParticles[indexParticle];
    Int_t nbParams = particle->getNPar();
    TMatrixD params( nbParams, 1 );
    for (Int_t index = 0; index < nbParams; index++) {
      params(index, 0) = _deltaA(index+offsetParam, 0);
    }
    particle->applycorr( &params );
    offsetParam+=nbParams;

  }

  return true;

}

Bool_t TKinFitter::applyDeltaY() {
  //apply corrections to measured particles

  int offsetParam = 0;
  for (unsigned int indexParticle = 0; indexParticle < _measParticles.size(); indexParticle++) {

    TAbsFitParticle* particle = _measParticles[indexParticle];
    Int_t nbParams = particle->getNPar();
    TMatrixD params( nbParams, 1 );
    for (Int_t index = 0; index < nbParams; index++) {
      params(index, 0) = _deltaY(index+offsetParam, 0);
    }
    particle->applycorr( &params );
    offsetParam+=nbParams;

  }

  for (unsigned int indexConstraint = 0; indexConstraint < _constraints.size(); indexConstraint++) {

    TAbsFitConstraint* constraint = _constraints[indexConstraint];
    Int_t nbParams = constraint->getNPar();
    if ( nbParams > 0 ) {
      TMatrixD params( nbParams, 1 );
      for (Int_t index = 0; index < nbParams; index++) {
	params(index, 0) = _deltaY(index+offsetParam, 0);
      }
      constraint->applyDeltaAlpha( &params );
      offsetParam+=nbParams;
    }

  }

  return true;

}
  
Double_t TKinFitter::getF() {
  // calculate current absolut value of constraints
  // F = Sum[ Abs(f_k( aStar, yStar)) ] 

  Double_t F = 0.;
  for (unsigned int indexConstr = 0; indexConstr < _constraints.size(); indexConstr++) {
    F += TMath::Abs( _constraints[indexConstr]->getCurrentValue() );
  }
  
  return F;

}

Double_t TKinFitter::getS() {
  // calculate current value of Chi2
  // S = deltaYT * V^-1 * deltaY

  Double_t S = 0.;

  if ( _nbIter > 0 ) {
    
    TMatrixD deltaYTVinv(_deltaY, TMatrixD::kTransposeMult, _Vinv);
    TMatrixD S2(deltaYTVinv, TMatrixD::kMult, _deltaY);
    S = S2(0,0);
         
  }

  return S;

}

Bool_t TKinFitter::converged( Double_t F, Double_t prevS, Double_t currS ) {
  // check whether convergence criteria are fulfilled

  Bool_t isConverged = false;
  
  // calculate F, delta(a) and delta(y) already applied
  isConverged = (F < _maxF);

  // Calculate current Chi^2 and delta(S)
  Double_t deltaS = currS - prevS;
  isConverged = isConverged && (TMath::Abs(deltaS) < _maxDeltaS);

  return isConverged;

}

TString TKinFitter::getStatusString() {

  TString statusstring = "";

  switch ( _status ) {
      
    case -1: {
      statusstring = "NO FIT PERFORMED";
      break;
    }
    case 10: {
      statusstring = "RUNNING";
      break;
    }
    case 0: {
      statusstring = "CONVERGED";
      break;
    }
    case 1: {
      statusstring = "NOT CONVERGED";
      break;
    }
  }
    
  return statusstring;

}

void TKinFitter::print() {

  cout << endl << endl;
  cout << setprecision( 4 );
  cout << "Status: " << getStatusString();
  cout << "   F=" << getF() << "   S=" << getS() << "   N=" << _nbIter << "   NDF=" << getNDF() << endl;
  cout << "measured particles:" << endl;
  Int_t parIndex = 0;
  for (unsigned int iP = 0; iP < _measParticles.size(); iP++) {
    TAbsFitParticle* particle = _measParticles[iP];
    Int_t nParP = particle->getNPar();
    const TMatrixD* par = particle->getParCurr();
    //const TMatrixD* parIni = particle->getParIni();
    const TMatrixD* covP = particle->getCovMatrix();
    cout << setw(3) << setiosflags(ios::right) << iP;
    cout << setw(15) << setiosflags(ios::right) << particle->GetName();
    cout << setw(3) << " ";
    for (int iPar = 0; iPar < nParP; iPar++) {
      if (iPar > 0) {
	cout << setiosflags(ios::right) << setw(21) << " ";
      }
      TString colstr = "";
      colstr += parIndex;
      colstr += ":";
      cout << setw(4) << colstr;
      cout << setw(2) << " ";   
      cout << setiosflags(ios::left) << setiosflags(ios::scientific) << setprecision(3);
      cout << setw(15) << (*par)(iPar, 0);
      if(_nbIter > 0 && _status < 10) {
	cout << setw(15) << TMath::Sqrt( _yaVFit(iPar, iPar) );
      } else {
	cout << setw(15) << " ";
      }
      cout << setw(15) << TMath::Sqrt( (*covP)(iPar, iPar) );
      cout << endl;
      parIndex++;
    }
    particle->print();
  }

  cout << "unmeasured particles:" << endl;
  parIndex = 0;
  for (unsigned int iP = 0; iP < _unmeasParticles.size(); iP++) {
    TAbsFitParticle* particle = _unmeasParticles[iP];
    Int_t nParP = particle->getNPar();
    const TMatrixD* par = particle->getParCurr();
    //const TMatrixD* parIni = particle->getParIni();
    cout << setw(3) << setiosflags(ios::right) << iP;
    cout << setw(15) << particle->GetName();
    cout << setw(3) << " ";
    for (int iPar = 0; iPar < nParP; iPar++) {
      if (iPar > 0) {
	cout << setiosflags(ios::right) << setw(21) << " ";
      }
      TString colstr = "";
      colstr += parIndex;
      colstr += ":";
      cout << setw(4) << colstr;
      cout << setw(2) << " ";

      cout << setiosflags(ios::left) << setiosflags(ios::scientific) << setprecision(3);
      cout << setw(15) << (*par)(iPar, 0);
      if(_nbIter > 0 && _status < 10) {
	cout << setw(15) << TMath::Sqrt( _yaVFit(iPar+_nParB, iPar+_nParB) );
      } else {
	cout << setw(15) << " ";
      }
      cout << endl;

      parIndex++;
    }
    particle->print();

  }
  cout << endl;
  cout << "constraints: "<< endl;
  for (unsigned int indexConstr = 0; indexConstr < _constraints.size(); indexConstr++) {
    _constraints[indexConstr]->print();
  }
  cout << endl;
}
