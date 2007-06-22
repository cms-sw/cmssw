// Classname: TFitConstraintMGaus
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TFitConstraintMGaus::
// --------------------
//
// Fit constraint: mass conservation ( m_i - m_j - alpha * MassConstraint == 0 )
//

using namespace std;

#include <iostream>
#include "PhysicsTools/KinFitter/interface/TFitConstraintMGaus.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "TLorentzVector.h"
#include "TClass.h"

ClassImp(TFitConstraintMGaus)

//----------------
// Constructor --
//----------------
TFitConstraintMGaus::TFitConstraintMGaus()
  : TFitConstraintM() 
{

  init();

}

TFitConstraintMGaus::TFitConstraintMGaus(vector<TAbsFitParticle*>* ParList1,
					 vector<TAbsFitParticle*>* ParList2, 
					 Double_t Mass,
					 Double_t Width)
  : TFitConstraintM(ParList1, ParList2, Mass ) 
{

  init();
  setMassConstraint( Mass, Width );

}

TFitConstraintMGaus::TFitConstraintMGaus(const TString &name, const TString &title,
					 vector<TAbsFitParticle*>* ParList1,
					 vector<TAbsFitParticle*>* ParList2, 
					 Double_t Mass,
					 Double_t Width)
  : TFitConstraintM( name, title, ParList1, ParList2, Mass )
{

  init();
  setMassConstraint( Mass, Width );

}

void
TFitConstraintMGaus::init() {

  _nPar = 1;
  _iniparameters.ResizeTo(1,1);
  _iniparameters(0,0) = 1.;
  _parameters.ResizeTo(1,1);
  _parameters = _iniparameters;

}


//--------------
// Destructor --
//--------------
TFitConstraintMGaus::~TFitConstraintMGaus() {

}

//--------------
// Operations --
//--------------

void TFitConstraintMGaus::setMassConstraint(Double_t Mass, Double_t Width) { 
  
  _TheMassConstraint = Mass;
  _width = Width;
  setCovMatrix( 0 );
  _covMatrix(0,0) = (Width*Width) / (Mass * Mass);

}

Double_t TFitConstraintMGaus::getInitValue() {
  // Get initial value of constraint (before the fit)

  Double_t InitValue = 
    CalcMass( &_ParList1, true ) - 
    CalcMass( &_ParList2, true ) - 
    _iniparameters(0,0) * _TheMassConstraint;

  return InitValue;

}

Double_t TFitConstraintMGaus::getCurrentValue() {
  // Get value of constraint after the fit

  Double_t CurrentValue =
    CalcMass(&_ParList1,false) - 
    CalcMass(&_ParList2,false) - 
    _parameters(0,0)*_TheMassConstraint;

  return CurrentValue;

}

TMatrixD* TFitConstraintMGaus::getDerivativeAlpha() { 
  // Calculate dF/dAlpha = -1 * M

  TMatrixD* DerivativeMatrix = new TMatrixD(1,1);
  DerivativeMatrix->Zero();

  (*DerivativeMatrix)(0,0) = -1. * _TheMassConstraint;

  return DerivativeMatrix;

}

void TFitConstraintMGaus::print() {

  cout << "__________________________" << endl << endl;
  cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << endl;

  cout << "initial value: " << getInitValue() << endl;
  cout << "current value: " << getCurrentValue() << endl;
  cout << "mean mass: " << _TheMassConstraint << endl;
  cout << "width: " << _width << endl;
  cout << "initial mass: " << _iniparameters(0,0)*_TheMassConstraint  << endl;
  cout << "current mass: " << _parameters(0,0)*_TheMassConstraint  << endl;

}

