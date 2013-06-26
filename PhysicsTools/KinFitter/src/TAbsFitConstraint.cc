// Classname: TAbsFitConstraint
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TAbsFitConstraint::
// --------------------
//
// Abstract base class for fit constraints
//

#include "PhysicsTools/KinFitter/interface/TAbsFitConstraint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include "TClass.h"


TAbsFitConstraint::TAbsFitConstraint() 
  : TNamed("NoName","NoTitle")
  ,_covMatrix()
  ,_covMatrixFit()
  ,_covMatrixDeltaAlpha()
  ,_iniparameters()
  ,_parameters()

{
  _nPar = 0;
}

TAbsFitConstraint::TAbsFitConstraint(const TString &name, const TString &title) 
  : TNamed(name, title)
  ,_covMatrix()
  ,_covMatrixFit()
  ,_covMatrixDeltaAlpha()
  ,_iniparameters()
  ,_parameters()

{
  _nPar = 0;
}

TAbsFitConstraint::~TAbsFitConstraint() {

}

void TAbsFitConstraint::reset() {
  // Reset parameters to initial values

  _parameters = _iniparameters;  
  setCovMatrixFit( 0 );

}

void TAbsFitConstraint::setCovMatrix(const TMatrixD* theCovMatrix) {
  // Set measured alpha covariance matrix

  _covMatrix.ResizeTo(_nPar, _nPar);
  if(theCovMatrix==0) {
    _covMatrix.Zero();
  } else if (theCovMatrix->GetNcols() ==_nPar && theCovMatrix->GetNrows() ==_nPar) {
    _covMatrix = (*theCovMatrix);
  } else {
    edm::LogError ("WrongMatrixSize")
      << GetName() << "::setCovMatrix - Measured alpha covariance matrix needs to be a "
      << _nPar << "x" << _nPar << " matrix.";
  }

}

void TAbsFitConstraint::setCovMatrixFit(const TMatrixD* theCovMatrixFit) {
  // Set the fitted covariance matrix

  _covMatrixFit.ResizeTo(_nPar, _nPar);
  if(theCovMatrixFit==0) {
    _covMatrixFit.Zero();
  } else if (theCovMatrixFit->GetNcols() ==_nPar && theCovMatrixFit->GetNrows() ==_nPar) {
    _covMatrixFit = (*theCovMatrixFit);
  } else {
    edm::LogError ("WrongMatrixSize")
      << GetName() << "::setCovMatrixFit - Fitted covariance matrix needs to be a "
      << _nPar << "x" << _nPar << " matrix.";
  }

}

void TAbsFitConstraint::calcCovMatrixDeltaAlpha() {
  // Calculates V(deltaAlpha) ==  V(alpha_meas) - V(alpha_fit)

  _covMatrixDeltaAlpha.ResizeTo( _nPar, _nPar );
  _covMatrixDeltaAlpha = _covMatrix;
  if(_covMatrixFit.GetNrows() == _nPar && _covMatrixFit.GetNcols() == _nPar)
    _covMatrixDeltaAlpha -= _covMatrixFit;
  else
    edm::LogError ("WrongMatrixSize")
      << GetName() << "::calcCovMatrixDeltaAlpha - _covMatrixFit probably not set.";
}


void TAbsFitConstraint::applyDeltaAlpha(TMatrixD* corrMatrix) {
  // Apply corrections to the parameters wrt. to the
  // initial parameters alpha* = alpha + delta(alpha)

  _parameters = _iniparameters;
  _parameters += (*corrMatrix);

}

void TAbsFitConstraint::setParIni(const TMatrixD* parini) {
  // Set initial parameter values (before the fit)

  if (parini == 0) return;
  else if( parini->GetNrows() == _iniparameters.GetNrows() &&
	   parini->GetNcols() == _iniparameters.GetNcols() )
    _iniparameters = (*parini) ;
  else {
    edm::LogError ("WrongMatrixSize")
      << GetName() << "::setParIni - Matrices don't fit.";
    return;
  }

}

const TMatrixD* TAbsFitConstraint::getCovMatrixDeltaAlpha() {
  // Returns covariance matrix delta(alpha)

  calcCovMatrixDeltaAlpha(); 
  return &_covMatrixDeltaAlpha; 

}

TString TAbsFitConstraint::getInfoString() {
  // Collect information to be used for printout

  std::stringstream info;
  info << std::scientific << std::setprecision(6);

  info << "__________________________" << std::endl
       << std::endl;
  info <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << std::endl;
  
  info << "initial value: " << getInitValue() << std::endl;
  info << "current value: " << getCurrentValue() << std::endl;

  return info.str();

}

void TAbsFitConstraint::print() {
  // Print constraint contents

  edm::LogVerbatim("KinFitter") << this->getInfoString();

}

