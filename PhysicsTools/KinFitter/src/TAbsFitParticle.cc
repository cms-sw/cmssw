// Classname: TAbsFitParticle
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TAbsFitParticle::
// --------------------
//
// Abstract base class for particles to be used with kinematic fitter
//


using namespace std;

#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include <iostream>
#include <iomanip>
#include "TClass.h"

ClassImp(TAbsFitParticle)

TAbsFitParticle::TAbsFitParticle():
  TNamed("NoName","NoTitle")
  ,_nPar(0) 	
  ,_u1()   
  ,_u2()
  ,_u3()
  ,_covMatrix()
  ,_covMatrixFit()
  ,_covMatrixDeltaY()
  ,_pull()
  ,_iniparameters(1,1)
  ,_parameters(1,1)
  ,_pini()
  ,_pcurr()
{
 
}

TAbsFitParticle::TAbsFitParticle(const TString &name, const TString &title ):
  TNamed(name,title)
  ,_nPar(0) 	
  ,_u1()   
  ,_u2()
  ,_u3()
  ,_covMatrix()
  ,_covMatrixFit()
  ,_covMatrixDeltaY()
  ,_pull()
  ,_iniparameters(1,1)
  ,_parameters(1,1)
  ,_pini()
  ,_pcurr()
{
 
}


TAbsFitParticle::~TAbsFitParticle() {

}

void 
TAbsFitParticle::print() {
  // Print particle contents

  cout << "__________________________" << endl << endl;
  cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << endl;

  cout << setw(22)<< "initial parameters:"  << setw(5)<<" " << setw(20)<< "current parameters:" << endl;
  for (int i = 0; i< _nPar ;i++){
    cout << "par[" << i << "] = " << setprecision(6) << setw(18)<< (*getParIni())(i,0) 
	 << setw(20) << setprecision(6)<< (*getParCurr())(i,0) <<endl;
  }

  cout << setw(22)<<"initial 4vector:" << setw(5)<<" " << setw(20) <<"current 4vector:"<< endl;
   for (int i = 0; i< 4 ;i++){
    cout << "p[" << i << "] = "<< setprecision(6) << setw(20) << (*getIni4Vec())[i] 
	 << setw(20)<< setprecision(6) << (*getCurr4Vec())[i] << endl;
   }
   cout <<  "mass = " << setprecision(6) << setw(20) << (*getIni4Vec()).M() 
	<< setw(20)<< setprecision(6)<< (*getCurr4Vec()).M()<< endl;

   cout << "u1  = " << _u1.X() << ", " << _u1.Y() << ", " << _u1.Z() << endl;
   cout << "u2  = " << _u2.X() << ", " << _u2.Y() << ", " << _u2.Z() << endl;
   cout << "u3  = " << _u3.X() << ", " << _u3.Y() << ", " << _u3.Z() << endl;


}

void TAbsFitParticle::reset() {
  // Reset particle to initial values

  _parameters = _iniparameters;  
  _pcurr = _pini;
  setCovMatrixFit( 0 );
  _pull.ResizeTo(_nPar, 1);
  _pull.Zero();

}

void TAbsFitParticle::setCovMatrix(const TMatrixD* theCovMatrix) {
  // Set the measured covariance matrix

  _covMatrix.ResizeTo(_nPar, _nPar);
  if(theCovMatrix==0) {
    _covMatrix.Zero();
  } else if (theCovMatrix->GetNcols() ==_nPar && theCovMatrix->GetNrows() ==_nPar) {
    _covMatrix = (*theCovMatrix);
  } else {
    cout <<"Something is wrong with the definition of the covariance matrix"<<endl;
  }

}


void TAbsFitParticle::setCovMatrixFit(const TMatrixD* theCovMatrixFit) {
  // Set the fitted covariance matrix

  _covMatrixFit.ResizeTo(_nPar, _nPar);
  if(theCovMatrixFit==0) {
    _covMatrixFit.Zero();
  } else if (theCovMatrixFit->GetNcols() ==_nPar && theCovMatrixFit->GetNrows() ==_nPar) {
    _covMatrixFit = (*theCovMatrixFit);
  } else {
    cout <<"Something is wrong with the definition of the fit covariance matrix"<<endl;
  }

}

void TAbsFitParticle::calcCovMatrixDeltaY() {
  // Calculates V(deltaY) ==  V(y_meas) - V(y_fit)

  _covMatrixDeltaY.ResizeTo( _nPar, _nPar );
  _covMatrixDeltaY = _covMatrix;
  if(_covMatrixFit.GetNrows() == _nPar && _covMatrixFit.GetNcols() == _nPar)
    _covMatrixDeltaY -= _covMatrixFit;
  else 
    cout << "_covMatrixFit probably not set" <<endl;  
}



const TMatrixD* TAbsFitParticle::getPull() {
  // Calculates the pull (y_fit - y_meas) / sigma
  // with sigma = Sqrt( sigma[y_meas]^2 - V[y_fit]^2 )
  // for all parameters


  _pull.ResizeTo( _nPar, 1 );
  _pull = _parameters;
  _pull -= _iniparameters;
  calcCovMatrixDeltaY(); 
  for (int i = 0; i<_nPar; i++) {
    Double_t sigmaDeltaY = _covMatrixDeltaY(i, i);
    if (sigmaDeltaY < 0) {
      cout << "V[deltaY] has negative diagonal element." << endl;
      _pull.Zero();
      return &_pull;
    } else {
      _pull(i,0) /= TMath::Sqrt( sigmaDeltaY );
    }
  }

  return &_pull;

}

void TAbsFitParticle::applycorr(TMatrixD* corrMatrix) {
  // Apply corrections to the parameters wrt. to the
  // initial parameters y* = y + delta(y)
  // This method will also calculate the fitted 
  // 4vector of the particle

  // update _parameters-Matrix
  _parameters = _iniparameters;
  _parameters += (*corrMatrix);

  // calculates new 4vec
  TLorentzVector* vec = calc4Vec( &_parameters );
  _pcurr = (*vec);
  delete vec;

}

void TAbsFitParticle::setParIni(const TMatrixD* parini) {
  if (parini == 0) return;
  else if( parini->GetNrows() == _iniparameters.GetNrows()
	   && parini->GetNcols() == _iniparameters.GetNcols() )
    _iniparameters = (*parini) ;
  else {
    cout << "TAbsFitParticle::setParIni : Matrices don't fit"<< endl;
    return;
      }
}

const TMatrixD* TAbsFitParticle::getCovMatrixDeltaY() {
  //
  calcCovMatrixDeltaY(); 
  return &_covMatrixDeltaY; 
}
