#include "Alignment/SurveyAnalysis/interface/DTSurveyChamber.h"
#include "Alignment/SurveyAnalysis/interface/Chi2.h" 
#include "Geometry/DTGeometry/interface/DTChamber.h" 
#include "TMath.h"

#define MAX_PUNTOS 16


//Constructor

DTSurveyChamber::DTSurveyChamber(int m_wheel, int m_station, int m_sector, long m_rawId) {
  
  //Coordinates of the chamber
  wheel = m_wheel;
  station = m_station;
  sector = m_sector;
  pointNumber = 0;
  rawId = m_rawId;  
   
}

DTSurveyChamber::~DTSurveyChamber() {}


void DTSurveyChamber::compute() {
  
  TMatrixD leftMatrix = makeMatrix();
  TMatrixD rightMatrix = makeVector();
  TMatrixD errors = makeErrors();
  
  Chi2 myChi2(leftMatrix, rightMatrix, errors);
  
  Solution.ResizeTo(6,1);
  Solution = myChi2.getSolution();
  Covariance.ResizeTo(6,6);
  Covariance = myChi2.getCovariance();

}


void DTSurveyChamber::addPoint(int code, TMatrixD r, TMatrixD disp, TMatrixD err) {
  
  
  pointNumber++;
  
  TMatrixD rLocal = r;
  TMatrixD rDisp = disp;
  TMatrixD rTeo = r-disp;
  TMatrixD rErr = err;

  points.push_back(rLocal);
  pointsDiff.push_back(rDisp);
  pointsError.push_back(rErr);
  pointsTheoretical.push_back(rTeo);

}
  

int DTSurveyChamber::getNumberPoints() {return pointNumber;}




TMatrixD & DTSurveyChamber::makeVector() {

  TMatrixD *result = new TMatrixD(3*getNumberPoints(),1);
  result->Zero();
  int real = 0;
  for(std::vector<TMatrixD >::iterator p = pointsDiff.begin(); p != pointsDiff.end(); ++p) {  
    (*result)(real*3,0) = (*p)(0,0);
    (*result)(real*3+1,0) = (*p)(1,0);
    (*result)(real*3+2,0) = (*p)(2,0);
    real++;
  }
  return *result;
}




TMatrixD & DTSurveyChamber::makeErrors() {
  
  TMatrixD *result = new TMatrixD(3*getNumberPoints(),3*getNumberPoints());
  result->Zero();
  int real = 0;
  for(std::vector<TMatrixD >::iterator p = pointsError.begin(); p != pointsError.end(); ++p) {
    double rmsn = 1.0/((*p)(0,0)*(*p)(0,0));
    (*result)(real*3,real*3) = rmsn;
    (*result)(real*3+1,real*3+1) = rmsn;
    (*result)(real*3+2,real*3+2) = rmsn;
    real++;
  }
  return *result;
}


TMatrixD & DTSurveyChamber::makeMatrix() {

  TMatrixD *result = new TMatrixD(3*getNumberPoints(), 6);
  result->Zero();
  int real = 0;
  for(std::vector<TMatrixD >::iterator p = pointsTheoretical.begin(); p != pointsTheoretical.end(); p++) {
    (*result)(real*3,0)= 1.0;
    (*result)(real*3,3) = (*p)(1,0);
    (*result)(real*3,4) = (*p)(2,0);
    (*result)(real*3+1,1) = 1.0;
    (*result)(real*3+1,3) = -(*p)(0,0);
    (*result)(real*3+1,5) = (*p)(2,0);
    (*result)(real*3+2,2) = 1.0;
    (*result)(real*3+2,4) = -(*p)(0,0);
    (*result)(real*3+2,5) = -(*p)(1,0);
    real++;
  }
  return *result;
}


long DTSurveyChamber::getId(){return rawId;}
float DTSurveyChamber::getDeltaX(){return Solution(0,0);}
float DTSurveyChamber::getDeltaY(){return Solution(1,0);}
float DTSurveyChamber::getDeltaZ(){return Solution(2,0);}
//My definition of the matrix rotation is not the same as the used in CMSSW   
float DTSurveyChamber::getAlpha(){return Solution(5,0);}
float DTSurveyChamber::getBeta(){return -1.0*Solution(4,0);}
float DTSurveyChamber::getGamma(){return Solution(3,0);}
float DTSurveyChamber::getDeltaXError(){return TMath::Sqrt(Covariance(0,0));}
float DTSurveyChamber::getDeltaYError(){return TMath::Sqrt(Covariance(1,1));}
float DTSurveyChamber::getDeltaZError(){return TMath::Sqrt(Covariance(2,2));}
float DTSurveyChamber::getAlphaError(){return TMath::Sqrt(Covariance(5,5));}
float DTSurveyChamber::getBetaError(){return TMath::Sqrt(Covariance(4,4));}
float DTSurveyChamber::getGammaError(){return TMath::Sqrt(Covariance(3,3));}



std::ostream &operator<<(std::ostream &flujo, DTSurveyChamber obj) {

  flujo << obj.getId() << " "
        << obj.getDeltaX() << " " << obj.getDeltaXError() << " "
        << obj.getDeltaY() << " " << obj.getDeltaYError() << " "
        << obj.getDeltaZ() << " " << obj.getDeltaZError() << " "
        << obj.getAlpha() << " " << obj.getAlphaError() << " "
        << obj.getBeta() << " " << obj.getBetaError() << " "
        << obj.getGamma() << " " << obj.getGammaError() << std::endl;
  return flujo;

}

