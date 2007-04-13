#ifndef Alignment_SurveyAnalysis_DTSurveyChamber_H
#define Alignment_SurveyAnalysis_DTSurveyChamber_H

#include "Alignment/SurveyAnalysis/interface/Chi2.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "TMatrixD.h" 

//This class implements a chamber in the context of Survey Measurements
//The reference system has to be provided trough a text file called geometry.txt
//which should provide: wheel station sector ID centerX centerY centerY Rotation(0,0) Rotation(1,0) Rotation(2,0) ... Rotation(2,2)
//Points are read using standard codes from photogrametry
//Results are provided in local coordinates



class DTSurveyChamber {
  
  
 public:
  //Constructor & Destructor
  DTSurveyChamber(int, int, int, long);
  ~DTSurveyChamber();
  

  //Add a point to the Chamber
  void addPoint(int, TMatrixD, TMatrixD, TMatrixD);
  
  //Begin the chi2 computation
  int getNumberPoints();
  void compute();
  void printChamberInfo(); 
  long getId();
  float getDeltaX();
  float getDeltaY();
  float getDeltaZ();
  float getDeltaXError();
  float getDeltaYError();
  float getDeltaZError();
  float getAlpha();
  float getBeta();
  float getGamma();
  float getAlphaError();
  float getBetaError();
  float getGammaError();

 
 private:
  
  TMatrixD & makeMatrix();
  TMatrixD & makeErrors();
  TMatrixD & makeVector();

  
  //Identifiers
  int wheel, station, sector , id;
  
  long rawId;


  //Points data
  std::vector<TMatrixD> points;
  std::vector<TMatrixD> pointsDiff;
  std::vector<TMatrixD> pointsError;
  std::vector<TMatrixD> pointsTheoretical;
  
  TMatrixD Solution;
  TMatrixD Covariance;
   
  //Number of points
  int pointNumber;
  
};

ostream & operator<<(ostream &, DTSurveyChamber);



#endif
