/** \class DTSurveyChamber
 *
 *  Implements a chamber in the context of Drift Tube Survey Measurements  
 *  and calculates displacements and rotations for it.
 *
 *  $Date: 2007/04/17 07:45:02 $
 *  $Revision: 1.2 $
 *  \author Pablo Martinez Ruiz del Arbol
 */


#ifndef Alignment_SurveyAnalysis_DTSurveyChamber_H
#define Alignment_SurveyAnalysis_DTSurveyChamber_H


#include "TMatrixD.h" 


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
