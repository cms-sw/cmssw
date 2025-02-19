/** \class DTSurveyChamber
 *
 *  Implements a chamber in the context of Drift Tube Survey Measurements  
 *  and calculates displacements and rotations for it.
 *
 *  $Date: 2011/09/14 15:03:57 $
 *  $Revision: 1.6 $
 *  \author Pablo Martinez Ruiz del Arbol
 */


#ifndef Alignment_SurveyAnalysis_DTSurveyChamber_H
#define Alignment_SurveyAnalysis_DTSurveyChamber_H

#include <vector>

#include "TMath.h"
#include "TMatrixD.h" 

class DTSurveyChamber {
  
  
 public:

  DTSurveyChamber(int, int, int, long);
  

  //Add a point to the Chamber
  void addPoint(int, const TMatrixD&, const TMatrixD&, const TMatrixD&);
  
  //Begin the chi2 computation
  int getNumberPoints() const { return pointNumber; }
  void compute();
  void printChamberInfo(); 
  long getId() const {return rawId;}
  float getDeltaX() const {return Solution(0,0);}
  float getDeltaY() const {return Solution(1,0);}
  float getDeltaZ() const {return Solution(2,0);}
  //My definition of the matrix rotation is not the same as the used in CMSSW  
  float getAlpha() const {return Solution(5,0);}
  float getBeta() const {return -1.0*Solution(4,0);}
  float getGamma() const {return Solution(3,0);}
  float getDeltaXError() const {return TMath::Sqrt(Covariance(0,0));}
  float getDeltaYError() const {return TMath::Sqrt(Covariance(1,1));}
  float getDeltaZError() const {return TMath::Sqrt(Covariance(2,2));}
  float getAlphaError() const {return TMath::Sqrt(Covariance(5,5));}
  float getBetaError() const {return TMath::Sqrt(Covariance(4,4));}
  float getGammaError() const {return TMath::Sqrt(Covariance(3,3));}

 
 private:

//   static const unsigned int MAX_PUNTOS = 16;

  TMatrixD & makeMatrix();
  TMatrixD & makeErrors();
  TMatrixD & makeVector();

  
  //Identifiers
  int wheel, station, sector;
  
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

std::ostream & operator<<(std::ostream &, const DTSurveyChamber&);



#endif
