#ifndef Alignment_SurveyAnalysis_Chi2_H
#define Alignment_SurveyAnalysis_Chi2_H

#include <TMatrixD.h>


class Chi2 {

 public:
  Chi2(TMatrixD &, TMatrixD &, TMatrixD &);
  ~Chi2();
  
  TMatrixD & getCovariance();
  TMatrixD & getSolution();
  double getChi2();
  int getDOF();
  
 private:
  
  double myChi2;
  int dof;
  TMatrixD covariance;
  TMatrixD leftMatrix;
  TMatrixD rightMatrix;
  TMatrixD solution;

};

#endif
