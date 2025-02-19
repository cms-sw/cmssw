#include "Alignment/SurveyAnalysis/interface/Chi2.h"

Chi2::Chi2(TMatrixD &m, TMatrixD &ym, TMatrixD &merrors) {
  
  TMatrixD mt = m;
  mt.T();
  TMatrixD yt = ym;
  yt.T();
  TMatrixD m_leftMatrix(mt*merrors*m);
  TMatrixD m_rightMatrix(mt*merrors*ym);
  leftMatrix.ResizeTo(m_leftMatrix.GetNrows(), m_leftMatrix.GetNcols());
  rightMatrix.ResizeTo(m_rightMatrix.GetNrows(), m_rightMatrix.GetNcols());
  covariance.ResizeTo(m_leftMatrix.GetNrows(), m_leftMatrix.GetNrows());
  rightMatrix = m_rightMatrix;
  leftMatrix = m_leftMatrix;
  covariance = m_leftMatrix.Invert();
  TMatrixD m_solution(covariance*m_rightMatrix);
  solution.ResizeTo(m_solution.GetNrows(), m_solution.GetNcols());
  solution = m_solution;
  TMatrixD m_Chi2((yt-m_solution.T()*mt)*merrors*(ym-m*solution));
  myChi2 = m_Chi2(0,0);
  dof = ym.GetNrows()-solution.GetNrows();


}

Chi2::~Chi2(){}

TMatrixD & Chi2::getCovariance() {return covariance;}
TMatrixD & Chi2::getSolution() {return solution;}
double Chi2::getChi2() {return myChi2;}
int Chi2::getDOF() {return dof;}

  
