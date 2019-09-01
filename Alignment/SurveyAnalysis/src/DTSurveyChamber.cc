#include <iostream>

#include "Alignment/SurveyAnalysis/interface/DTSurveyChamber.h"
#include "Alignment/SurveyAnalysis/interface/Chi2.h"

DTSurveyChamber::DTSurveyChamber(int m_wheel, int m_station, int m_sector, long m_rawId) {
  //Coordinates of the chamber
  wheel = m_wheel;
  station = m_station;
  sector = m_sector;
  pointNumber = 0;
  rawId = m_rawId;
}

void DTSurveyChamber::compute() {
  TMatrixD leftMatrix = makeMatrix();
  TMatrixD rightMatrix = makeVector();
  TMatrixD errors = makeErrors();

  Chi2 myChi2(leftMatrix, rightMatrix, errors);

  Solution.ResizeTo(6, 1);
  Solution = myChi2.getSolution();
  Covariance.ResizeTo(6, 6);
  Covariance = myChi2.getCovariance();
}

void DTSurveyChamber::addPoint(int code, const TMatrixD &r, const TMatrixD &disp, const TMatrixD &err) {
  ++pointNumber;

  points.push_back(r);
  pointsDiff.push_back(disp);
  pointsError.push_back(err);
  pointsTheoretical.push_back(r - disp);
}

TMatrixD &DTSurveyChamber::makeVector() {
  TMatrixD *result = new TMatrixD(3 * getNumberPoints(), 1);
  result->Zero();
  int real = 0;
  for (std::vector<TMatrixD>::iterator p = pointsDiff.begin(); p != pointsDiff.end(); ++p) {
    (*result)(real * 3, 0) = (*p)(0, 0);
    (*result)(real * 3 + 1, 0) = (*p)(1, 0);
    (*result)(real * 3 + 2, 0) = (*p)(2, 0);
    ++real;
  }
  return *result;
}

TMatrixD &DTSurveyChamber::makeErrors() {
  TMatrixD *result = new TMatrixD(3 * getNumberPoints(), 3 * getNumberPoints());
  result->Zero();
  int real = 0;
  for (std::vector<TMatrixD>::iterator p = pointsError.begin(); p != pointsError.end(); ++p) {
    double rmsn = 1.0 / ((*p)(0, 0) * (*p)(0, 0));
    (*result)(real * 3, real * 3) = rmsn;
    (*result)(real * 3 + 1, real * 3 + 1) = rmsn;
    (*result)(real * 3 + 2, real * 3 + 2) = rmsn;
    real++;
  }
  return *result;
}

TMatrixD &DTSurveyChamber::makeMatrix() {
  TMatrixD *result = new TMatrixD(3 * getNumberPoints(), 6);
  result->Zero();
  int real = 0;
  for (std::vector<TMatrixD>::iterator p = pointsTheoretical.begin(); p != pointsTheoretical.end(); p++) {
    (*result)(real * 3, 0) = 1.0;
    (*result)(real * 3, 3) = (*p)(1, 0);
    (*result)(real * 3, 4) = (*p)(2, 0);
    (*result)(real * 3 + 1, 1) = 1.0;
    (*result)(real * 3 + 1, 3) = -(*p)(0, 0);
    (*result)(real * 3 + 1, 5) = (*p)(2, 0);
    (*result)(real * 3 + 2, 2) = 1.0;
    (*result)(real * 3 + 2, 4) = -(*p)(0, 0);
    (*result)(real * 3 + 2, 5) = -(*p)(1, 0);
    real++;
  }
  return *result;
}

std::ostream &operator<<(std::ostream &flujo, const DTSurveyChamber &obj) {
  flujo << obj.getId() << " " << obj.getDeltaX() << " " << obj.getDeltaXError() << " " << obj.getDeltaY() << " "
        << obj.getDeltaYError() << " " << obj.getDeltaZ() << " " << obj.getDeltaZError() << " " << obj.getAlpha() << " "
        << obj.getAlphaError() << " " << obj.getBeta() << " " << obj.getBetaError() << " " << obj.getGamma() << " "
        << obj.getGammaError() << std::endl;
  return flujo;
}
