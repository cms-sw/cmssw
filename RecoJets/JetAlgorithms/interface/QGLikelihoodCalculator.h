#ifndef JetAlgorithms_QGLikelihoodCalculator_h
#define JetAlgorithms_QGLikelihoodCalculator_h

#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "FWCore/Framework/interface/ESHandle.h"

/**
 * The QGLikelihoodCalculater calculates the likelihood for a jet
 * It takes information on the valid range of the tool, the binning of the categories, and their PDFs from the QGLikelihoodObject
 * The variables in the vars vector should match with the variables in the QGLikelihoodObject, in which they are identified by the varIndex
 * Authors: andrea.carlo.marini@cern.ch, tom.cornelis@cern.ch, cms-qg-workinggroup@cern.ch
 */
class QGLikelihoodCalculator {
public:
  QGLikelihoodCalculator(){};
  ~QGLikelihoodCalculator(){};

  float computeQGLikelihood(
      edm::ESHandle<QGLikelihoodObject> &QGLParamsColl, float pt, float eta, float rho, std::vector<float> vars) const;
  float systematicSmearing(edm::ESHandle<QGLikelihoodSystematicsObject> &QGLParamsColl,
                           float pt,
                           float eta,
                           float rho,
                           float qgValue,
                           int qgIndex) const;

private:
  const QGLikelihoodObject::Entry *findEntry(std::vector<QGLikelihoodObject::Entry> const &data,
                                             float eta,
                                             float pt,
                                             float rho,
                                             int qgIndex,
                                             int varIndex) const;
  bool isValidRange(float pt, float rho, float eta, const QGLikelihoodCategory &qgValidRange) const;
  float smearingFunction(float x0, float a, float b, float min, float max) const;
};

#endif
