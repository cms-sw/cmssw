#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <math.h>

/// Compute likelihood for a jet using the QGLikelihoodObject information and a set of variables
float QGLikelihoodCalculator::computeQGLikelihood(edm::ESHandle<QGLikelihoodObject> &QGLParamsColl, float pt, float eta, float rho, std::vector<float> vars){
  if(!isValidRange(pt, rho, eta, QGLParamsColl->qgValidRange)) return -1;

  float Q=1., G=1.;
  for(unsigned int varIndex = 0; varIndex < vars.size(); ++varIndex){

    auto quarkEntry = findEntry(QGLParamsColl->data, eta, pt, rho, 0, varIndex);
    auto gluonEntry = findEntry(QGLParamsColl->data, eta, pt, rho, 1, varIndex);
    if(!quarkEntry || !gluonEntry) return -2;

    int binQ = quarkEntry->histogram.findBin(vars[varIndex]);
    float Qi = quarkEntry->histogram.binContent(binQ);

    int binG = gluonEntry->histogram.findBin(vars[varIndex]);
    float Gi = gluonEntry->histogram.binContent(binG);

    Q *= Qi;
    G *= Gi;
  }

  if(Q==0) return 0;
  return Q/(Q+G);
}


/// Find matching entry in vector for a given eta, pt, rho, qgIndex and varIndex
const QGLikelihoodObject::Entry* QGLikelihoodCalculator::findEntry(std::vector<QGLikelihoodObject::Entry> const &data, float eta, float pt, float rho, int qgIndex, int varIndex){
  QGLikelihoodParameters myParameters;
  myParameters.Rho = rho;
  myParameters.Pt = pt;
  myParameters.Eta = fabs(eta);
  myParameters.QGIndex = qgIndex;
  myParameters.VarIndex = varIndex;

  auto myDataObject = data.begin();
  while(!(myParameters == myDataObject->category)){
    ++myDataObject;
    if(myDataObject == data.end()){
      edm::LogWarning("QGLCategoryNotFound") << "Jet passed qgValidRange criteria, but no category found with rho=" << rho << ", pt=" << pt << ", eta=" << eta
                                             << "\nPlease contact cms-qg-workinggroup@cern.ch" << std::endl;
      return nullptr;
    }
  }
  return &*myDataObject;
}


/// Check the valid range of this qg tagger
bool QGLikelihoodCalculator::isValidRange(float pt, float rho, float eta, const QGLikelihoodCategory &qgValidRange){
  if(pt < qgValidRange.PtMin) return false;
  if(pt > qgValidRange.PtMax) return false;
  if(rho < qgValidRange.RhoMin) return false;
  if(rho > qgValidRange.RhoMax) return false;
  if(fabs(eta) < qgValidRange.EtaMin) return false;
  if(fabs(eta) > qgValidRange.EtaMax) return false;
  return true;
}


/// Return the smeared qgLikelihood value, given input x0 and parameters a, b, min and max
float QGLikelihoodCalculator::smearingFunction(float x0, float a ,float b,float min,float max){
  float x=(x0-min)/(max-min);
  if(x<0.) x=0.;
  if(x>1.) x=1.;

  float x1= tanh(a*atanh(2.*x-1.)+b)/2.+.5;
  if(x<=0.) x1=0.;
  if(x>=1.) x1=1.;

  return x1*(max-min)+min;
}

// Get systematic smearing
float QGLikelihoodCalculator::systematicSmearing(edm::ESHandle<QGLikelihoodSystematicsObject> &QGLSystematicsColl, float pt, float eta, float rho, float qgValue, int qgIndex){
  if(qgValue < 0 || qgValue > 1) return -1.;

  QGLikelihoodParameters myParameters;
  myParameters.Rho = rho;
  myParameters.Pt = pt;
  myParameters.Eta = fabs(eta);
  myParameters.QGIndex = qgIndex;
  myParameters.VarIndex = -1;

  auto myDataObject = QGLSystematicsColl->data.begin();
  while(!(myParameters == myDataObject->systCategory)){
    ++myDataObject;
    if(myDataObject == QGLSystematicsColl->data.end()) return -1; //Smearing not available in the whole qgValidRange: do not throw warnings or errors
  }
  return smearingFunction(qgValue, myDataObject->a, myDataObject->b, myDataObject->lmin, myDataObject->lmax);
}
