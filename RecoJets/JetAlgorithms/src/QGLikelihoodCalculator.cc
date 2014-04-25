#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"


/// Compute likelihood for a jet using the QGLikelihoodObject information and a set of variables
float QGLikelihoodCalculator::computeQGLikelihood(edm::ESHandle<QGLikelihoodObject> &QGLParamsColl, float pt, float eta, float rho, std::vector<float> vars){
  if(!isValidRange(pt, rho, eta, QGLParamsColl->qgValidRange)) return -1;

  float Q=1., G=1.;
  for(unsigned int varIndex = 0; varIndex < vars.size(); ++varIndex){

    auto qgEntry = findEntry(QGLParamsColl->data, eta, pt, rho, 0, varIndex); 
    if(!qgEntry) return -1; 
    float Qi = qgEntry->histogram.binContent(qgEntry->histogram.findBin(vars[varIndex]));
    float mQ = qgEntry->mean;

    qgEntry = findEntry(QGLParamsColl->data, eta, pt, rho, 1, varIndex); 
    if(!qgEntry) return -1;
    float Gi = qgEntry->histogram.binContent(qgEntry->histogram.findBin(vars[varIndex]));
    float mG = qgEntry->mean;

    float epsilon=0;
    float delta=0.000001;
    if(Qi <= epsilon && Gi <= epsilon){
      if(mQ>mG){
	if(vars[varIndex] > mQ){ Qi = 1-delta; Gi = delta;}
	else if(vars[varIndex] < mG){ Qi = delta; Gi = 1-delta;}
      }
      else if(mQ<mG){
	if(vars[varIndex]<mQ) { Qi = 1-delta; Gi = delta;}
	else if(vars[varIndex]>mG){Qi = delta;Gi = 1-delta;}
      }
    } 
    Q*=Qi;
    G*=Gi;	
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
