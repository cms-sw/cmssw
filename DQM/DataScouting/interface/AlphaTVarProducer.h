#ifndef AlphaTVarProducer_h
#define AlphaTVarProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TLorentzVector.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"

#include <vector>

class AlphaTVarProducer : public edm::EDProducer {
 public: 
  explicit AlphaTVarProducer(const edm::ParameterSet&);
  ~AlphaTVarProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  double CalcAlphaT(std::vector<TLorentzVector>);
  static double CalcHT(const std::vector<TLorentzVector>);
  static double CalcMHT(const std::vector<TLorentzVector>);
  static double deltaHt(const std::vector<double>&);
  static double alphaT(const double, const double, const double);
  
  
  edm::InputTag inputJetTag_; // input tag identifying product
};

#endif //AlphaTVarProducer_h
