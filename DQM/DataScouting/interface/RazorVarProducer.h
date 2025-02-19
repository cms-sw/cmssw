#ifndef RazorVarProducer_h
#define RazorVarProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TLorentzVector.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"

class RazorVarProducer : public edm::EDProducer {
 public: 
  explicit RazorVarProducer(const edm::ParameterSet&);
  ~RazorVarProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&);

  double CalcMR(TLorentzVector ja,TLorentzVector jb);
  double CalcR(double MR, TLorentzVector ja,TLorentzVector jb, edm::Handle<reco::CaloMETCollection> met, std::vector<math::XYZTLorentzVector> muons);

 private:

  edm::InputTag inputTag_; // input tag identifying product
  edm::InputTag inputMetTag_; // input tag identifying MET product
};

#endif //RazorVarProducer_h
