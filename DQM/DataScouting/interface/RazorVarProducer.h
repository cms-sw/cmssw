#ifndef RazorVarProducer_h
#define RazorVarProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include "TLorentzVector.h"
#include <vector>

class RazorVarProducer : public edm::EDProducer {
public:
  explicit RazorVarProducer(const edm::ParameterSet &);
  ~RazorVarProducer() override;
  void produce(edm::Event &, const edm::EventSetup &) override;

  double CalcMR(TLorentzVector ja, TLorentzVector jb);
  double CalcR(double MR,
               const TLorentzVector &ja,
               const TLorentzVector &jb,
               edm::Handle<reco::CaloMETCollection> met,
               const std::vector<math::XYZTLorentzVector> &muons);

private:
  edm::InputTag inputTag_;     // input tag identifying product
  edm::InputTag inputMetTag_;  // input tag identifying MET product

  // define Token(-s)
  edm::EDGetTokenT<std::vector<math::XYZTLorentzVector>> inputTagToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> inputMetTagToken_;
};

#endif  // RazorVarProducer_h
