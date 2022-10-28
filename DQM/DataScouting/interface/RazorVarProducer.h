#ifndef RazorVarProducer_h
#define RazorVarProducer_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include "TLorentzVector.h"
#include <vector>

class RazorVarProducer : public edm::global::EDProducer<> {
public:
  explicit RazorVarProducer(const edm::ParameterSet &);
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  double CalcMR(TLorentzVector ja, TLorentzVector jb) const;
  double CalcR(double MR,
               const TLorentzVector &ja,
               const TLorentzVector &jb,
               edm::Handle<reco::CaloMETCollection> met,
               const std::vector<math::XYZTLorentzVector> &muons) const;

private:
  edm::InputTag inputTag_;     // input tag identifying product
  edm::InputTag inputMetTag_;  // input tag identifying MET product

  // define Token(-s)
  edm::EDGetTokenT<std::vector<math::XYZTLorentzVector>> inputTagToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> inputMetTagToken_;
};

#endif  // RazorVarProducer_h
