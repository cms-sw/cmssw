#ifndef HLTPFJetIDProducer_h
#define HLTPFJetIDProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class HLTPFJetIDProducer : public edm::EDProducer {
 public:
  explicit HLTPFJetIDProducer(const edm::ParameterSet&);
  ~HLTPFJetIDProducer();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void beginJob() ; 
  virtual void produce(edm::Event &, const edm::EventSetup&);
 private:
  edm::InputTag jetsInput_;
  double min_NHEF_;         // minimum Neutral Hadron Energy Fraction
  double max_NHEF_;         // maximum NHEF
  double min_NEMF_;         // minimum Neutral EM Energy Fraction
  double max_NEMF_;         // maximum NEMF
  double min_CEMF_;         // minimum Charged EM Energy Fraction
  double max_CEMF_;         // maximum CEMF
  double min_CHEF_;         // minimum Charged Hadron Energy Fraction
  double max_CHEF_;         // maximum CHEF
  double min_pt_;           // pT cut
};

#endif
