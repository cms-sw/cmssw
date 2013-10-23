#ifndef EgammaHLTProducers_EgammaHLTPFChargedIsolationProducer_h
#define EgammaHLTProducers_EgammaHLTPFChargedIsolationProducer_h

//
// Original Author:  Matteo Sani (UCSD)
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTPFChargedIsolationProducer : public edm::EDProducer {
public:
  explicit EgammaHLTPFChargedIsolationProducer(const edm::ParameterSet&);
  ~EgammaHLTPFChargedIsolationProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
private:

  edm::InputTag electronProducer_;
  edm::InputTag pfCandidateProducer_;
  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag beamSpotProducer_;
  
  bool useGsfTrack_;
  bool useSCRefs_;
  
  double drMax_;
  double drVetoBarrel_;
  double drVetoEndcap_;
  double ptMin_;
  double dzMax_;
  double dxyMax_;
  int pfToUse_;

};

#endif
