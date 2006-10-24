#ifndef RecoEgamma_EgammaHLTProducers_EgammaHLTRecoEcalCandidateProducers_h
#define RecoEgamma_EgammaHLTProducers_EgammaHLTRecoEcalCandidateProducers_h
/** \class RecoEcalCandidateProducers
 **  
 ** $Id
 **  \author Monica Vazquez Acosta (CERN)
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"



// EgammaHLTRecoEcalCandidateProducers inherits from EDProducer, so it can be a module:
class EgammaHLTRecoEcalCandidateProducers : public edm::EDProducer {

 public:

  EgammaHLTRecoEcalCandidateProducers (const edm::ParameterSet& ps);
  ~EgammaHLTRecoEcalCandidateProducers();


  virtual void beginJob (edm::EventSetup const & es);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:
  std::string recoEcalCandidateCollection_;
  edm::InputTag scHybridBarrelProducer_;
  edm::InputTag scIslandEndcapProducer_;
  edm::ParameterSet conf_;
};
#endif

