#ifndef L1Trigger_ME0TriggerPrimitives_ME0TriggerPrimitivesProducer_h
#define L1Trigger_ME0TriggerPrimitives_ME0TriggerPrimitivesProducer_h

/** \class ME0TriggerPrimitivesProducer
 *
 * \author Sven Dildick (TAMU).
 *
 */

#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ME0TriggerPrimitivesBuilder;

class ME0TriggerPrimitivesProducer : public edm::one::EDProducer<edm::one::SharedResources>
{
 public:
  explicit ME0TriggerPrimitivesProducer(const edm::ParameterSet&);
  ~ME0TriggerPrimitivesProducer();

  //virtual void beginRun(const edm::EventSetup& setup);
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag me0PadDigiProducer_;
  edm::EDGetTokenT<ME0PadDigiCollection> me0_pad_token_;
 
  std::unique_ptr<ME0TriggerPrimitivesBuilder> lctBuilder_;
};

#endif
