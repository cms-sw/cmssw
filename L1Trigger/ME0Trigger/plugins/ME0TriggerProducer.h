#ifndef L1Trigger_ME0Trigger_ME0TriggerProducer_h
#define L1Trigger_ME0Trigger_ME0TriggerProducer_h

/** \class ME0TriggerProducer
 *
 * Takes ME0 pad clusters as input
 * Produces ME0 trigger objects
 *
 * \author Sven Dildick (TAMU).
 *
 */

#include "DataFormats/GEMDigi/interface/ME0PadDigiClusterCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ME0TriggerBuilder;

class ME0TriggerProducer : public edm::global::EDProducer<>
{
 public:
  explicit ME0TriggerProducer(const edm::ParameterSet&);
  ~ME0TriggerProducer();

  //virtual void beginRun(const edm::EventSetup& setup);
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

 private:
  edm::InputTag me0PadDigiClusterProducer_;
  edm::EDGetTokenT<ME0PadDigiClusterCollection> me0_pad_token_;
  edm::ParameterSet config_;
};

#endif
