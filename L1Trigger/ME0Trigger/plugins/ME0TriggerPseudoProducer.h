#ifndef L1Trigger_ME0Trigger_ME0TriggerPseudoProducer_h
#define L1Trigger_ME0Trigger_ME0TriggerPseudoProducer_h

/** \class ME0TriggerPseudoProducer
 *
 * Takes offline ME0 segment as input
 * Produces ME0 trigger objects
 *
 * \author Tao Huang (TAMU).
 *
 */

#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ME0TriggerPseudoBuilder;

class ME0TriggerPseudoProducer : public edm::global::EDProducer<> {
public:
  explicit ME0TriggerPseudoProducer(const edm::ParameterSet&);
  ~ME0TriggerPseudoProducer() override;

  //virtual void beginRun(const edm::EventSetup& setup);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::InputTag me0segmentProducer_;
  edm::EDGetTokenT<ME0SegmentCollection> me0segment_token_;
  edm::ParameterSet config_;
};

#endif
