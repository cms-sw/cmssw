#ifndef L1Trigger_L1TMuonEndCap_L1TMuonEndCapShowerProducer_h
#define L1Trigger_L1TMuonEndCap_L1TMuonEndCapShowerProducer_h

/*
  This EDProducer produces EMTF showers from showers in the CSC local trigger.
  These showers could indicate the passage of a long-lived particle decaying
  in the endcap muon system.

  The logic is executed in the SectorProcessorShower class. Multiple options
  are defined: "OneLoose", "TwoLoose", "OneNominal", "OneTight" 
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorShower.h"

// Class declaration
class L1TMuonEndCapShowerProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TMuonEndCapShowerProducer(const edm::ParameterSet&);
  ~L1TMuonEndCapShowerProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetToken tokenCSCShower_;
  emtf::sector_array<SectorProcessorShower> sector_processors_shower_;
};

#endif
