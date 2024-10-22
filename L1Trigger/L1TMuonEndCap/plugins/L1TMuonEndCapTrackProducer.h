#ifndef L1TMuonEndCap_L1TMuonEndCapTrackProducer_h
#define L1TMuonEndCap_L1TMuonEndCapTrackProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuonEndCap/interface/TrackFinder.h"
#include "L1Trigger/L1TMuonEndCap/interface/MicroGMTConverter.h"

// Class declaration
class L1TMuonEndCapTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TMuonEndCapTrackProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  std::unique_ptr<TrackFinder> track_finder_;
  std::unique_ptr<MicroGMTConverter> uGMT_converter_;
};

#endif
