#ifndef __L1Trigger_VertexFinder_VertexProducer_h__
#define __L1Trigger_VertexFinder_VertexProducer_h__

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "L1Trigger/VertexFinder/interface/AlgoSettings.h"
#include "L1Trigger/VertexFinder/interface/RecoVertex.h"
#include "L1Trigger/VertexFinder/interface/VertexFinder.h"

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace l1tVertexFinder {
  class AlgoSettings;
}

class VertexProducer : public edm::EDProducer {
public:
  explicit VertexProducer(const edm::ParameterSet&);
  ~VertexProducer() override {}

private:
  typedef edm::View<TTTrack<Ref_Phase2TrackerDigi_>> TTTrackCollectionView;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  const edm::EDGetTokenT<TTTrackCollectionView> l1TracksToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  const std::string outputCollectionName_;

  l1tVertexFinder::AlgoSettings settings_;
  std::vector<l1tVertexFinder::L1Track> l1Tracks;
};

#endif
