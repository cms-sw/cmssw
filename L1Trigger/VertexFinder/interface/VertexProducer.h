#ifndef __L1Trigger_VertexFinder_VertexProducer_h__
#define __L1Trigger_VertexFinder_VertexProducer_h__

#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "L1Trigger/VertexFinder/interface/AlgoSettings.h"
#include "L1Trigger/VertexFinder/interface/RecoVertex.h"

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace l1tVertexFinder {
  class AlgoSettings;
}

class VertexProducer : public edm::global::EDProducer<> {
public:
  explicit VertexProducer(const edm::ParameterSet&);
  ~VertexProducer() override {}

private:
  typedef edm::View<TTTrack<Ref_Phase2TrackerDigi_>> TTTrackCollectionView;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<TTTrackCollectionView> l1TracksToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  const std::string outputCollectionName_;

  l1tVertexFinder::AlgoSettings settings_;
  std::vector<l1tVertexFinder::L1Track> l1Tracks;
};

#endif
