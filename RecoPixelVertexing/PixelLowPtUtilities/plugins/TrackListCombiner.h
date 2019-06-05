#ifndef TrackListCombiner_H
#define TrackListCombiner_H

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include <vector>

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

class TrackListCombiner : public edm::global::EDProducer<> {
public:
  explicit TrackListCombiner(const edm::ParameterSet& ps);
  ~TrackListCombiner() override;
  void produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const override;

private:
  struct Tags {
    template <typename T1, typename T2>
    Tags(T1 t1, T2 t2) : trajectory(t1), assoMap(t2) {}
    edm::EDGetTokenT<std::vector<Trajectory>> trajectory;
    edm::EDGetTokenT<TrajTrackAssociationCollection> assoMap;
  };

  std::vector<Tags> trackProducers;
};
#endif
