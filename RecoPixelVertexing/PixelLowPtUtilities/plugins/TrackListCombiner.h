#ifndef TrackListCombiner_H
#define TrackListCombiner_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
using namespace std;

namespace edm { class Event; class EventSetup; }

class TrackListCombiner : public edm::EDProducer
{
public:
  explicit TrackListCombiner(const edm::ParameterSet& ps);
  ~TrackListCombiner();
  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:
  vector<string> trackProducers;
};
#endif

