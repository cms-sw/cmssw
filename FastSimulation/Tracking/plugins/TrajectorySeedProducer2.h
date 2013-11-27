#ifndef FastSimulation_Tracking_TrajectorySeedProducer2_h
#define FastSimulation_Tracking_TrajectorySeedProducer2_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"

#include <vector>
#include <string>

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TrajectorySeedProducer2 : public TrajectorySeedProducer
{
 public:
  
  explicit TrajectorySeedProducer2(const edm::ParameterSet& conf);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;

};

#endif
