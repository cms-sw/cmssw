
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Common/interface/EndPathStatus.h"

#include <memory>

namespace edm {
  EndPathStatusInserter::EndPathStatusInserter(unsigned int) { produces<EndPathStatus>(); }

  void EndPathStatusInserter::produce(StreamID, edm::Event& event, edm::EventSetup const&) const {
    event.put(std::make_unique<EndPathStatus>());
  }
}
