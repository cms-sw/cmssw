
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

namespace edm
{
  TriggerResultInserter::TriggerResultInserter(const ParameterSet& pset, unsigned int iNStreams) :
  resultsPerStream_(iNStreams),
  pset_id_(pset.id())
  {
    produces<TriggerResults>();
  }

  void
  TriggerResultInserter::setTrigResultForStream(unsigned int iStreamIndex, const TrigResPtr& trptr) {
    resultsPerStream_[iStreamIndex] =trptr;
  }

  void TriggerResultInserter::produce(StreamID id, edm::Event& e, edm::EventSetup const&) const
  {
    std::unique_ptr<TriggerResults>
      results(new TriggerResults(*resultsPerStream_[id.value()], pset_id_));

    e.put(std::move(results));
  }
}
