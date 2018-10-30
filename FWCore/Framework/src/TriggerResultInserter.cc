
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

namespace edm
{
  TriggerResultInserter::TriggerResultInserter(const ParameterSet& pset, unsigned int iNStreams) :
  resultsPerStream_(iNStreams),
  pset_id_(pset.id()),
  token_{produces<TriggerResults>()}
  {
  }

  void
  TriggerResultInserter::setTrigResultForStream(unsigned int iStreamIndex, const TrigResPtr& trptr) {
    resultsPerStream_[iStreamIndex] =trptr;
  }

  void TriggerResultInserter::produce(StreamID id, edm::Event& e, edm::EventSetup const&) const
  {
    e.emplace(token_,*resultsPerStream_[id.value()], pset_id_);
  }
}
