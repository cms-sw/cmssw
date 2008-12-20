
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

namespace edm
{
  TriggerResultInserter::TriggerResultInserter(const ParameterSet& pset, const TrigResPtr& trptr) :
    trptr_(trptr),
    pset_id_(pset.trackedID())
  {
    produces<TriggerResults>();
  }

  TriggerResultInserter::~TriggerResultInserter()
  {
  }  

  void TriggerResultInserter::produce(edm::Event& e, edm::EventSetup const&)
  {
    std::auto_ptr<TriggerResults>
      results(new TriggerResults(*trptr_, pset_id_));

    e.put(results);
  }
}
