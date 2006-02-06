
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"

using namespace std;

namespace edm
{
  TriggerResultInserter::TriggerResultInserter(ParameterSet const& pset,
					       BitMaskPtr mask):
    bits_(mask),
    pset_id_(pset.id()),
    path_names_(pset.getParameter<Strings>("@trigger_paths"))
  {
    produces<TriggerResults>();
    // calculate the number of bit used from the number of paths,
    // remember it so it can be passed on the TriggerResults ctor.
  }

  TriggerResultInserter::~TriggerResultInserter()
  {
  }  

  // Functions that gets called by framework every event
  void TriggerResultInserter::produce(edm::Event& e, edm::EventSetup const&)
  {

    // warning: the trigger results will be cleared as a result of inserting 
    // this object into the event

    std::auto_ptr<TriggerResults>
      results(new TriggerResults(*bits_,pset_id_,path_names_));

    e.put(results);
  }
}

