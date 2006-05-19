
#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

using namespace std;

namespace edm
{
  HLTPrescaler::HLTPrescaler(edm::ParameterSet const& ps):
    n_(ps.getParameter<int>("prescaleFactor")),
    count_()
  {
    produces<reco::HLTFilterObjectWithRefs>();
  }
    
  HLTPrescaler::~HLTPrescaler()
  {
  }

  bool HLTPrescaler::filter(edm::Event & e, const edm::EventSetup & es)
  {

    // prescaler decision
    ++count_;
    bool accept(count_%n_ == 0);

    // filter object
    auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs);
    filterproduct->setAccept(accept);
    e.put(filterproduct);

    return accept;

  }

}
