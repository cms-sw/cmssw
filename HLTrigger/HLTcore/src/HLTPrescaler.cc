
#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

using namespace std;

HLTPrescaler::HLTPrescaler(edm::ParameterSet const& ps):
  b_(ps.getParameter<bool>("makeFilterObject")),
  n_(ps.getParameter<unsigned int>("prescaleFactor")),
  count_()
{
  if (b_) produces<reco::HLTFilterObjectWithRefs>();
}
    
HLTPrescaler::~HLTPrescaler()
{
}

bool HLTPrescaler::filter(edm::Event & e, const edm::EventSetup & es)
{

  // prescaler decision
  ++count_;
  bool accept(count_%n_ == 0);

  // place filter object if requested
  if (b_) {
    auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs);
    e.put(filterproduct);
  }

  return accept;

}
