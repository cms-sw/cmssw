
#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

using namespace std;

HLTPrescaler::HLTPrescaler(edm::ParameterSet const& ps):
  b_(ps.getParameter<bool>("makeFilterObject")),
  n_(ps.getParameter<unsigned int>("prescaleFactor")),
  count_()
{
  if (n_==0) n_=1; // accept all!
  if (b_) produces<reco::HLTFilterObjectWithRefs>();
}
    
HLTPrescaler::~HLTPrescaler()
{
}

bool HLTPrescaler::filter(edm::Event & e, const edm::EventSetup & es)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  // prescaler decision
  ++count_;
  const bool accept(count_%n_ == 0);

  // construct and place filter object if requested
  if (b_) {
    auto_ptr<HLTFilterObjectWithRefs> 
      filterproduct (new HLTFilterObjectWithRefs(path(),module()));
    e.put(filterproduct);
  }

  return accept;

}
