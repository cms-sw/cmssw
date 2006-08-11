/** \class HLTPrescaler
 *
 *  
 *  See header file for documentation.
 *
 *  $Date: 2006/08/11 09:24:40 $
 *  $Revision: 1.7 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

HLTPrescaler::HLTPrescaler(edm::ParameterSet const& ps) :
  b_(ps.getParameter<bool>("makeFilterObject")),
  n_(ps.getParameter<unsigned int>("prescaleFactor")),
  o_(ps.getParameter<unsigned int>("eventOffset")),
  count_()
{
  if (b_) produces<reco::HLTFilterObjectBase>();
  if (n_==0) n_=1; // accept all!
  count_ += o_;    // event offset
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
    auto_ptr<HLTFilterObjectBase> 
      filterproduct (new HLTFilterObjectBase(path(),module()));
    e.put(filterproduct);
  }

  return accept;

}
