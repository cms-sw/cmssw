/** \class HLTTriggerTypeFilter
 *
 * See header file for documentation
 *
 *
 *  \author:  Giovanni FRANZONI
 *
 */

// include files
#include "HLTrigger/special/interface/HLTTriggerTypeFilter.h"

//
// constructors and destructor
//
HLTTriggerTypeFilter::HLTTriggerTypeFilter(const edm::ParameterSet& iConfig) :
  selectedTriggerType_(iConfig.getParameter<int>("SelectedTriggerType"))
{
}

HLTTriggerTypeFilter::~HLTTriggerTypeFilter()
{
}

void
HLTTriggerTypeFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("SelectedTriggerType", 2);
  descriptions.add("hltTriggerTypeFilter", desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTTriggerTypeFilter::filter(edm::StreamID, edm::Event & event, edm::EventSetup const & setup) const
{
  if (event.isRealData()) {
    return (event.experimentType() == selectedTriggerType_); 
  } else {
    return true;
  }
}

