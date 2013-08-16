/** \class HLTTriggerTypeFilter
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/21 15:00:22 $
 *  $Revision: 1.3 $
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
  SelectedTriggerType_(iConfig.getParameter<int>("SelectedTriggerType"))
{
}

HLTTriggerTypeFilter::~HLTTriggerTypeFilter()
{
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTTriggerTypeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if (iEvent.isRealData()) {
    return (iEvent.experimentType() == SelectedTriggerType_); 
  } else {
    return true;
  }
}

