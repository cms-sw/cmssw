/** \class HLTTriggerTypeFilter
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/22 22:20:49 $
 *  $Revision: 1.4 $
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

