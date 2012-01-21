/** \class HLTTriggerTypeFilter
 *
 * See header file for documentation
 *
 *  $Date: 2009/08/06 11:23:34 $
 *  $Revision: 1.2 $
 *
 *  \author:  Giovanni FRANZONI
 *
 */

// include files
#include "HLTrigger/special/interface/HLTTriggerTypeFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <string>
#include <iostream>

//
// constructors and destructor
//
HLTTriggerTypeFilter::HLTTriggerTypeFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  //now do what ever initialization is needed
  SelectedTriggerType_ = static_cast<unsigned short>(iConfig.getParameter<int>("SelectedTriggerType"));
}


HLTTriggerTypeFilter::~HLTTriggerTypeFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTTriggerTypeFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  if (iEvent.isRealData()) {
    return (iEvent.experimentType() == SelectedTriggerType_); 
  } else {
    return true;
  }
}

