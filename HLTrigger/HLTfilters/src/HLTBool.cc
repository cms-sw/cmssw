/** \class HLTBool
 *
 * See header file for documentation
 *
 *  $Date: 2011/02/16 17:25:34 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */


#include "HLTrigger/HLTfilters/interface/HLTBool.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constructors and destructor
//
HLTBool::HLTBool(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  result_(iConfig.getParameter<bool> ("result"))
{
  LogDebug("HLTBool") << " configured result is: " << result_;
}

HLTBool::~HLTBool()
{
}

void
HLTBool::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("result",false);
  descriptions.add("hltBool",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTBool::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
   return result_;
}
