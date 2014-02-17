/** \class HLTBool
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/22 22:15:43 $
 *  $Revision: 1.6 $
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
HLTBool::HLTBool(const edm::ParameterSet& iConfig) :
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
  desc.add<bool>("result", false);
  descriptions.add("hltBool", desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTBool::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   return result_;
}
