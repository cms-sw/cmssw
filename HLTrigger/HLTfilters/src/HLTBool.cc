/** \class HLTBool
 *
 * See header file for documentation
 *
 *  $Date: 2007/08/16 14:49:06 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */


#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "HLTrigger/HLTfilters/interface/HLTBool.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTBool::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   return result_;
}
