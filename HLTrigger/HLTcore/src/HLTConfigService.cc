/** \class HLTConfigService
 *
 *  
 *  This class provides a Service to get hold of the HLT Configuration
 *
 *  $Date: 2010/12/17 14:10:01 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "HLTrigger/HLTcore/interface/HLTConfigData.h"
#include "HLTrigger/HLTcore/interface/HLTConfigService.h"

#include<map>
#include<string>

//
// class declaration
//

namespace edm {

  namespace service {

    /// Initialisation
    bool HLTConfigService::init(const edm::Run& iRun,
				const edm::EventSetup& iSetup,
				const std::string& processName,
				bool& changed) {
      if (hltMap_.find(processName)==hltMap_.end()) {
	hltMap_[processName]=HLTConfigData();
	return hltMap_[processName].init(iRun,iSetup,processName,changed);
      } else if (hltMap_[processName].runID()!=iRun.id()) {
	return hltMap_[processName].init(iRun,iSetup,processName,changed);
      } else {
	changed=hltMap_[processName].changed();
	return  hltMap_[processName].inited();
      }
    }
    
    /// Access to config
    const HLTConfigData* HLTConfigService::hltConfigData(const std::string& processName) const {
      const std::map<std::string,HLTConfigData>::const_iterator index(hltMap_.find(processName));
      if (index==hltMap_.end()) {
	return 0;
      } else {
	return &(index->second);
      }
    }
        
  }
}
