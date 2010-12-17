/** \class HLTConfigService
 *
 *  
 *  This class provides a Service to get hold of the HLT Configuration
 *
 *  $Date: 2010/07/14 15:30:06 $
 *  $Revision: 1.29 $
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
    void HLTConfigService::init(const edm::Run& iRun,
				const edm::EventSetup& iSetup,
				const std::string& processName) {
      if (hltMap_.find(processName)==hltMap_.end()) {
	hltMap_[processName]=HLTConfigData();
	hltMap_[processName].init(iRun,iSetup,processName);
      } else if (hltMap_[processName].runID()!=iRun.id()) {
	hltMap_[processName].init(iRun,iSetup,processName);
      }
      return;
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
