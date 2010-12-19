#ifndef HLTcore_HLTConfigService_h
#define HLTcore_HLTConfigService_h

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

#include<map>
#include<string>

//
// class declaration
//

namespace edm {

  namespace service {

    class HLTConfigService {

    private:
      /// data member: HLT config, keyed on process name
      std::map<std::string,HLTConfigData> hltMap_;
      
    public:
      /// c'tor
      HLTConfigService (const edm::ParameterSet& iPSet,
			const edm::ActivityRegistry& iReg) : hltMap_() { }
	
    public:
      /// Initialisation
      bool init(const edm::Run& iRun, const edm::EventSetup& iSetup,
		const std::string& processName, bool& changed);
      
      /// Access to config
      const HLTConfigData* hltConfigData(const std::string& processName) const ;
    };

  }

}
#endif
