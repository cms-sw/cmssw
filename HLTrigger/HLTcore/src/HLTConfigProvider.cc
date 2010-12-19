/** \class HLTConfigProvider
 *
 * See header file for documentation
 *
 *  $Date: 2010/12/17 14:42:37 $
 *  $Revision: 1.52 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

bool HLTConfigProvider::init(const edm::Run& iRun,
			     const edm::EventSetup& iSetup,
			     const std::string& processName,
			     bool& changed) {
  processName_=processName;
  if (hltConfigService_==0) {
    changed=hltConfigData_->changed();
    return  hltConfigData_->inited();
  } else {
    const bool inited(hltConfigService_->init(iRun,iSetup,processName,changed));
    hltConfigData_=hltConfigService_->hltConfigData(processName);
    return inited;
  }
}
