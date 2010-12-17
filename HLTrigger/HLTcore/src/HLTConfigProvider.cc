/** \class HLTConfigProvider
 *
 * See header file for documentation
 *
 *  $Date: 2010/12/17 14:10:01 $
 *  $Revision: 1.51 $
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
  if (hltConfigService_!=0) {
    hltConfigService_->init(iRun,iSetup,processName);
    hltConfigData_=hltConfigService_->hltConfigData(processName);
  }
  changed=hltConfigData_->changed();
  return  hltConfigData_->inited();
}
