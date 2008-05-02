/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule_monitorDCC.cc
 *
 *    Description:  Monitor DCC method implementation.
 *
 *        Version:  1.0
 *        Created:  04/18/2008 03:38:42 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"

/**
 * @brief  MonitorDCC function that grabs DCCEventData and processes it.
 * @param  dccData DCC data
 * @return 
 */
void CSCMonitorModule::monitorDCC(const CSCDCCEventData& dccEvent){
  const std::vector<CSCDDUEventData> & dduData = dccEvent.dduData();

  for (int ddu = 0; ddu < (int)dduData.size(); ++ddu) {
    monitorDDU(dduData[ddu]);
  }

}
