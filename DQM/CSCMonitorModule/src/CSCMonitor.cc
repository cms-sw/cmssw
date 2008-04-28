/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitor.cc
 *
 *    Description:  Backward Compatible Object
 *
 *        Version:  1.0
 *        Created:  04/28/2008 09:51:42 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"

CSCMonitor::CSCMonitor(const edm::ParameterSet& ps) {
  mm = edm::Service<CSCMonitorModule>().operator->();
}

CSCMonitor::~CSCMonitor() {
  delete mm;
}

void CSCMonitor::process(CSCDCCExaminer * examiner, CSCDCCEventData * dccData) {
  mm->process(examiner, dccData);
}

