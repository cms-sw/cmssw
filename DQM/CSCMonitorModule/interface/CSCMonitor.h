/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitor.h
 *
 *    Description:  Backward compatibility object
 *
 *        Version:  1.0
 *        Created:  04/28/2008 09:42:57 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCMonitor_h 
#define CSCMonitor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

class CSCMonitor : public CSCMonitorInterface, public edm::EDAnalyzer {

  public:

    CSCMonitor( const edm::ParameterSet& ps);
    ~CSCMonitor();

    void process(CSCDCCExaminer * examiner, CSCDCCEventData * dccData);
    void analyze(const edm::Event& e, const edm::EventSetup& c){ };

  private:

    CSCMonitorModule *mm;

};

#endif
