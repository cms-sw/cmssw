#ifndef DTOfflineSummaryClients_test_H
#define DTOfflineSummaryClients_test_H


/** \class DTOfflineSummaryClients_test
 * *
 *  DQM Client for global summary
 *
 *  $Date: 2009/02/01 09:24:21 $
 *  $Revision: 1.0 $
 *  \author  M. Pelliccioni - INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

class DQMStore;
class MonitorElement;


class DTOfflineSummaryClients_test: public edm::EDAnalyzer{

public:

  /// Constructor
  DTOfflineSummaryClients_test(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTOfflineSummaryClients_test();

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// EndJob
  void endJob(void);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

protected:


private:

  int nevents;
  DQMStore* dbe;

  MonitorElement*  summaryReport;
  MonitorElement*  summaryReportMap;
  std::vector<MonitorElement*>  theSummaryContents;

};

#endif
