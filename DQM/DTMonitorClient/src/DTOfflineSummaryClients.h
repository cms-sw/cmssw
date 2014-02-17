#ifndef DTOfflineSummaryClients_H
#define DTOfflineSummaryClients_H


/** \class DTOfflineSummaryClients
 * *
 *  DQM Client for global summary
 *
 *  $Date: 2010/01/22 15:32:04 $
 *  $Revision: 1.4 $
 *  \author  M. Pelliccioni - INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

class DQMStore;
class MonitorElement;


class DTOfflineSummaryClients: public edm::EDAnalyzer{

public:

  /// Constructor
  DTOfflineSummaryClients(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTOfflineSummaryClients();

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// EndLumi
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

  /// EndJob
  void endJob(void);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// DQM Client Diagnostic
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

protected:


private:

  int nevents;
  DQMStore* dbe;

  MonitorElement*  summaryReport;
  MonitorElement*  summaryReportMap;
  std::vector<MonitorElement*>  theSummaryContents;

};

#endif
