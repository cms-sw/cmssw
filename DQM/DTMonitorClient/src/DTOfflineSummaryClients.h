#ifndef DTOfflineSummaryClients_H
#define DTOfflineSummaryClients_H


/** \class DTOfflineSummaryClients
 * *
 *  DQM Client for global summary
 *
 *  $Date: 2009/02/04 10:02:45 $
 *  $Revision: 1.1 $
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
