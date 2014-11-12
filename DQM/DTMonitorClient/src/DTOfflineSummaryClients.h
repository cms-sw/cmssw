#ifndef DTOfflineSummaryClients_H
#define DTOfflineSummaryClients_H


/** \class DTOfflineSummaryClients
 * *
 *  DQM Client for global summary
 *
 *  \author  M. Pelliccioni - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <DQMServices/Core/interface/DQMEDHarvester.h>

class DQMStore;
class MonitorElement;


//-class DTOfflineSummaryClients: public edm::EDAnalyzer{
class DTOfflineSummaryClients: public DQMEDHarvester{

public:

  /// Constructor
  DTOfflineSummaryClients(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTOfflineSummaryClients();

  /// BeginRun
//-  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void dqmBeginRun (const edm::Run& r, const edm::EventSetup& c);

  /// EndLumi
//-  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);

  /// EndJob
//-  void endJob(void);
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  /// Analyze
//-  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// DQM Client Diagnostic
//-  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

protected:


private:

  int nevents;
  DQMStore* dbe;

  bool bookingdone;

  MonitorElement*  summaryReport;
  MonitorElement*  summaryReportMap;
  std::vector<MonitorElement*>  theSummaryContents;

};

#endif
