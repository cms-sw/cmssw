#ifndef DQM_L1TMONITORCLIENT_L1TEMTFEventInfoClient_H
#define DQM_L1TMONITORCLIENT_L1TEMTFEventInfoClient_H

/**
 * \class L1TEMTFEventInfoClient
 *
 *
 * Description: fill L1 report summary for trigger L1T and emulator L1TEMU DQM.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *    Re-designed and fully rewritten class.
 *    Original version and authors: see CVS history
 *
 *
 */

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

// forward declarations
class DQMStore;

// class declaration
class L1TEMTFEventInfoClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TEMTFEventInfoClient(const edm::ParameterSet &);

  /// Destructor
  ~L1TEMTFEventInfoClient() override;

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                             DQMStore::IGetter &igetter,
                             const edm::LuminosityBlock &,
                             const edm::EventSetup &) override;

  /// end job
  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;

private:
  /// input parameters

  bool m_verbose;
  std::string m_monitorDir;
  std::string m_histDir;

  bool m_runInEventLoop;
  bool m_runInEndLumi;
  bool m_runInEndRun;
  bool m_runInEndJob;

  std::vector<edm::ParameterSet> m_trackObjects;
  std::vector<edm::ParameterSet> m_hitObjects;
  std::vector<std::string> m_disableTrackObjects;
  std::vector<std::string> m_disableHitObjects;

  std::vector<edm::ParameterSet> m_noisyStrip;
  std::vector<edm::ParameterSet> m_deadStrip;
  std::vector<std::string> m_disableNoisyStrip;
  std::vector<std::string> m_disableDeadStrip;

  /// private methods

  /// initialize properly all elements
  void initialize();

  /// dump the content of the monitoring elements defined in this module
  void dumpContentMonitorElements(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);

  /// book histograms
  void book(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);

  /// read quality test results
  void readQtResults(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);

  /// number of L1 trigger systems
  size_t m_nrTrackObjects;

  /// number of L1 trigger objects
  size_t m_nrHitObjects;

  /// number of L1 trigger noisy strips
  size_t m_nrNoisyStrip;

  /// number of L1 trigger dead strips
  size_t m_nrDeadStrip;

  /// total number of quality tests enabled for summary report for L1 trigger systems
  /// and L1 trigger objects
  size_t m_totalNrQtSummaryEnabled;

  std::vector<std::string> m_trackLabel;
  std::vector<std::string> m_trackLabelExt;
  std::vector<int> m_trackDisable;

  std::vector<std::vector<std::string> > m_trackQualityTestName;
  std::vector<std::vector<std::string> > m_trackQualityTestHist;
  std::vector<std::vector<unsigned int> > m_trackQtSummaryEnabled;

  std::vector<int> m_hitDisable;
  std::vector<std::string> m_hitLabel;
  std::vector<std::string> m_hitFolder;

  std::vector<std::vector<std::string> > m_hitQualityTestName;
  std::vector<std::vector<std::string> > m_hitQualityTestHist;
  std::vector<std::vector<unsigned int> > m_hitQtSummaryEnabled;

  std::vector<int> m_noisyDisable;
  std::vector<std::string> m_noisyLabel;
  std::vector<std::string> m_noisyFolder;

  std::vector<std::vector<std::string> > m_noisyQualityTestName;
  std::vector<std::vector<std::string> > m_noisyQualityTestHist;
  std::vector<std::vector<unsigned int> > m_noisyQtSummaryEnabled;

  std::vector<int> m_deadDisable;
  std::vector<std::string> m_deadLabel;
  std::vector<std::string> m_deadFolder;

  std::vector<std::vector<std::string> > m_deadQualityTestName;
  std::vector<std::vector<std::string> > m_deadQualityTestHist;
  std::vector<std::vector<unsigned int> > m_deadQtSummaryEnabled;

  /// summary report

  Float_t m_reportSummary;
  Float_t m_summarySum;
  std::vector<int> m_summaryContent;

  /// a summary report
  MonitorElement *m_meReportSummary;

  /// monitor elements to report content for all quality tests
  std::vector<MonitorElement *> m_meReportSummaryContent;

  /// report summary map
  MonitorElement *m_meReportSummaryMap;
  MonitorElement *m_meReportSummaryMap_chamberStrip;
};

#endif
