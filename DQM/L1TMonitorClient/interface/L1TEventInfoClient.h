#ifndef DQM_L1TMONITORCLIENT_L1TEventInfoClient_H
#define DQM_L1TMONITORCLIENT_L1TEventInfoClient_H

/**
 * \class L1TEventInfoClient
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
 * $Date: 2012/06/04 10:55:43 $
 * $Revision: 1.9 $
 *
 */

// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/MonitorElement.h"

// forward declarations
class DQMStore;

// class declaration
class L1TEventInfoClient: public edm::EDAnalyzer {

public:

    /// Constructor
    L1TEventInfoClient(const edm::ParameterSet&);

    /// Destructor
    virtual ~L1TEventInfoClient();

private:

    /// begin job
    void beginJob();

    /// begin run
    void beginRun(const edm::Run&, const edm::EventSetup&);

    /// begin luminosity block
    void beginLuminosityBlock(const edm::LuminosityBlock&,
            const edm::EventSetup&);

    /// analyze
    void analyze(const edm::Event&, const edm::EventSetup&);

    /// end luminosity block
    void
    endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

    /// end run
    void endRun(const edm::Run&, const edm::EventSetup&);

    /// end job
    void endJob();

private:

    /// input parameters

    bool m_verbose;
    std::string m_monitorDir;

    bool m_runInEventLoop;
    bool m_runInEndLumi;
    bool m_runInEndRun;
    bool m_runInEndJob;

    std::vector<edm::ParameterSet> m_l1Systems;
    std::vector<edm::ParameterSet> m_l1Objects;
    std::vector<std::string> m_disableL1Systems;
    std::vector<std::string> m_disableL1Objects;

    /// private methods

    /// initialize properly all elements
    void initialize();

    /// dump the content of the monitoring elements defined in this module
    void dumpContentMonitorElements();

    /// book histograms
    void bookHistograms();

    /// read quality test results
    void readQtResults();

    /// number of L1 trigger systems
    size_t m_nrL1Systems;

    /// number of L1 trigger objects
    size_t m_nrL1Objects;

    /// total number of quality tests enabled for summary report for L1 trigger systems
    /// and L1 trigger objects
    size_t m_totalNrQtSummaryEnabled;

    std::vector<std::string> m_systemLabel;
    std::vector<std::string> m_systemLabelExt;
    std::vector<int> m_systemDisable;

    std::vector<std::vector<std::string> > m_systemQualityTestName;
    std::vector<std::vector<std::string> > m_systemQualityTestHist;
    std::vector<std::vector<unsigned int> > m_systemQtSummaryEnabled;

    std::vector<int> m_objectDisable;
    std::vector<std::string> m_objectLabel;
    std::vector<std::string> m_objectFolder;

    std::vector<std::vector<std::string> > m_objectQualityTestName;
    std::vector<std::vector<std::string> > m_objectQualityTestHist;
    std::vector<std::vector<unsigned int> > m_objectQtSummaryEnabled;

    /// summary report

    Float_t m_reportSummary;
    Float_t m_summarySum;
    std::vector<int> m_summaryContent;


    /// a summary report
    MonitorElement* m_meReportSummary;

    /// monitor elements to report content for all quality tests
    std::vector<MonitorElement*> m_meReportSummaryContent;

    /// report summary map
    MonitorElement* m_meReportSummaryMap;

    //

    DQMStore* m_dbe;


};

#endif
