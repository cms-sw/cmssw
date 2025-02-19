#ifndef DQM_L1TMONITORCLIENT_L1EmulatorErrorFlagClient_H
#define DQM_L1TMONITORCLIENT_L1EmulatorErrorFlagClient_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/MonitorElement.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;

class L1EmulatorErrorFlagClient: public edm::EDAnalyzer {

public:

    /// Constructor
    L1EmulatorErrorFlagClient(const edm::ParameterSet&);

    /// Destructor
    virtual ~L1EmulatorErrorFlagClient();

private:

    /// begin job
    void beginJob();

    /// begin run
    void beginRun(const edm::Run&, const edm::EventSetup&);

    /// analyze
    void analyze(const edm::Event&, const edm::EventSetup&);

    void beginLuminosityBlock(const edm::LuminosityBlock&,
            const edm::EventSetup&);

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
    std::vector<edm::ParameterSet> m_l1Systems;
    std::vector<std::string> m_maskL1Systems;

    bool m_runInEventLoop;
    bool m_runInEndLumi;
    bool m_runInEndRun;
    bool m_runInEndJob;



    /// private methods

    void initialize();

    Float_t setSummary(const unsigned int&) const;

    /// number of L1 trigger systems
    size_t m_nrL1Systems;

    std::vector<std::string> m_systemLabel;
    std::vector<std::string> m_systemLabelExt;
    std::vector<int> m_systemMask;
    std::vector<std::string> m_systemFolder;

    std::vector<std::string> m_systemErrorFlag;

    /// summary report

    std::vector<Float_t> m_summaryContent;
    MonitorElement* m_meSummaryErrorFlagMap;

    //

    DQMStore* m_dbe;

};

#endif
