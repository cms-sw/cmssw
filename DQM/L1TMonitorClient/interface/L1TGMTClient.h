#ifndef DQM_L1TMONITORCLIENT_L1TGMTCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TGMTCLIENT_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class L1TGMTClient: public edm::EDAnalyzer {

public:

    /// Constructor
    L1TGMTClient(const edm::ParameterSet&);

    /// Destructor
    virtual ~L1TGMTClient();

protected:

    /// BeginJob
    void beginJob();

    /// BeginRun
    void beginRun(const edm::Run&, const edm::EventSetup&);

    /// Fake Analyze
    void analyze(const edm::Event&, const edm::EventSetup&);

    void beginLuminosityBlock(const edm::LuminosityBlock&,
            const edm::EventSetup&);

    /// DQM Client Diagnostic
    void
    endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

    /// EndRun
    void endRun(const edm::Run&, const edm::EventSetup&);

    /// Endjob
    void endJob();

private:

    void initialize();
    void processHistograms();
    void makeRatio1D(MonitorElement* mer, std::string h1Name,
            std::string h2Name);
    void makeEfficiency1D(MonitorElement *meeff, std::string heName,
            std::string hiName);
    void makeEfficiency2D(MonitorElement *meeff, std::string heName,
            std::string hiName);
    TH1F * get1DHisto(std::string meName, DQMStore* dbi);
    TH2F * get2DHisto(std::string meName, DQMStore* dbi);

    MonitorElement* bookClone1D(const std::string& name,
            const std::string& title, const std::string& hrefName);
    MonitorElement* bookClone1DVB(const std::string& name,
            const std::string& title, const std::string& hrefName);
    MonitorElement* bookClone2D(const std::string& name,
            const std::string& title, const std::string& hrefName);

    edm::ParameterSet parameters_;
    DQMStore* dbe_;
    std::string monitorName_;
    std::string input_dir_;
    std::string output_dir_;

    bool m_runInEventLoop;
    bool m_runInEndLumi;
    bool m_runInEndRun;
    bool m_runInEndJob;

    // -------- member data --------
    MonitorElement* eff_eta_dtcsc;
    MonitorElement* eff_eta_rpc;
    MonitorElement* eff_phi_dtcsc;
    MonitorElement* eff_phi_rpc;
    MonitorElement* eff_etaphi_dtcsc;
    MonitorElement* eff_etaphi_rpc;

};

#endif
