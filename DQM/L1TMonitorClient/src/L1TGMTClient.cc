#include "DQM/L1TMonitorClient/interface/L1TGMTClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <sstream>

L1TGMTClient::L1TGMTClient(const edm::ParameterSet& ps) {
    parameters_ = ps;
    initialize();
}

L1TGMTClient::~L1TGMTClient() {
    LogDebug("TriggerDQM") << "[TriggerDQM]: ending... ";
}

//--------------------------------------------------------
void L1TGMTClient::initialize() {

    // get back-end interface
    dbe_ = edm::Service<DQMStore>().operator->();

    // base folder for the contents of this job
    monitorName_ = parameters_.getUntrackedParameter<std::string> (
            "monitorName", "");
    LogDebug("TriggerDQM") << "Monitor name = " << monitorName_ << std::endl;

    output_dir_ = parameters_.getUntrackedParameter<std::string> ("output_dir",
            "");
    LogDebug("TriggerDQM") << "DQM output dir = " << output_dir_ << std::endl;

    input_dir_ = parameters_.getUntrackedParameter<std::string> ("input_dir",
            "");
    LogDebug("TriggerDQM") << "DQM input dir = " << input_dir_ << std::endl;

    m_runInEventLoop = parameters_.getUntrackedParameter<bool> (
            "runInEventLoop", false);
    m_runInEndLumi = parameters_.getUntrackedParameter<bool> ("runInEndLumi",
            false);
    m_runInEndRun = parameters_.getUntrackedParameter<bool> ("runInEndRun",
            false);
    m_runInEndJob = parameters_.getUntrackedParameter<bool> ("runInEndJob",
            false);

}

//--------------------------------------------------------
void L1TGMTClient::beginJob() {

    LogDebug("TriggerDQM") << "[TriggerDQM]: Begin Job";

}

//--------------------------------------------------------
void L1TGMTClient::beginRun(const edm::Run& r, const edm::EventSetup& evSetup) {

    // booking histograms in the output_dir_

    dbe_->setCurrentFolder(output_dir_);

    eff_eta_dtcsc = bookClone1DVB("eff_eta_dtcsc", "efficiency DTCSC vs eta",
            "eta_DTCSC_and_RPC");

    if (eff_eta_dtcsc != 0) {
        eff_eta_dtcsc->setAxisTitle("eta", 1);
        eff_eta_dtcsc->getTH1F()->Sumw2();

    }

    eff_eta_rpc = bookClone1DVB("eff_eta_rpc", "efficiency RPC vs eta",
            "eta_DTCSC_and_RPC");

    if (eff_eta_rpc != 0) {
        eff_eta_rpc->setAxisTitle("eta", 1);
        eff_eta_rpc->getTH1F()->Sumw2();

    }

    eff_phi_dtcsc = bookClone1D("eff_phi_dtcsc", "efficiency DTCSC vs phi",
            "phi_DTCSC_and_RPC");

    if (eff_phi_dtcsc != 0) {
        eff_phi_dtcsc->setAxisTitle("phi (deg)", 1);
        eff_phi_dtcsc->getTH1F()->Sumw2();

    }

    eff_phi_rpc = bookClone1D("eff_phi_rpc", "efficiency RPC vs phi",
            "phi_DTCSC_and_RPC");

    if (eff_phi_rpc != 0) {
        eff_phi_rpc->setAxisTitle("phi (deg)", 1);
        eff_phi_rpc->getTH1F()->Sumw2();

    }

    eff_etaphi_dtcsc = bookClone2D("eff_etaphi_dtcsc",
            "efficiency DTCSC vs eta and phi", "etaphi_DTCSC_and_RPC");

    if (eff_etaphi_dtcsc != 0) {
        eff_etaphi_dtcsc->setAxisTitle("eta", 1);
        eff_etaphi_dtcsc->setAxisTitle("phi (deg)", 2);
        eff_etaphi_dtcsc->getTH2F()->Sumw2();

    }

    eff_etaphi_rpc = bookClone2D("eff_etaphi_rpc",
            "efficiency RPC vs eta and phi", "etaphi_DTCSC_and_RPC");

    if (eff_etaphi_rpc != 0) {
        eff_etaphi_rpc->setAxisTitle("eta", 1);
        eff_etaphi_rpc->setAxisTitle("phi (deg)", 2);
        eff_etaphi_rpc->getTH2F()->Sumw2();

    }
}

//--------------------------------------------------------
void L1TGMTClient::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
        const edm::EventSetup& evSetup) {

    // empty

}
//--------------------------------------------------------

void L1TGMTClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
        const edm::EventSetup& evSetup) {

    if (m_runInEndLumi) {

        processHistograms();
    }

}

//--------------------------------------------------------
void L1TGMTClient::analyze(const edm::Event& iEvent,
        const edm::EventSetup& evSetup) {

    // there is no loop on events in the offline harvesting step
    // code here will not be executed offline

    if (m_runInEventLoop) {

        processHistograms();
    }

}

//--------------------------------------------------------
void L1TGMTClient::processHistograms() {

    LogDebug("TriggerDQM") << "L1TGMTClient: processing..." << std::endl;

    makeEfficiency1D(eff_eta_dtcsc, "eta_DTCSC_and_RPC", "eta_RPC_only");
    makeEfficiency1D(eff_eta_rpc, "eta_DTCSC_and_RPC", "eta_DTCSC_only");

    makeEfficiency1D(eff_phi_dtcsc, "phi_DTCSC_and_RPC", "phi_RPC_only");
    makeEfficiency1D(eff_phi_rpc, "phi_DTCSC_and_RPC", "phi_DTCSC_only");

    makeEfficiency2D(eff_etaphi_dtcsc, "etaphi_DTCSC_and_RPC",
            "etaphi_RPC_only");
    makeEfficiency2D(eff_etaphi_rpc, "etaphi_DTCSC_and_RPC",
            "etaphi_DTCSC_only");

}

//--------------------------------------------------------
void L1TGMTClient::endRun(const edm::Run& r, const edm::EventSetup& evSetup) {

    if (m_runInEndRun) {

        processHistograms();
    }

}

//--------------------------------------------------------
void L1TGMTClient::endJob() {

    if (m_runInEndJob) {

        processHistograms();
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////
void L1TGMTClient::makeRatio1D(MonitorElement* mer, std::string h1Name,
        std::string h2Name) {

    dbe_->setCurrentFolder(output_dir_);

    TH1F* h1 = get1DHisto(input_dir_ + "/" + h1Name, dbe_);
    TH1F* h2 = get1DHisto(input_dir_ + "/" + h2Name, dbe_);

    if (mer == 0) {
        LogDebug("TriggerDQM")
                << "\nL1TGMTClient::makeRatio1D: monitoring element zero, not able to retrieve histogram"
                << std::endl;
        return;
    }

    TH1F* hr = mer->getTH1F();

    if (hr && h1 && h2) {
        hr->Divide(h1, h2, 1., 1., " ");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void L1TGMTClient::makeEfficiency1D(MonitorElement* meeff, std::string heName,
        std::string hiName) {

    dbe_->setCurrentFolder(output_dir_);

    TH1F* he = get1DHisto(input_dir_ + "/" + heName, dbe_);
    TH1F* hi = get1DHisto(input_dir_ + "/" + hiName, dbe_);

    if (meeff == 0) {
        LogDebug("TriggerDQM")
                << "L1TGMTClient::makeEfficiency1D: monitoring element zero, not able to retrieve histogram"
                << std::endl;
        return;
    }

    TH1F* heff = meeff->getTH1F();

    if (heff && he && hi) {
        TH1F* hall = (TH1F*) he->Clone("hall");
        hall->Add(hi);
        heff->Divide(he, hall, 1., 1., "B");
        delete hall;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void L1TGMTClient::makeEfficiency2D(MonitorElement* meeff, std::string heName,
        std::string hiName) {

    dbe_->setCurrentFolder(output_dir_);

    TH2F* he = get2DHisto(input_dir_ + "/" + heName, dbe_);
    TH2F* hi = get2DHisto(input_dir_ + "/" + hiName, dbe_);

    if (meeff == 0) {
        LogDebug("TriggerDQM")
                << "\nL1TGMTClient::makeEfficiency2D: monitoring element zero, not able to retrieve histogram"
                << std::endl;
        return;
    }

    TH2F* heff = meeff->getTH2F();

    if (heff && he && hi) {
        TH2F* hall = (TH2F*) he->Clone("hall");
        hall->Add(hi);
        heff->Divide(he, hall, 1., 1., "B");
        delete hall;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
TH1F* L1TGMTClient::get1DHisto(std::string meName, DQMStore* dbi) {

    MonitorElement * me_ = dbi->get(meName);

    if (!me_) {
        LogDebug("TriggerDQM") << "\nL1TGMTClient: " << meName << " NOT FOUND.";
        return 0;
    }

    return me_->getTH1F();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
TH2F* L1TGMTClient::get2DHisto(std::string meName, DQMStore* dbi) {
    MonitorElement * me_ = dbi->get(meName);

    if (!me_) {
        LogDebug("TriggerDQM") << "\nL1TGMTClient: " << meName << " NOT FOUND.";
        return 0;
    }
    return me_->getTH2F();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
MonitorElement* L1TGMTClient::bookClone1D(const std::string& name,
        const std::string& title, const std::string& hrefName) {

    MonitorElement* me;

    TH1F* href = get1DHisto(input_dir_ + "/" + hrefName, dbe_);

    if (href) {
        LogDebug("TriggerDQM")
                << "\nL1TGMTClient::bookClone1D: booking histogram "
                << hrefName << std::endl;
        const unsigned nbx = href->GetNbinsX();
        const double xmin = href->GetXaxis()->GetXmin();
        const double xmax = href->GetXaxis()->GetXmax();
        me = dbe_->book1D(name, title, nbx, xmin, xmax);
    } else {
        LogDebug("TriggerDQM")
                << "\nL1TGMTClient::bookClone1D: not able to clone histogram "
                << hrefName << std::endl;
        me = 0;
    }

    return me;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
MonitorElement* L1TGMTClient::bookClone1DVB(const std::string& name,
        const std::string& title, const std::string& hrefName) {

    MonitorElement* me;

    TH1F* href = get1DHisto(input_dir_ + "/" + hrefName, dbe_);

    if (href) {
        LogDebug("TriggerDQM")
                << "\nL1TGMTClient::bookClone1DVB: booking histogram "
                << hrefName << std::endl;
        int nbx = href->GetNbinsX();
        if (nbx > 99)
            nbx = 99;
        float xbins[100];
        for (int i = 0; i < nbx; i++) {
            xbins[i] = href->GetBinLowEdge(i + 1);
        }
        xbins[nbx] = href->GetXaxis()->GetXmax();

        dbe_->setCurrentFolder(output_dir_);
        me = dbe_->book1D(name, title, nbx, xbins);

    } else {
        LogDebug("TriggerDQM")
                << "\nL1TGMTClient::bookClone1DVB: not able to clone histogram "
                << hrefName << std::endl;
        me = 0;
    }

    return me;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
MonitorElement* L1TGMTClient::bookClone2D(const std::string& name,
        const std::string& title, const std::string& hrefName) {

    MonitorElement* me;

    TH2F* href = get2DHisto(input_dir_ + "/" + hrefName, dbe_);

    if (href) {
        LogDebug("TriggerDQM")
                << "\nL1TGMTClient::bookClone2D: booking histogram "
                << hrefName << std::endl;
        const unsigned nbx = href->GetNbinsX();
        const double xmin = href->GetXaxis()->GetXmin();
        const double xmax = href->GetXaxis()->GetXmax();
        const unsigned nby = href->GetNbinsY();
        const double ymin = href->GetYaxis()->GetXmin();
        const double ymax = href->GetYaxis()->GetXmax();
        me = dbe_->book2D(name, title, nbx, xmin, xmax, nby, ymin, ymax);
    } else {
        LogDebug("TriggerDQM")
                << "\nL1TGMTClient::bookClone2D: not able to clone histogram "
                << hrefName << std::endl;
        me = 0;
    }

    return me;
}

//////////////////////////////////////////////////////////////////////////////////////////////////


