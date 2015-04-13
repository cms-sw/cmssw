#ifndef ZDCMonitorClient_H
#define ZDCMonitorClient_H

// Update on September 21, 2012 to match HcalMonitorClient
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"
#include "DQM/HcalMonitorClient/interface/HcalSummaryClient.h"

class DQMStore;
//class TH2F;
//class TH1F;
//class TFile;

class ZDCMonitorClient : public HcalBaseDQClient {

public:

	/// Constructors
	//ZDCMonitorClient();
	ZDCMonitorClient(std::string myname, const edm::ParameterSet& ps);

	/// Destructor
	virtual ~ZDCMonitorClient();

	/// Analyze
	virtual void analyze(DQMStore::IBooker &, DQMStore::IGetter &) override; // fill new histograms

	/// BeginJob
	// void beginJob();

	/// EndJob
	void endJob(void) override;

	/// BeginRun
	void beginRun() override;
	//  void beginRun(const edm::Run& r, const edm::EventSetup & c);

	/// EndRun
	//  void endRun();
	//  void endRun(const edm::Run & r, const edm::EventSetup & c);

	/// BeginLumiBlock
	//  void beginLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

	/// EndLumiBlock
	//  void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

	/// Reset
	void reset(void);

	/// Setup
	// void setup(void) override;

	/// Cleanup
	// void cleanup(void) override;

	/// SoftReset
	void softReset(bool flag);

	// Write channelStatus info
	void writeChannelStatus();

	// Write html output
	void writeHtml();

private:

	int ievt_; // all events
	int jevt_; // events in current run
	int run_;
	int evt_;
	bool begin_run_;
	bool end_run_;

	// parameter set inputs

	std::vector<double> ZDCGoodLumi_;
	std::string ZDCsubdir_;

	///////////////////New plots as of Fall 2012/////////////
	int LumiCounter;
	int PZDC_GoodLumiCounter;
	int NZDC_GoodLumiCounter;
	double PZDC_LumiRatio;
	double NZDC_LumiRatio;

	MonitorElement* ZDCChannelSummary_;
	MonitorElement* ZDCHotChannelFraction_;
	MonitorElement* ZDCColdChannelFraction_;
	MonitorElement* ZDCDeadChannelFraction_;
	MonitorElement* ZDCDigiErrorFraction_;
	MonitorElement* ZDCReportSummary_;
	/////////////new plots as of Fall 2012//////////////////
};

#endif
