#ifndef DQMOFFLINE_TRIGGER_TOPELECTRONHLTOFFLINECLIENT
#define DQMOFFLINE_TRIGGER_TOPELECTRONHLTOFFLINECLIENT

// -*- C++ -*-
//
// Package:		TopElectronHLTOfflineClient
// Class:			EgammaHLTOffline
// 
/*
 Description: This is a DQM client meant to plot high-level HLT trigger 
 quantities as stored in the HLT results object TriggerResults for the Egamma triggers

 Notes:
	Currently I would like to plot simple histograms of three seperate types of variables
	1) global event quantities: eg nr of electrons
	2) di-object quanities: transverse mass, di-electron mass
	3) single object kinematic and id variables: eg et,eta,isolation

*/
//
// Original Author:	Sam Harper
//				 Created:	June 2008
// 
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <vector>
#include <string>

class DQMStore;
class MonitorElement;

	
class TopElectronHLTOfflineClient : public edm::EDAnalyzer {

private:
	DQMStore* dbe_; // non-owned dqm store
	std::string dirName_;
	
	std::string hltTag_;
	
	std::vector<std::string> superMeNames_;
	std::vector<std::string> eleMeNames_;
	
	std::vector<std::string> electronIdNames_;
	std::vector<std::string> superTriggerNames_;
	std::vector<std::string> electronTriggerNames_;
	
	bool addExtraId_;
	
	bool runClientEndLumiBlock_;
	bool runClientEndRun_;
	bool runClientEndJob_;


	//disabling copying/assignment 
	TopElectronHLTOfflineClient(const TopElectronHLTOfflineClient& rhs){}
	TopElectronHLTOfflineClient& operator=(const TopElectronHLTOfflineClient& rhs){return *this;}

public:
	explicit TopElectronHLTOfflineClient(const edm::ParameterSet& );
	virtual ~TopElectronHLTOfflineClient();
	
	
	virtual void beginJob();
	virtual void analyze(const edm::Event&, const edm::EventSetup&); //dummy
	virtual void endJob();
	virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
	virtual void endRun(const edm::Run& run, const edm::EventSetup& c);

	virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& context){}
	// DQM Client Diagnostic
	virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c);

	MonitorElement* makeEffMonElemFromPassAndAll(const std::string& name,const MonitorElement* pass,const MonitorElement* fail);
	void createSingleEffHists(const std::string&, const std::string&, const std::string&);

private:
	void runClient_(); //master function which runs the client
	
};
 


#endif
