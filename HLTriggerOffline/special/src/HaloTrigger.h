#ifndef HaloTrigger_h
#define HaloTrigger_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <string>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TStyle.h>

class HaloTrigger : public edm::EDAnalyzer {

public:
	HaloTrigger(const edm::ParameterSet& ps);
	virtual ~HaloTrigger();

	bool first;
	std::vector<std::string> Namen;
	unsigned int majikNumber_HLThalo;

protected:
	void analyze(const edm::Event& e, const edm::EventSetup& es);
	void beginJob(const edm::EventSetup& es);
	void endJob(void);

private:
	DQMStore * dbe;
	MonitorElement* halocount;

	edm::InputTag HLTriggerTag, L1CSCTriggerTag, L1GTRR, GMTInputTag, trackProducer;
	std::string outFile;
	int gtHaloBit;
};

#endif
