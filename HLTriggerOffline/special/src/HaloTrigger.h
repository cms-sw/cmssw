#ifndef HaloTrigger_h
#define HaloTrigger_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
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
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"

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
	unsigned int hltHaloTriggers, hltHaloOver1, hltHaloOver2, hltHaloRing23, CscHalo_Gmt;
	unsigned int majikNumber0, majikNumber1, majikNumber2, majikNumber3;
	
protected:
	void analyze(const edm::Event& e, const edm::EventSetup& es);
	void beginJob(const edm::EventSetup& es);
	void endJob(void);
	
private:
	DQMStore * dbe;
	MonitorElement* TriggerChainEff;
	MonitorElement* haloDelEta23;
	MonitorElement* haloDelPhi23;
	
	CSCSectorReceiverLUT *srLUTs_[5];
	
	edm::InputTag lctProducer, HLTriggerTag, GMTInputTag;
	//L1CSCTriggerTag, L1GTRR, trackProducer
	std::string outFile;
	int gtHaloBit;
};

#endif
