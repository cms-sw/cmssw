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
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

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
	
	std::string SimVtxLabel;

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
	MonitorElement* PlusMe1BeamHaloOcc;
	MonitorElement* PlusMe1BeamHaloOccRing1;
	MonitorElement* PlusMe1BeamHaloOccRing2;
	MonitorElement* PlusMe1BeamHaloOccRing2or3;
	MonitorElement* PlusMe2BeamHaloOcc;
	MonitorElement* PlusMe2BeamHaloOccRing1;
	MonitorElement* PlusMe2BeamHaloOccRing2;
	MonitorElement* PlusMe2BeamHaloOccRing2or3;
	MonitorElement* PlusMe3BeamHaloOcc;
	MonitorElement* PlusMe3BeamHaloOccRing1;
	MonitorElement* PlusMe3BeamHaloOccRing2;
	MonitorElement* PlusMe3BeamHaloOccRing2or3;
	MonitorElement* PlusMe4BeamHaloOcc;
	MonitorElement* PlusMe4BeamHaloOccRing1;
	MonitorElement* PlusMe4BeamHaloOccRing2;
	MonitorElement* PlusMe4BeamHaloOccRing2or3;
	MonitorElement* PlusMe1BeamHaloOccRad;
	
	MonitorElement* MinusMe1BeamHaloOcc;
	MonitorElement* MinusMe1BeamHaloOccRing1;
	MonitorElement* MinusMe1BeamHaloOccRing2;
	MonitorElement* MinusMe1BeamHaloOccRing2or3;
	MonitorElement* MinusMe2BeamHaloOcc;
	MonitorElement* MinusMe2BeamHaloOccRing1;
	MonitorElement* MinusMe2BeamHaloOccRing2;
	MonitorElement* MinusMe2BeamHaloOccRing2or3;
	MonitorElement* MinusMe3BeamHaloOcc;
	MonitorElement* MinusMe3BeamHaloOccRing1;
	MonitorElement* MinusMe3BeamHaloOccRing2;
	MonitorElement* MinusMe3BeamHaloOccRing2or3;
	MonitorElement* MinusMe4BeamHaloOcc;
	MonitorElement* MinusMe4BeamHaloOccRing1;
	MonitorElement* MinusMe4BeamHaloOccRing2;
	MonitorElement* MinusMe4BeamHaloOccRing2or3;
	MonitorElement* MinusMe1BeamHaloOccRad;
	
	edm::ESHandle<CSCGeometry> m_cscGeometry;
	edm::InputTag lctProducer, HLTriggerTag, GMTInputTag, cscRecHitLabel;
	//L1CSCTriggerTag, L1GTRR, trackProducer
	std::string outFile;
	int gtHaloBit;
};

#endif
