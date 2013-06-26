#ifndef L1TdeCSCTF_h
#define L1TdeCSCTF_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <unistd.h>

#include "TTree.h"
#include "TFile.h"

class L1TdeCSCTF : public edm::EDAnalyzer {
private:
	edm::InputTag lctProducer, dataTrackProducer, emulTrackProducer, dataStubProducer, emulStubProducer;

	const L1MuTriggerScales *ts;
	CSCTFPtLUT* ptLUT_;
	edm::ParameterSet ptLUTset;
	CSCTFDTReceiver* my_dtrc;
	
	// Define Monitor Element Histograms
	////////////////////////////////////
	DQMStore * dbe;
	MonitorElement* phiComp, *etaComp, *occComp, *ptComp, *qualComp;
	MonitorElement* pt1Comp, *pt2Comp, *pt3Comp, *pt4Comp, *pt5Comp, *pt6Comp;
	MonitorElement* dtStubPhi, *badDtStubSector;
	
        MonitorElement* phiComp_1d, *etaComp_1d, *occComp_1d, *ptComp_1d, *qualComp_1d;
	MonitorElement* pt1Comp_1d, *pt2Comp_1d, *pt3Comp_1d, *pt4Comp_1d, *pt5Comp_1d, *pt6Comp_1d;
	MonitorElement* dtStubPhi_1d;
	
	// dqm folder name
	//////////////////
	std::string m_dirName;
	std::string outFile;
	

public:
	void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
	void endJob(void);
	void beginJob();

	explicit L1TdeCSCTF(edm::ParameterSet const& pset);
	virtual ~L1TdeCSCTF() {}
};

#endif

