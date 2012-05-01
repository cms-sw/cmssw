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
	edm::InputTag lctProducer, dataTrackProducer, emulTrackProducer;
	int nDataMuons, nEmulMuons;
	int eventNum;

	const L1MuTriggerScales *ts;
	CSCTFPtLUT* ptLUT_;
	edm::ParameterSet ptLUTset;
	
	DQMStore * dbe;
	MonitorElement* phiComp;
	MonitorElement* etaComp;
	MonitorElement* chargeComp;
	MonitorElement* trackCountComp;
	MonitorElement* badBitMode1,  *badBitMode2,  *badBitMode3,  *badBitMode4,  *badBitMode5;
	MonitorElement* badBitMode6,  *badBitMode7,  *badBitMode8,  *badBitMode9,  *badBitMode10;
	MonitorElement* badBitMode11, *badBitMode12, *badBitMode13, *badBitMode14, *badBitMode15;
	MonitorElement* pt1Comp, *pt2Comp, *pt3Comp, *pt4Comp, *pt5Comp, *pt6Comp;
	MonitorElement* ptLUTOutput;
	MonitorElement* mismatchSector, *mismatchTime, *mismatchEndcap;
	MonitorElement* mismatchPhi, *mismatchEta;
	MonitorElement* endTrackBadSector;
	MonitorElement* endTrackBadFR;
	MonitorElement* endTrackBadEta;
	MonitorElement* endTrackBadMode;
	MonitorElement* bxData, *bxEmu;
	MonitorElement* allLctBx;
	MonitorElement* mismatchDelPhi12, *mismatchDelPhi13, *mismatchDelPhi14, *mismatchDelPhi23, *mismatchDelPhi24, *mismatchDelPhi34;
	MonitorElement* mismatchDelEta12, *mismatchDelEta13, *mismatchDelEta14, *mismatchDelEta23, *mismatchDelEta24, *mismatchDelEta34	;
	
	// dqm folder name
	std::string m_dirName;
	
	std::string outFile;
	
	CSCSectorReceiverLUT *srLUTs_[2][6][5];

public:
	void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
	void endJob(void);
	void beginJob();

	explicit L1TdeCSCTF(edm::ParameterSet const& pset);
	virtual ~L1TdeCSCTF() {}
};

#endif

