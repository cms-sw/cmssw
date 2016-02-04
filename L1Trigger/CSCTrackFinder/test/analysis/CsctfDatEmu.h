#ifndef CsctfDatEmu_h
#define CsctfDatEmu_h

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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include <L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <unistd.h>

#include "TTree.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"

class CsctfDatEmu : public edm::EDAnalyzer {
public:
	explicit CsctfDatEmu(edm::ParameterSet const& ps);
	virtual ~CsctfDatEmu() {}
	virtual void analyze(edm::Event const& e, edm::EventSetup const& es);
	virtual void endJob();
	virtual void beginJob();
	int dtCount, cscCount, dcCount;
private:
	edm::InputTag  lctProducer, dataTrackProducer, emulTrackProducer;

	std::string outFile;
	CSCSectorReceiverLUT *srLUTs_[5];
	CSCTFDTReceiver* my_dtrc;
	
	TFile* fAnalysis;
	
	TH2F* modeComp, *phi12Comp, *phi23Comp, *etaComp, *signFrComp, *bxComp;
	TH2F* numTrkComp;
	TH1F* badPhiMode, *badEtaMode, *badEtaPhi, *badPhiEta;
	TH1F* badModePhi, *badModeEta, *badBxMode;
	TH1F* moreDatModeE, *moreEmuModeE, *moreDatModeD, *moreEmuModeD;
	TH1F* dtRBx, *dtRBx0, *dtRMlink, *dtRQ, *dtRBend, *dtRStrip;
	TH1F* dtBBx, *dtBBx0, *dtBMlink, *dtBQ, *dtBBend, *dtBStrip;
	TH2F* badSectorEnd;
	TH2F* stubDtStrip, *stubDtBx, *stubDtBX0, *stubDtMpc, *stubDtQual, *stubDtBend, *stubDtPhi, *stubBadPhiSec, *badRankSecEnd;
	TH1F* stubDtoTrack, *stubCtoTrack;
	
	TH2F* rankComp, *qualComp, *ptComp, *phiVComp, *etaVComp;
	TH1F* badRankMode, *badRankEta, *badRankLink, *badRankOcc;
	TH1F* stubEmuBx, *stubDatBx, *stubDifBx;

};

DEFINE_FWK_MODULE(CsctfDatEmu);

#endif
