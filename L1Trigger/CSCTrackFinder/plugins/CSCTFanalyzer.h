#ifndef CSCTFAnalyzer_h
#define CSCTFAnalyzer_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TTree.h"
#include "TFile.h"

class CSCTFanalyzer : public edm::EDAnalyzer {
private:
	edm::InputTag lctProducer, dataTrackProducer, emulTrackProducer;
	TTree *tree;
	TFile *file;
	int nDataMuons, nEmulMuons, verbose;
	int dphi1, deta1, dpt1, dch1, dbx1;
	int dphi2, deta2, dpt2, dch2, dbx2;
	int dphi3, deta3, dpt3, dch3, dbx3;
	int ephi1, eeta1, ept1, ech1, ebx1;
	int ephi2, eeta2, ept2, ech2, ebx2;
	int ephi3, eeta3, ept3, ech3, ebx3;

public:
	virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
	virtual void endJob(void);
	virtual void beginJob(edm::EventSetup const&){}

	explicit CSCTFanalyzer(edm::ParameterSet const& pset);
	virtual ~CSCTFanalyzer(void) {}
};

#endif

