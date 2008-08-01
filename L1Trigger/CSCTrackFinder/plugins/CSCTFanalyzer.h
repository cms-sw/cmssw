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
	edm::InputTag lctProducer, trackProducer;
	TTree *tree;
	TFile *file;
	int nMuons, verbose;
	int phi1, eta1, pt1, ch1, bx1;
	int phi2, eta2, pt2, ch2, bx2;

public:
	virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
	virtual void endJob(void);
	virtual void beginJob(edm::EventSetup const&){}

	explicit CSCTFanalyzer(edm::ParameterSet const& pset);
	virtual ~CSCTFanalyzer(void) {}
};

#endif

