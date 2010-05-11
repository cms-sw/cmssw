#ifndef CSCTFAnalyzer_h
#define CSCTFAnalyzer_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <TTree.h>
#include <TFile.h>

class CSCTFAnalyzer : public edm::EDAnalyzer {
private:
	edm::InputTag mbProducer, lctProducer, trackProducer, statusProducer;
	TTree *tree;
	TFile *file;
	int dtPhi[12][2];

public:
	void analyze(const edm::Event& e, const edm::EventSetup& c);

	explicit CSCTFAnalyzer(const edm::ParameterSet &conf);
	~CSCTFAnalyzer(void){ file->cd(); tree->Write(); file->Close(); }
};

#endif
