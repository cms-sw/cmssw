#ifndef CSCTFAnalyzer_h
#define CSCTFAnalyzer_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

//consumes
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

#include <TTree.h>
#include <TFile.h>

class CSCTFAnalyzer : public edm::EDAnalyzer {
private:
	edm::InputTag mbProducer, lctProducer, trackProducer, statusProducer;
	TTree *tree;
	TFile *file;
	int dtPhi[12][2];

	edm::EDGetTokenT<L1CSCStatusDigiCollection> L1CSCS_Tok;
	edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> > CSCTC_Tok;
	edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> CSCCDC_Tok;
	edm::EDGetTokenT<L1CSCTrackCollection> L1CST_Tok;


public:
	void analyze(const edm::Event& e, const edm::EventSetup& c) override;

	explicit CSCTFAnalyzer(const edm::ParameterSet &conf);
	~CSCTFAnalyzer(void) override{ file->cd(); tree->Write(); file->Close(); }
};

#endif
