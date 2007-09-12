#ifndef CSCTFAnalyzer_h
#define CSCTFAnalyzer_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


class CSCTFAnalyzer : public edm::EDAnalyzer {
private:
	edm::InputTag lctProducer, trackProducer;

public:
	void analyze(const edm::Event& e, const edm::EventSetup& c);

	explicit CSCTFAnalyzer(const edm::ParameterSet &conf);
	~CSCTFAnalyzer(void){}
};

#endif
