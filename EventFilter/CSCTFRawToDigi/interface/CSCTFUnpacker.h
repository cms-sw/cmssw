#ifndef CSCTFUnpacker_h
#define CSCTFUnpacker_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

//CSC Track Finder Raw Data Format
#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.h"

#include <vector>
#include <string>

class CSCTriggerMapping;

class CSCTFUnpacker: public edm::stream::EDProducer<> {
private:
	int  m_minBX, m_maxBX;
	bool swapME1strips;

	CSCTriggerMapping *mapping; // redundant, but needed

	CSCTFEvent tfEvent; // TF data container

	// geometry may not be properly set in CSC TF data
	// make an artificial assignment of each of 12 SPs (slots 6-11 and 16-21) to 12 sectors (1-12, 0-not assigned)
	std::vector<int> slot2sector;

	// label of the module which produced raw data
	edm::InputTag producer;

public:
	void produce(edm::Event& e, const edm::EventSetup& c);

	CSCTFUnpacker(const edm::ParameterSet& pset);
	~CSCTFUnpacker(void);
};

#endif
