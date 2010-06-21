#ifndef CSCTFSingleGen_h
#define CSCTFSingleGen_h

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

class CSCTriggerMapping;

class CSCTFSingleGen: public edm::EDProducer {
private:
	int  m_minBX, m_maxBX;
	int  endcap, sector, subSector, station, cscId, strip, wireGroup, pattern;
	CSCTriggerMapping *mapping; // redundant, but needed

public:
	void produce(edm::Event& e, const edm::EventSetup& c);

	CSCTFSingleGen(const edm::ParameterSet& pset);
	~CSCTFSingleGen(void);
};

#endif
