#ifndef CSCTFPacker_h
#define CSCTFPacker_h

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

#include <string>

class CSCTFPacker : public edm::one::EDProducer<> {
private:
	edm::InputTag lctProducer, mbProducer, trackProducer;

	bool zeroSuppression;
	unsigned short nTBINs;
	unsigned short activeSectors;
	bool putBufferToEvent;

	bool swapME1strips;

	FILE *file;

	int m_minBX, m_maxBX, central_lct_bx, central_sp_bx;

	edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> CSCCDC_Tok;
	edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> > CSCTC_Tok;
	edm::EDGetTokenT<L1CSCTrackCollection> L1CSCTr_Tok;

public:
	void produce(edm::Event& e, const edm::EventSetup& c) override;

	explicit CSCTFPacker(const edm::ParameterSet &conf);
	~CSCTFPacker(void) override;
};

#endif
