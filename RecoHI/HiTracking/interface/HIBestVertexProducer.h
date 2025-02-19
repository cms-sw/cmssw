#ifndef HIBestVertexProducer_H
#define HIBestVertexProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm { class Event; class EventSetup; }

class HIBestVertexProducer : public edm::EDProducer
{
public:
	explicit HIBestVertexProducer(const edm::ParameterSet& ps);
	~HIBestVertexProducer();
	virtual void produce(edm::Event& ev, const edm::EventSetup& es);
	
private:
	void beginJob();
	edm::ParameterSet theConfig;
        edm::InputTag theBeamSpotTag;
        edm::InputTag theMedianVertexCollection;
        edm::InputTag theAdaptiveVertexCollection;
};
#endif
