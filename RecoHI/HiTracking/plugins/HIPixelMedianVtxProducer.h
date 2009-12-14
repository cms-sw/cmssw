#ifndef HIPixelMedianVtxProducer_H
#define HIPixelMedianVtxProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm { class Event; class EventSetup; }

class HIPixelMedianVtxProducer : public edm::EDProducer
{
public:
	explicit HIPixelMedianVtxProducer(const edm::ParameterSet& ps);
	~HIPixelMedianVtxProducer();
	virtual void produce(edm::Event& ev, const edm::EventSetup& es);
	
private:
	void beginJob();
	
	edm::ParameterSet theConfig;
	double thePtMin;
};
#endif
