#ifndef HIBestVertexProducer_H
#define HIBestVertexProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

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
	edm::EDGetTokenT<reco::BeamSpot> theBeamSpotTag;
	edm::EDGetTokenT<reco::VertexCollection> theMedianVertexCollection;
	edm::EDGetTokenT<reco::VertexCollection> theAdaptiveVertexCollection;
};
#endif
