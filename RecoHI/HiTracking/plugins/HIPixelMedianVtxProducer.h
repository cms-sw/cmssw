#ifndef HIPixelMedianVtxProducer_H
#define HIPixelMedianVtxProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Utilities/interface/InputTag.h"

namespace edm { class Event; class EventSetup; }

class HIPixelMedianVtxProducer : public edm::EDProducer
{
public:
	explicit HIPixelMedianVtxProducer(const edm::ParameterSet& ps);
	~HIPixelMedianVtxProducer(){};
	virtual void produce(edm::Event& ev, const edm::EventSetup& es);
	
private:
	void beginJob(){};
	
	edm::InputTag theTrackCollection;
	double thePtMin;
	unsigned int thePeakFindThresh;
	double thePeakFindMaxZ;
	int thePeakFindBinning;
	int theFitThreshold;
	double theFitMaxZ;
	int theFitBinning;
};

struct ComparePairs
{
  bool operator() (const reco::Track * t1,
		   const reco::Track * t2)
  {
    return (t1->vz() < t2->vz());
  };
};

#endif
