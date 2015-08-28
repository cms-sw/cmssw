#ifndef GEMRecHit_ME0SegmentProducer_h
#define GEMRecHit_ME0SegmentProducer_h

/** \class ME0SegmentProducer derived by CSCSegmentProducer 
 * Produces a collection of ME0Segment's in endcap muon ME0s. 
 *
 * \author Marcello Maggi
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <DataFormats/GEMRecHit/interface/ME0RecHitCollection.h>

class ME0SegmentBuilder; 

class ME0SegmentProducer : public edm::stream::EDProducer<> {
public:
    /// Constructor
    explicit ME0SegmentProducer(const edm::ParameterSet&);
    /// Destructor
    ~ME0SegmentProducer();
    /// Produce the ME0Segment collection
    virtual void produce(edm::Event&, const edm::EventSetup&);

private:
    int iev; // events through
    edm::EDGetTokenT<ME0RecHitCollection> theME0RecHitToken;
    std::unique_ptr<ME0SegmentBuilder> segmentBuilder_;
};

#endif
