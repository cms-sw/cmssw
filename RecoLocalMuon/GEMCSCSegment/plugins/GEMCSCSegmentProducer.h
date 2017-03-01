#ifndef GEMCSCSegment_GEMCSCSegmentProducer_h
#define GEMCSCSegment_GEMCSCSegmentProducer_h

/** \class GEMCSCSegmentProducer 
 * Produces a collection of GEM-CSCSegments
 *
 * $Date: 2014/02/06 12:19:20 $
 *
 * \author Raffaella Radogna
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

class GEMCSCSegmentBuilder; 

class GEMCSCSegmentProducer : public edm::stream::EDProducer<> {
public:
    /// Constructor
    explicit GEMCSCSegmentProducer(const edm::ParameterSet&);
    /// Destructor
    ~GEMCSCSegmentProducer();
    /// Produce the GEM-CSCSegment collection
    virtual void produce(edm::Event&, const edm::EventSetup&);

private:
    int iev; // events through
    GEMCSCSegmentBuilder* segmentBuilder_;
    edm::EDGetTokenT<CSCSegmentCollection> csc_token;
    edm::EDGetTokenT<GEMRecHitCollection>  gem_token;
};

#endif
