#ifndef GEMRecHit_ME0SegmentProducer_h
#define GEMRecHit_ME0SegmentProducer_h

/** \class ME0SegmentProducer derived by CSCSegmentProducer 
 * Produces a collection of ME0Segment's in endcap muon ME0s. 
 *
 * $Date: 2014/02/03 23:48:11 $
 * $Revision: 1.1 $
 * \author Marcello Maggi
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ME0SegmentBuilder; 

class ME0SegmentProducer : public edm::EDProducer {
public:
    /// Constructor
    explicit ME0SegmentProducer(const edm::ParameterSet&);
    /// Destructor
    ~ME0SegmentProducer();
    /// Produce the ME0Segment collection
    virtual void produce(edm::Event&, const edm::EventSetup&);

private:
    int iev; // events through
    edm::InputTag inputObjectsTag; // input tag labelling rechits for input
    ME0SegmentBuilder* segmentBuilder_;
};

#endif
