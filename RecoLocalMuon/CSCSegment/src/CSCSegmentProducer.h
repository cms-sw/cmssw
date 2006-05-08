#ifndef CSCSegment_CSCSegmentProducer_h
#define CSCSegment_CSCSegmentProducer_h

/** \class CSCSegmentProducer 
 * Produces a collection of CSCSegment's in endcap muon CSCs. 
 *
 * $Date: 2006/04/03 10:10:10 $
 * $Revision: 1.2 $
 * \author M. Sani
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CSCSegmentBuilder; 

class CSCSegmentProducer : public edm::EDProducer {
public:
    /// Constructor
    explicit CSCSegmentProducer(const edm::ParameterSet&);
    /// Destructor
    ~CSCSegmentProducer();
    /// Produce the CSCSegment collection
    virtual void produce(edm::Event&, const edm::EventSetup&);

private:
    int iev; // events through
    std::string recHitProducer_;
    CSCSegmentBuilder* segmentBuilder_;
};

#endif
