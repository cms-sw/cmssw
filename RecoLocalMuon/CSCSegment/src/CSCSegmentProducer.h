#ifndef CSCSegment_CSCSegmentProducer_h
#define CSCSegment_CSCSegmentProducer_h

/** \class CSCSegmentProducer 
 * Produces a collection of CSCSegment's in endcap muon CSCs. 
 *
 * $Date: 2010/02/11 00:14:04 $
 * $Revision: 1.5 $
 * \author M. Sani
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
    edm::InputTag inputObjectsTag; // input tag labelling rechits for input
    CSCSegmentBuilder* segmentBuilder_;
};

#endif
