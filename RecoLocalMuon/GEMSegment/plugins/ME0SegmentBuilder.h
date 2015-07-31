#ifndef GEMRecHit_ME0SegmentBuilder_h
#define GEMRecHit_ME0SegmentBuilder_h

/** \class ME0SegmentBuilder derived by CSCSegmentBuilder
 * Algorithm to build ME0Segment's from ME0RecHit collection
 * by implementing a 'build' function required by ME0SegmentProducer.
 *
 * Implementation notes: <BR>
 * Configured via the Producer's ParameterSet. <BR>
 * Presume this might become an abstract base class one day. <BR>
 *
 * $Date: 2014/02/04 13:45:31 $
 * $Revision: 1.1 $
 * \author Marcello Maggi
 *
 */

#include <DataFormats/GEMRecHit/interface/ME0RecHitCollection.h>
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>
#include <Geometry/GEMGeometry/interface/ME0Geometry.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

class ME0SegmentAlgorithm;

class ME0SegmentBuilder {
public:
   
    /** Configure the algorithm via ctor.
     * Receives ParameterSet percolated down from EDProducer
     * which owns this Builder.
     */
    explicit ME0SegmentBuilder(const edm::ParameterSet&);
    /// Destructor
    ~ME0SegmentBuilder();

    /** Find rechits in each ensemble of 6 ME0 layers, build ME0Segment's ,
     *  and fill into output collection.
     */
    void build(const ME0RecHitCollection* rechits, ME0SegmentCollection& oc);

    /** Cache pointer to geometry _for current event_
     */
    void setGeometry(const ME0Geometry* g);

private:
    ME0SegmentAlgorithm* algo;
    const ME0Geometry* geom_; 
};

#endif
