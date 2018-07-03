#ifndef GEMRecHit_GEMSegmentBuilder_h
#define GEMRecHit_GEMSegmentBuilder_h

/** \class GEMSegmentBuilder derived by CSCSegmentBuilder
 * Algorithm to build GEMSegment's from GEMRecHit collection
 * by implementing a 'build' function required by GEMSegmentProducer.
 *
 * Implementation notes: <BR>
 * Configured via the Producer's ParameterSet. <BR>
 * Presume this might become an abstract base class one day. <BR>
 *
 * \author Piet Verwilligen
 *
 */

#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GEMSegmentAlgorithmBase;

class GEMSegmentBuilder {
public:
   
    /** Configure the algorithm via ctor.
     * Receives ParameterSet percolated down from EDProducer
     * which owns this Builder.
     */
    explicit GEMSegmentBuilder(const edm::ParameterSet&);
    /// Destructor
    ~GEMSegmentBuilder();

    /** Find rechits in each ensemble of 6 GEM layers, build GEMSegment's ,
     *  and fill into output collection.
     */
    void build(const GEMRecHitCollection* rechits, GEMSegmentCollection& oc);

    /** Cache pointer to geometry _for current event_
     */
    void setGeometry(const GEMGeometry* g);

private:

    std::string algoName;
    edm::ParameterSet segAlgoPSet;
    std::unique_ptr<GEMSegmentAlgorithmBase> algo;
    const GEMGeometry* geom_; 
};

#endif
