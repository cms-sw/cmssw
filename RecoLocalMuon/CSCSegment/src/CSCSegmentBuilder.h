#ifndef CSCSegment_CSCSegmentBuilder_h
#define CSCSegment_CSCSegmentBuilder_h

/** \class CSCSegmentBuilder 
 * Algorithm to build CSCSegment's from CSCRecHit2D collection
 * by implementing a 'build' function required by CSCSegmentProducer.
 *
 * Implementation notes: <BR>
 * Configured via the Producer's ParameterSet. <BR>
 * Presume this might become an abstract base class one day. <BR>
 *
 * $Date: 2006/05/08 17:45:31 $
 * $Revision: 1.3 $
 * \author M. Sani
 *
 *
 */

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCGeometry;
class CSCSegmentAlgorithm;

class CSCSegmentBuilder {
public:
   
    /** Configure the algorithm via ctor.
     * Receives ParameterSet percolated down from EDProducer
     * which owns this Builder.
     */
    explicit CSCSegmentBuilder(const edm::ParameterSet&);
    /// Destructor
    ~CSCSegmentBuilder();

    /** Find rechits in each CSCChamber, build CSCSegment's in each chamber,
     *  and fill into output collection.
     */
    void build(const CSCRecHit2DCollection* rechits, CSCSegmentCollection& oc);

    /** Cache pointer to geometry _for current event_
     */
    void setGeometry(const CSCGeometry* geom);

private:

    const CSCGeometry* geom_;
    std::map<std::string, CSCSegmentAlgorithm*> algoMap;
};

#endif
