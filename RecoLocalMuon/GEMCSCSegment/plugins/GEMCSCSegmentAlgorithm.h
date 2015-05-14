#ifndef GEMRecHit_GEMCSCSegmentAlgorithm_h
#define GEMRecHit_GEMCSCSegmentAlgorithm_h

/** \class GEMCSCSegmentAlgo
 *
 * An abstract base class for algorithmic classes used to 
 * build segments combining CSC and GEM information.
 *
 * $Date:  $
 * $Revision: 1.7 $
 * \author Raffaella Radogna
 *
 */

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>

#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMCSCSegment.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <vector>
#include <map>

class GEMCSCSegmentAlgorithm {

    public:

    /// Constructor
    explicit GEMCSCSegmentAlgorithm(const edm::ParameterSet&) {};
    /// Destructor
    virtual ~GEMCSCSegmentAlgorithm() {};
    /// Run the algorithm = build segments 
    virtual std::vector<GEMCSCSegment> run(std::map<uint32_t, const CSCLayer*> csclayermap, std::map<uint32_t, const GEMEtaPartition*> gemrollmap,
					   std::vector<const CSCSegment*> cscsegments, std::vector<const GEMRecHit*> gemrechits) = 0;
    private:
};

#endif
