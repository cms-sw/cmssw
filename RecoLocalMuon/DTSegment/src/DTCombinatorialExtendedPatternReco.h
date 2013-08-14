#ifndef DTSegment_DTCombinatorialExtendedPatternReco_h
#define DTSegment_DTCombinatorialExtendedPatternReco_h

/** \class DTCombinatorialExtendedPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a combinatorial approach
 *  
 * $Date: 2008/12/03 12:52:22 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DBaseAlgo.h"

/* Collaborating Class Declarations */
namespace edm {
  class ParameterSet;
  class EventSetup;
  //  class ESHandle;
}
class DTSegmentUpdator;
class DTSegmentCleaner;
class DTHitPairForFit;
class DTSegmentCand;
class DTSegmentExtendedCand;
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"

/* C++ Headers */
#include <vector>
#include <deque>
#include <utility>

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

/* ====================================================================== */

/* Class DTCombinatorialExtendedPatternReco Interface */

class DTCombinatorialExtendedPatternReco : private DTRecSegment2DBaseAlgo {

  public:

    /// Constructor
    DTCombinatorialExtendedPatternReco(const edm::ParameterSet& pset) ;

    /// Destructor
    virtual ~DTCombinatorialExtendedPatternReco() ;

    /* Operations */

    /// this function is called in the producer
    virtual edm::OwnVector<DTSLRecSegment2D>
      reconstruct(const DTSuperLayer* sl,
                  const std::vector<DTRecHit1DPair>& hits);

    /// return the algo name
    virtual std::string algoName() const { return theAlgoName; }

    /// Through this function the EventSetup is percolated to the
    /// objs which request it
    virtual void setES(const edm::EventSetup& setup);

    // pass clusters to algo
    void setClusters(std::vector<DTSLRecCluster> clusters);

  protected:

  private:
    typedef std::pair<DTHitPairForFit*, DTEnums::DTCellSide> AssPoint;

    // create the DTHitPairForFit from the pairs for easy use
    std::vector<DTHitPairForFit*> initHits(const DTSuperLayer* sl,
                                           const std::vector<DTRecHit1DPair>& hits);

    // search for candidate, starting from pairs of hits in different layers
    std::vector<DTSegmentCand*> buildSegments(const DTSuperLayer* sl,
                                              const std::vector<DTHitPairForFit*>& hits);

    // find all the hits compatible with the candidate
    std::vector<AssPoint> findCompatibleHits(const LocalPoint& pos,
                                             const LocalVector& dir,
                                             const std::vector<DTHitPairForFit*>& hits);

    // build segments from hits collection
    DTSegmentExtendedCand* buildBestSegment(std::vector<AssPoint>& assHits,
                                            const DTSuperLayer* sl) ;

    bool checkDoubleCandidates(std::vector<DTSegmentCand*>& segs,
                               DTSegmentCand* seg);

    /** build collection of compatible hits for L/R hits: the candidates is
     * updated with the segment candidates found */
    void buildPointsCollection(std::vector<AssPoint>& points, 
                               std::deque<DTHitPairForFit* >& pointsNoLR,
                               std::vector<DTSegmentCand*>& candidates,
                               const DTSuperLayer* sl);

    /** extend the candidates with clusters from external SL */
    std::vector<DTSegmentExtendedCand*> extendCandidates(std::vector<DTSegmentCand*>& candidates,
                                                          const DTSuperLayer* sl);

    bool closeSL(const DTSuperLayerId& id1, const DTSuperLayerId& id2);

   private:

    std::string theAlgoName;
    unsigned int theMaxAllowedHits;
    double theAlphaMaxTheta;
    double theAlphaMaxPhi;
    bool debug;
    bool usePairs;
    DTSegmentUpdator* theUpdator; // the updator and fitter
    DTSegmentCleaner* theCleaner; // the cleaner

    edm::ESHandle<DTGeometry> theDTGeometry; // the DT geometry

  private:

    std::vector<std::vector<int> > theTriedPattern;
    std::vector<DTSLRecCluster> theClusters;
};
#endif // DTSegment_DTCombinatorialExtendedPatternReco_h
