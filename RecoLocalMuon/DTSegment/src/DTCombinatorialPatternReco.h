#ifndef DTSegment_DTCombinatorialPatternReco_h
#define DTSegment_DTCombinatorialPatternReco_h

/** \class DTCombinatorialPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a combinatorial approach
 *  
 * $Date: 2006/04/21 14:25:38 $
 * $Revision: 1.5 $
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

/* C++ Headers */
#include <vector>
#include <deque>
#include <utility>

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

/* ====================================================================== */

/* Class DTCombinatorialPatternReco Interface */

class DTCombinatorialPatternReco : public DTRecSegment2DBaseAlgo {

  public:

/// Constructor
    DTCombinatorialPatternReco(const edm::ParameterSet& pset) ;

/// Destructor
    virtual ~DTCombinatorialPatternReco() ;

/* Operations */ 
    virtual edm::OwnVector<DTRecSegment2D>
      reconstruct(const DTSuperLayer* sl,
                  const std::vector<DTRecHit1DPair>& hits);
	
    
    virtual std::string algoName() const { return theAlgoName; }
    
    virtual void setES(const edm::EventSetup& setup);
 protected:

 private:
    friend class DTCombinatorialPatternReco4D;

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
    DTSegmentCand* buildBestSegment(std::vector<AssPoint>& assHits,
                                    const DTSuperLayer* sl) ;

    bool checkDoubleCandidates(std::vector<DTSegmentCand*>& segs,
                               DTSegmentCand* seg);

    /** build collection of compatible hits for L/R hits: the candidates is
     * updated with the segment candidates found */
    void buildPointsCollection(std::vector<AssPoint>& points, 
                               std::deque<DTHitPairForFit* >& pointsNoLR,
                               std::vector<DTSegmentCand*>& candidates,
                               const DTSuperLayer* sl);


    // FIXME it isnt't in the abstract interface!!
    // builds the superPhi-segments candidates
    //std::vector<DTSegmentCand*> buildPhiSuperSegments(const std::vector<DTRecSegment2DPhi>& segments2DPhi1,
    //const std::vector<DTRecSegment2DPhi>& segments2DPhi2);
    //friend class DTCombinatorialPatternReco4D;
    //

  private:
    std::string theAlgoName;
    unsigned int theMaxAllowedHits;
    double theAlphaMaxTheta;
    double theAlphaMaxPhi;
    bool debug;
    DTSegmentUpdator* theUpdator; // the updator and fitter
    DTSegmentCleaner* theCleaner; // the cleaner
    
    edm::ESHandle<DTGeometry> theDTGeometry; // the DT geometry
};
#endif // DTSegment_DTCombinatorialPatternReco_h
