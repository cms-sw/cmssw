#ifndef DTSegment_DTCombinatorialPatternReco_h
#define DTSegment_DTCombinatorialPatternReco_h

/** \class DTCombinatorialPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a combinatorial approach
 *  
 * $Date: 2009/11/27 11:59:48 $
 * $Revision: 1.12 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DBaseAlgo.h"
#include <boost/unordered_set.hpp>

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

    /// this function is called in the producer
    virtual edm::OwnVector<DTSLRecSegment2D>
      reconstruct(const DTSuperLayer* sl,
                  const std::vector<DTRecHit1DPair>& hits);

    /// return the algo name
    virtual std::string algoName() const { return theAlgoName; }

    /// Through this function the EventSetup is percolated to the
    /// objs which request it
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

  public:
    // The type must be public, as otherwise the global 'hash_value' function can't locate it
    class TriedPattern {
        public:
            typedef std::vector<short unsigned int> values;
            
            // empty costructor
            TriedPattern() : hash_(1) { values_.reserve(8); }

            // equality operator
            bool operator==(const TriedPattern & other) const { 
                return (hash_ == other.hash_) &&   // cheap
                       (values_ == other.values_); // expensive last resort
            }

            /// push back value, and update the hash
            void push_back(short unsigned int i) { 
                boost::hash_combine(hash_,i);
                values_.push_back(i);
            }
            /// return the hash: equal objects MUST have the same hash, 
            ///  different ones should have different ones
            size_t hash() const { return hash_; }

            // some extra methods to look like a std::vector
            typedef values::const_iterator const_iterator;
            const_iterator begin() const { return values_.begin(); }
            const_iterator end() const { return values_.end(); }
            values::size_type size() const { return values_.size(); }
        private:
            values values_;
            size_t hash_;
    };
    typedef boost::unordered_set<TriedPattern> TriedPatterns;
 private:

    TriedPatterns theTriedPattern;
};

inline std::size_t hash_value(const DTCombinatorialPatternReco::TriedPattern &t) { return t.hash(); }
#endif // DTSegment_DTCombinatorialPatternReco_h
