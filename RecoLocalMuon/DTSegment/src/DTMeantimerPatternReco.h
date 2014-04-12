#ifndef DTSegment_DTMeantimerPatternReco_h
#define DTSegment_DTMeantimerPatternReco_h

/** \class DTMeantimerPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a combinatorial approach with
 * a T0 estimation produced along the way
 *  
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 * \author Piotr Traczyk - SINS Warsaw <ptraczyk@fuw.edu.pl>
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
class DTLinearFit;

/* C++ Headers */
#include <vector>
#include <deque>
#include <utility>

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"

/* ====================================================================== */

/* Class DTMeantimerPatternReco Interface */


class DTMeantimerPatternReco : public DTRecSegment2DBaseAlgo {

 public:

  /// Constructor
  DTMeantimerPatternReco(const edm::ParameterSet& pset) ;

  /// Destructor
  virtual ~DTMeantimerPatternReco() ;

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
  DTLinearFit* theFitter; // the linear fitter

  friend class DTMeantimerPatternReco4D;

  // typedef std::pair<DTHitPairForFit*, DTEnums::DTCellSide> AssPoint;
    
  // create the DTHitPairForFit from the pairs for easy use
  std::vector<std::shared_ptr<DTHitPairForFit>> initHits(const DTSuperLayer* sl,
	    						 const std::vector<DTRecHit1DPair>& hits);

  // search for candidate, starting from pairs of hits in different layers
  std::vector<DTSegmentCand*> buildSegments(const DTSuperLayer* sl,
					    const std::vector<std::shared_ptr<DTHitPairForFit>>& hits);

  // try adding more hits to a candidate
  void addHits(const DTSuperLayer* sl, 
               std::vector<DTSegmentCand::AssPoint>& assHits, 
               const std::vector<std::shared_ptr<DTHitPairForFit>>& hits, 
               std::vector<DTSegmentCand*> &result);

  // fit a set of left/right hits, calculate t0 and chi^2
  std::unique_ptr<DTSegmentCand> fitWithT0(const DTSuperLayer* sl,
                                           const std::vector<DTSegmentCand::AssPoint> &assHits, 
                                           double &chi2, 
                                           double &t0_corr, 
                                           const bool fitdebug);

  // check if two hist can be considered in one segment (come from different layers, not too far away etc.)
  bool geometryFilter( const DTWireId first, const DTWireId second ) const;

  bool checkDoubleCandidates(std::vector<DTSegmentCand*>& segs,
			     DTSegmentCand* seg);

  void printPattern( std::vector<DTSegmentCand::AssPoint>& assHits, const DTHitPairForFit* hit);


 private:

  std::string theAlgoName;
  unsigned int theMaxAllowedHits;
  double theAlphaMaxTheta;
  double theAlphaMaxPhi;
  double theMaxChi2;
  bool debug;
  DTSegmentUpdator* theUpdator; // the updator and fitter
  DTSegmentCleaner* theCleaner; // the cleaner
  
  unsigned int maxfound;

  edm::ESHandle<DTGeometry> theDTGeometry; // the DT geometry
};
#endif // DTSegment_DTMeantimerPatternReco_h
