#ifndef DTSegment_DTMeantimerPatternReco_h
#define DTSegment_DTMeantimerPatternReco_h

/** \class DTMeantimerPatternReco
 *
 * Algo for reconstructing 2d segment in DT using a combinatorial approach with
 * a T0 estimation produced along the way
 *  
 * $Date: 2008/03/10 11:18:20 $
 * $Revision: 1.2 $
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

/* C++ Headers */
#include <vector>
#include <deque>
#include <utility>

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

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
  friend class DTMeantimerPatternReco4D;

  typedef std::pair<DTHitPairForFit*, DTEnums::DTCellSide> AssPoint;
    
  // create the DTHitPairForFit from the pairs for easy use
  std::vector<DTHitPairForFit*> initHits(const DTSuperLayer* sl,
					 const std::vector<DTRecHit1DPair>& hits);

  // search for candidate, starting from pairs of hits in different layers
  std::vector<DTSegmentCand*> buildSegments(const DTSuperLayer* sl,
					    const std::vector<DTHitPairForFit*>& hits);

  // try adding more hits to a candidate
  void addHits(const DTSuperLayer* sl, 
               std::vector<AssPoint>& assHits, 
               const std::vector<DTHitPairForFit*>& hits, 
               std::vector<DTSegmentCand*> &result,
               std::vector<AssPoint>& usedHits);

  // fit a set of left/right hits, calculate t0 and chi^2
  bool fitWithT0(const std::vector<AssPoint> &assHits, double &chi2, double &t0_corr, const bool fitdebug);

  // check if two hist can be considered in one segment (come from different layers, not too far away etc.)
  bool geometryFilter( const DTWireId first, const DTWireId second ) const;

  // a generic least-square fit to a set of points
  void rawFit(double &a, double &b, const std::vector< std::pair<double,double> > &hits);

  bool checkDoubleCandidates(std::vector<DTSegmentCand*>& segs,
			     DTSegmentCand* seg);

 private:

  std::string theAlgoName;
  unsigned int theMaxAllowedHits;
  double theAlphaMaxTheta;
  double theAlphaMaxPhi;
  double theMaxChi2;
  double theMaxT0;
  double theMinT0;
  bool debug;
  DTSegmentUpdator* theUpdator; // the updator and fitter
  DTSegmentCleaner* theCleaner; // the cleaner
  
  unsigned int maxfound;

  edm::ESHandle<DTGeometry> theDTGeometry; // the DT geometry
};
#endif // DTSegment_DTMeantimerPatternReco_h
