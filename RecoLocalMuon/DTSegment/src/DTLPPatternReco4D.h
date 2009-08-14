#ifndef DTSegment_DTLPPatternReco4D_h
#define DTSegment_DTLPPatternReco4D_h

/** \class DTLPPatternReco4D
 *
 * Algo for reconstructing 4d segment in DT using a linear programming approach
 *  
 * $Date: 2009/08/13 18:43:47 $
 * $Revision: 0.1 $
 * \author Enzo Busseti - SNS Pisa <enzo.busseti@sns.it>
 *
 */

// Base Class Headers
#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DBaseAlgo.h"

class DTRecSegment2DBaseAlgo;

// Collaborating Class Declarations
namespace edm {
  class ParameterSet;
  class EventSetup;
}
class DTSegmentUpdator;
//class DTSegmentCleaner;

// C++ Headers
#include <vector>
//#include <utility>

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

// ====================================================================== 
class DTSegmentCand;
class DTLPPatternReco;
class DTHitPairForFit;

// Class DTLPPatternReco4D Interface 

class DTLPPatternReco4D : public DTRecSegment4DBaseAlgo {

 public:

  /// Constructor
  DTLPPatternReco4D(const edm::ParameterSet& pset) ;
 
  /// Destructor
  virtual ~DTLPPatternReco4D();
    
  /// Operations  
  virtual edm::OwnVector<DTRecSegment4D> reconstruct();
    
  virtual std::string algoName() const { return "DTLPPatternReco4D"; }

  virtual void setES(const edm::EventSetup& setup);
  virtual void setDTRecHit1DContainer(edm::Handle<DTRecHitCollection> all1DHits);
  virtual void setDTRecSegment2DContainer(edm::Handle<DTRecSegment2DCollection> all2DSegments);
  virtual void setChamber(const DTChamberId &chId);
  virtual bool wants2DSegments(){return !allDTRecHits;}

 protected:

 private:
  std::vector<DTSegmentCand*> buildPhiSuperSegmentsCandidates(std::vector<DTHitPairForFit*> &pairPhiOwned);
  //DTRecSegment4D* segmentSpecialZed(const DTRecSegment4D* seg);


  bool debug;
  // DTSegmentUpdator* theUpdator; // the updator and fitter
  // DTSegmentCleaner* theCleaner; // the cleaner
    
  edm::ESHandle<DTGeometry> theDTGeometry; // the DT geometry

  // The reconstruction 2D algorithm
  // For the 2D reco I use thei reconstructor!
  DTLPPatternReco* the2DAlgo;
  
  // the updator
  DTSegmentUpdator *theUpdator;

  const DTChamber *theChamber;

  //the input type
  bool allDTRecHits;
  bool applyT0corr;
  bool computeT0corr;

  //  std::vector<DTRecHit1D> the1DPhiHits;
  std::vector<DTSLRecSegment2D> theSegments2DTheta; 
  std::vector<DTRecHit1DPair> theHitsFromPhi1;
  std::vector<DTRecHit1DPair> theHitsFromTheta;
  std::vector<DTRecHit1DPair> theHitsFromPhi2;
};
#endif
