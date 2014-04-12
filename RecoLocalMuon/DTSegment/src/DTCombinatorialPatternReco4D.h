#ifndef DTSegment_DTCombinatorialPatternReco4D_h
#define DTSegment_DTCombinatorialPatternReco4D_h

/** \class DTCombinatorialPatternReco4D
 *
 * Algo for reconstructing 4d segment in DT using a combinatorial approach
 *  
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
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
class DTCombinatorialPatternReco;
class DTHitPairForFit;

// Class DTCombinatorialPatternReco4D Interface 

class DTCombinatorialPatternReco4D : public DTRecSegment4DBaseAlgo {

 public:

  /// Constructor
  DTCombinatorialPatternReco4D(const edm::ParameterSet& pset) ;
  
  /// Destructor
  virtual ~DTCombinatorialPatternReco4D();
    
  /// Operations  
  virtual edm::OwnVector<DTRecSegment4D> reconstruct();
    
  virtual std::string algoName() const { return theAlgoName; }

  virtual void setES(const edm::EventSetup& setup);
  virtual void setDTRecHit1DContainer(edm::Handle<DTRecHitCollection> all1DHits);
  virtual void setDTRecSegment2DContainer(edm::Handle<DTRecSegment2DCollection> all2DSegments);
  virtual void setChamber(const DTChamberId &chId);
  virtual bool wants2DSegments(){return !allDTRecHits;}

 protected:

 private:
  std::vector<DTSegmentCand*> buildPhiSuperSegmentsCandidates(std::vector<std::shared_ptr<DTHitPairForFit>> &pairPhiOwned);
  DTRecSegment4D* segmentSpecialZed(const DTRecSegment4D* seg);

  std::string theAlgoName;

  bool debug;
  // DTSegmentUpdator* theUpdator; // the updator and fitter
  // DTSegmentCleaner* theCleaner; // the cleaner
    
  edm::ESHandle<DTGeometry> theDTGeometry; // the DT geometry

  // The reconstruction 2D algorithm
  // For the 2D reco I use thei reconstructor!
  DTCombinatorialPatternReco* the2DAlgo;
  
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
