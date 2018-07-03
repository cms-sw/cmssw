#ifndef DTSegment_DTRefitAndCombineReco4D_h
#define DTSegment_DTRefitAndCombineReco4D_h

/** \class DTRefitAndCombineReco4D
 *
 * Algo for reconstructing 4d segment in DT refitting the 2D phi SL hits and combining
 * the results with the theta view.
 *  
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

// Base Class Headers
#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DBaseAlgo.h"

//class DTRecSegment2DBaseAlgo;

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
//#include "DataFormats/DTRecHit/interface/DTRecSegment2DPhi.h"

// Class DTRefitAndCombineReco4D Interface 

class DTRefitAndCombineReco4D : public DTRecSegment4DBaseAlgo {

 public:

  /// Constructor
  DTRefitAndCombineReco4D(const edm::ParameterSet& pset) ;
  
  /// Destructor
  ~DTRefitAndCombineReco4D() override{};
    
  /// Operations  
  edm::OwnVector<DTRecSegment4D>
    reconstruct() override;
    
  std::string algoName() const override { return theAlgoName; }
    
  void setES(const edm::EventSetup& setup) override;

  void setDTRecHit1DContainer(edm::Handle<DTRecHitCollection> all1DHits) override {};
  void setDTRecSegment2DContainer(edm::Handle<DTRecSegment2DCollection> all2DSegments) override;
  void setChamber(const DTChamberId &chId) override;
  bool wants2DSegments() override{return true;}

 protected:

 private:
  std::vector<DTChamberRecSegment2D> refitSuperSegments();

  std::string theAlgoName;

  double theMaxChi2forPhi;

  bool debug;
  // DTSegmentUpdator* theUpdator; // the updator and fitter
  // DTSegmentCleaner* theCleaner; // the cleaner
    
  edm::ESHandle<DTGeometry> theDTGeometry; // the DT geometry

  //   // The reconstruction 2D algorithm 
  // DTRecSegment2DBaseAlgo* the2DAlgo; 

  // the updator
  DTSegmentUpdator *theUpdator;
  
  const DTChamber *theChamber;
  std::vector<DTSLRecSegment2D> theSegments2DPhi1;
  std::vector<DTSLRecSegment2D> theSegments2DTheta; 
  std::vector<DTSLRecSegment2D> theSegments2DPhi2;
  
};
#endif
