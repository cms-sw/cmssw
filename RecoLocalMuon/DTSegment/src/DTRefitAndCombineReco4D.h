#ifndef DTSegment_DTRefitAndCombineReco4D_h
#define DTSegment_DTRefitAndCombineReco4D_h

/** \class DTRefitAndCombineReco4D
 *
 * Algo for reconstructing 4d segment in DT refitting the 2D phi SL hits and combining
 * the results with the theta view.
 *  
 * $Date: 2006/05/04 09:17:36 $
 * $Revision: 1.5 $
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
  virtual ~DTRefitAndCombineReco4D(){};
    
  /// Operations  
  virtual edm::OwnVector<DTRecSegment4D>
    reconstruct();
    
  virtual std::string algoName() const { return theAlgoName; }
    
  virtual void setES(const edm::EventSetup& setup);

  virtual void setDTRecHit1DContainer(edm::Handle<DTRecHitCollection> all1DHits) {};
  virtual void setDTRecSegment2DContainer(edm::Handle<DTRecSegment2DCollection> all2DSegments);
  virtual void setChamber(const DTChamberId &chId);
  virtual bool wants2DSegments(){return true;}

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
