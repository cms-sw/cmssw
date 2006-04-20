#ifndef DTSegment_DTRefitAndCombineReco4D_h
#define DTSegment_DTRefitAndCombineReco4D_h

/** \class DTRefitAndCombineReco4D
 *
 * Algo for reconstructing 4d segment in DT refitting the 2D phi SL hits and combining
 * the results with the theta view.
 *  
 * $Date: 2006/04/19 15:00:33 $
 * $Revision: 1.1 $
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
    reconstruct(const DTChamber* chamber,
		const std::vector<DTRecSegment2D>& segments2DPhi1,
		const std::vector<DTRecSegment2D>& segments2DTheta,
		const std::vector<DTRecSegment2D>& segments2DPhi2);
    
  virtual std::string algoName() const { return theAlgoName; }
    
  virtual void setES(const edm::EventSetup& setup);
 protected:

 private:
  std::vector<DTRecSegment2DPhi> refitSuperSegments(const std::vector<DTRecSegment2D>& segments2DPhi1,
						    const std::vector<DTRecSegment2D>& segments2DPhi2);

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

};
#endif
