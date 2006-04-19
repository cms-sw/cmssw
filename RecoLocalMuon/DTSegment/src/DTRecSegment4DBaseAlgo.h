#ifndef DTSegment_DTRecSegment4DBaseAlgo_h
#define DTSegment_DTRecSegment4DBaseAlgo_h

/** \class DTRecSegment4DBaseAlgo
 *
 * Abstract algo class to reconstruct 4D-segments in chamber given a set of 2D-segment
 *
 * $Date:  $
 * $Revision: $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

// Collaborating Class Declarations
namespace edm {
  class ParameterSet;
  class EventSetup;
}
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"

class DTChamber;

// C++ Headers
#include <vector>
#include <string>

// ====================================================================== 

// Class DTRecSegment4DBaseAlgo Interface

class DTRecSegment4DBaseAlgo{

 public:

  /// Constructor
  DTRecSegment4DBaseAlgo(const edm::ParameterSet& ) {}
    
  /// Destructor
  virtual ~DTRecSegment4DBaseAlgo() {}
    
  // Operations
  virtual edm::OwnVector<DTRecSegment4D>
    reconstruct(const DTChamber* ch,
		const std::vector<DTRecSegment2D>& recSegments2DPhi1,
		const std::vector<DTRecSegment2D>& recSegments2DTheta,
		const std::vector<DTRecSegment2D>& recSegments2DPhi2) = 0;
    
  virtual std::string algoName() const = 0;
    
  virtual  void setES(const edm::EventSetup& setup) = 0;
    
 protected:
    
 private:

};
#endif 
