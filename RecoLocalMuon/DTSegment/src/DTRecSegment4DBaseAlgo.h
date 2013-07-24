#ifndef DTSegment_DTRecSegment4DBaseAlgo_h
#define DTSegment_DTRecSegment4DBaseAlgo_h

/** \class DTRecSegment4DBaseAlgo
 *
 * Abstract algo class to reconstruct 4D-segments in chamber given a set of 2D-segment
 *
 * $Date: 2007/03/10 16:14:43 $
 * $Revision: 1.5 $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

// Collaborating Class Declarations
namespace edm {
  class ParameterSet;
  class EventSetup;
}
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

#include "DataFormats/Common/interface/Handle.h"

class DTChamberId;

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
  virtual edm::OwnVector<DTRecSegment4D> reconstruct() = 0;
    
  virtual std::string algoName() const = 0;
  
  virtual void setES(const edm::EventSetup& setup) = 0;
  virtual void setDTRecHit1DContainer(edm::Handle<DTRecHitCollection> all1DHits) = 0;
  virtual void setDTRecSegment2DContainer(edm::Handle<DTRecSegment2DCollection> all2DSegments) = 0;
  virtual void setChamber(const DTChamberId &chId) = 0;
  virtual bool wants2DSegments() = 0;
   
 protected:

    
 private:

};
#endif 
