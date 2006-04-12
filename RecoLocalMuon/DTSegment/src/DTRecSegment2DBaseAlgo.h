#ifndef DTSegment_DTRecSegment2DBaseAlgo_h
#define DTSegment_DTRecSegment2DBaseAlgo_h

/** \class DTRecSegment2DBaseAlgo
 *
 * Abstract aglo class to reconstruct segments in SL given a set of hits
 *
 * $Date: 2006/03/30 16:53:18 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
namespace edm {
  class ParameterSet;
  class EventSetup;
}
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
class DTSuperLayer;

/* C++ Headers */
#include <vector>
#include <string>

/* ====================================================================== */

/* Class DTRecSegment2DBaseAlgo Interface */

class DTRecSegment2DBaseAlgo{

  public:

/// Constructor
    DTRecSegment2DBaseAlgo(const edm::ParameterSet& ) {}

/// Destructor
    virtual ~DTRecSegment2DBaseAlgo() {}

/* Operations */ 
    virtual edm::OwnVector<DTRecSegment2D>
      reconstruct(const DTSuperLayer* sl,
                  const std::vector<DTRecHit1DPair>& hits,
                  const edm::EventSetup& setup) = 0;

    virtual std::string algoName() const = 0;
    
    //    virtual  void setES(const edm::EventSetup& setup) = 0;
    
 protected:
    
  private:

};
#endif // DTSegment_DTRecSegment2DBaseAlgo_h
