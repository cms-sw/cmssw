#ifndef DTRecHit_DTRecSegment2DProducer_h
#define DTRecHit_DTRecSegment2DProducer_h

/** \class DTRecSegment2DProducer
 *
 * Producer for DT segment in one projection.
 *  
 * $Date: 2006/03/30 16:53:18 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

/* Collaborating Class Declarations */
namespace edm {
  class ParameterSet;
}
class DTRecSegment2DBaseAlgo;

/* C++ Headers */

/* ====================================================================== */

/* Class DTRecSegment2DProducer Interface */

class DTRecSegment2DProducer : public edm::EDProducer {

  public:

/// Constructor
    DTRecSegment2DProducer(const edm::ParameterSet&) ;

/// Destructor
    virtual ~DTRecSegment2DProducer() ;

/* Operations */ 
    /// The method which produces the rec segments
    virtual void produce(edm::Event& event, const edm::EventSetup& setup);

  protected:

  private:
    // Switch on verbosity
    static bool debug;
    // The reconstruction algorithm
    DTRecSegment2DBaseAlgo* theAlgo;
    //static std::string theAlgoName;
    //
    std::string theRecHits1DLabel;

};
#endif // DTRecHit_DTRecSegment2DProducer_h

