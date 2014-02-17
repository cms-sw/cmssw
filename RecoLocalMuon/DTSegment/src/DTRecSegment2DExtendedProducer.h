#ifndef DTSegment_DTRecSegment2DExtendedProducer_h
#define DTSegment_DTRecSegment2DExtendedProducer_h

/** \class DTRecSegment2DExtendedProducer
 *
 * Producer for DT segment in one projection.
 *  
 * $Date: 2010/02/16 17:08:17 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DTCombinatorialExtendedPatternReco;

/* C++ Headers */

/* ====================================================================== */

/* Class DTRecSegment2DExtendedProducer Interface */

class DTRecSegment2DExtendedProducer : public edm::EDProducer {

 public:

  /// Constructor
  DTRecSegment2DExtendedProducer(const edm::ParameterSet&) ;

  /// Destructor
  virtual ~DTRecSegment2DExtendedProducer() ;
    
  // Operations

  /// The method which produces the 2D-segments
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);

 protected:

 private:
  // Switch on verbosity
  bool debug;

  // The 2D-segments reconstruction algorithm
  DTCombinatorialExtendedPatternReco* theAlgo;

  //static std::string theAlgoName;
  edm::InputTag theRecHits1DLabel;
  edm::InputTag theRecClusLabel;
};
#endif // DTRecHit_DTRecSegment2DExtendedProducer_h

