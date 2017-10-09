#ifndef DTSegment_DTRecSegment2DExtendedProducer_h
#define DTSegment_DTRecSegment2DExtendedProducer_h

/** \class DTRecSegment2DExtendedProducer
 *
 * Producer for DT segment in one projection.
 *  
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecClusterCollection.h"

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
  ~DTRecSegment2DExtendedProducer() override ;
    
  // Operations

  /// The method which produces the 2D-segments
  void produce(edm::Event& event, const edm::EventSetup& setup) override;

 protected:

 private:
  // Switch on verbosity
  bool debug;

  // The 2D-segments reconstruction algorithm
  DTCombinatorialExtendedPatternReco* theAlgo;

  //static std::string theAlgoName;
  edm::EDGetTokenT<DTRecHitCollection> recHits1DToken_;
  edm::EDGetTokenT<DTRecClusterCollection> recClusToken_;
};
#endif // DTRecHit_DTRecSegment2DExtendedProducer_h

