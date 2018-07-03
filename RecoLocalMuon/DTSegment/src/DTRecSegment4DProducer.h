#ifndef DTSegment_DTRecSegment4DProducer_H
#define DTSegment_DTRecSegment4DProducer_H

/** \class DTRecSegment4DProducer
 *  Builds the segments in the DT chambers.
 *
 * \author Riccardo Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}
class DTRecSegment4DBaseAlgo;

class DTRecSegment4DProducer: public edm::stream::EDProducer<> {
public:
  /// Constructor
  DTRecSegment4DProducer(const edm::ParameterSet&) ;

  /// Destructor
  ~DTRecSegment4DProducer() override;

  // Operations

  /// The method which produces the 4D rec segments
  void produce(edm::Event& event, const edm::EventSetup& setup) override;
  

protected:

private:

  // Switch on verbosity
  bool debug;

  edm::EDGetTokenT<DTRecHitCollection> recHits1DToken_;
  //static std::string theAlgoName;
  edm::EDGetTokenT<DTRecSegment2DCollection> recHits2DToken_;
  // The 4D-segments reconstruction algorithm
  DTRecSegment4DBaseAlgo* the4DAlgo;
};
#endif


