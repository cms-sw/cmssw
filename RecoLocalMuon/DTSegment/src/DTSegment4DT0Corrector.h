#ifndef DTSegment_DTSegment4DT0Corrector_H
#define DTSegment_DTSegment4DT0Corrector_H

/** \class DTSegment4DT0Corrector
 *  Builds the segments in the DT chambers.
 *
 * \author Mario Pelliccioni - INFN Torino <pellicci@cern.ch>
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DTSegment4DT0Corrector: public edm::stream::EDProducer<> {

 public:
  /// Constructor
  DTSegment4DT0Corrector(const edm::ParameterSet&) ;

  /// Destructor
  ~DTSegment4DT0Corrector() override;

  // Operations

  /// The method which produces the 4D rec segments corrected for t0 offset
  void produce(edm::Event& event, const edm::EventSetup& setup) override;
  

 protected:

 private:

  // Switch on verbosity
  bool debug;

  edm::EDGetTokenT<DTRecSegment4DCollection> recHits4DToken_;

  // the updator
  DTSegmentUpdator *theUpdator;

};
#endif


