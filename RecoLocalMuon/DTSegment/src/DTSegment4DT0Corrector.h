#ifndef DTSegment_DTSegment4DT0Corrector_H
#define DTSegment_DTSegment4DT0Corrector_H

/** \class DTSegment4DT0Corrector
 *  Builds the segments in the DT chambers.
 *
 *  $Date: 2010/02/16 17:08:20 $
 *  $Revision: 1.2 $
 * \author Mario Pelliccioni - INFN Torino <pellicci@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DTSegment4DT0Corrector: public edm::EDProducer {

 public:
  /// Constructor
  DTSegment4DT0Corrector(const edm::ParameterSet&) ;

  /// Destructor
  virtual ~DTSegment4DT0Corrector();

  // Operations

  /// The method which produces the 4D rec segments corrected for t0 offset
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);
  

 protected:

 private:

  // Switch on verbosity
  bool debug;

  edm::InputTag theRecHits4DLabel;

  // the updator
  DTSegmentUpdator *theUpdator;

};
#endif


