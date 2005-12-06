#ifndef RecoLocalMuon_DTRecHitProducer_h
#define RecoLocalMuon_DTRecHitProducer_h

/** \class DTRecHitProducer
 *
 *
 *  $Date: $
 *  $Revision: $
 *  \author
 */

#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class DTRecHitProducer : public edm::EDProducer {
public:
  /// Constructor
  DTRecHitProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~DTRecHitProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

};
#endif

