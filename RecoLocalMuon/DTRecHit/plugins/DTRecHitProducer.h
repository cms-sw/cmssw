#ifndef RecoLocalMuon_DTRecHitProducer_h
#define RecoLocalMuon_DTRecHitProducer_h

/** \class DTRecHitProducer
 *  Module for 1D DTRecHitPairs production. The concrete reconstruction algorithm
 *  is specified with the parameter "recAlgo" and must be configured with the
 *  "recAlgoConfig" parameter set.
 *
 *  $Date: 2010/02/20 21:00:58 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DTRecHitBaseAlgo;

class DTRecHitProducer : public edm::EDProducer {
public:
  /// Constructor
  DTRecHitProducer(const edm::ParameterSet&);

  /// Destructor
  virtual ~DTRecHitProducer();

  /// The method which produces the rechits
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);

private:
  // Switch on verbosity
  static bool debug;
  // The label to be used to retrieve DT digis from the event
  edm::InputTag theDTDigiLabel;
  // The reconstruction algorithm
  DTRecHitBaseAlgo *theAlgo;
//   static string theAlgoName;

};
#endif

