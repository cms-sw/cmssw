#ifndef L3MuonIsolationProducer_L3MuonIsolationProducer_H
#define L3MuonIsolationProducer_L3MuonIsolationProducer_H

/**  \class L3MuonIsolationProducer
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoMuon/MuonIsolation/interface/Cuts.h"

#include "RecoMuon/MuonIsolation/src/TrackExtractor.h"

#include <string>

class L3MuonIsolationProducer : public edm::EDProducer {

public:

  /// constructor with config
  L3MuonIsolationProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3MuonIsolationProducer(); 
  
  /// Produce isolation maps
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  // Muon track Collection Label
  std::string theMuonCollectionLabel;

  // Isolation cuts
  muonisolation::Cuts theCuts;

  // Option to write MuIsoDeposits into the event
  double optOutputIsoDeposits;

  // MuIsoExtractor
  muonisolation::TrackExtractor theTrackExtractor;

};

#endif
