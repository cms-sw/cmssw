#ifndef L3MuonIsolationProducer_L3MuonIsolationProducer_H
#define L3MuonIsolationProducer_L3MuonIsolationProducer_H

/**  \class L3MuonIsolationProducer
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include <string>
#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}
namespace reco { class MuIsoDeposit; }
namespace muonisolation { class Direction; }

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/MuonIsolation/interface/Cuts.h"



class L3MuonIsolationProducer : public edm::EDProducer {

public:

  /// constructor with config
  L3MuonIsolationProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3MuonIsolationProducer(); 
  
  /// Produce isolation maps
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  double theDiff_r, theDiff_z, theDR_Match, theDR_Veto, theDR_Max;
  std::string theMuonCollectionLabel;  // Muon track Collection Label
  std::string theTrackCollectionLabel; // Isolation track Collection Label
  std::string theDepositLabel;         // name for deposit

  muonisolation::Cuts theCuts;

};

#endif
