#ifndef L3MuonIsolationProducer_L3MuonIsolationProducer_H
#define L3MuonIsolationProducer_L3MuonIsolationProducer_H

/**  \class L3MuonIsolationProducer
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "RecoMuon/MuonIsolation/interface/Cuts.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class L3MuonIsolationProducer : public edm::EDProducer {

public:

  /// constructor with config
  L3MuonIsolationProducer(const edm::ParameterSet&);

  /// destructor
  virtual ~L3MuonIsolationProducer();

  /// Produce isolation maps
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  edm::ParameterSet theConfig;

  // Muon track Collection Label
  edm::InputTag theMuonCollectionLabel;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> theMuonCollectionToken;

  // Isolation cuts
  muonisolation::Cuts theCuts;

  // Option to write MuIsoDeposits into the event
  double optOutputIsoDeposits;

  // MuIsoExtractor
  reco::isodeposit::IsoDepositExtractor * theExtractor;

  //! pt cut to consider track in sumPt after extracting iso deposit
  //! better split this off into a filter
  double theTrackPt_Min;

  //! max number of tracks to allow in the sum
  //! count <= maxN
  int theMaxNTracks;

  //! apply or not the maxN cut on top of the sumPt (or nominall eff) < cuts
  bool theApplyCutsORmaxNTracks;

};

#endif
