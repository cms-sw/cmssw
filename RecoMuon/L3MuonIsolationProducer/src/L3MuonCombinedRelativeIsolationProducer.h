#ifndef L3MuonCombinedRelativeIsolationProducer_L3MuonCombinedRelativeIsolationProducer_H
#define L3MuonCombinedRelativeIsolationProducer_L3MuonCombinedRelativeIsolationProducer_H

/**  \class L3MuonCombinedRelativeIsolationProducer
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/MuonIsolation/interface/Cuts.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class L3MuonCombinedRelativeIsolationProducer : public edm::EDProducer {

public:

  /// constructor with config
  L3MuonCombinedRelativeIsolationProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3MuonCombinedRelativeIsolationProducer(); 

  /// initialisation
  virtual void beginJob();
  
  /// Produce isolation maps
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  edm::ParameterSet theConfig;

  // Muon track Collection Label
  edm::InputTag theMuonCollectionLabel;

  // Isolation cuts
  muonisolation::Cuts theCuts;

  // Option to write MuIsoDeposits into the event
  bool optOutputIsoDeposits;

  // Option to use rho-corrected calo deposits (ONLY if already available)
  bool useRhoCorrectedCaloDeps;
  edm::InputTag theCaloDepsLabel;

  // MuIsoExtractor
  reco::isodeposit::IsoDepositExtractor * caloExtractor;
  reco::isodeposit::IsoDepositExtractor * trkExtractor;

  //! pt cut to consider track in sumPt after extracting iso deposit
  //! better split this off into a filter
  double theTrackPt_Min;

  //! max number of tracks to allow in the sum
  //! count <= maxN
  int theMaxNTracks;

  //! apply or not the maxN cut on top of the sumPt (or nominall eff) < cuts
  bool theApplyCutsORmaxNTracks;

  // Print out debug info

  bool printDebug;

};

#endif
