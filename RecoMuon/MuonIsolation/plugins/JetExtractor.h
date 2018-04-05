#ifndef MuonIsolation_JetExtractor_H
#define MuonIsolation_JetExtractor_H

/** \class JetExtractor
 *  Extracts deposits in each calorimeter section (ECAL, HCAL, HO)
 *  vetoes are set based on expected crossed DetIds (xtals, towers)
 *  these can later be subtracted from deposits in a cone.
 *  All work is done by TrackDetectorAssociator. Because of the heavy
 *  weight of the tool, all extractions can (should?) be placed in a single place.
 *
 *  \author S. Krutelyov
 */

#include <string>

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class TrackAssociatorParameters;
class TrackDetectorAssociator;
class MuonServiceProxy;

namespace muonisolation {

class JetExtractor : public reco::isodeposit::IsoDepositExtractor {

public:

  JetExtractor(){};
  JetExtractor(const edm::ParameterSet& par, edm::ConsumesCollector && iC);

  ~JetExtractor() override;

  void fillVetos (const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks) override;
  reco::IsoDeposit
    deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const override;

private:
  edm::EDGetTokenT<reco::CaloJetCollection> theJetCollectionToken;

  std::string thePropagatorName;

  // Cone cuts and thresholds
  double theThreshold;
  double theDR_Veto;
  double theDR_Max;

  //excludes sumEt of towers that are inside muon veto cone
  bool theExcludeMuonVeto;

  //! the event setup proxy, it takes care the services update
  MuonServiceProxy* theService;

  TrackAssociatorParameters* theAssociatorParameters;
  TrackDetectorAssociator* theAssociator;

  bool thePrintTimeReport;

};

}

#endif
