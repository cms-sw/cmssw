#ifndef MuonIsolation_JetExtractor_H
#define MuonIsolation_JetExtractor_H

/** \class JetExtractor
 *  Extracts deposits in each calorimeter section (ECAL, HCAL, HO)
 *  vetoes are set based on expected crossed DetIds (xtals, towers)
 *  these can later be subtracted from deposits in a cone.
 *  All work is done by TrackDetectorAssociator. Because of the heavy
 *  weight of the tool, all extractions can (should?) be placed in a single place.
 *  
 *  $Date: 2009/06/18 07:27:42 $
 *  $Revision: 1.5 $
 *  $Id: JetExtractor.h,v 1.5 2009/06/18 07:27:42 slava77 Exp $
 *  \author S. Krutelyov
 */

#include <string>

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class TrackAssociatorParameters;
class TrackDetectorAssociator;
class MuonServiceProxy;

namespace muonisolation {

class JetExtractor : public reco::isodeposit::IsoDepositExtractor {

public:

  JetExtractor(){};
  JetExtractor(const edm::ParameterSet& par);

  virtual ~JetExtractor();

  virtual void fillVetos (const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks);
  virtual reco::IsoDeposit 
    deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const;

private:
  edm::InputTag theJetCollectionLabel;

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
