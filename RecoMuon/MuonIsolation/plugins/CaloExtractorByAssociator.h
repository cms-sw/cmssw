#ifndef MuonIsolation_CaloExtractorByAssociator_H
#define MuonIsolation_CaloExtractorByAssociator_H

/** \class CaloExtractorByAssociator
 *  Extracts deposits in each calorimeter section (ECAL, HCAL, HO)
 *  vetoes are set based on expected crossed DetIds (xtals, towers)
 *  these can later be subtracted from deposits in a cone.
 *  All work is done by TrackDetectorAssociator. Because of the heavy
 *  weight of the tool, all extractions can (should?) be placed in a single place.
 *  
 *  $Date: 2007/03/31 20:32:23 $
 *  $Revision: 1.2 $
 *  $Id: CaloExtractorByAssociator.h,v 1.2 2007/03/31 20:32:23 slava77 Exp $
 *  \author S. Krutelyov
 */

#include <string>

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class TrackAssociatorParameters;
class TrackDetectorAssociator;
class Propagator;

namespace muonisolation {

class CaloExtractorByAssociator : public MuIsoExtractor {

public:

  CaloExtractorByAssociator(){};
  CaloExtractorByAssociator(const edm::ParameterSet& par);

  virtual ~CaloExtractorByAssociator();

  virtual void fillVetos (const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks);
  virtual reco::MuIsoDeposit 
    deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const;
  virtual std::vector<reco::MuIsoDeposit> 
    deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const;

private:

  //use towers or rec hits
  bool theUseRecHitsFlag;
  
  // Label of deposit
  std::string theDepositLabel;

  //multiple deposits: labels
  std::vector<std::string> theDepositInstanceLabels;

  std::string thePropagatorName;

  // Cone cuts and thresholds
  double theThreshold_E;
  double theThreshold_H;
  double theThreshold_HO;
  double theDR_Veto_E;
  double theDR_Veto_H;
  double theDR_Veto_HO;
  double theDR_Max;

  double theNoise_EB;
  double theNoise_EE;
  double theNoise_HB;
  double theNoise_HE;
  double theNoise_HO;
  double theNoiseTow_EB;
  double theNoiseTow_EE;

  // Vector of calo Ids to veto
  std::vector<DetId> theVetoCollection;

  TrackAssociatorParameters* theAssociatorParameters;
  TrackDetectorAssociator* theAssociator;  
  mutable Propagator* thePropagator; 

  bool thePrintTimeReport;

  // Determine noise for HCAL and ECAL (take some defaults for the time being)
  double noiseEcal(const CaloTower& tower) const;
  double noiseHcal(const CaloTower& tower) const;
  double noiseHOcal(const CaloTower& tower) const;
  double noiseRecHit(const DetId& detId) const;

  // Function to ensure that phi and theta are in range
  static double PhiInRange(const double& phi);

  // DeltaR function
  template <class T, class U> static double deltaR(const T& t, const U& u);
};

}

#endif
