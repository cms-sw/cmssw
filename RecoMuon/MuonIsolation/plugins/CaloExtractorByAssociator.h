#ifndef MuonIsolation_CaloExtractorByAssociator_H
#define MuonIsolation_CaloExtractorByAssociator_H

/** \class CaloExtractorByAssociator
 *  Extracts deposits in each calorimeter section (ECAL, HCAL, HO)
 *  vetoes are set based on expected crossed DetIds (xtals, towers)
 *  these can later be subtracted from deposits in a cone.
 *  All work is done by TrackDetectorAssociator. Because of the heavy
 *  weight of the tool, all extractions can (should?) be placed in a single place.
 *  
 *  $Date: 2009/06/18 07:27:42 $
 *  $Revision: 1.6 $
 *  $Id: CaloExtractorByAssociator.h,v 1.6 2009/06/18 07:27:42 slava77 Exp $
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
  
  class CaloExtractorByAssociator : public reco::isodeposit::IsoDepositExtractor {

  public:

    //! constructors
    CaloExtractorByAssociator(){};
    CaloExtractorByAssociator(const edm::ParameterSet& par);

    //! destructor
    virtual ~CaloExtractorByAssociator();

    //! allows to set extra vetoes (in addition to the muon) -- no-op at this point
    virtual void fillVetos (const edm::Event & ev, const edm::EventSetup & evSetup, const reco::TrackCollection & tracks);
    //! no-op: by design of this extractor the deposits are pulled out all at a time
    virtual reco::IsoDeposit 
      deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const;
    //! return deposits for 3 calorimeter subdetectors (ecal, hcal, ho) -- in this order
    virtual std::vector<reco::IsoDeposit> 
      deposits(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Track & track) const;

  private:

    //! use towers or rec hits
    bool theUseRecHitsFlag;
  
    //! Label of deposit -- suggest to set to "" (all info is in collection name anyways)
    std::string theDepositLabel;

    //! multiple deposits: labels -- expect 3 labels beginning with "e", "h", "ho"
    std::vector<std::string> theDepositInstanceLabels;

    //! propagator name to feed into the track associator
    std::string thePropagatorName;

    //! Cone cuts and thresholds
    //! min values of Et to be included in deposits
    double theThreshold_E;
    double theThreshold_H;
    double theThreshold_HO;

    //! cone sizes inside which the Et (towers) are not counted
    double theDR_Veto_E;
    double theDR_Veto_H;
    double theDR_Veto_HO;
    //! centers the cone on the veto direction -- makes more sense for
    //! very displaced tracks like in cosmics
    bool theCenterConeOnCalIntersection;
    //! max cone size in which towers are considered
    double theDR_Max;

    //! the noise "sigmas" for a hit or tower to be considered
    //! consider if Energy > 3.*sigma
    double theNoise_EB;
    double theNoise_EE;
    double theNoise_HB;
    double theNoise_HE;
    double theNoise_HO;
    double theNoiseTow_EB;
    double theNoiseTow_EE;

    //! Vector of calo Ids to veto -- not used
    std::vector<DetId> theVetoCollection;

    //! the event setup proxy, it takes care the services update
    MuonServiceProxy* theService;


    //! associator, its' parameters and the propagator
    TrackAssociatorParameters* theAssociatorParameters;
    TrackDetectorAssociator* theAssociator;  

    //! flag to turn on/off printing of a time report
    bool thePrintTimeReport;

    //! Determine noise for HCAL and ECAL (take some defaults for the time being)
    double noiseEcal(const CaloTower& tower) const;
    double noiseHcal(const CaloTower& tower) const;
    double noiseHOcal(const CaloTower& tower) const;
    double noiseRecHit(const DetId& detId) const;

  };

}

#endif
