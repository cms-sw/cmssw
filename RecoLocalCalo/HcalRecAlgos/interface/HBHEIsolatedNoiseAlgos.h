#ifndef __HBHE_ISOLATED_NOISE_ALGOS_H__
#define __HBHE_ISOLATED_NOISE_ALGOS_H__

/*
Description: Isolation algorithms used to identify anomalous noise in the HB/HE.
             These algorithms will be used to reflag HB/HE rechits as noise.

             There are 6 objects defined here:
             1) ObjectValidatorAbs
             2) ObjectValidator
	     3) PhysicsTower
	     4) PhysicsTowerOrganizer
	     5) HBHEHitMap
	     6) HBHEHitMapOrganizer
	     See comments below for details.

Original Author: John Paul Chou (Brown University)
                 Thursday, September 2, 2010
*/

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalFrontEndMap.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/JetReco/interface/TrackExtrapolation.h"

#include <vector>
#include <set>
#include <map>

// forward declarations
class HBHERecHit;
class EcalRecHit;
class HcalChannelQuality;
class HcalSeverityLevelComputer;
class EcalSeverityLevelAlgo;
class CaloGeometry;

//////////////////////////////////////////////////////////////////////////////
//
// ObjectValidator
//
// determines if an ECAL hit, HCAL hit, or track is valid or not.
// Note that various objects need to be passed to the validator before use.
// ObjectValidatorAbs provides a base class in case you want to change the
// hit/track validation algorithms.
//////////////////////////////////////////////////////////////////////////////

class ObjectValidatorAbs {
public:
  ObjectValidatorAbs() {}
  virtual ~ObjectValidatorAbs() {}

  virtual bool validHit(const HBHERecHit&) const = 0;
  virtual bool validHit(const EcalRecHit&) const = 0;
  virtual bool validTrack(const reco::Track&) const = 0;
};

class ObjectValidator : public ObjectValidatorAbs {
public:
  explicit ObjectValidator(const edm::ParameterSet&);
  ObjectValidator(double HBThreshold,
                  double HESThreshold,
                  double HEDThreshold,
                  double EBThreshold,
                  double EEThreshold,
                  uint32_t HcalAcceptSeverityLevel,
                  uint32_t EcalAcceptSeverityLevel,
                  bool UseHcalRecoveredHits,
                  bool UseEcalRecoveredHits,
                  double MinValidTrackPt,
                  double MinValidTrackPtBarrel,
                  int MinValidTrackNHits)
      : HBThreshold_(HBThreshold),
        HESThreshold_(HESThreshold),
        HEDThreshold_(HEDThreshold),
        EBThreshold_(EBThreshold),
        EEThreshold_(EEThreshold),
        HcalAcceptSeverityLevel_(HcalAcceptSeverityLevel),
        EcalAcceptSeverityLevel_(EcalAcceptSeverityLevel),
        UseHcalRecoveredHits_(UseHcalRecoveredHits),
        UseEcalRecoveredHits_(UseEcalRecoveredHits),
        MinValidTrackPt_(MinValidTrackPt),
        MinValidTrackPtBarrel_(MinValidTrackPtBarrel),
        MinValidTrackNHits_(MinValidTrackNHits),
        theHcalChStatus_(nullptr),
        theEcalChStatus_(nullptr),
        theHcalSevLvlComputer_(nullptr),
        theEcalSevLvlAlgo_(nullptr),
        theEBRecHitCollection_(nullptr),
        theEERecHitCollection_(nullptr) {}
  ~ObjectValidator() override;

  inline void setHcalChannelQuality(const HcalChannelQuality* q) { theHcalChStatus_ = q; }
  inline void setEcalChannelStatus(const EcalChannelStatus* q) { theEcalChStatus_ = q; }
  inline void setHcalSeverityLevelComputer(const HcalSeverityLevelComputer* q) { theHcalSevLvlComputer_ = q; }
  inline void setEcalSeverityLevelAlgo(const EcalSeverityLevelAlgo* q) { theEcalSevLvlAlgo_ = q; }
  inline void setEBRecHitCollection(const EcalRecHitCollection* q) { theEBRecHitCollection_ = q; }
  inline void setEERecHitCollection(const EcalRecHitCollection* q) { theEERecHitCollection_ = q; }

  bool validHit(const HBHERecHit&) const override;
  bool validHit(const EcalRecHit&) const override;
  bool validTrack(const reco::Track&) const override;

private:
  double HBThreshold_;   // energy threshold for HB hits
  double HESThreshold_;  // energy threshold for 5-degree (phi) HE hits
  double HEDThreshold_;  // energy threshold for 10-degree (phi) HE hits
  double EBThreshold_;   // energy threshold for EB hits
  double EEThreshold_;   // energy threshold for EE hits

  uint32_t HcalAcceptSeverityLevel_;  // severity level to accept HCAL hits
  uint32_t EcalAcceptSeverityLevel_;  // severity level to accept ECAL hits
  bool UseHcalRecoveredHits_;         // whether or not to use recovered HCAL hits
  bool UseEcalRecoveredHits_;         // whether or not to use recovered HCAL hits
  bool UseAllCombinedRechits_;        // whether to use all "Plan 1" combined rechits

  double MinValidTrackPt_;        // minimum valid track pT
  double MinValidTrackPtBarrel_;  // minimum valid track pT in the Barrel
  int MinValidTrackNHits_;        // minimum number of hits needed for a valid track

  // for checking the status of ECAL and HCAL channels stored in the DB
  const HcalChannelQuality* theHcalChStatus_;
  const EcalChannelStatus* theEcalChStatus_;

  // calculator of severety level for ECAL and HCAL
  const HcalSeverityLevelComputer* theHcalSevLvlComputer_;
  const EcalSeverityLevelAlgo* theEcalSevLvlAlgo_;

  // needed to determine an ECAL channel's validity
  const EcalRecHitCollection* theEBRecHitCollection_;
  const EcalRecHitCollection* theEERecHitCollection_;
};

//////////////////////////////////////////////////////////////////////////////
//
// PhysicsTower
//
// basic structure to hold information relevant at the tower level
// consists of hcal hits, ecal hits, and tracks
//////////////////////////////////////////////////////////////////////////////

struct PhysicsTower {
  CaloTowerDetId id;
  std::set<const HBHERecHit*> hcalhits;
  std::set<const EcalRecHit*> ecalhits;
  std::set<const reco::Track*> tracks;
};

//////////////////////////////////////////////////////////////////////////////
//
// PhysicsTowerOrganizer
//
// Organizers the PhysicsTowers into CaloTower (ieta,iphi) space.
// Also provides methods to find a towers nearest neighbors.
// For simplicity, merges the ieta=28 and ieta=29 towers together (calls them
// tower "28", collectively).
//////////////////////////////////////////////////////////////////////////////

class PhysicsTowerOrganizer {
public:
  struct towercmp {
    bool operator()(const PhysicsTower& lhs, const PhysicsTower& rhs) const { return (lhs.id < rhs.id); }
  };

  PhysicsTowerOrganizer(const edm::Handle<HBHERecHitCollection>& hbhehitcoll_h,
                        const edm::Handle<EcalRecHitCollection>& ebhitcoll_h,
                        const edm::Handle<EcalRecHitCollection>& eehitcoll_h,
                        const edm::Handle<std::vector<reco::TrackExtrapolation> >& trackextrapcoll_h,
                        const ObjectValidatorAbs& objectvalidator,
                        const CaloTowerConstituentsMap& ctcm,
                        const CaloGeometry& geo);

  virtual ~PhysicsTowerOrganizer() {}

  // find a PhysicsTower by some coordinate
  inline const PhysicsTower* findTower(const CaloTowerDetId& id) const;
  inline const PhysicsTower* findTower(int ieta, int iphi) const;

  // get the neighbors +/- 1 in eta-space or +/- 1 in phi-space
  // (accounts for change in phi-segmentation starting with eta=21)
  void findNeighbors(const CaloTowerDetId& id, std::set<const PhysicsTower*>& neighbors) const;
  void findNeighbors(const PhysicsTower* twr, std::set<const PhysicsTower*>& neighbors) const;
  void findNeighbors(int ieta, int iphi, std::set<const PhysicsTower*>& neighbors) const;

private:
  // the non-const, private version of findTower()
  PhysicsTower* findTower(const CaloTowerDetId& id);
  PhysicsTower* findTower(int ieta, int iphi);

  void insert_(CaloTowerDetId& id, const HBHERecHit* hit);
  void insert_(CaloTowerDetId& id, const EcalRecHit* hit);
  void insert_(CaloTowerDetId& id, const reco::Track* hit);

  std::set<PhysicsTower, towercmp> towers_;
};

//////////////////////////////////////////////////////////////////////////////
//
// HBHEHitMap
//
// Collection of HBHERecHits and their associated PhysicsTowers.
// Typically, these maps are organized into RBXs, HPDs, dihits, or monohits.
// From this one may calculate the hcal, ecal, and track energy associated
// with these hits.
//
// N.B. Many, of the operations can be computationally expensive and should be
// used with care.
//////////////////////////////////////////////////////////////////////////////

class HBHEHitMap {
public:
  typedef std::map<const HBHERecHit*, const PhysicsTower*>::const_iterator hitmap_const_iterator;
  typedef std::set<const PhysicsTower*>::const_iterator neighbor_const_iterator;

  struct twrinfo {
    double hcalInMap;
    double hcalOutOfMap;
    double ecal;
    double track;
  };

  HBHEHitMap();
  virtual ~HBHEHitMap() {}

  // energy of the hits in this collection
  double hitEnergy(void) const;

  // energy of the hits in a region fiducial to tracks
  double hitEnergyTrackFiducial(void) const;

  // number of hits in this collection
  int nHits(void) const;

  // same as above, except for the HCAL hits, ECAL hits, and tracks in the same towers as the hits
  // note that HCAL hits may be present in the same tower, but not be a part of the collection
  double hcalEnergySameTowers(void) const;
  double ecalEnergySameTowers(void) const;
  double trackEnergySameTowers(void) const;
  int nHcalHitsSameTowers(void) const;
  int nEcalHitsSameTowers(void) const;
  int nTracksSameTowers(void) const;
  void hcalHitsSameTowers(std::set<const HBHERecHit*>& v) const;
  void ecalHitsSameTowers(std::set<const EcalRecHit*>& v) const;
  void tracksSameTowers(std::set<const reco::Track*>& v) const;

  // same as above except for the hits and tracks in the neighboring towers to the hits in the collection
  double hcalEnergyNeighborTowers(void) const;
  double ecalEnergyNeighborTowers(void) const;
  double trackEnergyNeighborTowers(void) const;
  int nHcalHitsNeighborTowers(void) const;
  int nEcalHitsNeighborTowers(void) const;
  int nTracksNeighborTowers(void) const;
  void hcalHitsNeighborTowers(std::set<const HBHERecHit*>& v) const;
  void ecalHitsNeighborTowers(std::set<const EcalRecHit*>& v) const;
  void tracksNeighborTowers(std::set<const reco::Track*>& v) const;

  // gives the total energy due to hcal, ecal, and tracks tower-by-tower
  // separates the hcal energy into that which is a hit in the collection, and that which is not
  void byTowers(std::vector<twrinfo>& v) const;

  // find a hit in the hitmap
  hitmap_const_iterator findHit(const HBHERecHit* hit) const { return hits_.find(hit); }

  // find a neighbor
  neighbor_const_iterator findNeighbor(const PhysicsTower* twr) const { return neighbors_.find(twr); }

  // add a hit to the hitmap
  void insert(const HBHERecHit* hit, const PhysicsTower* twr, std::set<const PhysicsTower*>& neighbors);

  // access to the private maps and sets
  inline hitmap_const_iterator beginHits(void) const { return hits_.begin(); }
  inline hitmap_const_iterator endHits(void) const { return hits_.end(); }

  inline neighbor_const_iterator beginNeighbors(void) const { return neighbors_.begin(); }
  inline neighbor_const_iterator endNeighbors(void) const { return neighbors_.end(); }

private:
  std::map<const HBHERecHit*, const PhysicsTower*> hits_;
  std::set<const PhysicsTower*> neighbors_;

  void calcHits_(void) const;
  mutable double hitEnergy_;
  mutable double hitEnergyTrkFid_;
  mutable int nHits_;

  void calcHcalSameTowers_(void) const;
  mutable double hcalEnergySameTowers_;
  mutable int nHcalHitsSameTowers_;

  void calcEcalSameTowers_(void) const;
  mutable double ecalEnergySameTowers_;
  mutable int nEcalHitsSameTowers_;

  void calcTracksSameTowers_(void) const;
  mutable double trackEnergySameTowers_;
  mutable int nTracksSameTowers_;

  void calcHcalNeighborTowers_(void) const;
  mutable double hcalEnergyNeighborTowers_;
  mutable int nHcalHitsNeighborTowers_;

  void calcEcalNeighborTowers_(void) const;
  mutable double ecalEnergyNeighborTowers_;
  mutable int nEcalHitsNeighborTowers_;

  void calcTracksNeighborTowers_(void) const;
  mutable double trackEnergyNeighborTowers_;
  mutable int nTracksNeighborTowers_;
};

//////////////////////////////////////////////////////////////////////////////
//
// HBHEHitMapOrganizer
//
// Organizers the HBHEHitMaps into RBXs, HPDs, dihits, and monohits
//////////////////////////////////////////////////////////////////////////////

class HBHEHitMapOrganizer {
public:
  HBHEHitMapOrganizer(const edm::Handle<HBHERecHitCollection>& hbhehitcoll_h,
                      const ObjectValidatorAbs& objvalidator,
                      const PhysicsTowerOrganizer& pto,
                      const HcalFrontEndMap* hfemap);

  virtual ~HBHEHitMapOrganizer() {}

  void getRBXs(std::vector<HBHEHitMap>& v, double energy) const;
  void getHPDs(std::vector<HBHEHitMap>& v, double energy) const;
  void getDiHits(std::vector<HBHEHitMap>& v, double energy) const;
  void getMonoHits(std::vector<HBHEHitMap>& v, double energy) const;

private:
  const HcalFrontEndMap* hfemap_;
  std::map<int, HBHEHitMap> rbxs_, hpds_;
  std::vector<HBHEHitMap> dihits_, monohits_;

  // helper functions
  // finds all of the hits which are neighbors and in the same HPD as the reference hit
  void getHPDNeighbors(const HBHERecHit* hit,
                       std::vector<const HBHERecHit*>& neighbors,
                       const PhysicsTowerOrganizer& pto);
};

#endif
