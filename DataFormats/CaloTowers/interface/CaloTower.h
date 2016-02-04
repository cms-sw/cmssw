#ifndef DATAFORMATS_CALOTOWERS_CALOTOWER_H
#define DATAFORMATS_CALOTOWERS_CALOTOWER_H 1

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Rtypes.h" 
#include <vector>
#include <cmath>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

/** \class CaloTower
    
$Date: 2011/02/25 22:03:20 $
$Revision: 1.18 $
\author J. Mans - Minnesota
*/

// Original author: J. Mans - Minnesota
//
// Modified: Anton Anastassov (Northwestern)
//    Make CaloTower inherit from LeafCandidate,
//    add new members and accessors.


class CaloTower : public reco::LeafCandidate {
public:
  typedef CaloTowerDetId key_type; // for SortedCollection

  // Default constructor
  CaloTower();

  // Constructors from values

  CaloTower(const CaloTowerDetId& id, 
	    double emE, double hadE, double outerE,
	    int ecal_tp, int hcal_tp,
	    const PolarLorentzVector p4,
      GlobalPoint emPosition, GlobalPoint hadPosition);

  CaloTower(const CaloTowerDetId& id, 
	    double emE, double hadE, double outerE,
	    int ecal_tp, int hcal_tp,
	    const LorentzVector p4,
      GlobalPoint emPosition, GlobalPoint hadPosition);


   // setters
  void addConstituent( DetId id ) { constituents_.push_back( id ); }
  void addConstituents( const std::vector<DetId>& ids );
  void setEcalTime(int t) { ecalTime_ = t; };
  void setHcalTime(int t) { hcalTime_ = t; };

  // set CaloTower status based on the number of
  // bad/recovered/problematic cells in ECAL and HCAL

  void setCaloTowerStatus(unsigned int numBadHcalChan,unsigned int numBadEcalChan, 
			  unsigned int numRecHcalChan,unsigned int numRecEcalChan,
			  unsigned int numProbHcalChan,unsigned int numProbEcalChan);

  void setCaloTowerStatus(uint32_t s) { twrStatusWord_ = s; }

  // set the hottest cell energy in the tower
  void setHottestCellE(double e) { hottestCellE_ = e; }




  // getters
  CaloTowerDetId id() const { return id_; }
  const std::vector<DetId>& constituents() const { return constituents_; }
  size_t constituentsSize() const { return constituents_.size(); }
  DetId constituent( size_t i ) const { return constituents_[ i ]; }

  // energy contributions from different detectors
  // energy in HO ("outerEnergy")is not included in "hadEnergy"
  double emEnergy() const { return emE_ ; }
  double hadEnergy() const { return hadE_ ; }
  double outerEnergy() const { return (id_.ietaAbs()<16)? outerE_ : 0.0; }

  // transverse energies wrt to vtx (0,0,0)
  double emEt() const { return emE_ * sin( theta() ); }
  double hadEt() const { return hadE_ * sin( theta() ); }
  double outerEt() const { return (id_.ietaAbs()<16)? outerE_ * sin( theta() ) : 0.0; }


  // preserve the inherited default accessors where applicable
  // (user gets default p4 wrt to vtx (0,0,0) using p4(), etc.

  using LeafCandidate::p4;
  using LeafCandidate::p;
  using LeafCandidate::et; 


  // recalculated wrt user provided vertex Z position;

  math::PtEtaPhiMLorentzVector p4(double vtxZ) const;
  double p (double vtxZ) const { return p4(vtxZ).P(); }
  double et(double vtxZ) const { return p4(vtxZ).Et(); }

  double emEt(double vtxZ)  const { return  emE_ * sin(p4(vtxZ).theta()); }
  double hadEt(double vtxZ) const { return  hadE_ * sin(p4(vtxZ).theta()); }
  double outerEt(double vtxZ) const { return (id_.ietaAbs()<16)? outerE_ * sin(p4(vtxZ).theta()) : 0.0; }

  // recalculated wrt vertex provided as 3D point

  math::PtEtaPhiMLorentzVector p4(Point v) const;
  double p (Point v) const { return p4(v).P(); }
  double et(Point v) const { return p4(v).Et(); }

  double emEt(Point v)  const { return  emE_ * sin(p4(v).theta()); }
  double hadEt(Point v) const { return  hadE_ * sin(p4(v).theta()); }
  double outerEt(Point v) const { return (id_.ietaAbs()<16)? outerE_ * sin(p4(v).theta()) : 0.0; }

  double hottestCellE() const { return hottestCellE_; }

  // Access to p4 comming from HO alone: requested by JetMET to add/subtract HO contributions
  // to the tower for cases when the tower collection was created without/with HO   

  math::PtEtaPhiMLorentzVector p4_HO() const;  
  math::PtEtaPhiMLorentzVector p4_HO(double vtxZ) const;
  math::PtEtaPhiMLorentzVector p4_HO(Point v) const;


  // the reference poins in ECAL and HCAL for direction determination
  // algorithm and parameters for selecting these points are set in the CaloTowersCreator
  const GlobalPoint& emPosition()  const { return emPosition_ ; }
  const GlobalPoint& hadPosition() const { return hadPosition_ ; }

  int emLvl1() const { return emLvl1_; }
  int hadLv11() const { return hadLvl1_; }

  // energy contained in depths>1 in the HE for 18<|iEta|<29
  double hadEnergyHeOuterLayer() const { return (id_.ietaAbs()<18 || id_.ietaAbs()>29)? 0 : outerE_; }
  double hadEnergyHeInnerLayer() const { return (id_.ietaAbs()<18 || id_.ietaAbs()>29)? 0 : hadE_ - outerE_; }

  // energy in the tower by HCAL subdetector
  // This is trivial except for tower 16
  // needed by JetMET cleanup in AOD.
  double energyInHB() const; // { return (id_.ietaAbs()<16)? outerE_ : 0.0; }
  double energyInHE() const;
  double energyInHF() const;
  double energyInHO() const;



  // time (ns) in ECAL/HCAL components of the tower based on weigted sum of the times in the contributing RecHits
  float ecalTime() const { return float(ecalTime_) * 0.01; }
  float hcalTime() const { return float(hcalTime_) * 0.01; }

  // position information on the tower
  int ieta() const { return id_.ieta(); }
  int ietaAbs() const { return id_.ietaAbs(); }
  int iphi() const { return id_.iphi(); }
  int zside() const { return id_.zside(); }

  int numCrystals() const; 

  // methods to retrieve status information from the CaloTower:
  // number of bad/recovered/problematic cells in the tower
  // separately for ECAL and HCAL

  unsigned int numBadEcalCells() const { return (twrStatusWord_ & 0x1F); }
  unsigned int numRecoveredEcalCells() const { return ((twrStatusWord_ >> 5) & 0x1F); }
  unsigned int numProblematicEcalCells() const { return ((twrStatusWord_ >> 10) & 0x1F); }

  unsigned int numBadHcalCells() const { return ( (twrStatusWord_ >> 15)& 0x7); }
  unsigned int numRecoveredHcalCells() const { return ((twrStatusWord_ >> 18) & 0x7); }
  unsigned int numProblematicHcalCells() const { return ((twrStatusWord_ >> 21) & 0x7); }

  // the status word itself
  uint32_t towerStatusWord() const { return twrStatusWord_; }


private:
  CaloTowerDetId id_;
 
  uint32_t twrStatusWord_;

   // positions of assumed EM and HAD shower positions
   GlobalPoint emPosition_;
   GlobalPoint hadPosition_;

   // time
   int ecalTime_;
   int hcalTime_;

  float emE_, hadE_, outerE_;
  // for Jet ID use: hottest cell (ECAl or HCAL)
  float hottestCellE_;

  int emLvl1_,hadLvl1_;
  std::vector<DetId> constituents_;

  // vertex correction of EM and HAD momentum components:
  // internally used in the transformation of the CaloTower p4

  // for 3D vertex
  math::PtEtaPhiMLorentzVector hadP4(Point v) const;
  math::PtEtaPhiMLorentzVector emP4(Point v) const;

  // taking only z-component
  math::PtEtaPhiMLorentzVector hadP4(double vtxZ) const;
  math::PtEtaPhiMLorentzVector emP4(double vtxZ) const;
};

std::ostream& operator<<(std::ostream& s, const CaloTower& ct);

inline bool operator==( const CaloTower & t1, const CaloTower & t2 ) {
  return t1.id() == t2.id();
} 

#endif
