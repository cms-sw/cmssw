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
    
$Date: 2007/07/31 15:19:57 $
$Revision: 1.5 $
\author J. Mans - Minnesota
*/

//
// Original author: J. Mans - Minnesota
// Created: 2007/07/31 15:19:57
//
// Modified: Anton Anastassov (Northwestern)
// Reason:   Make CaloTower inherit from LeafCandidate
// 
//

using namespace reco;

class CaloTower : public LeafCandidate {
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


  CaloTowerDetId id() const { return id_; }
  void addConstituent( DetId id ) { constituents_.push_back( id ); }
  void addConstituents( const std::vector<DetId>& ids );
  size_t constituentsSize() const { return constituents_.size(); }
  DetId constituent( size_t i ) const { return constituents_[ i ]; }

  double emEnergy() const { return emE_ ; }
  double hadEnergy() const { return hadE_ ; }
  double outerEnergy() const { return outerE_; }
  double emEt() const { return emE_ * sin( theta() ); }
  double hadEt() const { return hadE_ * sin( theta() ); }
  double outerEt() const { return outerE_ * sin( theta() ); }

  GlobalPoint emPosition()  const { return emPosition_ ; }
  GlobalPoint hadPosition() const { return hadPosition_ ; }

  int emLvl1() const { return emLvl1_; }
  int hadLv11() const { return hadLvl1_; }

private:
  CaloTowerDetId id_;
 
   // positions of assumed EM and HAD shower positions
   GlobalPoint emPosition_;
   GlobalPoint hadPosition_;

  Double32_t emE_, hadE_, outerE_;

  int emLvl1_,hadLvl1_;
  std::vector<DetId> constituents_;
};

std::ostream& operator<<(std::ostream& s, const CaloTower& ct);

inline bool operator==( const CaloTower & t1, const CaloTower & t2 ) {
  return t1.id() == t2.id();
} 

#endif
