#ifndef DATAFORMATS_CALOTOWERS_CALOTOWER_H
#define DATAFORMATS_CALOTOWERS_CALOTOWER_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Rtypes.h" 
#include <vector>
#include <cmath>

/** \class CaloTower
    
$Date: 2006/03/29 21:26:26 $
$Revision: 1.4 $
\author J. Mans - Minnesota
*/
class CaloTower {
public:
  typedef CaloTowerDetId key_type; // for SortedCollection
  typedef math::RhoEtaPhiVector Vector;

  CaloTower();
  CaloTower(const CaloTowerDetId& id, const Vector& vec, 
	    double emEt, double hadEt, double outerEt,
	    int ecal_tp, int hcal_tp);

  CaloTowerDetId id() const { return id_; }
  void addConstituent( DetId id ) { constituents_.push_back( id ); }
  void addConstituents( const std::vector<DetId>& ids );
  size_t constituentsSize() const { return constituents_.size(); }
  DetId constituent( size_t i ) const { return constituents_[ i ]; }

  const Vector & momentum() const { return momentum_; }
  double et() const { return momentum_.rho(); }
  double energy() const { return momentum_.r(); }
  double eta() const { return momentum_.eta(); }
  double phi() const { return momentum_.phi(); }
  double emEnergy() const { return emEt_ * cosh( eta() ); }
  double hadEnergy() const { return hadEt_ * cosh( eta() ); }
  double outerEnergy() const { return outerEt_ * cosh( eta() ); }
  double outerEt() const { return outerEt_; }
  double emEt() const { return emEt_; }
  double hadEt() const { return hadEt_; }

  int emLvl1() const { return emLvl1_; }
  int hadLv11() const { return hadLvl1_; }

private:
  CaloTowerDetId id_;
  Vector momentum_;
  Double32_t emEt_, hadEt_, outerEt_;
  int emLvl1_,hadLvl1_;
  std::vector<DetId> constituents_;
};

std::ostream& operator<<(std::ostream& s, const CaloTower& ct);

inline bool operator==( const CaloTower & t1, const CaloTower & t2 ) {
  return t1.id() == t2.id();
} 

#endif
