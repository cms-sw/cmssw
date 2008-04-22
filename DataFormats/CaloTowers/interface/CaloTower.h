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
    
$Date: 2008/04/14 06:16:22 $
$Revision: 1.6 $
\author J. Mans - Minnesota
*/

// Original author: J. Mans - Minnesota
//
// Modified: Anton Anastassov (Northwestern)
//    Make CaloTower inherit from LeafCandidate,
//    add new members and accessors.

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


   // setters
  void addConstituent( DetId id ) { constituents_.push_back( id ); }
  void addConstituents( const std::vector<DetId>& ids );
  void setEcalTime(int t) { ecalTime_ = t; };
  void setHcalTime(int t) { hcalTime_ = t; };

  // getters
  CaloTowerDetId id() const { return id_; }
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

  double hadEnergyHeOuterLayer() { return (id_.ietaAbs()<=17)? 0 : outerE_; }
  double hadEnergyHeInnerLayer() { return (id_.ietaAbs()<=17)? 0 : hadE_ - outerE_; }

  float ecalTime() const { return float(ecalTime_) * 0.01; }
  float hcalTime() const { return float(hcalTime_) * 0.01; }

  int ieta() const { return id_.ieta(); }
  int ietaAbs() const { return id_.ietaAbs(); }
  int iphi() const { return id_.iphi(); }
  int zside() const { return id_.zside(); }

  int numCrystals(); 

private:
  CaloTowerDetId id_;
 
   // positions of assumed EM and HAD shower positions
   GlobalPoint emPosition_;
   GlobalPoint hadPosition_;

   // time
   int ecalTime_;
   int hcalTime_;

  Double32_t emE_, hadE_, outerE_;

  int emLvl1_,hadLvl1_;
  std::vector<DetId> constituents_;
};

std::ostream& operator<<(std::ostream& s, const CaloTower& ct);

inline bool operator==( const CaloTower & t1, const CaloTower & t2 ) {
  return t1.id() == t2.id();
} 

#endif
