#ifndef HcalTrigTowerGeometry_h
#define HcalTrigTowerGeometry_h

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
class HcalTrigTowerDetId;
class HcalDetId;

class HcalTrigTowerGeometry {
public:

  HcalTrigTowerGeometry();

  void setupHF(bool useShortFibers, bool useQuadRings);
  
  /// the mapping to and from DetIds
  std::vector<HcalTrigTowerDetId> towerIds(const HcalDetId & cellId) const;
  std::vector<HcalDetId> detIds(const HcalTrigTowerDetId &) const;

  /// an interface for CaloSubdetectorGeometry
  //std::vector<DetId> getValidDetIds(DetId::Detector det, int subdet) const;

  /// the number of phi bins in this eta ring
  int nPhiBins(int ieta) const {
    return (abs(ieta) < firstHFTower()) ? 72 : 18;
  }

  int firstHFTower() const {return 29;} 
  int nTowers() const {return 32;}

  /// the number of HF eta rings in this trigger tower
  /// ieta starts at firstHFTower()
  int hfTowerEtaSize(int ieta) const;

  /// since the towers are irregular in eta in HF
  int firstHFRingInTower(int ietaTower) const;

  /// where this tower begins and ends in eta
  void towerEtaBounds(int ieta, double & eta1, double & eta2) const;

private:
  HcalTopology theTopology;
  bool useShortFibers_;
  bool useHFQuadPhiRings_;
  bool useUpgradeConfigurationHFTowers_;
};

#endif

