#ifndef __L1Trigger_L1THGCal_HGCalTriggerTowerGeometryHelper_h__
#define __L1Trigger_L1THGCal_HGCalTriggerTowerGeometryHelper_h__

/** \class HGCalTriggerTowerGeometryHelper
 *  Handles the mapping between TCs and TTs.
 *  The mapping can be provided externally (via a mapping file)
 *  or can be derived on the fly based on the TC eta-phi coordinates.
 *  The bin boundaries need anyhow to be provided to establish the eta-phi coordinates
 *  of the towers (assumed as the Tower Center for the moment)
 */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerID.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerModuleDetId.h"

#include <vector>
#include <unordered_map>

namespace l1t {
  class HGCalTowerID;
  struct HGCalTowerCoord;
}  // namespace l1t

class HGCalTriggerTowerGeometryHelper {
public:
  HGCalTriggerTowerGeometryHelper(const edm::ParameterSet& conf);

  ~HGCalTriggerTowerGeometryHelper() {}

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }

  unsigned packLayerSubdetWaferId(int subdet, int layer, int moduleU, int moduleV) const;
  unsigned packTowerIDandShare(int towerEta, int towerPhi, int towerShare) const;
  void unpackTowerIDandShare(unsigned towerIDandShare, int& towerEta_raw, int& towerPhi_raw, int& towerShare) const;
  int moveToCorrectSector(int towerPhi_raw, int sector) const;
  void reverseXaxis(int& towerPhi) const;

  const std::vector<l1t::HGCalTowerCoord>& getTowerCoordinates() const;

  unsigned short getTriggerTowerFromEtaPhi(const float& eta, const float& phi) const;
  std::unordered_map<unsigned short, float> getTriggerTower(const l1t::HGCalTriggerCell&) const;
  std::unordered_map<unsigned short, float> getTriggerTower(const l1t::HGCalTriggerSums&) const;

  const bool isNose() { return doNose_; }

private:
  static const int towerShareMask = 0x7F;
  static const int towerShareShift = 14;
  static const int signMask = 0x1;
  static const int sign1Shift = 21;
  static const int sign2Shift = 22;
  std::vector<l1t::HGCalTowerCoord> tower_coords_;
  std::unordered_map<unsigned, short> cells_to_trigger_towers_;
  std::unordered_map<unsigned, std::vector<unsigned>> modules_to_trigger_towers_;

  bool doNose_;
  double minEta_;
  double maxEta_;
  double minPhi_;
  double maxPhi_;
  unsigned int nBinsEta_;
  unsigned int nBinsPhi_;

  std::vector<double> binsEta_;
  std::vector<double> binsPhi_;

  bool splitModuleSum_;
  int splitDivisorSilic_;
  int splitDivisorScint_;
  int rotate180Deg_;
  int rotate120Deg_;
  int reverseX_;

  HGCalTriggerTools triggerTools_;
};

#endif
