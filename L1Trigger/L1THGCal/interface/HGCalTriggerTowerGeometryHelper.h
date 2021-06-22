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

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

  const std::vector<l1t::HGCalTowerCoord>& getTowerCoordinates() const;

  unsigned short getTriggerTowerFromEtaPhi(const float& eta, const float& phi) const;
  unsigned short getTriggerTower(const l1t::HGCalTriggerCell&) const;
  unsigned short getTriggerTower(const l1t::HGCalTriggerSums&) const;

  const bool isNose() { return doNose_; }

private:
  std::vector<l1t::HGCalTowerCoord> tower_coords_;
  std::unordered_map<unsigned, short> cells_to_trigger_towers_;

  bool doNose_;
  double minEta_;
  double maxEta_;
  double minPhi_;
  double maxPhi_;
  unsigned int nBinsEta_;
  unsigned int nBinsPhi_;

  std::vector<double> binsEta_;
  std::vector<double> binsPhi_;

  HGCalTriggerTools triggerTools_;
};

#endif
