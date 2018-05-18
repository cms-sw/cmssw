#ifndef __L1Trigger_L1THGCal_HGCalTriggerTowerGeometryHelper_h__
#define __L1Trigger_L1THGCal_HGCalTriggerTowerGeometryHelper_h__

/** \class HGCalTriggerTowerGeometryHelper
 *  The trigger tower map is defined esternally to CMSSW
 *  Assuming a regular binning, to map a given tower to a position we need to know
 *  - the reference surface,
 *  - the kind of binning (x-y or eta-phi)
 *  - the bin pitch
 *  - the first bin definition
 *  NOTE: more exotic binnning strategies need a mapping between bin-ids and positions
 */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerID.h"

#include <vector>

enum HGCalTriggerTowerType {
  regular_xy_generic = 0,
  regular_etaphi_generic = 1,
  regular_etaphi = 2
};

namespace l1t {
  class HGCalTowerID;
  class HGCalTowerCoord;
}


class HGCalTriggerTowerGeometryHelper {
  public:
    HGCalTriggerTowerGeometryHelper(const edm::ParameterSet& conf);

    void setRefCoordinates(float refCoord1,
                           float refCoord2,
                           float refZ,
                           float binSize1,
                           float binSize2) {
      refCoord1_ = refCoord1;
      refCoord2_ = refCoord2;
      referenceZ_ = refZ;
      binSizeCoord1_ = binSize1;
      binSizeCoord2_ = binSize2;
    }

    ~HGCalTriggerTowerGeometryHelper() {}

    // FIXME: is this still needed????
    // void createTowerCoordinates(const std::vector<unsigned short>& tower_ids);

    GlobalPoint getPositionAtReferenceSurface(const l1t::HGCalTowerID& towerId) const;
    const std::vector<l1t::HGCalTowerCoord>& getTowerCoordinates() const;

    unsigned short getTriggerTowerFromTriggerCell(const unsigned) const;

  private:

    const HGCalTriggerTowerType type_;

    float refCoord1_;
    float refCoord2_;
    float referenceZ_;
    float binSizeCoord1_;
    float binSizeCoord2_;

    std::vector<l1t::HGCalTowerCoord> tower_coords_;
    std::map<unsigned, short> cells_to_trigger_towers_;

  };


#endif
