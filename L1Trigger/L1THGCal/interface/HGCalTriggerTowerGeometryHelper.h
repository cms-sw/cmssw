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
  regular_xy = 0,
  regular_etaphi = 1
};

namespace l1t {
  class HGCalTowerID;
  class HGCalTowerCoord;
}


class HGCalTriggerTowerGeometryHelper {
  public:
    HGCalTriggerTowerGeometryHelper(const edm::ParameterSet& conf) : HGCalTriggerTowerGeometryHelper(conf.getParameter<double>("refCoord1"),
                                                                                                     conf.getParameter<double>("refCoord2"),
                                                                                                     conf.getParameter<double>("refZ"),
                                                                                                     conf.getParameter<double>("binSizeCoord1"),
                                                                                                     conf.getParameter<double>("binSizeCoord2"),
                                                                                                     static_cast<HGCalTriggerTowerType>(conf.getParameter<int>("type"))) {}


    HGCalTriggerTowerGeometryHelper(float refCoord1,
                                    float refCoord2,
                                    float refZ,
                                    float binSize1,
                                    float binSize2,
                                    HGCalTriggerTowerType type) : refCoord1_(refCoord1),
                                                                 refCoord2_(refCoord2),
                                                                 referenceZ_(refZ),
                                                                 binSizeCoord1_(binSize1),
                                                                 binSizeCoord2_(binSize2),
                                                                 type_(type) {}
    ~HGCalTriggerTowerGeometryHelper() {}


    void createTowerCoordinates(const std::vector<unsigned short>& tower_ids);

    GlobalPoint getPositionAtReferenceSurface(const l1t::HGCalTowerID& towerId) const;
    const std::vector<l1t::HGCalTowerCoord>& getTowerCoordinates() const;


  private:

    const float refCoord1_;
    const float refCoord2_;
    const float referenceZ_;
    const float binSizeCoord1_;
    const float binSizeCoord2_;
    const HGCalTriggerTowerType type_;
    std::vector<l1t::HGCalTowerCoord> coord;

  };


#endif
