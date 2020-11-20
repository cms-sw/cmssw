#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometryESCommon.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"

namespace CTPPSGeometryESCommon {

  std::unique_ptr<DetGeomDesc> applyAlignments(const DetGeomDesc& idealDetRoot,
                                               const CTPPSRPAlignmentCorrectionsData* alignments) {
    std::deque<const DetGeomDesc*> bufferIdealGeo;
    bufferIdealGeo.emplace_back(&idealDetRoot);

    std::deque<DetGeomDesc*> bufferAlignedGeo;
    DetGeomDesc* alignedDetRoot = new DetGeomDesc(idealDetRoot, DetGeomDesc::cmWithoutChildren);
    bufferAlignedGeo.emplace_back(alignedDetRoot);

    while (!bufferIdealGeo.empty()) {
      const DetGeomDesc* idealDet = bufferIdealGeo.front();
      DetGeomDesc* alignedDet = bufferAlignedGeo.front();
      bufferIdealGeo.pop_front();
      bufferAlignedGeo.pop_front();

      const std::string name = alignedDet->name();

      // Is it sensor? If yes, apply full sensor alignments
      if (name == DDD_TOTEM_RP_SENSOR_NAME || name == DDD_CTPPS_DIAMONDS_SEGMENT_NAME ||
          name == DDD_CTPPS_UFSD_SEGMENT_NAME || name == DDD_CTPPS_PIXELS_SENSOR_NAME ||
          name == DDD_CTPPS_PIXELS_SENSOR_NAME_2x2 ||
          std::regex_match(name, std::regex(DDD_TOTEM_TIMING_SENSOR_TMPL))) {
        unsigned int plId = alignedDet->geographicalID();

        if (alignments) {
          const auto& ac = alignments->getFullSensorCorrection(plId);
          alignedDet->applyAlignment(ac);
        }
      }

      // Is it RP box? If yes, apply RP alignments
      if (name == DDD_TOTEM_RP_RP_NAME || name == DDD_CTPPS_DIAMONDS_RP_NAME || name == DDD_CTPPS_PIXELS_RP_NAME ||
          name == DDD_TOTEM_TIMING_RP_NAME) {
        unsigned int rpId = alignedDet->geographicalID();

        if (alignments) {
          const auto& ac = alignments->getRPCorrection(rpId);
          alignedDet->applyAlignment(ac);
        }
      }

      // create and add children
      const auto& idealDetChildren = idealDet->components();
      for (unsigned int i = 0; i < idealDetChildren.size(); i++) {
        const DetGeomDesc* idealDetChild = idealDetChildren[i];
        bufferIdealGeo.emplace_back(idealDetChild);

        // create new node with the same information as in idealDetChild and add it as a child of alignedDet
        DetGeomDesc* alignedDetChild = new DetGeomDesc(*idealDetChild, DetGeomDesc::cmWithoutChildren);
        alignedDet->addComponent(alignedDetChild);

        bufferAlignedGeo.emplace_back(alignedDetChild);
      }
    }
    return std::unique_ptr<DetGeomDesc>(alignedDetRoot);
  }

}  // namespace CTPPSGeometryESCommon