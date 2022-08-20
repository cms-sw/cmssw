#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

MTDTimeCalib::MTDTimeCalib(edm::ParameterSet const& conf, const MTDGeometry* geom, const MTDTopology* topo)
    : geom_(geom),
      topo_(topo),
      btlTimeOffset_(conf.getParameter<double>("BTLTimeOffset")),
      etlTimeOffset_(conf.getParameter<double>("ETLTimeOffset")),
      btlLightCollTime_(conf.getParameter<double>("BTLLightCollTime")),
      btlLightCollSlope_(conf.getParameter<double>("BTLLightCollSlope")) {}

float MTDTimeCalib::getTimeCalib(const MTDDetId& id) const {
  if (id.subDetector() != MTDDetId::FastTime) {
    throw cms::Exception("MTDTimeCalib") << "MTDDetId: " << std::hex << id.rawId() << " is invalid!" << std::dec
                                         << std::endl;
  }

  float time_calib = 0.;

  if (id.mtdSubDetector() == MTDDetId::BTL) {
    time_calib += btlTimeOffset_;
    BTLDetId hitId(id);
    //for BTL topology gives different layout id
    DetId geoId = hitId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topo_->getMTDTopologyMode()));
    const MTDGeomDet* thedet = geom_->idToDet(geoId);

    if (thedet == nullptr) {
      throw cms::Exception("MTDTimeCalib") << "GeographicalID: " << std::hex << geoId.rawId() << " (" << id.rawId()
                                           << ") is invalid!" << std::dec << std::endl;
    }
    const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
    const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

    BTLDetId::CrysLayout btlL = MTDTopologyMode::crysLayoutFromTopoMode(topo_->getMTDTopologyMode());
    if (btlL == BTLDetId::CrysLayout::tile) {
      time_calib -= btlLightCollTime_;  //simply remove the offset introduced at sim level
    } else if (btlL == BTLDetId::CrysLayout::bar || btlL == BTLDetId::CrysLayout::barphiflat ||
               btlL == BTLDetId::CrysLayout::tdr) {
      //for bars in phi
      time_calib -= 0.5 * topo.pitch().first * btlLightCollSlope_;  //time offset for bar time is L/2v
    } else if (btlL == BTLDetId::CrysLayout::barzflat) {
      //for bars in z
      time_calib -= 0.5 * topo.pitch().second * btlLightCollSlope_;  //time offset for bar time is L/2v
    }
  } else if (id.mtdSubDetector() == MTDDetId::ETL) {
    time_calib += etlTimeOffset_;
  } else {
    throw cms::Exception("MTDTimeCalib") << "MTDDetId: " << std::hex << id.rawId() << " is invalid!" << std::dec
                                         << std::endl;
  }

  return time_calib;
}

#include "FWCore/Utilities/interface/typelookup.h"

//--- Now use the Framework macros to set it all up:
TYPELOOKUP_DATA_REG(MTDTimeCalib);
