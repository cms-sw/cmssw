#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyMap.h"

void PixelTopologyMap::printAll(std::ostream& os) const {
  for (unsigned int i = 1; i <= m_pxbMap.size(); i++) {
    os << "PXB layer " << std::setw(2) << i << " has: " << std::setw(2) << getPXBLadders(i) << " ladders and "
       << std::setw(2) << getPXBModules(i) << " modules" << std::endl;
  }

  for (unsigned int j = 1; j <= m_pxfMap.size(); j++) {
    os << "PXF disk  " << std::setw(2) << j << " has: " << std::setw(2) << getPXFBlades(j) << " blades  and "
       << std::setw(2) << getPXFModules(j) << " modules" << std::endl;
  }
}

void PixelTopologyMap::buildTopologyMaps() {
  // build barrel
  const auto& nlay = m_trackerGeom->numberOfLayers(PixelSubdetector::PixelBarrel);
  std::vector<unsigned> maxLadder, maxModule;
  maxLadder.resize(nlay);
  maxModule.resize(nlay);
  for (unsigned int i = 1; i <= nlay; i++) {
    maxLadder.push_back(0);
    maxModule.push_back(0);
  }

  for (auto det : m_trackerGeom->detsPXB()) {
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);

    const auto& layer = m_trackerTopo->pxbLayer(pixelDet->geographicalId());
    const auto& ladder = m_trackerTopo->pxbLadder(pixelDet->geographicalId());
    const auto& module = m_trackerTopo->pxbModule(pixelDet->geographicalId());

    if (ladder > maxLadder[layer]) {
      maxLadder[layer] = ladder;
    }

    if (module > maxModule[layer]) {
      maxModule[layer] = module;
    }
  }

  for (unsigned int i = 1; i <= nlay; i++) {
    m_pxbMap[i] = std::make_pair(maxLadder[i], maxModule[i]);
  }

  // build endcaps
  const auto& ndisk = m_trackerGeom->numberOfLayers(PixelSubdetector::PixelEndcap);
  std::vector<unsigned> maxBlade, maxPXFModule;
  maxBlade.resize(ndisk);
  maxPXFModule.resize(ndisk);
  for (unsigned int i = 1; i <= ndisk; i++) {
    maxBlade.push_back(0);
    maxPXFModule.push_back(0);
  }

  for (auto det : m_trackerGeom->detsPXF()) {
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);

    const auto& disk = m_trackerTopo->pxfDisk(pixelDet->geographicalId());
    const auto& blade = m_trackerTopo->pxfBlade(pixelDet->geographicalId());
    const auto& pxf_module = m_trackerTopo->pxfModule(pixelDet->geographicalId());

    if (blade > maxBlade[disk]) {
      maxBlade[disk] = blade;
    }

    if (pxf_module > maxPXFModule[disk]) {
      maxPXFModule[disk] = pxf_module;
    }
  }

  for (unsigned int i = 1; i <= ndisk; i++) {
    m_pxfMap[i] = std::make_pair(maxBlade[i], maxPXFModule[i]);
  }
}
