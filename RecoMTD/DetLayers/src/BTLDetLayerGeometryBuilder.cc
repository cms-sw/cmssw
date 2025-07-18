//#define EDM_ML_DEBUG

#include "RecoMTD/DetLayers/interface/BTLDetLayerGeometryBuilder.h"

#include <DataFormats/ForwardDetId/interface/BTLDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h>
#include <RecoMTD/DetLayers/interface/MTDDetTray.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

BTLDetLayerGeometryBuilder::BTLDetLayerGeometryBuilder() {}

BTLDetLayerGeometryBuilder::~BTLDetLayerGeometryBuilder() {}

vector<DetLayer*> BTLDetLayerGeometryBuilder::buildLayers(const MTDGeometry& geo, const MTDTopology& topo) {
  vector<DetLayer*> detlayers;
  vector<MTDTrayBarrelLayer*> result;

  vector<const DetRod*> btlDetTrays;

  vector<const GeomDet*> geomDets;

  // logical tracking trays are now rows along z of modules, 3 per each mechanical tray, running from -z to z
  // MTDGeometry is already built with the proper ordering, it is enough to exploit that
  geomDets.reserve(topo.btlModulesPerRod());

  uint32_t index(0);
  for (const auto& det : geo.detsBTL()) {
    index++;
    geomDets.emplace_back(det);
    if (index == topo.btlModulesPerRod()) {
      btlDetTrays.emplace_back(new MTDDetTray(geomDets));
      LogTrace("MTDDetLayers") << "  New BTLDetTray with " << geomDets.size()
                               << " modules at R=" << btlDetTrays.back()->position().perp()
                               << ", phi=" << btlDetTrays.back()->position().phi();
      index = 0;
      geomDets.clear();
    }
  }

  result.emplace_back(new MTDTrayBarrelLayer(btlDetTrays));
  LogTrace("MTDDetLayers") << "BTLDetLayerGeometryBuilder: new MTDTrayBarrelLayer with " << btlDetTrays.size()
                           << " rods, at R " << result.back()->specificSurface().radius();

  for (vector<MTDTrayBarrelLayer*>::const_iterator it = result.begin(); it != result.end(); it++)
    detlayers.push_back((DetLayer*)(*it));

  return detlayers;
}
