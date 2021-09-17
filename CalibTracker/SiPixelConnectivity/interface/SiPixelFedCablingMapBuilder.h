#ifndef SiPixelFedCablingMapBuilder_H
#define SiPixelFedCablingMapBuilder_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <vector>
#include <string>

class PixelModuleName;
class PixelGeomDetUnit;

class SiPixelFedCablingMapBuilder {
public:
  //SiPixelFedCablingMapBuilder(const std::string & associatorName);
  SiPixelFedCablingMapBuilder(edm::ConsumesCollector&& iCC, const std::string fileName, const bool phase1 = false);
  SiPixelFedCablingTree* produce(const edm::EventSetup& setup);

private:
  struct FedSpec {
    int fedId;                            // fed ID
    std::vector<PixelModuleName*> names;  // names of modules
    std::vector<uint32_t> rawids;         // modules corresponding to names
  };
  //std::string theAssociatorName;
  std::string fileName_;
  std::string myprint(const PixelGeomDetUnit* pxUnit);
  bool phase1_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
};

#endif
