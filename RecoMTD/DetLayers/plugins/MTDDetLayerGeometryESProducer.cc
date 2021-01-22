/** \file
 *
 *  ESProducer for MTDDetLayerGeometry in RecoMTD/DetLayers
 *
 *  \author L. Gray - FNAL
 *
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "ETLDetLayerGeometryBuilder.h"
#include "BTLDetLayerGeometryBuilder.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <iostream>

class MTDDetLayerGeometryESProducer : public edm::ESProducer {
public:
  /// Constructor
  MTDDetLayerGeometryESProducer(const edm::ParameterSet& p);

  /// Produce MuonDeLayerGeometry.
  std::unique_ptr<MTDDetLayerGeometry> produce(const MTDRecoGeometryRecord& record);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
};

using namespace edm;

MTDDetLayerGeometryESProducer::MTDDetLayerGeometryESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);
  geomToken_ = cc.consumes();
  mtdtopoToken_ = cc.consumes();
}

std::unique_ptr<MTDDetLayerGeometry> MTDDetLayerGeometryESProducer::produce(const MTDRecoGeometryRecord& record) {
  auto mtdDetLayerGeometry = std::make_unique<MTDDetLayerGeometry>();

  if (auto mtd = record.getHandle(geomToken_)) {
    // Build BTL layers
    mtdDetLayerGeometry->addBTLLayers(BTLDetLayerGeometryBuilder::buildLayers(*mtd));
    // Build ETL layers, depends on the scenario
    if (auto mtdtopo = record.getHandle(mtdtopoToken_)) {
      mtdDetLayerGeometry->addETLLayers(ETLDetLayerGeometryBuilder::buildLayers(*mtd, mtdtopo->getMTDTopologyMode()));
    } else {
      LogWarning("MTDDetLayers") << "No MTD topology  is available.";
    }
  } else {
    LogWarning("MTDDetLayers") << "No MTD geometry is available.";
  }

  // Sort layers properly
  mtdDetLayerGeometry->sortLayers();

  return mtdDetLayerGeometry;
}

void MTDDetLayerGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription ps;
  desc.addDefault(ps);
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDDetLayerGeometryESProducer);
