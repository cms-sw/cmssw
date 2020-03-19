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
};

using namespace edm;

MTDDetLayerGeometryESProducer::MTDDetLayerGeometryESProducer(const edm::ParameterSet& p) {
  setWhatProduced(this).setConsumes(geomToken_);
}

std::unique_ptr<MTDDetLayerGeometry> MTDDetLayerGeometryESProducer::produce(const MTDRecoGeometryRecord& record) {
  auto mtdDetLayerGeometry = std::make_unique<MTDDetLayerGeometry>();

  if (auto mtd = record.getHandle(geomToken_)) {
    // Build BTL layers
    mtdDetLayerGeometry->addBTLLayers(BTLDetLayerGeometryBuilder::buildLayers(*mtd));
    // Build ETL layers
    mtdDetLayerGeometry->addETLLayers(ETLDetLayerGeometryBuilder::buildLayers(*mtd));
  } else {
    const std::string metname = "MTD|RecoMTD|RecoMTDDetLayers|MTDDetLayerGeometryESProducer";
    LogInfo(metname) << "No MTD geometry is available.";
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
