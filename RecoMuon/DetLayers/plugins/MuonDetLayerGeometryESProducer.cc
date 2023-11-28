/** \file
 *
 *  \author N. Amapane - CERN
 *
 *  \modified by R. Radogna & C. Calabria & A. Sharma
 *  \modified by D. Nash
 */

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h"
#include "RecoMuon/DetLayers/src/MuonRPCDetLayerGeometryBuilder.h"
#include "RecoMuon/DetLayers/src/MuonDTDetLayerGeometryBuilder.h"
#include "RecoMuon/DetLayers/src/MuonGEMDetLayerGeometryBuilder.h"
#include "RecoMuon/DetLayers/src/MuonME0DetLayerGeometryBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>

class MuonDetLayerGeometryESProducer : public edm::ESProducer {
public:
  /// Constructor
  MuonDetLayerGeometryESProducer(const edm::ParameterSet& p);

  /// Produce MuonDeLayerGeometry.
  std::unique_ptr<MuonDetLayerGeometry> produce(const MuonRecoGeometryRecord& record);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemToken_;
  edm::ESGetToken<ME0Geometry, MuonGeometryRecord> me0Token_;
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcToken_;
};

using namespace edm;

MuonDetLayerGeometryESProducer::MuonDetLayerGeometryESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);
  dtToken_ = cc.consumes();
  cscToken_ = cc.consumes();
  gemToken_ = cc.consumes();
  me0Token_ = cc.consumes();
  rpcToken_ = cc.consumes();
}

std::unique_ptr<MuonDetLayerGeometry> MuonDetLayerGeometryESProducer::produce(const MuonRecoGeometryRecord& record) {
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonDetLayerGeometryESProducer";
  auto muonDetLayerGeometry = std::make_unique<MuonDetLayerGeometry>();

  // Build DT layers
  if (auto dt = record.getHandle(dtToken_)) {
    muonDetLayerGeometry->addDTLayers(MuonDTDetLayerGeometryBuilder::buildLayers(*dt));
  } else {
    LogInfo(metname) << "No DT geometry is available.";
  }

  // Build CSC layers
  if (auto csc = record.getHandle(cscToken_)) {
    muonDetLayerGeometry->addCSCLayers(MuonCSCDetLayerGeometryBuilder::buildLayers(*csc));
  } else {
    LogInfo(metname) << "No CSC geometry is available.";
  }

  // Build GEM layers
  if (auto gem = record.getHandle(gemToken_)) {
    muonDetLayerGeometry->addGEMLayers(MuonGEMDetLayerGeometryBuilder::buildEndcapLayers(*gem));
  } else {
    LogInfo(metname) << "No GEM geometry is available.";
  }

  // Build ME0 layers
  if (auto me0 = record.getHandle(me0Token_)) {
    muonDetLayerGeometry->addME0Layers(MuonME0DetLayerGeometryBuilder::buildEndcapLayers(*me0));
  } else {
    LogDebug(metname) << "No ME0 geometry is available.";
  }

  // Build RPC layers
  if (auto rpc = record.getHandle(rpcToken_)) {
    muonDetLayerGeometry->addRPCLayers(MuonRPCDetLayerGeometryBuilder::buildBarrelLayers(*rpc),
                                       MuonRPCDetLayerGeometryBuilder::buildEndcapLayers(*rpc));
  } else {
    LogInfo(metname) << "No RPC geometry is available.";
  }

  // Sort layers properly
  muonDetLayerGeometry->sortLayers();

  return muonDetLayerGeometry;
}

void MuonDetLayerGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  //no parameters are used
  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonDetLayerGeometryESProducer);
