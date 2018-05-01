/** \file
 *
 *  \author L. Gray - FNAL
 *
 */

#include <RecoMTD/DetLayers/plugins/MTDDetLayerGeometryESProducer.h>
#include <Geometry/Records/interface/MTDDigiGeometryRecord.h>

#include <Geometry/MTDGeometryBuilder/interface/MTDGeometry.h>

#include <RecoMTD/DetLayers/src/ETLDetLayerGeometryBuilder.h>
#include <RecoMTD/DetLayers/src/BTLDetLayerGeometryBuilder.h>

#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Framework/interface/NoProxyException.h>

#include <memory>
#include <iostream>


using namespace edm;

MTDDetLayerGeometryESProducer::MTDDetLayerGeometryESProducer(const edm::ParameterSet & p){
  setWhatProduced(this);
}


MTDDetLayerGeometryESProducer::~MTDDetLayerGeometryESProducer(){}


std::shared_ptr<MTDDetLayerGeometry>
MTDDetLayerGeometryESProducer::produce(const MTDRecoGeometryRecord & record) {

  const std::string metname = "MTD|RecoMTD|RecoMTDDetLayers|MTDDetLayerGeometryESProducer";
  auto mtdDetLayerGeometry = std::make_shared<MTDDetLayerGeometry>();
  
  edm::ESHandle<MTDGeometry> mtd;
  record.getRecord<MTDDigiGeometryRecord>().get(mtd);
  if (mtd.isValid()) {
    // Build BTL layers  
    mtdDetLayerGeometry->addBTLLayers(BTLDetLayerGeometryBuilder::buildLayers(*mtd));
    // Build ETL layers  
    mtdDetLayerGeometry->addETLLayers(ETLDetLayerGeometryBuilder::buildLayers(*mtd));
  } else {
    LogInfo(metname) << "No MTD geometry is available."; 
  }
  
  // Sort layers properly
  mtdDetLayerGeometry->sortLayers();

  return mtdDetLayerGeometry;
}
