#include "HcalTrigTowerGeometryESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include <memory>

HcalTrigTowerGeometryESProducer::HcalTrigTowerGeometryESProducer( const edm::ParameterSet & config )
{
  setWhatProduced( this );
}

HcalTrigTowerGeometryESProducer::~HcalTrigTowerGeometryESProducer( void ) 
{}

std::shared_ptr<HcalTrigTowerGeometry>
HcalTrigTowerGeometryESProducer::produce( const CaloGeometryRecord & iRecord )
{
    edm::ESHandle<HcalTopology> hcalTopology;
    iRecord.getRecord<HcalRecNumberingRecord>().get( hcalTopology );

    m_hcalTrigTowerGeom = std::make_shared<HcalTrigTowerGeometry>( &*hcalTopology);
    HcalTopologyMode::TriggerMode tmode=hcalTopology->triggerMode();
    bool enableRCTHF=(tmode==HcalTopologyMode::tm_LHC_RCT || tmode==HcalTopologyMode::tm_LHC_RCT_and_1x1);
    bool enable1x1HF=(tmode==HcalTopologyMode::tm_LHC_1x1 || tmode==HcalTopologyMode::tm_LHC_RCT_and_1x1);
    m_hcalTrigTowerGeom->setupHFTowers(enableRCTHF,enable1x1HF);

    return m_hcalTrigTowerGeom;
}

void HcalTrigTowerGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
   edm::ParameterSetDescription desc;
   descriptions.add("HcalTrigTowerGeometryESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE( HcalTrigTowerGeometryESProducer );
