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

std::unique_ptr<HcalTrigTowerGeometry>
HcalTrigTowerGeometryESProducer::produce( const CaloGeometryRecord & iRecord )
{
    edm::ESHandle<HcalTopology> hcalTopology;
    iRecord.getRecord<HcalRecNumberingRecord>().get( hcalTopology );

    return std::make_unique<HcalTrigTowerGeometry>(&*hcalTopology);
}

void HcalTrigTowerGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
   edm::ParameterSetDescription desc;
   descriptions.add("HcalTrigTowerGeometryESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE( HcalTrigTowerGeometryESProducer );
