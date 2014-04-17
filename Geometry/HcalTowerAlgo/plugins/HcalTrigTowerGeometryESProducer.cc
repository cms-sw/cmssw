#include "HcalTrigTowerGeometryESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <memory>

HcalTrigTowerGeometryESProducer::HcalTrigTowerGeometryESProducer( const edm::ParameterSet & config )
{
  enable1x1HF_=config.getParameter<bool>("enableHF1x1");
  enableRCTHF_=config.getParameter<bool>("enableHFRCT");

  setWhatProduced( this );
}

HcalTrigTowerGeometryESProducer::~HcalTrigTowerGeometryESProducer( void ) 
{}

boost::shared_ptr<HcalTrigTowerGeometry>
HcalTrigTowerGeometryESProducer::produce( const CaloGeometryRecord & iRecord )
{
    edm::ESHandle<HcalTopology> hcalTopology;
    iRecord.getRecord<IdealGeometryRecord>().get( hcalTopology );

    m_hcalTrigTowerGeom =
	boost::shared_ptr<HcalTrigTowerGeometry>( new HcalTrigTowerGeometry( &*hcalTopology));
    m_hcalTrigTowerGeom->setupHFTowers(enableRCTHF_,enable1x1HF_);

    return m_hcalTrigTowerGeom;
}

void HcalTrigTowerGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
   edm::ParameterSetDescription desc;
   desc.add<bool>("enableHF1x1", true);
   desc.add<bool>("enableHFRCT", true);
   descriptions.add("HcalTrigTowerGeometryESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE( HcalTrigTowerGeometryESProducer );
