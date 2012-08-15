#include "HcalTrigTowerGeometryESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include <memory>

HcalTrigTowerGeometryESProducer::HcalTrigTowerGeometryESProducer( const edm::ParameterSet & config )
    :  m_pSet( config )
{
    setWhatProduced( this );
}

HcalTrigTowerGeometryESProducer::~HcalTrigTowerGeometryESProducer( void ) 
{}

void
HcalTrigTowerGeometryESProducer::fillDescription( edm::ConfigurationDescriptions & descriptions )
{}

boost::shared_ptr<HcalTrigTowerGeometry>
HcalTrigTowerGeometryESProducer::produce( const CaloGeometryRecord & iRecord )
{
    const edm::ParameterSet hcalTopoConsts = m_pSet.getParameter<edm::ParameterSet>( "hcalTopologyConstants" );
    m_hcalTrigTowerGeom =
	boost::shared_ptr<HcalTrigTowerGeometry>( new HcalTrigTowerGeometry( new HcalTopology((HcalTopology::Mode) StringToEnumValue<HcalTopology::Mode>(hcalTopoConsts.getParameter<std::string>("mode")),
											      hcalTopoConsts.getParameter<int>("maxDepthHB"),
											      hcalTopoConsts.getParameter<int>("maxDepthHE"))));

    return m_hcalTrigTowerGeom;
}

DEFINE_FWK_EVENTSETUP_MODULE(HcalTrigTowerGeometryESProducer);
