#include "HcalTrigTowerGeometryESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include <memory>

HcalTrigTowerGeometryESProducer::HcalTrigTowerGeometryESProducer( const edm::ParameterSet & config )
    :  m_pSet( config )
{
    setWhatProduced( this );
}

HcalTrigTowerGeometryESProducer::~HcalTrigTowerGeometryESProducer( void ) 
{}

void
HcalTrigTowerGeometryESProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
{
  edm::ParameterSetDescription hcalTopologyConstants;
  hcalTopologyConstants.add<std::string>( "mode", "HcalTopologyMode::LHC" );
  hcalTopologyConstants.add<int>( "maxDepthHB", 2 );
  hcalTopologyConstants.add<int>( "maxDepthHE", 3 );  

  edm::ParameterSetDescription hcalSLHCTopologyConstants;
  hcalSLHCTopologyConstants.add<std::string>( "mode", "HcalTopologyMode::SLHC" );
  hcalSLHCTopologyConstants.add<int>( "maxDepthHB", 7 );
  hcalSLHCTopologyConstants.add<int>( "maxDepthHE", 7 );

  edm::ParameterSetDescription desc;
  desc.addOptional<edm::ParameterSetDescription>( "hcalTopologyConstants", hcalTopologyConstants );
  descriptions.add( "hcalTrigTowerGeometry", desc );

  edm::ParameterSetDescription descSLHC;
  descSLHC.addOptional<edm::ParameterSetDescription>( "hcalTopologyConstants", hcalSLHCTopologyConstants );
  descriptions.add( "hcalTrigTowerGeometrySLHC", descSLHC );
}

boost::shared_ptr<HcalTrigTowerGeometry>
HcalTrigTowerGeometryESProducer::produce( const CaloGeometryRecord & iRecord )
{
    const edm::ParameterSet hcalTopoConsts = m_pSet.getParameter<edm::ParameterSet>( "hcalTopologyConstants" );

    StringToEnumParser<HcalTopologyMode::Mode> parser;
    HcalTopologyMode::Mode mode = (HcalTopologyMode::Mode) parser.parseString(hcalTopoConsts.getParameter<std::string>("mode"));

    m_hcalTrigTowerGeom =
	boost::shared_ptr<HcalTrigTowerGeometry>( new HcalTrigTowerGeometry( new HcalTopology(mode,
											      hcalTopoConsts.getParameter<int>("maxDepthHB"),
											      hcalTopoConsts.getParameter<int>("maxDepthHE"))));

    return m_hcalTrigTowerGeom;
}

DEFINE_FWK_EVENTSETUP_MODULE( HcalTrigTowerGeometryESProducer );
