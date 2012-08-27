#include "HcalGeometryESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

HcalGeometryESProducer::HcalGeometryESProducer( const edm::ParameterSet & p )
    :  m_pSet( p )
{
  setWhatProduced( this );
}

HcalGeometryESProducer::~HcalGeometryESProducer( void )
{}

void
HcalGeometryESProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) 
{
  edm::ParameterSetDescription hcalTopologyConstants;
  hcalTopologyConstants.add<std::string>( "mode", "HcalTopologyMode::LHC" );
  hcalTopologyConstants.add<int>( "maxDepthHB", 2 );
  hcalTopologyConstants.add<int>( "maxDepthHE", 3 );  
  descriptions.add( "hcalTopologyConstants", hcalTopologyConstants );

  edm::ParameterSetDescription hcalSLHCTopologyConstants;
  hcalSLHCTopologyConstants.add<std::string>( "mode", "HcalTopologyMode::SLHC" );
  hcalSLHCTopologyConstants.add<int>( "maxDepthHB", 7 );
  hcalSLHCTopologyConstants.add<int>( "maxDepthHE", 7 );
  descriptions.add( "hcalSLHCTopologyConstants", hcalSLHCTopologyConstants );
}

boost::shared_ptr<HcalGeometry>
HcalGeometryESProducer::produce( const IdealGeometryRecord & iRecord )
{
  HcalTopologyMode::Mode mode = HcalTopologyMode::LHC;
  int maxDepthHB = 2;
  int maxDepthHE = 3;
  if( m_pSet.exists( "hcalTopologyConstants" ))
  {
    const edm::ParameterSet hcalTopoConsts = m_pSet.getParameter<edm::ParameterSet>( "hcalTopologyConstants" );
    StringToEnumParser<HcalTopologyMode::Mode> parser;
    mode = (HcalTopologyMode::Mode) parser.parseString(hcalTopoConsts.getParameter<std::string>("mode"));
    maxDepthHB = hcalTopoConsts.getParameter<int>("maxDepthHB");
    maxDepthHE = hcalTopoConsts.getParameter<int>("maxDepthHE");
  }

  m_hcal = boost::shared_ptr<HcalGeometry>( new HcalGeometry( new HcalTopology( mode, maxDepthHB, maxDepthHE )));

  return m_hcal;
}

DEFINE_FWK_EVENTSETUP_MODULE( HcalGeometryESProducer );
