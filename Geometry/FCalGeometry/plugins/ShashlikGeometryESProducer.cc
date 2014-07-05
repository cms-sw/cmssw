#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/FCalGeometry/interface/ShashlikGeometry.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include "Geometry/FCalGeometry/src/ShashlikGeometryBuilderFromDDD.h"

class ShashlikGeometryESProducer : public edm::ESProducer
{
public:

  ShashlikGeometryESProducer( const edm::ParameterSet & );
  virtual ~ShashlikGeometryESProducer( void );

  typedef boost::shared_ptr<CaloSubdetectorGeometry> ReturnType;
  
  ReturnType produce( const ShashlikGeometryRecord & );

  static void fillDescriptions( edm::ConfigurationDescriptions & );
};

ShashlikGeometryESProducer::ShashlikGeometryESProducer( const edm::ParameterSet & )
{
  std::cout << "ShashlikGeometryESProducer loaded.\n";
  setWhatProduced( this );
}

ShashlikGeometryESProducer::~ShashlikGeometryESProducer( void ) 
{}

ShashlikGeometryESProducer::ReturnType
ShashlikGeometryESProducer::produce( const ShashlikGeometryRecord & record )
{
  std::cout << "ShashlikGeometryESProducer::produce Shashlik geometry.\n";
  
  edm::ESTransientHandle<DDCompactView> cpv;
  record.getRecord<IdealGeometryRecord>().get( cpv );
  edm::ESHandle<ShashlikTopology> shdc;
  record.getRecord<ShashlikNumberingRecord>().get( shdc );
  ShashlikGeometryBuilderFromDDD builder;
  return ReturnType( builder.build( &(*cpv), *shdc ));
}

void
ShashlikGeometryESProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
{
  edm::ParameterSetDescription desc;
  descriptions.add( "ShashlikGeometryESProducer", desc );
}

DEFINE_FWK_EVENTSETUP_MODULE( ShashlikGeometryESProducer );
