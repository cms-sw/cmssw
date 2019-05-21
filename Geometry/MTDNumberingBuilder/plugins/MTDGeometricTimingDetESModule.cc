#include "Geometry/MTDNumberingBuilder/plugins/DDDCmsMTDConstruction.h"
#include "Geometry/MTDNumberingBuilder/plugins/CondDBCmsMTDConstruction.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>


class MTDGeometricTimingDetESModule : public edm::ESProducer
{
public:
  MTDGeometricTimingDetESModule( const edm::ParameterSet & p );

  std::unique_ptr<GeometricTimingDet> produce( const IdealGeometryRecord & );

  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );
  
private:
  const bool fromDDD_;

  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddCompactToken_;
  edm::ESGetToken<PGeometricTimingDet, IdealGeometryRecord> pGTDetToken_;
};

using namespace edm;

MTDGeometricTimingDetESModule::MTDGeometricTimingDetESModule( const edm::ParameterSet & p ) 
  : fromDDD_( p.getParameter<bool>( "fromDDD" ))
{
  auto cc = setWhatProduced( this );
  if(fromDDD_) {
    ddCompactToken_ = cc.consumes<DDCompactView>(edm::ESInputTag());
  } else {
    pGTDetToken_ = cc.consumes<PGeometricTimingDet>(edm::ESInputTag());
  }
}

void
MTDGeometricTimingDetESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
{
  edm::ParameterSetDescription descDB;
  descDB.add<bool>( "fromDDD", false );
  descriptions.add( "mtdNumberingGeometryDB", descDB );

  edm::ParameterSetDescription desc;
  desc.add<bool>( "fromDDD", true );
  descriptions.add( "mtdNumberingGeometry", desc );
}

std::unique_ptr<GeometricTimingDet> 
MTDGeometricTimingDetESModule::produce( const IdealGeometryRecord & iRecord )
{ 
  if( fromDDD_ )
  {
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle( ddCompactToken_ );
    return DDDCmsMTDConstruction::construct((*cpv), dbl_to_int( DDVectorGetter::get( "detIdShifts" )));
  }
  else
  {
    PGeometricTimingDet const& pgd = iRecord.get( pGTDetToken_ );
    return CondDBCmsMTDConstruction::construct( pgd );
  }
}

DEFINE_FWK_EVENTSETUP_MODULE( MTDGeometricTimingDetESModule );
