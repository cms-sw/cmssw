#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryParsFromDD.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"

#include <iostream>

using namespace std;

class GEMRecoIdealDBLoader : public edm::EDAnalyzer
{
public:
  
  explicit GEMRecoIdealDBLoader( const edm::ParameterSet& );
  ~GEMRecoIdealDBLoader( void );
  
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) {}
  virtual void endJob( void ) {};
};

GEMRecoIdealDBLoader::GEMRecoIdealDBLoader( const edm::ParameterSet& )
{
  std::cout << "GEMRecoIdealDBLoader::GEMRecoIdealDBLoader" << std::endl;
}

GEMRecoIdealDBLoader::~GEMRecoIdealDBLoader()
{
  std::cout << "GEMRecoIdealDBLoader::~GEMRecoIdealDBLoader" << std::endl;
}

void
GEMRecoIdealDBLoader::beginRun( const edm::Run&, edm::EventSetup const& es ) 
{
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable())
  {
    edm::LogError( "GEMRecoIdealDBLoader" ) << "PoolDBOutputService unavailable";
    return;
  }

  if( mydbservice->isNewTagRequest( "GEMRecoGeometryRcd" ))
  {
    edm::ESTransientHandle<DDCompactView> pDD;
    edm::ESHandle<MuonDDDConstants> pMNDC;
    es.get<IdealGeometryRecord>().get( pDD );
    es.get<MuonNumberingRecord>().get( pMNDC );

    const DDCompactView& cpv = *pDD;
    GEMGeometryParsFromDD rpcpd;

    RecoIdealGeometry* rig = new RecoIdealGeometry;
    rpcpd.build( &cpv, *pMNDC, *rig );

    mydbservice->createNewIOV<RecoIdealGeometry>( rig,
						  mydbservice->beginOfTime(),                                                 
						  mydbservice->endOfTime(),
						  "GEMRecoGeometryRcd" );
  }
  else
  {
    edm::LogError( "GEMRecoIdealDBLoader" ) << "GEMRecoGeometryRcd Tag is already present";
  }
}

DEFINE_FWK_MODULE( GEMRecoIdealDBLoader );
