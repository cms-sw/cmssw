#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Fireworks/Geometry/interface/FWTGeoRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWTGeoRecoGeometryRecord.h"

#include "TGeoManager.h"
#include "TFile.h"
#include "TTree.h"
#include "TError.h"

class DumpFWTGeoRecoGeometry : public edm::EDAnalyzer
{
public:
  explicit DumpFWTGeoRecoGeometry( const edm::ParameterSet& config );
  virtual ~DumpFWTGeoRecoGeometry( void ) {}

private:
  virtual void analyze( const edm::Event& event, const edm::EventSetup& eventSetup ) override;
  virtual void beginJob( void ) override;
  virtual void endJob( void ) override;

  int m_level;
};

DumpFWTGeoRecoGeometry::DumpFWTGeoRecoGeometry( const edm::ParameterSet& config )
  : m_level( config.getUntrackedParameter<int>( "level", 1 ))
{}

void
DumpFWTGeoRecoGeometry::analyze( const edm::Event& event, const edm::EventSetup& eventSetup )
{
  using namespace edm;

  ESTransientHandle<FWTGeoRecoGeometry> geoh;
  eventSetup.get<FWTGeoRecoGeometryRecord>().get( geoh );
  TGeoManager *geom = geoh.product()->manager();//const_cast<TGeoManager*>( geoh.product()->manager());

  TFile file( "cmsTRecoGeom.root", "RECREATE" );
   
 
  file.WriteTObject( &*geom );
  file.Close();
}

void 
DumpFWTGeoRecoGeometry::beginJob( void )
{}

void 
DumpFWTGeoRecoGeometry::endJob( void )
{}

DEFINE_FWK_MODULE( DumpFWTGeoRecoGeometry );
