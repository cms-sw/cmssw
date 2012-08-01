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
  virtual void analyze( const edm::Event& event, const edm::EventSetup& eventSetup );
  virtual void beginJob( void );
  virtual void endJob( void );

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

  std::stringstream s;
  s << "cmsTGeoRecoGeom" << m_level << ".root";
  TFile file( s.str().c_str(), "RECREATE" );
   
  TTree *tree = new TTree( "idToGeo", "Raw detector id association with geometry" );
  UInt_t v_id;
  TString *v_path( new TString );
  char v_name[1000];
  Float_t v_vertex[24];
  Float_t v_params[9];

  tree->SetBranchStyle( 0 );
  tree->Branch( "id", &v_id, "id/i" );
  tree->Branch( "path", &v_name, "path/C" );
  tree->Branch( "points", &v_vertex, "points[24]/F" );
  tree->Branch( "topology", &v_params, "topology[9]/F" );

  for( std::map<unsigned int, FWTGeoRecoGeometry::Info>::const_iterator it = geoh.product()->idToName.begin(),
								       end = geoh.product()->idToName.end();
       it != end; ++it )
  {
    v_id = it->first;
    *v_path = it->second.name.c_str();
    for( unsigned int i = 0; i < 24; ++i )
      v_vertex[i] = it->second.points[i];
    for( unsigned int i = 0; i < 9; ++i )
      v_params[i] = it->second.topology[i];
    assert( it->second.name.size() < 1000 );
    strncpy( v_name, it->second.name.c_str(), 1000 );
    tree->Fill();
  }
  file.WriteTObject( &*geom );
  file.WriteTObject( tree );
  file.Close();
}

void 
DumpFWTGeoRecoGeometry::beginJob( void )
{}

void 
DumpFWTGeoRecoGeometry::endJob( void )
{}

DEFINE_FWK_MODULE( DumpFWTGeoRecoGeometry );
