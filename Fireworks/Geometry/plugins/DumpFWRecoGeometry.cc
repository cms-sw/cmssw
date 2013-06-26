#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Fireworks/Geometry/interface/FWRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWRecoGeometryRecord.h"

#include "TFile.h"
#include "TTree.h"
#include "TError.h"

class DumpFWRecoGeometry : public edm::EDAnalyzer
{
public:
  explicit DumpFWRecoGeometry( const edm::ParameterSet& config );
  virtual ~DumpFWRecoGeometry( void ) {}

private:
  virtual void analyze( const edm::Event& event, const edm::EventSetup& eventSetup );
  virtual void beginJob( void );
  virtual void endJob( void );

  int m_level;
};

DumpFWRecoGeometry::DumpFWRecoGeometry( const edm::ParameterSet& config )
  : m_level( config.getUntrackedParameter<int>( "level", 1 ))
{}

void
DumpFWRecoGeometry::analyze( const edm::Event& event, const edm::EventSetup& eventSetup )
{
  using namespace edm;

  ESTransientHandle<FWRecoGeometry> geoh;
  eventSetup.get<FWRecoGeometryRecord>().get( geoh );
  
  std::stringstream s;
  s << "cmsRecoGeom" << m_level << ".root";
  TFile file( s.str().c_str(), "RECREATE" );
   
  TTree *tree = new TTree( "idToGeo", "Raw detector id association with geometry" );
  UInt_t v_id;
  Float_t v_vertex[24];
  Float_t v_params[9];
  Float_t v_shape[5];
  Float_t v_translation[3];
  Float_t v_matrix[9];

  tree->SetBranchStyle( 0 );
  tree->Branch( "id", &v_id, "id/i" );
  tree->Branch( "points", &v_vertex, "points[24]/F" );
  tree->Branch( "topology", &v_params, "topology[9]/F" );
  tree->Branch( "shape", &v_shape, "shape[5]/F" );
  tree->Branch( "translation", &v_translation, "translation[3]/F" );
  tree->Branch( "matrix", &v_matrix, "matrix[9]/F" );

  for( FWRecoGeom::InfoMapItr it = geoh.product()->idToName.begin(),
			     end = geoh.product()->idToName.end();
       it != end; ++it )
  {
    v_id = it->id;
    for( unsigned int i = 0; i < 24; ++i )
      v_vertex[i] = it->points[i];
    for( unsigned int i = 0; i < 9; ++i )
      v_params[i] = it->topology[i];
    for( unsigned int i = 0; i < 5; ++i )
      v_shape[i] = it->shape[i];
    for( unsigned int i = 0; i < 3; ++i )
      v_translation[i] = it->translation[i];
    for( unsigned int i = 0; i < 9; ++i )
      v_matrix[i] = it->matrix[i];
    tree->Fill();
  }
  file.WriteTObject( tree );
  file.Close();
}

void 
DumpFWRecoGeometry::beginJob( void )
{}

void 
DumpFWRecoGeometry::endJob( void )
{}

DEFINE_FWK_MODULE( DumpFWRecoGeometry );
