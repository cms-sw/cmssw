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
#include "TSystem.h"

class DumpFWRecoGeometry : public edm::EDAnalyzer
{
public:
  explicit DumpFWRecoGeometry( const edm::ParameterSet& config );
  virtual ~DumpFWRecoGeometry( void ) {}

private:
  virtual void analyze( const edm::Event& event, const edm::EventSetup& eventSetup ) override;
  virtual void beginJob( void ) override;
  virtual void endJob( void ) override;

  int m_level;
  std::string m_tag;
  std::string m_outputFileName;
};

DumpFWRecoGeometry::DumpFWRecoGeometry( const edm::ParameterSet& config )
  : m_level( config.getUntrackedParameter<int>( "level", 1 )),
    m_tag( config.getUntrackedParameter<std::string>( "tagInfo", "unknown" )),
    m_outputFileName( config.getUntrackedParameter<std::string>( "outputFileName", "cmsRecoGeo.root" ))
{}

void
DumpFWRecoGeometry::analyze( const edm::Event& event, const edm::EventSetup& eventSetup )
{
  using namespace edm;

  ESTransientHandle<FWRecoGeometry> geoh;
  eventSetup.get<FWRecoGeometryRecord>().get( geoh );
  TFile file( m_outputFileName.c_str(), "RECREATE" );

  TTree *tree = new TTree("idToGeo", "raw detector id association with geometry ANT");

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


  file.WriteTObject(new TNamed("CMSSW_VERSION", gSystem->Getenv( "CMSSW_VERSION" )));
  file.WriteTObject(new TNamed("tag", m_tag.c_str()));
  file.WriteTObject(&geoh.product()->extraDet, "ExtraDetectors");
  file.WriteTObject(new TNamed("PRODUCER_VERSION", "1")); // version 2 changes pixel parameters



  file.Close();
}

void 
DumpFWRecoGeometry::beginJob( void )
{}

void 
DumpFWRecoGeometry::endJob( void )
{}

DEFINE_FWK_MODULE( DumpFWRecoGeometry );
