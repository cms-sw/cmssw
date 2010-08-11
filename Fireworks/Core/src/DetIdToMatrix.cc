#include "TGeoManager.h"
#include "TFile.h"
#include "TTree.h"
#include "TEveGeoNode.h"
#include "TPRegexp.h"
#include "TSystem.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>
#include <cassert>
#include <sstream>
#include <stdexcept>

DetIdToMatrix::~DetIdToMatrix( void )
{
   // ATTN: not sure I own the manager
   // if ( m_manager ) delete m_manager;
}

TFile*
DetIdToMatrix::findFile( const char* fileName )
{
   TString file;
   if( fileName[0] == '/' )
   {
      file = fileName;
   }
   else
   {
      if( const char* cmspath = gSystem->Getenv( "CMSSW_BASE" ))
      {
         file += cmspath;
         file += "/";
      }
      file += fileName;
   }
   if( !gSystem->AccessPathName( file.Data()))
   {
      return TFile::Open( file );
   }

   const char* searchpath = gSystem->Getenv( "CMSSW_SEARCH_PATH" );
   if( searchpath == 0 )
     return 0;
   TString paths( searchpath );
   TObjArray* tokens = paths.Tokenize( ":" );
   for( int i = 0; i < tokens->GetEntries(); ++i )
   {
      TObjString* path = (TObjString*)tokens->At( i );
      TString fullFileName( path->GetString());
      fullFileName += "/Fireworks/Geometry/data/";
      fullFileName += fileName;
      if( !gSystem->AccessPathName( fullFileName.Data()))
         return TFile::Open( fullFileName.Data());
   }
   return 0;
}

void
DetIdToMatrix::loadGeometry( const char* fileName )
{
   m_manager = 0;
   if( TFile* file = findFile( fileName ))
   {
      // load geometry
      m_manager = static_cast<TGeoManager*>(file->Get( "cmsGeo" ));
      file->Close();
   }
   else
   {
      throw std::runtime_error( "ERROR: failed to find geometry file. Initialization failed." );
   }
   if( !m_manager )
   {
      throw std::runtime_error( "ERROR: cannot find geometry in the file. Initialization failed." );
      return;
   }
}

void
DetIdToMatrix::loadMap( const char* fileName )
{
   if( ! m_manager )
   {
      throw std::runtime_error( "ERROR: CMS detector geometry is not available. DetId to Matrix map Initialization failed." );
      return;
   }

   TFile* file = findFile( fileName );
   if( ! file )
   {
      throw std::runtime_error( "ERROR: failed to find geometry file. Initialization failed." );
      return;
   }
   TTree* tree = static_cast<TTree*>(file->Get( "idToGeo" ));
   if( ! tree )
   {
      throw std::runtime_error( "ERROR: cannot find detector id map in the file. Initialization failed." );
      return;
   }
   
   unsigned int id;
   char path[1000];
   Float_t points[24];
   Float_t topology[9];
   bool loadPoints = tree->GetBranch( "points" ) != 0;
   bool loadParameters = tree->GetBranch( "topology" ) != 0;
   tree->SetBranchAddress( "id", &id );
   tree->SetBranchAddress( "path", &path );
   if( loadPoints )
      tree->SetBranchAddress( "points", &points );
   if( loadParameters )
      tree->SetBranchAddress( "topology", &topology );
   
   for( unsigned int i = 0; i < tree->GetEntries(); ++i)
   {
      tree->GetEntry( i );
      m_idToInfo[id].path = path;
      if( loadPoints )
      {
         std::vector<Float_t> p( 24 );
	 for( unsigned int j = 0; j < 24; ++j )
	    p[j] = points[j];
	 m_idToInfo[id].points.swap( p );
      }
      if( loadParameters )
      {
         std::vector<Float_t> t( 9 );
	 for( unsigned int j = 0; j < 9; ++j )
	    t[j] = topology[j];
	 m_idToInfo[id].parameters.swap( t );
      }      
   }
   file->Close();
}

void
DetIdToMatrix::initMap( FWRecoGeom::InfoMap imap )
{
  for( std::map<unsigned int, FWRecoGeom::Info>::const_iterator it = imap.begin(),
							       end = imap.end();
       it != end; ++it )
  {
    unsigned int rawid = it->first;
    m_idToInfo[rawid].path = it->second.name;
    std::vector<float> points = it->second.points;
    m_idToInfo[rawid].points.swap( points );
    std::vector<float> topology = it->second.topology;
    m_idToInfo[rawid].parameters.swap( topology );
  }
}

const TGeoHMatrix*
DetIdToMatrix::getMatrix( unsigned int id ) const
{
   std::map<unsigned int, TGeoHMatrix>::const_iterator it = m_idToMatrix.find( id );
   if( it != m_idToMatrix.end()) return &( it->second );

   const char* path = getPath( id );
   if( !path )
      return 0;
   if( !m_manager->cd(path))
   {
      DetId detId( id );
      fwLog( fwlog::kError ) << "incorrect path " << path << "\nfor DetId: " << detId.det() << " : " << id << std::endl;
      return 0;
   }

   TGeoHMatrix m = *( m_manager->GetCurrentMatrix());

   m_idToMatrix[id] = m;
   return &m_idToMatrix[id];
}

const char*
DetIdToMatrix::getPath( unsigned int id ) const
{
   std::map<unsigned int, RecoGeomInfo>::const_iterator it = m_idToInfo.find( id );
   if( it != m_idToInfo.end())
      return it->second.path.c_str();
   else
      return 0;
}

const TGeoVolume*
DetIdToMatrix::getVolume( unsigned int id ) const
{
   std::map<unsigned int, RecoGeomInfo>::const_iterator it = m_idToInfo.find( id );
   if( it != m_idToInfo.end())
   {
      m_manager->cd( it->second.path.c_str());
      return m_manager->GetCurrentVolume();
   }
   else
      return 0;
}

std::vector<unsigned int>
DetIdToMatrix::getMatchedIds( const char* regular_expression ) const
{
   std::vector<unsigned int> ids;
   TPRegexp regexp( regular_expression );
   for( std::map<unsigned int, RecoGeomInfo>::const_iterator it = m_idToInfo.begin(), itEnd = m_idToInfo.end();
	it != itEnd; ++it )
      if( regexp.MatchB( it->second.path )) ids.push_back( it->first );
   return ids;
}


TEveGeoShape*
DetIdToMatrix::getShape( const char* path, const char* name, const TGeoMatrix* matrix /* = 0 */) const
{
   if( ! m_manager || ! path || ! name )
      return 0;
   m_manager->cd( path );
   // it's possible to get a corrected matrix from outside
   // if it's not provided, we take whatever the geo manager has
   if( ! matrix )
      matrix = m_manager->GetCurrentMatrix();

   TEveGeoShape* shape = new TEveGeoShape( name, path );
   shape->SetElementTitle( name );
   shape->SetTransMatrix( *matrix );
   TGeoShape* gs = m_manager->GetCurrentVolume()->GetShape();
   
   //------------------------------------------------------------------------------//
   // FIXME !!!!!!!!!!!!!!
   // hack zone to make CSC complex shape visible
   // loop over bool shapes till we get something non-composite on the left side.
   if( TGeoCompositeShape* composite = dynamic_cast<TGeoCompositeShape*>( gs ))
   {
      int depth = 0;
      TGeoShape* nextShape( gs );
      do
      {
	 if( depth > 10 ) break;
	 nextShape = composite->GetBoolNode()->GetLeftShape();
	 composite = dynamic_cast<TGeoCompositeShape*>( nextShape );
	 ++depth;
      } while( depth < 10 && composite != 0 );
      if( composite == 0 )
         gs = nextShape;
   }
   //------------------------------------------------------------------------------//
   UInt_t id = TMath::Max( gs->GetUniqueID(), UInt_t( 1 ));
   gs->SetUniqueID( id );
   shape->SetShape( gs );
   TGeoVolume* volume = m_manager->GetCurrentVolume();
   shape->SetMainColor( volume->GetLineColor());
   shape->SetRnrSelf( kTRUE );
   shape->SetRnrChildren( kTRUE );
   return shape;
}

TEveGeoShape*
DetIdToMatrix::getShape( unsigned int id,
			 bool corrected /* = false */ ) const
{
   std::ostringstream s;
   s << id;
   if( corrected )
      return getShape( getPath(id), s.str().c_str(), getMatrix( id ));
   else
      return getShape( getPath(id), s.str().c_str());
}

std::vector<TEveVector>
DetIdToMatrix::getPoints( unsigned int id ) const
{
   // reco geometry points
   std::map<unsigned int, RecoGeomInfo>::const_iterator it = m_idToInfo.find( id );
   if( it == m_idToInfo.end())
   {
      fwLog(fwlog::kWarning) << "no reco geometry is found for id " <<  id << std::endl;
      return std::vector<TEveVector>();
   }
   else
   {
      if( it->second.corners.empty() && ! it->second.points.empty())
      {
	 fillCorners( id );
      }

      return it->second.corners;
   }
}

std::vector<Float_t>
DetIdToMatrix::getParameters( unsigned int id ) const
{
   // reco geometry parameters
   std::map<unsigned int, RecoGeomInfo>::const_iterator it = m_idToInfo.find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco parameters are found for id " <<  id << std::endl;
      return std::vector<Float_t>();
   }
   else
   {
      return it->second.parameters;
   }
}

void
DetIdToMatrix::fillCorners( unsigned int id ) const
{
   std::vector<TEveVector> p(8);
   for( unsigned int j = 0; j < 8; ++j )
      p[j].Set( m_idToInfo[id].points[3 * j], m_idToInfo[id].points[3 * j + 1], m_idToInfo[id].points[3 * j + 2] );
   m_idToInfo[id].corners.swap( p );
}
