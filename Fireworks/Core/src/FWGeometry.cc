#include "TFile.h"
#include "TTree.h"
#include "TEveGeoNode.h"
#include "TPRegexp.h"
#include "TSystem.h"
#include "TGeoArb8.h"
#include "TObjArray.h"

#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <algorithm>

FWGeometry::FWGeometry( void )
{}

FWGeometry::~FWGeometry( void )
{}

TFile*
FWGeometry::findFile( const char* fileName )
{
   std::string searchPath = ".";

   if (gSystem->Getenv( "CMSSW_SEARCH_PATH" ))
   {    

       TString paths = gSystem->Getenv( "CMSSW_SEARCH_PATH" );

       TObjArray* tokens = paths.Tokenize( ":" );
       for( int i = 0; i < tokens->GetEntries(); ++i )
       {
           TObjString* path = (TObjString*)tokens->At( i );
           searchPath += ":";
           searchPath += path->GetString();
           if (gSystem->Getenv("CMSSW_VERSION"))
               searchPath += "/Fireworks/Geometry/data/";
       }
   }

   TString fn = fileName;
   const char* fp = gSystem->FindFile(searchPath.c_str(), fn, kFileExists);
   return fp ? TFile::Open( fp) : 0;
}

void
FWGeometry::loadMap( const char* fileName )
{  
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
   Float_t points[24];
   Float_t topology[9];
   Float_t shape[5];
   Float_t translation[3];
   Float_t matrix[9];
   bool loadPoints = tree->GetBranch( "points" ) != 0;
   bool loadParameters = tree->GetBranch( "topology" ) != 0;
   bool loadShape = tree->GetBranch( "shape" ) != 0;
   bool loadTranslation = tree->GetBranch( "translation" ) != 0;
   bool loadMatrix = tree->GetBranch( "matrix" ) != 0;
   tree->SetBranchAddress( "id", &id );
   if( loadPoints )
      tree->SetBranchAddress( "points", &points );
   if( loadParameters )
      tree->SetBranchAddress( "topology", &topology );
   if( loadShape )
      tree->SetBranchAddress( "shape", &shape );
   if( loadTranslation )
      tree->SetBranchAddress( "translation", &translation );
   if( loadMatrix )
      tree->SetBranchAddress( "matrix", &matrix );
   
   unsigned int treeSize = tree->GetEntries();
   if( m_idToInfo.size() != treeSize )
       m_idToInfo.resize( treeSize );
   for( unsigned int i = 0; i < treeSize; ++i )
   {
      tree->GetEntry( i );

      m_idToInfo[i].id = id;
      if( loadPoints )
      {
	 for( unsigned int j = 0; j < 24; ++j )
	    m_idToInfo[i].points[j] = points[j];
      }
      if( loadParameters )
      {
	 for( unsigned int j = 0; j < 9; ++j )
	    m_idToInfo[i].parameters[j] = topology[j];
      }
      if( loadShape )
      {
	 for( unsigned int j = 0; j < 5; ++j )
	    m_idToInfo[i].shape[j] = shape[j];
      }
      if( loadTranslation )
      {
	 for( unsigned int j = 0; j < 3; ++j )
	    m_idToInfo[i].translation[j] = translation[j];
      }
      if( loadMatrix )
      {
	 for( unsigned int j = 0; j < 9; ++j )
	    m_idToInfo[i].matrix[j] = matrix[j];
      }
   }


   m_versionInfo.productionTag  = static_cast<TNamed*>(file->Get( "tag" ));
   m_versionInfo.cmsswVersion   = static_cast<TNamed*>(file->Get( "CMSSW_VERSION" ));
   m_versionInfo.extraDetectors = static_cast<TObjArray*>(file->Get( "ExtraDetectors" ));
  
   TString path = file->GetPath();
   if (path.EndsWith(":/"))  path.Resize(path.Length() -2);

   if (m_versionInfo.productionTag)
      fwLog( fwlog::kInfo ) << Form("Load %s %s from %s\n",  tree->GetName(),  m_versionInfo.productionTag->GetName(), path.Data());  
   else 
      fwLog( fwlog::kInfo ) << Form("Load %s from %s\n",  tree->GetName(), path.Data());  

   file->Close();
}

void
FWGeometry::initMap( const FWRecoGeom::InfoMap& map )
{
  FWRecoGeom::InfoMapItr begin = map.begin();
  FWRecoGeom::InfoMapItr end = map.end();
  unsigned int mapSize = map.size();
  if( m_idToInfo.size() != mapSize )
    m_idToInfo.resize( mapSize );
  unsigned int i = 0;
  for( FWRecoGeom::InfoMapItr it = begin;
       it != end; ++it, ++i )
  {
    m_idToInfo[i].id = it->id;
    for( unsigned int j = 0; j < 24; ++j )
      m_idToInfo[i].points[j] = it->points[j];
    for( unsigned int j = 0; j < 9; ++j )
      m_idToInfo[i].parameters[j] = it->topology[j];
    for( unsigned int j = 0; j < 5; ++j )
      m_idToInfo[i].shape[j] = it->shape[j];
    for( unsigned int j = 0; j < 3; ++j )
      m_idToInfo[i].translation[j] = it->translation[j];
    for( unsigned int j = 0; j < 9; ++j )
      m_idToInfo[i].matrix[j] = it->matrix[j];
  }
}

const TGeoMatrix*
FWGeometry::getMatrix( unsigned int id ) const
{
   std::map<unsigned int, TGeoMatrix*>::iterator mit = m_idToMatrix.find( id );
   if( mit != m_idToMatrix.end()) return mit->second;
   
   IdToInfoItr it = FWGeometry::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geometry found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      const GeomDetInfo& info = *it;
      TGeoTranslation trans( info.translation[0], info.translation[1], info.translation[2] );
      TGeoRotation rotation;
      const Double_t matrix[9] = { info.matrix[0], info.matrix[1], info.matrix[2],
				   info.matrix[3], info.matrix[4], info.matrix[5],
				   info.matrix[6], info.matrix[7], info.matrix[8]
      };
      rotation.SetMatrix( matrix );

      m_idToMatrix[id] = new TGeoCombiTrans( trans, rotation );
      return m_idToMatrix[id];
   }
}

std::vector<unsigned int>
FWGeometry::getMatchedIds( Detector det, SubDetector subdet ) const
{
   std::vector<unsigned int> ids;
   unsigned int mask = ( det << 4 ) | ( subdet );
   for( IdToInfoItr it = m_idToInfo.begin(), itEnd = m_idToInfo.end();
	it != itEnd; ++it )
   {
      if( FWGeometry::match_id( *it, mask ))
	 ids.push_back(( *it ).id );
   }
   
   return ids;
}

TGeoShape*
FWGeometry::getShape( unsigned int id ) const
{
   IdToInfoItr it = FWGeometry::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geoemtry found for id " <<  id << std::endl;
      return 0;
   }
   else 
   {
      return getShape( *it );
   }
}

TGeoShape*
FWGeometry::getShape( const GeomDetInfo& info ) const 
{
   TEveGeoManagerHolder gmgr( TEveGeoShape::GetGeoMangeur());
   TGeoShape* geoShape = 0;
   if( info.shape[0] == 1 ) 
   {
      geoShape = new TGeoTrap(
	 info.shape[3], //dz
	 0,             //theta
	 0,             //phi
	 info.shape[4], //dy1
	 info.shape[1], //dx1
	 info.shape[2], //dx2
	 0,             //alpha1
	 info.shape[4], //dy2
	 info.shape[1], //dx3
	 info.shape[2], //dx4
	 0);            //alpha2
   }
   else
      geoShape = new TGeoBBox( info.shape[1], info.shape[2], info.shape[3] );
      
   return geoShape;
}

TEveGeoShape*
FWGeometry::getEveShape( unsigned int id  ) const
{
   IdToInfoItr it = FWGeometry::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geoemtry found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      const GeomDetInfo& info = *it;
      double array[16] = { info.matrix[0], info.matrix[3], info.matrix[6], 0.,
			   info.matrix[1], info.matrix[4], info.matrix[7], 0.,
			   info.matrix[2], info.matrix[5], info.matrix[8], 0.,
			   info.translation[0], info.translation[1], info.translation[2], 1.
      };
      TEveGeoManagerHolder gmgr( TEveGeoShape::GetGeoMangeur());
      TEveGeoShape* shape = new TEveGeoShape(TString::Format("RecoGeom Id=%u", id));
      TGeoShape* geoShape = getShape( info );
      shape->SetShape( geoShape );
      // Set transformation matrix from a column-major array
      shape->SetTransMatrix( array );
      return shape;
   }
}

const float*
FWGeometry::getCorners( unsigned int id ) const
{
   // reco geometry points
   IdToInfoItr it = FWGeometry::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geometry found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      return ( *it ).points;
   }
}

const float*
FWGeometry::getParameters( unsigned int id ) const
{
   // reco geometry parameters
   IdToInfoItr it = FWGeometry::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geometry found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      return ( *it ).parameters;
   }
}

const float*
FWGeometry::getShapePars( unsigned int id ) const
{
   // reco geometry parameters
   IdToInfoItr it = FWGeometry::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geometry found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      return ( *it ).shape;
   }
}

void
FWGeometry::localToGlobal( unsigned int id, const float* local, float* global, bool translatep ) const
{
   IdToInfoItr it = FWGeometry::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geometry found for id " <<  id << std::endl;
   }
   else
   {
      localToGlobal( *it, local, global, translatep );
   }
}

void
FWGeometry::localToGlobal( unsigned int id, const float* local1, float* global1,
                           const float* local2, float* global2, bool translatep ) const
{
   IdToInfoItr it = FWGeometry::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geometry found for id " <<  id << std::endl;
   }
   else
   {
      localToGlobal( *it, local1, global1, translatep );
      localToGlobal( *it, local2, global2, translatep );
   }
}

FWGeometry::IdToInfoItr
FWGeometry::find( unsigned int id ) const
{
  FWGeometry::IdToInfoItr begin = m_idToInfo.begin();
  FWGeometry::IdToInfoItr end = m_idToInfo.end();
  return std::lower_bound( begin, end, id );
}

void
FWGeometry::localToGlobal( const GeomDetInfo& info, const float* local, float* global, bool translatep ) const
{
   for( int i = 0; i < 3; ++i )
   {
      global[i]  = translatep ? info.translation[i] : 0;
      global[i] +=   local[0] * info.matrix[3 * i]
		   + local[1] * info.matrix[3 * i + 1]
		   + local[2] * info.matrix[3 * i + 2];
   }
}

//______________________________________________________________________________

bool FWGeometry::VersionInfo::haveExtraDet(const char* det) const
{
   
   return (extraDetectors && extraDetectors->FindObject(det)) ? true : false;
}
