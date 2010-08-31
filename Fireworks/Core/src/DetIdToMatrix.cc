#include "TFile.h"
#include "TTree.h"
#include "TEveGeoNode.h"
#include "TPRegexp.h"
#include "TSystem.h"
#include "TGeoArb8.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <algorithm>

DetIdToMatrix::DetIdToMatrix( void )
  : m_idToInfo( 260000 )
{}

DetIdToMatrix::~DetIdToMatrix( void )
{}

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
DetIdToMatrix::loadMap( const char* fileName )
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
	 for( unsigned int j = 0; j < 9; ++j )
	    m_idToInfo[i].translation[j] = translation[j];
      }
      if( loadMatrix )
      {
	 for( unsigned int j = 0; j < 9; ++j )
	    m_idToInfo[i].matrix[j] = matrix[j];
      }
   }
   file->Close();
}

void
DetIdToMatrix::initMap( const FWRecoGeom::InfoMap& map )
{
  FWRecoGeom::InfoMapItr begin = map.begin();
  FWRecoGeom::InfoMapItr end = map.end();
  unsigned int mapSize = map.size();
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
DetIdToMatrix::getMatrix( unsigned int id ) const
{
   std::map<unsigned int, TGeoMatrix*>::iterator mit = m_idToMatrix.find( id );
   if( mit != m_idToMatrix.end()) return mit->second;
   
   IdToInfoItr it = DetIdToMatrix::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geometry is found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      TGeoTranslation trans(( *it ).translation[0], ( *it ).translation[1], ( *it ).translation[2] );
      TGeoRotation rotation;
      const Double_t matrix[9] = { it->matrix[0], it->matrix[1], it->matrix[2],
				   it->matrix[3], it->matrix[4], it->matrix[5],
				   it->matrix[6], it->matrix[7], it->matrix[8]
      };
      rotation.SetMatrix( matrix );

      m_idToMatrix[id] = new TGeoCombiTrans( trans, rotation );
      return m_idToMatrix[id];
   }
}

const TGeoVolume*
DetIdToMatrix::getVolume( unsigned int id ) const
{
  std::cout << "DetIdToMatrix::getVolume: no volume!!!" << std::endl;
  return 0;
}

std::vector<unsigned int>
DetIdToMatrix::getMatchedIds( Detector det, SubDetector subdet ) const
{
   std::vector<unsigned int> ids;
   unsigned int mask = ( det << 4 ) | ( subdet );
   for( IdToInfoItr it = m_idToInfo.begin(), itEnd = m_idToInfo.end();
	it != itEnd; ++it )
   {
      if( DetIdToMatrix::match_id( *it, mask ))
	 ids.push_back(( *it ).id );
   }
   
   return ids;
}

TGeoShape*
DetIdToMatrix::getShape( unsigned int id ) const
{
   IdToInfoItr it = DetIdToMatrix::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geoemtry found for id " <<  id << std::endl;
      return 0;
   }
   else 
   {
      TEveGeoManagerHolder gmgr( TEveGeoShape::GetGeoMangeur());
      TGeoShape* geoShape = 0;
      if(( *it ).shape[0] == 1 ) 
      {
	 geoShape = new TGeoTrap(
	   ( *it ).shape[3], //dz
	   0, 	             //theta
	   0, 	             //phi
	   ( *it ).shape[4], //dy1
	   ( *it ).shape[1], //dx1
	   ( *it ).shape[2], //dx2
	   0, 	    	     //alpha1
	   ( *it ).shape[4], //dy2
	   ( *it ).shape[1], //dx3
	   ( *it ).shape[2], //dx4
	   0);               //alpha2
      }
      else
	 geoShape = new TGeoBBox(( *it ).shape[1], ( *it ).shape[2], ( *it ).shape[3] );
      
      return geoShape;
   }
}

TEveGeoShape*
DetIdToMatrix::getEveShape( unsigned int id  ) const
{
   IdToInfoItr it = DetIdToMatrix::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco geoemtry found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      TEveGeoManagerHolder gmgr( TEveGeoShape::GetGeoMangeur());
      TEveGeoShape* shape = new TEveGeoShape;
      TGeoShape* geoShape = getShape( id );
      shape->SetShape( geoShape );
      const TGeoMatrix* matrix = getMatrix( id );
      shape->SetTransMatrix( *matrix );
      return shape;
   }
}

const float*
DetIdToMatrix::getCorners( unsigned int id ) const
{
   // reco geometry points
   IdToInfoItr it = DetIdToMatrix::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco corners geometry is found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      return ( *it ).points;
   }
}

const float*
DetIdToMatrix::getParameters( unsigned int id ) const
{
   // reco geometry parameters
   IdToInfoItr it = DetIdToMatrix::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco parameters are found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      return ( *it ).parameters;
   }
}

const float*
DetIdToMatrix::getShapePars( unsigned int id ) const
{
   // reco geometry parameters
   IdToInfoItr it = DetIdToMatrix::find( id );
   if( it == m_idToInfo.end())
   {
      fwLog( fwlog::kWarning ) << "no reco parameters are found for id " <<  id << std::endl;
      return 0;
   }
   else
   {
      return ( *it ).shape;
   }
}

DetIdToMatrix::IdToInfoItr
DetIdToMatrix::find( unsigned int id ) const
{
  DetIdToMatrix::IdToInfoItr begin = m_idToInfo.begin();
  DetIdToMatrix::IdToInfoItr end = m_idToInfo.end();
  return std::lower_bound( begin, end, id );
}
