#ifndef Fireworks_Core_DetIdToMatrix_h
#define Fireworks_Core_DetIdToMatrix_h
//
//  Description: service class that implements access to geometry of a given DetId
//
//  Primary usage: raw detector id to full transformation matrix mapping
//
//  Original Author: D.Kovalskyi
//
class TGeoManager;
class TGeoVolume;
class TEveGeoShape;
class TFile;

#include "TEveVSDStructs.h"
#include "TGeoMatrix.h"

#include "Fireworks/Core/interface/FWRecoGeom.h"

#include <map>
#include <string>
#include <vector>

class DetIdToMatrix
{
public:
   static const int kDetOffset          = 28;
   static const int kSubdetOffset       = 25;

   enum Detector { Tracker = 1, Muon = 2, Ecal = 3, Hcal = 4, Calo = 5 };
   enum SubDetector { PixelBarrel = 1, PixelEndcap = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6, CSC = 7, DT = 8, RPCBarrel = 9, RPCEndcap = 10 };

   struct Range {
      double min1;
      double max1;
      double min2;
      double max2;
      Range( void ) : min1( 9999 ), max1( -9999 ), min2( 9999 ), max2( -9999 ) {
      }
   };

   DetIdToMatrix( void )
     : m_manager( 0 )       
    {
      m_idToInfo.reserve( 260000 );
    }
   ~DetIdToMatrix( void );

   // load full CMS geometry
   void loadGeometry( const char* fileName );

   // load DetId to RecoGeomInfo map
   void loadMap( const char* fileName );

   void initMap( FWRecoGeom::InfoMapItr begin, FWRecoGeom::InfoMapItr end );

   void manager( TGeoManager* geom )
    {
      m_manager = geom;
    }

   // get matrix for full local to global transformation
   const TGeoHMatrix* getMatrix( unsigned int id ) const;

   // get path to the detector volume in the geometry manager
   const char* getPath( unsigned int id ) const;

   static TFile* findFile( const char* fileName );

   // extract globally positioned shape for stand alone use
   // note: transformations are fixed for known differences
   //       between Sim and Reco geometries
   TEveGeoShape* getShape( unsigned int id, bool corrected = false  ) const;

   // extract globally positioned shape for stand alone use
   // note: if matrix is not provided, it will be extracted from
   //       the geo manager and no correction will be applied
   //       to fix Sim/Reco geometry differences.
   //       For expert use only!
   TEveGeoShape* getShape( const char* path, const char* name,
                           const TGeoMatrix* matrix = 0 ) const;

   // get the detector volume in the geometry manager
   const TGeoVolume* getVolume( unsigned int id ) const;

   // get all known detector ids with path matching regular expression
   std::vector<unsigned int> getMatchedIds( const char* selection ) const;

   // get all known detector ids with id matching mask
   std::vector<unsigned int> getMatchedIds( Detector det, SubDetector subdet ) const;

   // get reco geometry
   const std::vector<TEveVector>& getPoints( unsigned int id ) const;

   // get reco geometry
   const float* getCorners( unsigned int id ) const;

   // get reco topology/parameters
   const float* getParameters( unsigned int id ) const;

   TGeoManager* getManager( void ) const {
      return m_manager;
   }
  
   struct RecoGeomInfo
   {
      unsigned int id;
      std::string path;
      float points[24];
      float parameters[9];

      bool operator< ( const RecoGeomInfo& o ) const { 
         return ( this->id < o.id );
      }
   };

   struct find_id : std::unary_function<RecoGeomInfo, bool> {
      unsigned int id;
      find_id( unsigned int id ) : id( id ) {}
      bool operator () ( const RecoGeomInfo& o ) const {
         return o.id == id;
      }
   };
  
   typedef std::vector<DetIdToMatrix::RecoGeomInfo> IdToInfo;
   typedef std::vector<DetIdToMatrix::RecoGeomInfo>::const_iterator IdToInfoItr;

private:
  
   mutable std::map<unsigned int, TGeoHMatrix> m_idToMatrix;
  
   IdToInfo m_idToInfo;
  
   mutable TGeoManager* m_manager;
  
   IdToInfoItr find( unsigned int ) const;

   std::vector<TEveVector> m_eveVector;
};

#endif

