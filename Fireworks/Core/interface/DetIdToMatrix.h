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
    {}
   ~DetIdToMatrix( void );

   // load full CMS geometry
   void loadGeometry( const char* fileName );

   // load DetId to RecoGeomInfo map
   void loadMap( const char* fileName );

   void initMap( FWRecoGeom::InfoMap );

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

   // get reco geometry
   const std::vector<TEveVector>& getPoints( unsigned int id ) const;

   // get reco topology/parameters
   const std::vector<Float_t>& getParameters( unsigned int id ) const;

   TGeoManager* getManager( void ) const {
      return m_manager;
   }
private:
  void fillCorners( unsigned int id ) const;
  
  mutable std::map<unsigned int, TGeoHMatrix> m_idToMatrix;

  struct RecoGeomInfo
  {
    std::string path;
    std::vector<Float_t> points;
    std::vector<Float_t> parameters;
    std::vector<TEveVector> corners;
  };
  mutable std::map<unsigned int, RecoGeomInfo> m_idToInfo;
  
  mutable TGeoManager* m_manager;
  
  std::vector<TEveVector> m_eveVector;
  std::vector<Float_t> m_float;
};

#endif

