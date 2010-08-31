#ifndef Fireworks_Core_DetIdToMatrix_h
#define Fireworks_Core_DetIdToMatrix_h
//
//  Description: service class that implements access to geometry of a given DetId
//
//  Primary usage: raw detector id to full transformation matrix mapping
//
//  Original Author: D.Kovalskyi
//
class TEveGeoShape;
class TGeoVolume;
class TGeoShape;
class TFile;

#include "TEveVSDStructs.h"
#include "TGeoMatrix.h"

#include "Fireworks/Core/interface/FWRecoGeom.h"

#include <map>
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

   DetIdToMatrix( void );

   ~DetIdToMatrix( void );

   // load DetId to RecoGeomInfo map
   void loadMap( const char* fileName );

   void initMap( const FWRecoGeom::InfoMap& map );

   // get matrix for full local to global transformation
   const TGeoMatrix* getMatrix( unsigned int id ) const;

   static TFile* findFile( const char* fileName );

   // extract locally positioned shape for stand alone use
   TGeoShape* getShape( unsigned int id ) const;
  
   // extract globally positioned shape for stand alone use
   TEveGeoShape* getEveShape( unsigned int id  ) const;
  
   // get shape description parameters
   const float* getShapePars( unsigned int id  ) const;

   // get the detector volume in the geometry manager
   const TGeoVolume* getVolume( unsigned int id ) const;

   // get all known detector ids with id matching mask
   std::vector<unsigned int> getMatchedIds( Detector det, SubDetector subdet ) const;

   // get reco geometry
   const float* getCorners( unsigned int id ) const;

   // get reco topology/parameters
   const float* getParameters( unsigned int id ) const;

   struct RecoGeomInfo
   {
      unsigned int id;
      float points[24];
      float parameters[9];
      float shape[5];
      float translation[3];
      float matrix[9];
     
      bool operator< ( unsigned int id ) const { 
         return ( this->id < id );
      }
   };
  
   bool match_id( const RecoGeomInfo& o, unsigned int mask ) const {
     unsigned int id = o.id;
     return ((((( id >> kDetOffset ) & 0xF ) << 4) | (( id >> kSubdetOffset ) & 0x7 )) == mask );
   }
  
   typedef std::vector<DetIdToMatrix::RecoGeomInfo> IdToInfo;
   typedef std::vector<DetIdToMatrix::RecoGeomInfo>::const_iterator IdToInfoItr;

private:
   mutable std::map<unsigned int, TGeoMatrix*> m_idToMatrix;

   IdToInfo m_idToInfo;
  
   IdToInfoItr find( unsigned int ) const;
};

#endif

