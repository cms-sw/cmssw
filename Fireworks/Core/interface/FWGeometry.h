#ifndef Fireworks_Core_FWGeometry_h
#define Fireworks_Core_FWGeometry_h
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
class TObjArray;

#include <map>
#include <vector>
#include <memory>


#include "TEveVSDStructs.h"
#include "TGeoMatrix.h"
#include "TGeoXtru.h"

#include "Fireworks/Core/interface/FWRecoGeom.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
class FWGeometry
{
public:
   static const int kDetOffset          = 28;
   static const int kSubdetOffset       = 25;

   enum Detector { Tracker = 1, Muon = 2, Ecal = 3, Hcal = 4, Calo = 5, HGCalEE=8, HGCalHSi=9, HGCalHSc=10, HGCalTrigger=11 };
   enum SubDetector { PixelBarrel = 1, PixelEndcap = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6, CSC = 7, DT = 8, RPCBarrel = 9, RPCEndcap = 10, GEM = 11, ME0 = 12};

   struct Range {
      double min1;
      double max1;
      double min2;
      double max2;
      Range( void ) : min1( 9999 ), max1( -9999 ), min2( 9999 ), max2( -9999 ) {
      }
   };

   class VersionInfo {
   public:
      TNamed* productionTag;
      TNamed* cmsswVersion;
      TObjArray* extraDetectors;

      VersionInfo() : productionTag(nullptr), cmsswVersion(nullptr), extraDetectors(nullptr) {}
      bool haveExtraDet(const char*)const;
   };

   FWGeometry( void );

   ~FWGeometry( void );

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
   TEveGeoShape* getHGCSiliconEveShape( unsigned int id  ) const;
   TEveGeoShape* getHGCScintillatorEveShape( unsigned int id  ) const;
  
   // get shape description parameters
   const float* getShapePars( unsigned int id  ) const;

   // get all known detector ids with id matching mask
   std::vector<unsigned int> getMatchedIds( Detector det, SubDetector subdet ) const;
   std::vector<unsigned int> getMatchedIds( Detector det ) const;

   // get reco geometry
   const float* getCorners( unsigned int id ) const;

   // get reco topology/parameters
   const float* getParameters( unsigned int id ) const;

   void localToGlobal( unsigned int id, const float* local,  float* global, bool translatep=true ) const;
   void localToGlobal( unsigned int id, const float* local1, float* global1, const float* local2, float* global2, bool translatep=true ) const;

   struct GeomDetInfo
   {
      unsigned int id;       // DetId
      float points[24];      // 3*8 x,y,z points defining its shape (can be undefined, e.g. 0s) 
      float parameters[9];   // specific DetId dependent parameters, e.g. topology (can be undefined, e.g. 0s)
      float shape[5];        // shape description: 0 - shape type, For Trap: 1 - dx1, 2 - dx2, 3 - dz, 4 - dy1; for Box: dx, dy, dz (can be undefined, e.g. 0s) 
      float translation[3];  // translation x, y, z (can be undefined, e.g. 0s) 
      float matrix[9];       // transformation matrix xx, yx, zx, xy, yy, zy, xz, yz, zz (can be undefined, e.g. 0s) 
     
      bool operator< ( unsigned int id ) const { 
         return ( this->id < id );
      }
   };
  
   bool match_id( const GeomDetInfo& o, unsigned int mask ) const {
     unsigned int id = o.id;
     return ((((( id >> kDetOffset ) & 0xF ) << 4) | (( id >> kSubdetOffset ) & 0x7 )) == mask );
   }
  
   typedef std::vector<FWGeometry::GeomDetInfo> IdToInfo;
   typedef std::vector<FWGeometry::GeomDetInfo>::const_iterator IdToInfoItr;


   bool contains( unsigned int id ) const {
     return FWGeometry::find( id ) != m_idToInfo.end();
   }

   IdToInfoItr mapEnd() const {return m_idToInfo.end();}

   void clear( void ) { m_idToInfo.clear(); m_idToMatrix.clear(); }
   IdToInfoItr find( unsigned int ) const;
   void localToGlobal( const GeomDetInfo& info, const float* local, float* global, bool translatep=true ) const;

   const VersionInfo& versionInfo() const { return m_versionInfo; }

   int getProducerVersion() const {return m_producerVersion;}
   
   TGeoShape* getShape( const GeomDetInfo& info ) const;

   const TrackerTopology* getTrackerTopology() const { return m_trackerTopology.get(); }

private:
   mutable std::map<unsigned int, TGeoMatrix*> m_idToMatrix;

   IdToInfo m_idToInfo;

   std::string m_prodTag;

   VersionInfo  m_versionInfo;

   int m_producerVersion;

   std::unique_ptr<TrackerTopology> m_trackerTopology;
};

#endif

