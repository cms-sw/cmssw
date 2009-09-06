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
class TEveElementList;
class TFile;

#include "TGeoMatrix.h"
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
      Range(): min1(9999),max1(-9999),min2(9999),max2(-9999){}
     };
   
   DetIdToMatrix() : manager_(0){
   }
   ~DetIdToMatrix();

   // load full CMS geometry
   void loadGeometry(const char* fileName);

   // load DetId to Matrix map
   void loadMap(const char* fileName);

   // get matrix for full local to global transformation
   const TGeoHMatrix* getMatrix( unsigned int id ) const;

   // get path to the detector volume in the geometry manager
   const char* getPath( unsigned int id ) const;
   
   static TFile* findFile(const char* fileName);

   // extract globally positioned shape for stand alone use
   // note: transformations are fixed for known differences
   //       between Sim and Reco geometries
   TEveGeoShape* getShape( unsigned int id, bool corrected = false  ) const;

   // extract globally positioned shape for stand alone use
   // note: if matrix is not provided, it will be extracted from
   //       the geo manager and no correction will be applied
   //       to fix Sim/Reco geometry differences.
   //       For expert use only!
   TEveGeoShape* getShape(const char* path, const char* name,
                          const TGeoMatrix* matrix = 0) const;

   // get the detector volume in the geometry manager
   const TGeoVolume* getVolume( unsigned int id ) const;

   // get all known detector ids
   std::vector<unsigned int> getAllIds() const;

   // get all known detector ids with path matching regular expression
   std::vector<unsigned int> getMatchedIds( const char* selection ) const;

   // extract shapes of all known elements
   TEveElementList* getAllShapes(const char* elementListName = "CMS") const;

   // get eta-phi size
   Range getEtaPhiRange(unsigned int id) const;
   Range getXYRange(unsigned int id) const;
   void printEtaPhiRange(unsigned int id) const;
   
   TGeoManager* getManager() const {
      return manager_;
   }
private:
   mutable std::map<unsigned int, TGeoHMatrix> idToMatrix_;
   std::map<unsigned int, std::string> idToPath_;
   mutable TGeoManager* manager_;
   void updateEtaPhiRange(const TGeoHMatrix* matrix,
			  double local_x,
			  double local_y,
			  double local_z,
			  Range& range) const;
   void updateXYRange(const TGeoHMatrix* matrix,
		      double local_x,
		      double local_y,
		      double local_z,
		      Range& range) const;
};

#endif

