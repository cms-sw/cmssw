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
#include "TGeoMatrix.h" 
#include <map>
#include <string>
#include <vector>

class DetIdToMatrix
{
 public:
   DetIdToMatrix():manager_(0){}
   ~DetIdToMatrix();
   
   // load full CMS geometry
   void loadGeometry(const char* fileName);
   
   // load DetId to Matrix map
   void loadMap(const char* fileName);

   // get matrix for full local to global transformation
   const TGeoHMatrix* getMatrix( unsigned int id );

   // get path to the detector volume in the geometry manager
   const char* getPath( unsigned int id );
   
   // get the detector volume in the geometry manager
   const TGeoVolume* getVolume( unsigned int id );
   
   // get all known detector ids
   std::vector<unsigned int> getAllIds();

 private:
   std::map<unsigned int, TGeoHMatrix> idToMatrix_;
   std::map<unsigned int, std::string> idToPath_; 
   TGeoManager* manager_;
	
};

#endif
  
