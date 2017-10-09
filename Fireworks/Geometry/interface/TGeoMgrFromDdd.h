#ifndef Fireworks_Geometry_TGeoMgrFromDdd_h
#define Fireworks_Geometry_TGeoMgrFromDdd_h
// -*- C++ -*-
//
// Package:     Geometry
// Class  :     TGeoMgrFromDdd
// 
/**\class TGeoMgrFromDdd TGeoMgrFromDdd.h Fireworks/Geometry/interface/TGeoMgrFromDdd.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Fri Jul  2 16:11:33 CEST 2010
//

// system include files
#include <string>
#include <map>

#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

// forward declarations

namespace edm
{
   class ParameterSet;
}

class DDSolid;
class DDMaterial;
class DisplayGeomRecord;

class TGeoManager;
class TGeoShape;
class TGeoVolume;
class TGeoMaterial;
class TGeoMedium;

class TGeoMgrFromDdd : public edm::ESProducer
{
public:
   TGeoMgrFromDdd(const edm::ParameterSet&);
   virtual ~TGeoMgrFromDdd();

   typedef std::shared_ptr<TGeoManager> ReturnType;

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   ReturnType produce(const DisplayGeomRecord&);

private:
   TGeoMgrFromDdd(const TGeoMgrFromDdd&);                  // stop default
   const TGeoMgrFromDdd& operator=(const TGeoMgrFromDdd&); // stop default


   TGeoManager*  createManager(int level);

   TGeoShape*    createShape(const std::string& iName,
                             const DDSolid& iSolid);
   TGeoVolume*   createVolume(const std::string& iName,
                              const DDSolid& iSolid,
                              const DDMaterial& iMaterial);
   TGeoMaterial* createMaterial(const DDMaterial& iMaterial);

   // ---------- member data --------------------------------

   int                      m_level;
   bool                     m_verbose;

   std::map<std::string, TGeoShape*>    nameToShape_;
   std::map<std::string, TGeoVolume*>   nameToVolume_;
   std::map<std::string, TGeoMaterial*> nameToMaterial_;
   std::map<std::string, TGeoMedium*>   nameToMedium_;
};

#endif
