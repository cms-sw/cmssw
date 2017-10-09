#ifndef Fireworks_Geometry_TGeoFromDddService_h
#define Fireworks_Geometry_TGeoFromDddService_h
// -*- C++ -*-
//
// Package:     Geometry
// Class  :     TGeoFromDddService
// 
/**\class TGeoFromDddService TGeoFromDddService.h Fireworks/Geometry/interface/TGeoFromDddService.h

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


// user include files

// forward declarations

namespace edm
{
   class ParameterSet;
   class ActivityRegistry;
   class Run;
   class EventSetup;
}

class DDSolid;
class DDMaterial;

class TGeoManager;
class TGeoShape;
class TGeoVolume;
class TGeoMaterial;
class TGeoMedium;

class TGeoFromDddService
{
public:
   TGeoFromDddService(const edm::ParameterSet&, edm::ActivityRegistry&);
   virtual ~TGeoFromDddService();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   void postBeginRun(const edm::Run&, const edm::EventSetup&);
   void postEndRun  (const edm::Run&, const edm::EventSetup&);

   TGeoManager* getGeoManager();

private:
   TGeoFromDddService(const TGeoFromDddService&);                  // stop default
   const TGeoFromDddService& operator=(const TGeoFromDddService&); // stop default


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
   const edm::EventSetup   *m_eventSetup;
   TGeoManager             *m_geoManager;

   std::map<std::string, TGeoShape*>    nameToShape_;
   std::map<std::string, TGeoVolume*>   nameToVolume_;
   std::map<std::string, TGeoMaterial*> nameToMaterial_;
   std::map<std::string, TGeoMedium*>   nameToMedium_;
};

#endif
