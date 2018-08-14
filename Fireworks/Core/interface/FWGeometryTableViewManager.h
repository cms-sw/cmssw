#ifndef Fireworks_Core_FWGeometryTableViewManager_h
#define Fireworks_Core_FWGeometryTableViewManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableViewManager
// 
/**\class FWGeometryTableViewManager FWGeometryTableViewManager.h Fireworks/Core/interface/FWGeometryTableViewManager.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Jul  8 00:40:50 CEST 2011
//

class FWViewBase;
class FWGUIManager;
class TEveWindowSlot;
class TGeoManager;

#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWGeometryTableViewBase.h"

class FWGeometryTableViewManager : public FWViewManagerBase
{
public:
  FWGeometryTableViewManager(FWGUIManager*, std::string fileName, std::string geoName);
   ~FWGeometryTableViewManager() override;

   // dummy functions of FWViewManagerBase
   FWTypeToRepresentations supportedTypesAndRepresentations() const override { return FWTypeToRepresentations();}
   void newItem(const FWEventItem*) override {}  

   FWViewBase *buildView (TEveWindowSlot *iParent, const std::string& type);
   void colorsChanged() override;

   TList*  getListOfVolumes() const;
   TGeoNode* getTopTGeoNode();

   static TGeoManager* getGeoMangeur();
   static void setGeoManagerRuntime(TGeoManager*);

protected:
   // dummy functions of FWViewManagerBase
   void modelChangesComing() override {}
   void modelChangesDone() override {}

   std::vector<std::shared_ptr<FWGeometryTableViewBase> > m_views;

private:
   FWGeometryTableViewManager(const FWGeometryTableViewManager&); // stop default
   const FWGeometryTableViewManager& operator=(const FWGeometryTableViewManager&); // stop default
   void beingDestroyed(const FWViewBase* iView);

   static TGeoManager *s_geoManager;
   std::string m_fileName;
   std::string m_TGeoName;
   void setGeoManagerFromFile();
};


#endif
