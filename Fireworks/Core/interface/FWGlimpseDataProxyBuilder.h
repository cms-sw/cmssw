#ifndef Fireworks_Core_FWGlimpseDataProxyBuilder_h
#define Fireworks_Core_FWGlimpseDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseDataProxyBuilder
//
/**\class FWGlimpseDataProxyBuilder FWGlimpseDataProxyBuilder.h Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Sat Jan  5 15:02:03 EST 2008
// $Id: FWGlimpseDataProxyBuilder.h,v 1.6 2008/12/02 21:15:07 chrjones Exp $
//

// system include files
#include <vector>

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilderFactory.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class FWEventItem;
class TEveElementList;
class TEveElement;
class FWModelId;
class TEveCalo3D;
class FWEveValueScaler;

class FWGlimpseDataProxyBuilder
{

public:
   FWGlimpseDataProxyBuilder();
   virtual ~FWGlimpseDataProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static std::string typeOfBuilder();

   // ---------- member functions ---------------------------
   void setItem(const FWEventItem* iItem);
   void setHaveAWindow(bool iFlag);
   void build();

   void modelChanges(const FWModelIds&);
   void itemChanged(const FWEventItem*);

   TEveElement* usedInScene();

   void setScaler(FWEveValueScaler* iScaler) {
      m_scaler = iScaler;
   }

   FWEveValueScaler* scaler() const {
      return m_scaler;
   }
protected:
   const FWEventItem* item() const {
      return m_item;
   }

   std::vector<FWModelId>& ids() {
      return m_ids;
   }
private:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product) = 0 ;


   //Override this if you need to special handle selection or other changes
   virtual void modelChanges(const FWModelIds&, TEveElement*);
   virtual void applyChangesToAllModels(TEveElement* iElements);
   virtual void itemChangedImp(const FWEventItem*);

   virtual void itemBeingDestroyed(const FWEventItem*);

   void applyChangesToAllModels();

   FWGlimpseDataProxyBuilder(const FWGlimpseDataProxyBuilder&); // stop default

   const FWGlimpseDataProxyBuilder& operator=(const FWGlimpseDataProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
   const FWEventItem* m_item;
   FWEvePtr<TEveElementList> m_elementHolder;   //Used as a smart pointer for the item created by the builder
   std::vector<FWModelId> m_ids;

   bool m_modelsChanged;
   bool m_haveViews;
   FWEveValueScaler* m_scaler;
   bool m_mustBuild;
};


#endif
