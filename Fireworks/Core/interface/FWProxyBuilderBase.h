#ifndef Fireworks_Core_FWProxyBuilderBase_h
#define Fireworks_Core_FWProxyBuilderBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWProxyBuilderBase
// 
/**\class FWProxyBuilderBase FWProxyBuilderBase.h Fireworks/Core/interface/FWProxyBuilderBase.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones, Alja Mrak-Tadel
//         Created:  Thu Mar 18 14:12:12 CET 2010
// $Id: FWProxyBuilderBase.h,v 1.1 2010/04/06 20:00:35 amraktad Exp $
//

// system include files

// user include files

// user include files
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/FWProxyBuilderFactory.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWModelIdFromEveSelector.h"

// forward declarations

class FWEventItem;
class TEveElementList;
class TEveElement;
class FWModelId;

namespace fireworks {
   class Context;
}

class FWProxyBuilderBase
{

public:
   FWProxyBuilderBase();
   virtual ~FWProxyBuilderBase();

   // ---------- const member functions ---------------------

   const fireworks::Context& context() const;

   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static std::string typeOfBuilder();

   // ---------- member functions ---------------------------
   void setItem(const FWEventItem* iItem);
   void setHaveAWindow(bool iFlag);
   void build();

   void modelChanges(const FWModelIds&);
   void itemChanged(const FWEventItem*);

   virtual bool canHandle(const FWEventItem&);//note pass FWEventItem to see if type and container match
   virtual void attachToScene(const FWViewType&, const std::string& purpose, TEveElementList* sceneHolder);
   virtual void releaseFromSceneGraph(const FWViewType&);
   virtual bool willHandleInteraction();
   virtual void setInteractionList(std::vector<TEveCompound* > const *, std::string& purpose);

   // getters
   int layer() const;
   TEveElementList* getProduct();

protected:
   const FWEventItem* item() const {
      return m_item;
   }

   std::vector<FWModelIdFromEveSelector>& ids() {
      return m_ids;
   }
   
   //Override this if you need to special handle selection or other changes
   virtual bool specialModelChangeHandling(const FWModelId&, TEveElement*);
   virtual void applyChangesToAllModels(TEveElement* iElements);

   virtual void itemChangedImp(const FWEventItem*);
   virtual void itemBeingDestroyed(const FWEventItem*);

   virtual void modelChanges(const FWModelIds&, TEveElement*);

private:
   FWProxyBuilderBase(const FWProxyBuilderBase&); // stop default
   const FWProxyBuilderBase& operator=(const FWProxyBuilderBase&); // stop default

   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product) = 0 ;

   void applyChangesToAllModels();

   // ---------- member data --------------------------------
   const FWEventItem* m_item;
   TEveElementList* m_elementHolder;   //Used as a smart pointer for the item created by the builder
   std::vector<FWModelIdFromEveSelector> m_ids;

   bool m_modelsChanged;
   bool m_haveViews;
   bool m_mustBuild;
};

#endif
