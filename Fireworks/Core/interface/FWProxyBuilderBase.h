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
// Original Author:  Chris Jones, Matevz Tadel, Alja Mrak-Tadel
//         Created:  Thu Mar 18 14:12:12 CET 2010
// $Id: FWProxyBuilderBase.h,v 1.17 2011/07/30 04:55:46 amraktad Exp $
//

// system include files
#include "sigc++/connection.h"

// user include files

// user include files
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/FWProxyBuilderFactory.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWModelIdFromEveSelector.h"
#include "Fireworks/Core/interface/FWViewContext.h"

// forward declarations

class FWEventItem;
class TEveElementList;
class TEveElement;
class FWModelId;
class FWInteractionList;

namespace fireworks {
   class Context;
}

class FWProxyBuilderBase
{
public:

   struct Product
   {
      FWViewType::EType     m_viewType;
      const FWViewContext*  m_viewContext;
      TEveElementList*      m_elements;
      sigc::connection      m_scaleConnection;

      Product(FWViewType::EType t, const FWViewContext* c);
      ~Product();
   };

   FWProxyBuilderBase();
   virtual ~FWProxyBuilderBase();

   // ---------- const member functions ---------------------

   const fireworks::Context& context() const;
   const FWEventItem* item() const {
      return m_item;
   }
   // ---------- static member functions --------------------
   ///Used by the plugin system to determine how the proxy uses the data from FWEventItem
   static std::string typeOfBuilder();

   ///Used by the plugin system to determine precidence of different proxy builders for same type
   /// this returns 'true' if the proxy builder is specialized to only show a sub-part of the object
   /// as opposed to showing the object as a whole
   static bool representsSubPart();
   // ---------- member functions ---------------------------
   virtual void setItem(const FWEventItem* iItem);
   void setHaveWindow(bool iFlag);
   void build();

   void modelChanges(const FWModelIds&);
   void itemChanged(const FWEventItem*);
   void scaleChanged(const FWViewContext*);

   virtual bool canHandle(const FWEventItem&);//note pass FWEventItem to see if type and container match


   virtual void setInteractionList(FWInteractionList*, const std::string&);
   virtual void itemBeingDestroyed(const FWEventItem*);

   // const member functions   
   virtual bool haveSingleProduct() const { return true; }
   virtual bool havePerViewProduct(FWViewType::EType) const { return false; }
   virtual bool willHandleInteraction() const { return false; }

   TEveElementList* createProduct(FWViewType::EType, const FWViewContext*);
   void removePerViewProduct(FWViewType::EType, const FWViewContext* vc);

   bool getHaveWindow() const { return m_haveWindow; }
   void setupElement(TEveElement* el, bool color = true) const;
   void setupAddElement(TEveElement* el, TEveElement* parent,  bool set_color = true) const;
   int  layer() const;

 
protected:   
   // Override this if visibility changes can cause (re)-creation of proxies.
   // Returns true if new proxies were created.
   virtual bool visibilityModelChanges(const FWModelId&, TEveElement*, FWViewType::EType,
                                       const FWViewContext*);

   // Override this if you need special handling of selection or other changes.
   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

   virtual void modelChanges(const FWModelIds&, Product*);

   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc) {};

   FWProxyBuilderBase(const FWProxyBuilderBase&); // stop default
   const FWProxyBuilderBase& operator=(const FWProxyBuilderBase&); // stop default

   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
   virtual void buildViewType(const FWEventItem* iItem, TEveElementList*, FWViewType::EType, const FWViewContext*);

   virtual void clean();
   virtual void cleanLocal();

   void increaseComponentTransparency(unsigned int index, TEveElement* holder,
                                      const std::string& name, Char_t transpOffset);

   // utility
   TEveCompound* createCompound(bool set_color=true, bool propagate_color_to_all_children=false) const;

   // ---------- member data --------------------------------
   typedef std::vector<Product*>::iterator Product_it;

   std::vector<Product*> m_products;

private:
   void cleanProduct(Product* p);
   void setProjectionLayer(float);

   // ---------- member data --------------------------------

   FWInteractionList*    m_interactionList;

   const FWEventItem* m_item;

   bool m_modelsChanged;
   bool m_haveWindow;
   bool m_mustBuild;

   float m_layer;
};

#endif
