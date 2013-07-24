#ifndef Fireworks_Core_FWEventItem_h
#define Fireworks_Core_FWEventItem_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEventItem
//
/**\class FWEventItem FWEventItem.h Fireworks/Core/interface/FWEventItem.h

   Description: Stand in for a top level item in an Event

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Jan  3 14:02:21 EST 2008
// $Id: FWEventItem.h,v 1.51 2012/08/03 18:20:27 wmtan Exp $
//

// system include files
#include <string>
#include <vector>
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include <boost/shared_ptr.hpp>
#include <sigc++/connection.h>

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWItemChangeSignal.h"

#include "Fireworks/Core/interface/FWModelFilter.h"
#include "Fireworks/Core/interface/FWItemValueGetter.h"

#include "Fireworks/Core/interface/Context.h"

// forward declarations
class TClass;
class FWModelChangeManager;
class FWSelectionManager;
class FWGeometry;
class TVirtualCollectionProxy;
class FWItemAccessorBase;
class FWProxyBuilderConfiguration;
class FWConfiguration;

namespace edm {
   class EventBase;
}
namespace fireworks {
   class Context;
}

class FWEventItem
{
public:
   struct ModelInfo {
      FWDisplayProperties m_displayProperties;
      bool m_isSelected;
      ModelInfo(const FWDisplayProperties& iProps, bool iIsSelected) :
         m_displayProperties(iProps),
         m_isSelected(iIsSelected) {
      }

      const FWDisplayProperties& displayProperties() const {
         return m_displayProperties;
      }
      bool isSelected() const {
         return m_isSelected;
      }
   };

   FWEventItem(fireworks::Context* iContext,
               unsigned int iItemId,
               boost::shared_ptr<FWItemAccessorBase> iAccessor,
               const FWPhysicsObjectDesc& iDesc,  const FWConfiguration* pbConf = 0);
   virtual ~FWEventItem();

   // ---------- const member functions ---------------------
#if !defined(__CINT__) && !defined(__MAKECINT__)
   template<class T>
   void get(const T*& oData) const {
      oData=reinterpret_cast<const T*>(data(typeid(T)));
   }
#endif
   const void* data(const std::type_info&) const;
   const FWDisplayProperties& defaultDisplayProperties() const;

   /** objects with a larger layer number are drawn on top of objects with a lower number */
   int layer() const;
   ///returns true if item is in front of all other items
   bool isInFront() const;
   ///returns true if item is behind all other items
   bool isInBack() const;

   const std::string& filterExpression() const;
   /**Unique ID for the item. This number starts at 0 and increments by one for each
      new item.*/
   unsigned int id() const;
   const std::string& name() const;
   const TClass* type() const;
   /** Since the same C++ type can be used for multiple purposes, this string disambiguates them.*/
   const std::string& purpose() const;

   const std::string& moduleLabel() const;
   const std::string& productInstanceLabel() const;
   const std::string& processName() const;

   const TClass* modelType() const;
   ModelInfo modelInfo(int iIndex) const;    //return copy for now since want to be able to change visibility
   size_t size() const;
   const void* modelData(int iIndex) const;
   std::string modelName(int iIndex) const;

   ///one value from the model which is normally used for the popup
  const  FWItemValueGetter& valueGetter() const { return m_interestingValueGetter; }
   bool haveInterestingValue() const;
   const std::string& modelInterestingValueAsString(int iIndex) const;

   bool isCollection() const;

   //convenience methods

   const fireworks::Context& context () const {
      return *m_context;
   }

   FWModelChangeManager* changeManager() const {
      return m_context->modelChangeManager();
   }
   FWSelectionManager* selectionManager() const {
      return m_context->selectionManager();
   }
   
   FWColorManager* colorManager() const {
      return m_context->colorManager();
   }

   bool hasEvent() const {
      return 0 != m_event;
   }

   // hackery methods
   const edm::EventBase *getEvent () const {
      return m_event;
   }


   ///returns true if failed to get data for this event
   bool hasError() const;
   ///returns error string if there was a problem this event
   const std::string& errorMessage() const;
   
   // ---------- static member functions --------------------

   static int minLayerValue();
   static int maxLayerValue();

   // ---------- member functions ---------------------------
   void setEvent(const edm::EventBase* iEvent);

   const FWGeometry* getGeom() const;
   FWProxyBuilderConfiguration* getConfig() const { return m_proxyBuilderConfig; }

   void setLabels(const std::string& iModule,
                  const std::string& iProductInstance,
                  const std::string& iProcess);
   void setName(const std::string& iName);
   void setDefaultDisplayProperties(const FWDisplayProperties&);
   /**Throws an FWExpresionException if there is a problem with the expression */
   void setFilterExpression(const std::string& );

   /**Select the item (i.e. container) itself*/
   void selectItem();
   void unselectItem();
   void toggleSelectItem();
   bool itemIsSelected() const;
   
   /**change layering*/
   void moveToFront();
   void moveToBack();
   void moveToLayer(int layer);

   void proxyConfigChanged();

   void unselect(int iIndex) const;
   void select(int iIndex) const;
   void toggleSelect(int iIndex) const;
   void setDisplayProperties(int iIndex, const FWDisplayProperties&) const;

   void destroy() const;
   /** connect to this signal if you want to know when models held by the item change */
   mutable FWModelChangeSignal changed_;

   /** connect to this signal if you want to know when the data underlying the item changes */
   mutable FWItemChangeSignal itemChanged_;

   /** connect to this signal if you want to know immediately when the data underlying the item changes
      only intended to be used by the FWSelectionManager
    */
   mutable FWItemChangeSignal preItemChanged_;

   /** connect to this signal if you want to know that the default display properties of the item have changed.
      This is only useful if you are displaying these properties and not just the underlying models.*/
   mutable FWItemChangeSignal defaultDisplayPropertiesChanged_;
   
   /** connect to this signal if you want to know that the filter being applied to the item was changed. */
   mutable FWItemChangeSignal filterChanged_;

   /** connect to this signal if you need to know that this item is going to be destroyed.
    */
   mutable FWItemChangeSignal goingToBeDestroyed_;
private:
   //FWEventItem(const FWEventItem&); // stop default

   //const FWEventItem& operator=(const FWEventItem&); // stop default
   void setData(const edm::ObjectWithDict& ) const;

   void getPrimaryData() const;
   void runFilter();
   void handleChange();
   // ---------- member data --------------------------------
   const fireworks::Context* m_context;
   unsigned int m_id;
   std::string m_name;
   const TClass* m_type;
   std::string m_purpose;
   boost::shared_ptr<FWItemAccessorBase> m_accessor;
   FWDisplayProperties m_displayProperties;
   int m_layer;
   mutable std::vector<ModelInfo> m_itemInfos;

   //This will probably moved to a FWEventItemRetriever class
   std::string m_moduleLabel;
   std::string m_productInstanceLabel;
   std::string m_processName;
   const edm::EventBase* m_event;
   edm::TypeWithDict m_wrapperType;
   FWItemValueGetter m_interestingValueGetter;

   FWModelFilter m_filter;
   mutable bool m_printedErrorThisEvent;
   mutable std::string m_errorMessage;
   
   bool m_isSelected;


   FWProxyBuilderConfiguration*  m_proxyBuilderConfig;
};


#endif
