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
// $Id: FWEventItem.h,v 1.3 2008/01/19 04:51:12 dmytro Exp $
//

// system include files
#include <string>
#include <vector>
#include "Reflex/Type.h"
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"

// forward declarations
class TClass;
class FWModelChangeManager;
class FWSelectionManager;
class DetIdToMatrix;
class TVirtualCollectionProxy;

namespace fwlite {
  class Event;
}

class FWEventItem
{

   public:
      struct ModelInfo {
         FWDisplayProperties m_displayProperties;
         bool m_isSelected;
         ModelInfo(const FWDisplayProperties& iProps, bool iIsSelected):
         m_displayProperties(iProps),
         m_isSelected(iIsSelected) {}
      };
   
      FWEventItem(FWModelChangeManager* iCM,
                  FWSelectionManager* iSM,
                  const std::string& iName,
		  const TClass* iClass,
		  const FWDisplayProperties& iProperties =
		  FWDisplayProperties(),
		  const std::string& iModuleLabel = std::string(),
		  const std::string& iProductInstanceLabel = std::string(),
		  const std::string& iProcessName = std::string());
   
      FWEventItem(FWModelChangeManager* iCM,
                  FWSelectionManager* iSM,
                  const FWPhysicsObjectDesc& iDesc);
      //virtual ~FWEventItem();

      // ---------- const member functions ---------------------
#if !defined(__CINT__) && !defined(__MAKECINT__)
      template<class T>
	void get(const T*& oData) const {
	oData=reinterpret_cast<const T*>(data(typeid(T)));
      }
#endif
      const void* data(const std::type_info&) const;
      const FWDisplayProperties& defaultDisplayProperties() const;

      const std::string& name() const;
      const TClass* type() const;

      const std::string& moduleLabel() const;
      const std::string& productInstanceLabel() const;
      const std::string& processName() const;
   
      const ModelInfo& modelInfo(int iIndex) const;
      size_t size() const;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setEvent(const fwlite::Event* iEvent);
      void setGeom(const DetIdToMatrix* geom){ m_detIdToGeo = geom; }
      const DetIdToMatrix* getGeom() const { return m_detIdToGeo; }

      void setLabels(const std::string& iModule,
		     const std::string& iProductInstance,
		     const std::string& iProcess);
      void setName(const std::string& iName);

      void unselect(int iIndex) const;
      void select(int iIndex) const;
      void toggleSelect(int iIndex) const;
      void setDisplayProperties(int iIndex, const FWDisplayProperties&) const;
   
   
   private:
      //FWEventItem(const FWEventItem&); // stop default

      //const FWEventItem& operator=(const FWEventItem&); // stop default
      void setData(const void* ) const;
   
      // ---------- member data --------------------------------
      FWModelChangeManager* m_changeManager;
      FWSelectionManager* m_selectionManager;
      std::string m_name;
      const TClass* m_type;
      boost::shared_ptr<TVirtualCollectionProxy> m_colProxy; //should be something other than shared_ptr 
      mutable const void * m_data;
      FWDisplayProperties m_displayProperties;
      mutable std::vector<ModelInfo> m_itemInfos;

      //This will probably moved to a FWEventItemRetriever class
      std::string m_moduleLabel;
      std::string m_productInstanceLabel;
      std::string m_processName;
      const fwlite::Event* m_event;
      ROOT::Reflex::Type m_wrapperType;
      const DetIdToMatrix* m_detIdToGeo;
};


#endif
