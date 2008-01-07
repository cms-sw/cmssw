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
// $Id$
//

// system include files
#include <string>
#include "Reflex/Type.h"

// user include files
#include "Fireworks/Core/interface/FWDisplayProperties.h"

// forward declarations
class TClass;
namespace fwlite {
  class Event;
}

class FWEventItem
{

   public:
      FWEventItem(const std::string& iName,
		  const TClass* iClass,
		  const FWDisplayProperties& iProperties =
		  FWDisplayProperties(),
		  const std::string& iModuleLabel = std::string(),
		  const std::string& iProductInstanceLabel = std::string(),
		  const std::string& iProcessName = std::string());
      //virtual ~FWEventItem();

      // ---------- const member functions ---------------------
#if !defined(__CINT__) && !defined(__MAKECINT__)
      template<class T>
	void get(const T*& oData) const {
	oData=reinterpret_cast<const T*>(data(typeid(T)));
      }
#endif
      const void* data(const std::type_info&) const;
      const FWDisplayProperties& displayProperties() const;

      const std::string& name() const;
      const TClass* type() const;

      const std::string& moduleLabel() const;
      const std::string& productInstanceLabel() const;
      const std::string& processName() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setEvent(const fwlite::Event* iEvent);

      void setLabels(const std::string& iModule,
		     const std::string& iProductInstance,
		     const std::string& iProcess);
      void setName(const std::string& iName);

   private:
      //FWEventItem(const FWEventItem&); // stop default

      //const FWEventItem& operator=(const FWEventItem&); // stop default

      // ---------- member data --------------------------------
      std::string m_name;
      const TClass* m_type;
      mutable const void * m_data;
      FWDisplayProperties m_displayProperties;

      //This will probably moved to a FWEventItemRetriever class
      std::string m_moduleLabel;
      std::string m_productInstanceLabel;
      std::string m_processName;
      const fwlite::Event* m_event;
      ROOT::Reflex::Type m_wrapperType;
};


#endif
