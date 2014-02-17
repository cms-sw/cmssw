#ifndef Fireworks_Core_FWViewManagerManager_h
#define Fireworks_Core_FWViewManagerManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewManagerManager
//
/**\class FWViewManagerManager FWViewManagerManager.h Fireworks/Core/interface/FWViewManagerManager.h

   Description: Manages all the FWViewManagerBase instances

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Jan 15 10:26:23 EST 2008
// $Id: FWViewManagerManager.h,v 1.12 2009/11/05 22:06:02 dmytro Exp $
//

// system include files
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include <set>
#include <string>

// user include files

// forward declarations
class FWViewManagerBase;
class FWEventItem;
class FWModelChangeManager;
class FWColorManager;
class FWTypeToRepresentations;

class FWViewManagerManager
{

public:
   FWViewManagerManager(FWModelChangeManager*, FWColorManager*);
   virtual ~FWViewManagerManager();

   // ---------- const member functions ---------------------
   FWTypeToRepresentations supportedTypesAndRepresentations() const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void add( boost::shared_ptr<FWViewManagerBase>);
   void registerEventItem(const FWEventItem*iItem);
   void removeEventItem(const FWEventItem*iItem);
   void eventBegin();
   void eventEnd();

private:
   FWViewManagerManager(const FWViewManagerManager&);    // stop default

   const FWViewManagerManager& operator=(const FWViewManagerManager&);    // stop default

   // ---------- member data --------------------------------
   std::vector<boost::shared_ptr<FWViewManagerBase> > m_viewManagers;
   FWModelChangeManager* m_changeManager;
   FWColorManager* m_colorManager;
   std::map<std::string, const FWEventItem*> m_typeToItems;    //use this to tell view managers registered after the item

};


#endif
