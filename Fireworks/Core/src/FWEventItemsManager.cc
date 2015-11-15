// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEventItemsManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Fri Jan  4 10:38:18 EST 2008
//

// system include files
#include <sstream>
#include <boost/bind.hpp>
#include "TClass.h"

// user include files
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWItemAccessorFactory.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/fwLog.h"
#include <cassert>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEventItemsManager::FWEventItemsManager(FWModelChangeManager* iManager) :
   m_changeManager(iManager),
   m_context(0),
   m_event(0),
   m_accessorFactory(new FWItemAccessorFactory())
{
}

// FWEventItemsManager::FWEventItemsManager(const FWEventItemsManager& rhs)
// {
//    // do actual copying here;
// }

/** FWEventItemsManager has ownership of the items it contains.

    Note that because of the way we keep track of removed items,
    m_items[i] could actually be 0 for indices corresponding
    to removed items.
 */
FWEventItemsManager::~FWEventItemsManager()
{
   for (size_t i = 0, e = m_items.size(); i != e; ++i)
      delete m_items[i];

   m_items.clear();
}

//
// assignment operators
//
// const FWEventItemsManager& FWEventItemsManager::operator=(const FWEventItemsManager& rhs)
// {
//   //An exception safe implementation is
//   FWEventItemsManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
const FWEventItem*
FWEventItemsManager::add(const FWPhysicsObjectDesc& iItem, bool showFilteredInTable, const FWConfiguration* pbc)
{
   FWPhysicsObjectDesc temp(iItem);
   
   if(! m_context->colorManager()->colorHasIndex(temp.displayProperties().color())) {
      FWDisplayProperties prop(temp.displayProperties());
      fwLog(fwlog::kWarning) << Form("FWEventItemsManager::add(const FWPhysicsObjectDesc& iItem), color index  not valid. Set Color idex to %d\n", FWColorManager::getDefaultStartColorIndex());
      prop.setColor(FWColorManager::getDefaultStartColorIndex());
      temp.setDisplayProperties(prop);
   }
   
   m_items.push_back(new FWEventItem(m_context,m_items.size(),m_accessorFactory->accessorFor(temp.type()),
                                     temp, showFilteredInTable, pbc));
   newItem_(m_items.back());
   m_items.back()->goingToBeDestroyed_.connect(boost::bind(&FWEventItemsManager::removeItem,this,_1));
   if(m_event) {
      FWChangeSentry sentry(*m_changeManager);
      m_items.back()->setEvent(m_event);
   }
   return m_items.back();
}

/** Prepare to handle a new event by associating
    all the items to watch it.
  */
void
FWEventItemsManager::newEvent(const edm::EventBase* iEvent)
{
   FWChangeSentry sentry(*m_changeManager);
   m_event = iEvent;
   for(size_t i = 0, e = m_items.size(); i != e; ++i)
   {
      FWEventItem *item = m_items[i];
      if(item)
         item->setEvent(iEvent);
   }
}

/** Clear all the items in the model. 
    
    Notice that a previous implementation was setting all the items to 0, I
    guess to track accessing delete items.
  */
void
FWEventItemsManager::clearItems(void)
{
   for (size_t i = 0, e = m_items.size(); i != e; ++i)
   {
      FWEventItem *item = m_items[i];
      if (item) {
         item->destroy();
      }
      m_items[i]=0;
   }
   goingToClearItems_();

   m_items.clear();
}

static const std::string kType("type");
static const std::string kModuleLabel("moduleLabel");
static const std::string kProductInstanceLabel("productInstanceLabel");
static const std::string kProcessName("processName");
static const std::string kFilterExpression("filterExpression");
static const std::string kShowFilteredEntriesInTable("showFilteredEntriesInTable");
static const std::string kColor("color");
static const std::string kIsVisible("isVisible");
static const std::string kTrue("t");
static const std::string kFalse("f");
static const std::string kLayer("layer");
static const std::string kPurpose("purpose");
static const std::string kTransparency("transparency");

void
FWEventItemsManager::addTo(FWConfiguration& iTo) const
{
   FWColorManager* cm = m_context->colorManager();
   assert(0!=cm);
   for(std::vector<FWEventItem*>::const_iterator it = m_items.begin();
       it != m_items.end();
       ++it)
   {
      if(!*it) continue;
      FWConfiguration conf(7);
      edm::TypeWithDict dataType((*((*it)->type()->GetTypeInfo())));
      assert(dataType != edm::TypeWithDict() );

      conf.addKeyValue(kType,FWConfiguration(dataType.name()));
      conf.addKeyValue(kModuleLabel,FWConfiguration((*it)->moduleLabel()));
      conf.addKeyValue(kProductInstanceLabel, FWConfiguration((*it)->productInstanceLabel()));
      conf.addKeyValue(kProcessName, FWConfiguration((*it)->processName()));
      conf.addKeyValue(kFilterExpression, FWConfiguration((*it)->filterExpression()));
      {
         std::ostringstream os;
         os << (*it)->defaultDisplayProperties().color();
         conf.addKeyValue(kColor, FWConfiguration(os.str()));
      }
      conf.addKeyValue(kIsVisible, FWConfiguration((*it)->defaultDisplayProperties().isVisible() ? kTrue : kFalse));
      {
         std::ostringstream os;
         os << (*it)->layer();
         conf.addKeyValue(kLayer,FWConfiguration(os.str()));
      }
      conf.addKeyValue(kPurpose,(*it)->purpose());
      {
         std::ostringstream os;
         os << static_cast<int>((*it)->defaultDisplayProperties().transparency());
         conf.addKeyValue(kTransparency, FWConfiguration(os.str()));
      }
      
      conf.addKeyValue(kShowFilteredEntriesInTable, FWConfiguration((*it)->showFilteredEntries() ? kTrue : kFalse));

      FWConfiguration pbTmp;
      (*it)->getConfig()->addTo(pbTmp);
      conf.addKeyValue("PBConfig",pbTmp, true);

      iTo.addKeyValue((*it)->name(), conf, true);
   }
}

/** This is responsible for resetting the status of items from configuration  
  */
void
FWEventItemsManager::setFrom(const FWConfiguration& iFrom)
{
 
   FWColorManager* cm = m_context->colorManager();
   assert(0!=cm);

   clearItems();
   const FWConfiguration::KeyValues* keyValues =  iFrom.keyValues();

   if (keyValues == 0) return;

   for (FWConfiguration::KeyValues::const_iterator it = keyValues->begin();
        it != keyValues->end();
        ++it)
   {
      const std::string& name = it->first;
      const FWConfiguration& conf = it->second;
      const FWConfiguration::KeyValues* keyValues =  conf.keyValues();
      assert(0!=keyValues);
      const std::string& type = (*keyValues)[0].second.value();
      const std::string& moduleLabel = (*keyValues)[1].second.value();
      const std::string& productInstanceLabel = (*keyValues)[2].second.value();
      const std::string& processName = (*keyValues)[3].second.value();
      const std::string& filterExpression = (*keyValues)[4].second.value();
      const std::string& sColor = (*keyValues)[5].second.value();
      const bool isVisible = (*keyValues)[6].second.value() == kTrue;

      unsigned int colorIndex;
      if(conf.version() < 5)
      {
         std::istringstream is(sColor);
         Color_t color;
         is >> color;
         colorIndex = cm->oldColorToIndex(color, conf.version());
      }
      else
      {
         // In version 4 we assume:
         //   fireworks colors start at ROOT index 1000
         //   geometry  colors start at ROOT index 1100
         // We save them as such -- no conversions needed.
         std::istringstream is(sColor);
         is >> colorIndex;
      }
      
      int transparency = 0;

      // Read transparency from file. We don't care about checking errors
      // because strtol returns 0 in that case.
      if (conf.version() > 3)
         transparency = strtol((*keyValues)[9].second.value().c_str(), 0, 10);

      FWDisplayProperties dp(colorIndex, isVisible, true, transparency);

      unsigned int layer = strtol((*keyValues)[7].second.value().c_str(), 0, 10);

      //For older configs assume name is the same as purpose
      std::string purpose(name);
      if (conf.version() > 1)
         purpose = (*keyValues)[8].second.value();

      FWConfiguration* proxyConfig = (FWConfiguration*) conf.valueForKey("PBConfig") ? new FWConfiguration(*conf.valueForKey("PBConfig")) : 0;

      // beckward compatibilty for obsolete proxy builders
      if (conf.version() < 6)
      {
         assert(proxyConfig == 0);
         if (purpose == "VerticesWithTracks")
         {
            purpose = "Vertices";
            proxyConfig = new FWConfiguration();
            FWConfiguration vTmp; vTmp.addKeyValue("Draw Tracks", FWConfiguration("1"));
            proxyConfig->addKeyValue("Var", vTmp,true);
         }
      }
      
      bool showFilteredInTable = true;
      if (conf.version() > 6) {
         showFilteredInTable = (*keyValues)[10].second.value() == kTrue;
      }

      FWPhysicsObjectDesc desc(name,
                               TClass::GetClass(type.c_str()),
                               purpose,
                               dp,
                               moduleLabel,
                               productInstanceLabel,
                               processName,
                               filterExpression,
                               layer);
      
      add(desc, showFilteredInTable, proxyConfig );
   }
}

/** Remove one item. 
  
    Notice that rather than erasing the item from the list, it is preferred to
    set it to zero, I guess to catch accesses to remove items and to avoid 
    having to recalculate the current selection.
    
    GE: I think this is a broken way of handling removal of objects.  The object
        should be properly deleted and the current selection should be updated
        accordingly.
  */
void
FWEventItemsManager::removeItem(const FWEventItem* iItem)
{
   assert(iItem->id() < m_items.size());
   m_items[iItem->id()] = 0;
}

void
FWEventItemsManager::setContext(fireworks::Context* iContext)
{
   m_context = iContext;
}

//
// const member functions
//
FWEventItemsManager::const_iterator
FWEventItemsManager::begin() const
{
   return m_items.begin();
}
FWEventItemsManager::const_iterator
FWEventItemsManager::end() const
{
   return m_items.end();
}

/** Look up an item by name.
  */
const FWEventItem*
FWEventItemsManager::find(const std::string& iName) const
{
   for (size_t i = 0, e = m_items.size(); i != e; ++i)
   {
      const FWEventItem *item = m_items[i];
      if (item && item->name() == iName)
         return item;
   }
   return 0;
}

//
// static member functions
//
