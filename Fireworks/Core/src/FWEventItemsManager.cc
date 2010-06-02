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
// $Id: FWEventItemsManager.cc,v 1.21 2009/07/26 17:19:01 chrjones Exp $
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
   m_event(0),
   m_geom(0),
   m_accessorFactory(new FWItemAccessorFactory())
{
}

// FWEventItemsManager::FWEventItemsManager(const FWEventItemsManager& rhs)
// {
//    // do actual copying here;
// }

FWEventItemsManager::~FWEventItemsManager()
{
   for(std::vector<FWEventItem*>::iterator it = m_items.begin();
       it != m_items.end();
       ++it) {
      delete *it;
   }
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
FWEventItemsManager::add(const FWPhysicsObjectDesc& iItem)
{
   FWPhysicsObjectDesc temp(iItem);
   if(! m_context->colorManager()->colorHasIndex(temp.displayProperties().color())) {
      FWDisplayProperties prop(temp.displayProperties());
      prop.setColor( m_context->colorManager()->indexToColor(0));
      temp.setDisplayProperties(prop);
   }
   m_items.push_back(new FWEventItem(m_context,m_items.size(),m_accessorFactory->accessorFor(temp.type()),
                                     temp) );
   newItem_(m_items.back());
   m_items.back()->goingToBeDestroyed_.connect(boost::bind(&FWEventItemsManager::removeItem,this,_1));
   m_items.back()->setGeom(m_geom);
   if(m_event) {
      FWChangeSentry sentry(*m_changeManager);
      m_items.back()->setEvent(m_event);
   }
   return m_items.back();
}

void
FWEventItemsManager::newEvent(const fwlite::Event* iEvent)
{
   FWChangeSentry sentry(*m_changeManager);
   m_event = iEvent;
   for(std::vector<FWEventItem*>::iterator it = m_items.begin();
       it != m_items.end();
       ++it) {
      if(*it) {
         (*it)->setEvent(iEvent);
      }
   }
}

void
FWEventItemsManager::setGeom(const DetIdToMatrix* geom)
{
   // cache the geometry (in case items are added later)
   m_geom = geom;
   for(std::vector<FWEventItem*>::iterator it = m_items.begin();
       it != m_items.end();
       ++it) {
      if(*it) {
         (*it)->setGeom(geom);
      }
   }
}

void
FWEventItemsManager::clearItems()
{
   goingToClearItems_();
   for(std::vector<FWEventItem*>::iterator it = m_items.begin();
       it != m_items.end();
       ++it) {
      delete *it;
   }
   m_items.clear();
}

static const std::string kType("type");
static const std::string kModuleLabel("moduleLabel");
static const std::string kProductInstanceLabel("productInstanceLabel");
static const std::string kProcessName("processName");
static const std::string kFilterExpression("filterExpression");
static const std::string kColor("color");
static const std::string kIsVisible("isVisible");
static const std::string kTrue("t");
static const std::string kFalse("f");
static const std::string kLayer("layer");
static const std::string kPurpose("purpose");

void
FWEventItemsManager::addTo(FWConfiguration& iTo) const
{
   FWColorManager* cm = m_context->colorManager();
   assert(0!=cm);
   for(std::vector<FWEventItem*>::const_iterator it = m_items.begin();
       it != m_items.end();
       ++it) {
      if(!*it) continue;
      FWConfiguration conf(3);
      ROOT::Reflex::Type dataType( ROOT::Reflex::Type::ByTypeInfo(*((*it)->type()->GetTypeInfo())));
      assert(dataType != ROOT::Reflex::Type() );

      conf.addKeyValue(kType,FWConfiguration(dataType.Name(ROOT::Reflex::SCOPED)));
      conf.addKeyValue(kModuleLabel,FWConfiguration((*it)->moduleLabel()));
      conf.addKeyValue(kProductInstanceLabel, FWConfiguration((*it)->productInstanceLabel()));
      conf.addKeyValue(kProcessName, FWConfiguration((*it)->processName()));
      conf.addKeyValue(kFilterExpression, FWConfiguration((*it)->filterExpression()));
      {
         std::ostringstream os;
         os << cm->colorToIndex((*it)->defaultDisplayProperties().color());
         conf.addKeyValue(kColor, FWConfiguration(os.str()));
      }
      conf.addKeyValue(kIsVisible, FWConfiguration((*it)->defaultDisplayProperties().isVisible() ? kTrue : kFalse));
      {
         std::ostringstream os;
         os << (*it)->layer();
         conf.addKeyValue(kLayer,FWConfiguration(os.str()));
      }
      conf.addKeyValue(kPurpose,(*it)->purpose());
      iTo.addKeyValue((*it)->name(), conf, true);
   }
}

void
FWEventItemsManager::setFrom(const FWConfiguration& iFrom)
{
   FWColorManager* cm = m_context->colorManager();
   assert(0!=cm);

   clearItems();
   const FWConfiguration::KeyValues* keyValues =  iFrom.keyValues();
   assert(0!=keyValues);
   for(FWConfiguration::KeyValues::const_iterator it = keyValues->begin();
       it != keyValues->end();
       ++it) {
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
      if(conf.version()<3) {
         std::istringstream is(sColor);
         Color_t color;
         is >> color;
         colorIndex = cm->oldColorToIndex(color);
      } else {
         std::istringstream is(sColor);
         is >> colorIndex;
      }

      FWDisplayProperties disp(cm->indexToColor(colorIndex), isVisible);

      unsigned int layer;
      const std::string& sLayer =(*keyValues)[7].second.value();
      std::istringstream isl(sLayer);
      isl >> layer;
      //For older configs assume name is the same as purpose
      std::string purpose(name);
      if(conf.version()>1) {
         purpose = (*keyValues)[8].second.value();
      }
      FWPhysicsObjectDesc desc(name,
                               TClass::GetClass(type.c_str()),
                               purpose,
                               disp,
                               moduleLabel,
                               productInstanceLabel,
			       processName,
                               filterExpression,
                               layer);
      add(desc);
   }
}

void
FWEventItemsManager::removeItem(const FWEventItem* iItem)
{
   m_items[iItem->id()]=0;
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

const FWEventItem*
FWEventItemsManager::find(const std::string& iName) const
{
   for(std::vector<FWEventItem*>::const_iterator it = m_items.begin();
       it != m_items.end();
       ++it) {
      if( *it && (*it)->name() == iName) {
         return *it;
      }
   }
   return 0;
}

//
// static member functions
//
