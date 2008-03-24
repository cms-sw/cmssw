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
// $Id: FWEventItemsManager.cc,v 1.9 2008/03/24 07:10:01 dmytro Exp $
//

// system include files
#include <sstream>
#include "TClass.h"

// user include files
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"

#include "Fireworks/Core/interface/FWConfiguration.h"

#include "Fireworks/Core/interface/FWDisplayProperties.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEventItemsManager::FWEventItemsManager(FWModelChangeManager* iManager, 
FWSelectionManager* iSelMgr):
m_changeManager(iManager),
m_selectionManager(iSelMgr)
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
  m_items.push_back(new FWEventItem(m_changeManager,m_selectionManager,m_items.size(),iItem) );
  newItem_(m_items.back());
  return m_items.back();
}

void 
FWEventItemsManager::newEvent(const fwlite::Event* iEvent)
{    
  FWChangeSentry sentry(*m_changeManager);
  for(std::vector<FWEventItem*>::iterator it = m_items.begin();
      it != m_items.end();
      ++it) {
    (*it)->setEvent(iEvent);
  }
}

void 
FWEventItemsManager::setGeom(const DetIdToMatrix* geom)
{
  for(std::vector<FWEventItem*>::iterator it = m_items.begin();
      it != m_items.end();
      ++it) {
    (*it)->setGeom(geom);
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

void 
FWEventItemsManager::addTo(FWConfiguration& iTo) const
{
   for(std::vector<FWEventItem*>::const_iterator it = m_items.begin();
       it != m_items.end();
       ++it) {
      FWConfiguration conf(1);
      ROOT::Reflex::Type dataType( ROOT::Reflex::Type::ByTypeInfo(*((*it)->type()->GetTypeInfo())));
      assert(dataType != ROOT::Reflex::Type() );

      conf.addKeyValue(kType,FWConfiguration(dataType.Name(ROOT::Reflex::SCOPED)));
      conf.addKeyValue(kModuleLabel,FWConfiguration((*it)->moduleLabel()));
      conf.addKeyValue(kProductInstanceLabel, FWConfiguration((*it)->productInstanceLabel()));
      conf.addKeyValue(kProcessName, FWConfiguration((*it)->processName()));
      conf.addKeyValue(kFilterExpression, FWConfiguration((*it)->filterExpression()));
      {
         std::ostringstream os;
         os << (*it)->defaultDisplayProperties().color();
         conf.addKeyValue(kColor, FWConfiguration(os.str()));
      }
      conf.addKeyValue(kIsVisible, FWConfiguration((*it)->defaultDisplayProperties().isVisible()?kTrue:kFalse));
      {
         std::ostringstream os;
         os << (*it)->layer();
         conf.addKeyValue(kLayer,FWConfiguration(os.str()));
      }
      iTo.addKeyValue((*it)->name(), conf, true);
   }
}

void 
FWEventItemsManager::setFrom(const FWConfiguration& iFrom)
{
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

      std::istringstream is(sColor);
      Color_t color;
      is >> color;
      
      FWDisplayProperties disp(color, isVisible);

      unsigned int layer;
      const std::string& sLayer =(*keyValues)[7].second.value();
      std::istringstream isl(sLayer);
      isl >> layer;
      FWPhysicsObjectDesc desc(name,
                               TClass::GetClass(type.c_str()),
                               disp,
                               moduleLabel,
                               productInstanceLabel,
                               processName,
			       filterExpression,
                               layer);
      add(desc);
   }
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
    if( (*it)->name() == iName) {
      return *it;
    }
  }
  return 0;
}
//
// static member functions
//
