// -*- C++ -*-
//
// Package:     Core
// Class  :     FWConfigurationManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Feb 24 14:42:32 EST 2008
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWConfigurable.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWConfigurationManager::FWConfigurationManager()
{
}

// FWConfigurationManager::FWConfigurationManager(const FWConfigurationManager& rhs)
// {
//    // do actual copying here;
// }

FWConfigurationManager::~FWConfigurationManager()
{
}

//
// assignment operators
//
// const FWConfigurationManager& FWConfigurationManager::operator=(const FWConfigurationManager& rhs)
// {
//   //An exception safe implementation is
//   FWConfigurationManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
void 
FWConfigurationManager::setFrom(const FWConfiguration& iConfig) const
{
   assert(0!=iConfig.keyValues());
   for(FWConfiguration::KeyValues::const_iterator it = iConfig.keyValues()->begin(),
       itEnd = iConfig.keyValues()->end();
       it != itEnd;
       ++it) {
      std::map<std::string,FWConfigurable*>::const_iterator itFound = m_configurables.find(it->first);
      assert(itFound != m_configurables.end());
      itFound->second->setFrom(it->second);
   }
}
void 
FWConfigurationManager::to(FWConfiguration& oConfig) const
{
   FWConfiguration config;
   for(std::map<std::string,FWConfigurable*>::const_iterator it = m_configurables.begin(), 
       itEnd = m_configurables.end();
       it != itEnd;
       ++it) {
      it->second->addTo(config);
      oConfig.addKeyValue(it->first, config, true);
   }
}

//
// static member functions
//
