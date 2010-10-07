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
// $Id: FWConfigurationManager.cc,v 1.8 2009/06/29 19:16:28 amraktad Exp $
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "TROOT.h"
#include "TInterpreter.h"

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
void
FWConfigurationManager::add(const std::string& iName, FWConfigurable* iConf)
{
   assert(0!=iConf);
   m_configurables[iName]=iConf;
}

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

void
FWConfigurationManager::writeToFile(const std::string& iName) const
{
   try
   {
      ofstream file(iName.c_str());
      if(not file) {
         std::string message("unable to open file %s ", iName.c_str());
         fflush(stdout);
         message += iName;
         throw std::runtime_error(message.c_str());
      }
      FWConfiguration top;
      to(top);

      printf("Writing to file %s ...\n", iName.c_str());
      fflush(stdout);
      const std::string topName("top");
      file <<"FWConfiguration* fwConfig() {\n"
           <<"  FWConfiguration* "<<topName<<"_p = new FWConfiguration("<<top.version()<<");\n"
           <<"  FWConfiguration& "<<topName<<" = *"<<topName<<"_p;\n";

      for(FWConfiguration::KeyValues::const_iterator it = top.keyValues()->begin();
          it != top.keyValues()->end();
          ++it) {
         addToCode(topName,it->first,it->second, file);
      }
      file<<"\n  return "<<topName<<"_p;\n}\n"<<std::flush;
   }
   catch (std::runtime_error &e) { std::cout << e.what() << std::endl; }
}

void
FWConfigurationManager::readFromFile(const std::string& iName) const
{
   Int_t error=0;

   // Int_t value =
   gROOT->LoadMacro( iName.c_str(), &error );
   if(0 != error) {
      std::string message("unable to load macro file ");
      message += iName;
      throw std::runtime_error(message.c_str());
   }

   const std::string command("(Long_t)(fwConfig() )");

   error = 0;
   Long_t lConfig = gROOT->ProcessLineFast(command.c_str(),
                                           &error);

   {
      //need to unload this macro so that we can load a new configuration
      // which uses the same function name in the macro
      Int_t error = 0;
      gROOT->ProcessLineSync((std::string(".U ")+iName).c_str(), &error);
   }
   if(0 != error) {
      std::string message("unable to properly parse configuration file ");
      message += iName;
      throw std::runtime_error(message.c_str());
   }
   std::auto_ptr<FWConfiguration> config( reinterpret_cast<FWConfiguration*>(lConfig) );

   setFrom( *config);
}

//
// static member functions
//
