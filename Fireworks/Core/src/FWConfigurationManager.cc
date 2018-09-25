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
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "TROOT.h"
#include "TSystem.h"
#include "TStopwatch.h"

// user include files
#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWConfigurable.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/src/SimpleSAXParser.h"
#include "Fireworks/Core/interface/FWJobMetadataManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWXMLConfigParser.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWConfigurationManager::FWConfigurationManager():m_ignore(false)
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
   assert(nullptr!=iConf);
   m_configurables[iName]=iConf;
}

//
// const member functions
//
void
FWConfigurationManager::setFrom(const FWConfiguration& iConfig) const
{
   assert(nullptr!=iConfig.keyValues());
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
      std::ofstream file(iName.c_str());
      if(not file) {
         std::string message = "unable to open file " + iName;
         message += iName;
         throw std::runtime_error(message.c_str());
      }
      FWConfiguration top;
      to(top);
      fwLog(fwlog::kInfo) << "Writing to file "<< iName.c_str() << "...\n";
      fflush(stdout);

      FWConfiguration::streamTo(file, top, "top");
   }
   catch (std::runtime_error &e)
   { 
      fwLog(fwlog::kError) << "FWConfigurationManager::writeToFile() " << e.what() << std::endl;
   }
}

void
FWConfigurationManager::readFromOldFile(const std::string& iName) const
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
   std::unique_ptr<FWConfiguration> config( reinterpret_cast<FWConfiguration*>(lConfig) );

   setFrom( *config);
}


/** Reads the configuration specified in @a iName and creates the internal 
    representation in terms of FWConfigutation objects.
    
    Notice that if the file does not start with '<' the old CINT macro based
    system is used.
  */
void
FWConfigurationManager::readFromFile(const std::string& iName) const
{
   std::ifstream f(iName.c_str());
   if (f.peek() != (int) '<')
      return readFromOldFile(iName);
   
   // Check that the syntax is correct.
   SimpleSAXParser syntaxTest(f);
   syntaxTest.parse();
   f.close();
   
   // Read again, this time actually parse.
   std::ifstream g(iName.c_str());
   // Actually parse the results.
   FWXMLConfigParser parser(g);
   parser.parse();
   setFrom(*parser.config());
}

std::string
FWConfigurationManager::guessAndReadFromFile( FWJobMetadataManager* dataMng) const
{
    struct CMatch {
        std::string file;
        int cnt;
        const FWConfiguration* cfg;

        CMatch(std::string f):file(f), cnt(0), cfg(nullptr) {}
        bool operator < (const CMatch& x) const { return cnt < x.cnt; }
    };

    std::vector<CMatch> clist;
    clist.push_back(CMatch("reco.fwc"));
    clist.push_back(CMatch("miniaod.fwc"));
    clist.push_back(CMatch("aod.fwc"));
    std::vector<FWJobMetadataManager::Data> & sdata = dataMng->usableData();

    for (std::vector<CMatch>::iterator c = clist.begin(); c != clist.end(); ++c ) {
        std::string iName = gSystem->Which(TROOT::GetMacroPath(), c->file.c_str(), kReadPermission);
        std::ifstream f(iName.c_str());
        if (f.peek() != (int) '<') {
            fwLog(fwlog::kWarning) << "FWConfigurationManager::guessAndReadFromFile can't open "<<  iName << std::endl ;        
            continue;
        }
   
        // Read again, this time actually parse.
        std::ifstream g(iName.c_str());
        FWXMLConfigParser* parser = new FWXMLConfigParser(g);
        parser->parse();

        c->cfg = parser->config();
        const FWConfiguration::KeyValues* keyValues = nullptr;
        for(FWConfiguration::KeyValues::const_iterator it = c->cfg->keyValues()->begin(),
                itEnd = c->cfg->keyValues()->end();  it != itEnd; ++it) {
            if (it->first == "EventItems" )  {
                keyValues = it->second.keyValues();
                break;
            }
        }
  
        for (FWConfiguration::KeyValues::const_iterator it = keyValues->begin(); it != keyValues->end(); ++it)
        {
            const FWConfiguration& conf = it->second;
            const FWConfiguration::KeyValues* keyValues =  conf.keyValues();
            const std::string& type = (*keyValues)[0].second.value();
            for(std::vector<FWJobMetadataManager::Data>::iterator di = sdata.begin(); di != sdata.end(); ++di)
            {
                if (di->type_ == type) {
                    c->cnt++;
                    break;
                }
            } 
        }
        // printf("%s file %d matches\n", iName.c_str(), c->cnt);
    }
    std::sort(clist.begin(), clist.end());
    fwLog(fwlog::kInfo) << "Loading configuration file "  << clist.back().file << std::endl;
    setFrom(*(clist.back().cfg));

    return clist.back().file;
}

//
// static member functions
//
