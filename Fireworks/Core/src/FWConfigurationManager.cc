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
// $Id: FWConfigurationManager.cc,v 1.16 2011/02/22 18:37:31 amraktad Exp $
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "TROOT.h"

// user include files
#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWConfigurable.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/src/SimpleSAXParser.h"

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
      fwLog(fwlog::kInfo) << "Writing to file "<< iName.c_str() << "...\n";
      fflush(stdout);

      streamTo(file, top, "top");
   }
   catch (std::runtime_error &e) { std::cout << e.what() << std::endl; }
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
   std::auto_ptr<FWConfiguration> config( reinterpret_cast<FWConfiguration*>(lConfig) );

   setFrom( *config);
}

void
debug_config_state_machine(const char *where, const std::string &tag, int state)
{
#ifdef FW_CONFIG_PARSER_DEBUG
  static char *debug_states[] = {
     "IN_BEGIN_DOCUMENT",
     "IN_PUSHED_CONFIG",
     "IN_POPPED_CONFIG",
     "IN_BEGIN_STRING",
     "IN_STORED_STRING"
   };

  std::cerr << "  " << where << " tag/data " << tag << "in state " << debug_states[state] << std::endl;
#endif
}

/** Helper class which reads the XML configuration and constructs the
   FWConfiguration classes.
   
   State machine for the parser can be found by cut and pasting the following 
   in graphviz.
   
   digraph {
    IN_BEGIN_DOCUMENT->IN_PUSHED_CONFIG [label = "beginElement(config)"]

    IN_PUSHED_CONFIG->IN_PUSHED_CONFIG [label = "beginElement(config)"]
    IN_PUSHED_CONFIG->IN_POPPED_CONFIG [label = "endElement(config)"]
    IN_PUSHED_CONFIG->IN_BEGIN_STRING [label = "beginElement(string)"]

    IN_POPPED_CONFIG->IN_PUSHED_CONFIG [label = "beginElement(config)"]
    IN_POPPED_CONFIG->IN_POPPED_CONFIG [label = "endElement(config)"]
    IN_POPPED_CONFIG->DONE [label = "top level config popped"]

    IN_BEGIN_STRING->IN_STORED_STRING [label = "data()"];
    IN_BEGIN_STRING->IN_PUSHED_CONFIG [label = "endElement(string)"]

    IN_STORED_STRING->IN_PUSHED_CONFIG [label = "endElement(string)"]
   }
 */
class FWXMLConfigParser : public SimpleSAXParser
{
   enum STATES {
        IN_BEGIN_DOCUMENT,
        IN_PUSHED_CONFIG,
        IN_POPPED_CONFIG,
        IN_BEGIN_STRING,
        IN_STORED_STRING
      };

public:
   FWXMLConfigParser(istream &f) 
   : SimpleSAXParser(f),
     m_state(IN_BEGIN_DOCUMENT),
     m_first(0)
   {}

   /** Pushes the configuration on stack eventually */
   void pushConfig(Attributes &attributes)
   {
      std::string name;
      int version = 0;
      for (size_t i = 0, e = attributes.size(); i != e; ++i)
      {
         Attribute &attr = attributes[i];
         if (attr.key == "name")
            name = attr.value;
         else if (attr.key == "version")
         {
           char *endptr;
           version = strtol(attr.value.c_str(), &endptr, 10);
           if (endptr == attr.value.c_str())
             throw ParserError("Version must be an integer.");
         }
         else
            throw ParserError("Unexpected attribute " + attr.key);
      }
      m_configs.push_back(std::make_pair(name, new FWConfiguration(version)));
   }
   
   
   /** Executes any transaction in the state machine which happens when the 
       xml parser finds an new element.
     */
   virtual void startElement(const std::string &tag, Attributes &attributes)
   {
      debug_config_state_machine("start", tag, m_state);
      if (m_state == IN_BEGIN_DOCUMENT)
      {
         if (tag != "config")
            throw ParserError("Expecting toplevel <config> tag");
         pushConfig(attributes);
         m_first.reset(m_configs.back().second);
         m_state = IN_PUSHED_CONFIG;
      }
      else if (m_state == IN_PUSHED_CONFIG)
      {
         if (tag == "config")
            pushConfig(attributes);
         else if (tag == "string")
            m_state = IN_BEGIN_STRING;
         else
            throw ParserError("Unexpected element " + tag);
      }
      else if (m_state == IN_POPPED_CONFIG)
      {
         if (tag != "config")
            throw ParserError("Unexpected element " + tag);
         pushConfig(attributes);
         m_state = IN_PUSHED_CONFIG;
      }
      else
         throw ParserError("Wrong opening tag found " + tag);
   }

   /** Executes any transaction in the state machine which happens when the 
       xml parser closes an element.

       Notice that we need to do addKeyValue on endElement (and carry around
       the FWConfigutation name) because of the "copy by value"
       policy of addKeyValue addition which would add empty
       FWConfiguration objects if done on startElement.
     */
   virtual void endElement(const std::string &tag)
   {
      debug_config_state_machine("end", tag, m_state);
      if (m_state == IN_PUSHED_CONFIG || m_state == IN_POPPED_CONFIG)
      {
         if (tag != "config")
            throw ParserError("Wrong closing tag found " + tag);
         
         FWConfiguration *current = m_configs.back().second;
         std::string key = m_configs.back().first;
         m_configs.pop_back();
         if (!m_configs.empty())
            m_configs.back().second->addKeyValue(key, *current);
         m_state = IN_POPPED_CONFIG;
      }
      else if (m_state == IN_BEGIN_STRING && tag == "string")
      {
         m_configs.back().second->addValue("");
         m_state = IN_PUSHED_CONFIG;
      }       
      else if (m_state == IN_STORED_STRING && tag == "string")
         m_state = IN_PUSHED_CONFIG;
      else
         throw ParserError("Wrong closing tag found " + tag);
   }

   /** Executes any transaction in the state machine which happens when
       the xml parser finds some data (i.e. text) between tags
       This is mainly used to handle <string> element contents
       but also whitespace between tags.
     */
   virtual void data(const std::string &data)
   {
      debug_config_state_machine("data", data, m_state);
      // We ignore whitespace but complain about any text which is not 
      // in the <string> tag.
      if (m_state == IN_BEGIN_STRING)
      {
         m_configs.back().second->addValue(data);
         m_state = IN_STORED_STRING;
      }
      else if (strspn(data.c_str(), " \t\n") != data.size())
         throw ParserError("Unexpected text " + data);
   }
   
   /** The parsed configuration. Notice that the parser owns it and destroys
       it when destroyed.
     */
   FWConfiguration *config(void)
   {
      return m_first.get();
   }
   
private:
   std::vector<std::pair<std::string, FWConfiguration *> > m_configs;
   enum STATES                                             m_state;
   std::auto_ptr<FWConfiguration>                          m_first;
   //   unsigned int                                            m_currentConfigVersion;
   std::string                                             m_currentConfigName;
};

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

//
// static member functions
//
