#ifndef  Fireworks_Core_FWXMLConfigParser
#define  Fireworks_Core_FWXMLConfigParser
#include <istream>
#include <iostream>
#include "Fireworks/Core/src/SimpleSAXParser.h"

#include "Fireworks/Core/interface/FWConfiguration.h"


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
   FWXMLConfigParser(std::istream &f) 
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
   virtual void startElement(const std::string &tag, Attributes &attributes) override
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
   virtual void endElement(const std::string &tag) override
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
   virtual void data(const std::string &data) override
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

private:
   std::vector<std::pair<std::string, FWConfiguration *> > m_configs;
   enum STATES                                             m_state;
   std::auto_ptr<FWConfiguration>                          m_first;
   //   unsigned int                                            m_currentConfigVersion;
   std::string                                             m_currentConfigName;
};
#endif
