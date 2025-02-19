#ifndef __SIMPLE_SAX_PARSER_H_
#define __SIMPLE_SAX_PARSER_H_
/*  A simple SAX-like parser. 

    And yes, I know the S in SAX stands for Simple.
        
    Licensed under GPLv3 license.
    
    TODO: incomplete support for entities.
    TODO: no support for DTD nor <?xml> preamble.
 */

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

bool
fgettoken(std::istream &in, char **buffer, size_t *maxSize, const char *separators,
          int *firstChar);

/** A simple SAX parser which is able to parse the configuration.

    State machine for the parser can be drawn by cut and pasting the following
    to graphviz:
  
    digraph {
    IN_DOCUMENT->IN_BEGIN_TAG [label="nextChar == '<'"];
    IN_DOCUMENT->IN_DATA [label="nextChar != '<'"];
    
    IN_BEGIN_TAG->IN_BEGIN_ELEMENT [label="nextChar >= 'a' && nextChar < 'Z'"];
    IN_BEGIN_TAG->IN_END_ELEMENT [label= "nextChar == '/'"];
    
    IN_BEGIN_ELEMENT->IN_END_ELEMENT [label="nextChar == '/'"];
    IN_BEGIN_ELEMENT->IN_ELEMENT_WHITESPACE [label="nextChar == ' '"];
    IN_BEGIN_ELEMENT->IN_END_TAG [label="nextChar == '>'"];
    
    IN_ELEMENT_WHITESPACE->IN_ELEMENT_WHITESPACE [ label = "nextChar == \"\\ \\t\\n\""]
    IN_ELEMENT_WHITESPACE->IN_ATTRIBUTE_KEY [ label = "nextChar >= 'a' && nextChar < 'Z'"]
    IN_ELEMENT_WHITESPACE->IN_END_ELEMENT [label="nextChar == '/'"]
    
    IN_END_ELEMENT->IN_END_TAG [label = "nextChar == '>'"];
    
    IN_END_TAG->IN_BEGIN_TAG [label="nextChar == '<'"];
    IN_END_TAG->IN_DATA [label="nextChar != '<'"]
    
    IN_DATA->IN_BEGIN_TAG [label="nextChar == '<'"];
    IN_DATA->IN_DATA_ENTITY [label="nextChar == '&'"];
    IN_DATA->IN_DONE [label = "nextChar == EOF"];
    
    IN_DATA_ENTITY->IN_DATA [label="nextChar == ';'"];
    
    IN_ATTRIBUTE_KEY->IN_BEGIN_ATTRIBUTE_VALUE [label = "nextChar == '='"]
    
    IN_BEGIN_ATTRIBUTE_VALUE->IN_STRING [label = "nextChar == '\"' || nextChar == '\'' "]
    
    IN_STRING->IN_END_ATTRIBUTE_VALUE [label = "nextChar == quote"]
    IN_STRING->IN_STRING_ENTITY [label = "nextChar == '&'"]
    
    IN_END_ATTRIBUTE_VALUE->IN_ELEMENT_WHITESPACE [label = "nextChar == ' '"]
    IN_END_ATTRIBUTE_VALUE->IN_END_ELEMENT [label = "nextChar == '/'"]
    IN_END_ATTRIBUTE_VALUE->IN_END_TAG [label = "nextChar == '>'"]
    
    IN_STRING_ENTITY->IN_STRING [label = "nextChar == ';'"]
    }    
    */
class SimpleSAXParser
{
public:
   struct Attribute
   {
      std::string    key;
      std::string    value;

      Attribute(const std::string &iKey, const std::string &iValue)
      :key(iKey), value(iValue)
      {}
      
      Attribute(const Attribute &attr)
      :key(attr.key), value(attr.value)
      {}
      
      bool operator<(const Attribute &attribute) const
      {
         return this->key < attribute.key;
      }
   };

   typedef std::vector<Attribute> Attributes;
   class ParserError
   {
   public:
      ParserError(const std::string &error)
      :m_error(error)
      {}
      
      const char *error() { return m_error.c_str(); }
   private:
      std::string m_error;
   };
   
   enum PARSER_STATES {
      IN_DOCUMENT,
      IN_BEGIN_TAG,
      IN_DONE,
      IN_BEGIN_ELEMENT,
      IN_ELEMENT_WHITESPACE,
      IN_END_ELEMENT,
      IN_ATTRIBUTE_KEY,
      IN_END_TAG,
      IN_DATA,
      IN_BEGIN_ATTRIBUTE_VALUE,
      IN_STRING,
      IN_END_ATTRIBUTE_VALUE,
      IN_STRING_ENTITY,
      IN_DATA_ENTITY
   };
   
   SimpleSAXParser(std::istream &f)
   : m_in(f),
     m_bufferSize(1024),
     m_buffer(new char[m_bufferSize]),
     m_nextChar(m_in.get())
   {}

   virtual ~SimpleSAXParser();
   
   void parse(void);
   
   virtual void startElement(const std::string &/*name*/, 
                             Attributes &/*attributes*/) {}
   virtual void endElement(const std::string &/*name*/) {}
   virtual void data(const std::string &/*data*/) {}

private:
   SimpleSAXParser(const SimpleSAXParser&);    // stop default
   const SimpleSAXParser& operator=(const SimpleSAXParser&);    // stop default
   
   std::string parseEntity(const std::string &entity);
   std::string getToken(const char *delim)
      {
         fgettoken(m_in, &m_buffer, &m_bufferSize, delim, &m_nextChar);
         return m_buffer;
      }

   std::string getToken(const char delim)
      {
         char buf[2] = {delim, 0};
         fgettoken(m_in, &m_buffer, &m_bufferSize, buf, &m_nextChar);
         m_nextChar = m_in.get();
         return m_buffer;
      }
   
   bool skipChar(int c) 
      { 
         if (m_nextChar != c)
            return false;
         m_nextChar = m_in.get();
         return true;
      }
   
   int nextChar(void) { return m_nextChar; }

   std::istream                        &m_in;
   size_t                              m_bufferSize;
   char                                *m_buffer;
   int                                 m_nextChar;
   std::vector<std::string>            m_elementTags;
   Attributes                          m_attributes;
};

// NOTE: put in a .cc if this file is used in more than one place.
#endif // __SIMPLE_SAX_PARSER_H_
