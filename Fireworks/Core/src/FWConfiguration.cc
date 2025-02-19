// -*- C++ -*-
//
// Package:     Core
// Class  :     FWConfiguration
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 22 15:54:29 EST 2008
// $Id: FWConfiguration.cc,v 1.7 2011/11/18 02:57:07 amraktad Exp $
//

// system include files
#include <stdexcept>
#include <algorithm>
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWConfiguration.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//FWConfiguration::FWConfiguration()
//{
//}

FWConfiguration::FWConfiguration(const FWConfiguration& rhs) :
   m_stringValues( rhs.m_stringValues ? new std::vector<std::string>(*(rhs.m_stringValues)) : 0),
   m_keyValues( rhs.m_keyValues ? new KeyValues(*(rhs.m_keyValues)) : 0),
   m_version(rhs.m_version)
{
}

FWConfiguration::~FWConfiguration()
{
}

//
// assignment operators
//
FWConfiguration& FWConfiguration::operator=(const FWConfiguration& rhs)
{
   //An exception safe implementation is
   FWConfiguration temp(rhs);
   swap(temp);

   return *this;
}

//
// member functions
//
FWConfiguration&
FWConfiguration::addKeyValue(const std::string& iKey, const FWConfiguration& iConfig)
{
   if( m_stringValues ) {
      throw std::runtime_error("adding key/value to configuration containing string values");
   }
   if(not m_keyValues) {
      m_keyValues.reset(new KeyValues(1, std::make_pair(iKey,iConfig) ) );
   } else {
      m_keyValues->push_back(std::make_pair(iKey,iConfig));
   }
   return *this;
}
FWConfiguration&
FWConfiguration::addKeyValue(const std::string&iKey, FWConfiguration& iConfig, bool iDoSwap)
{
   if( m_stringValues ) {
      throw std::runtime_error("adding key/value to configuration containing string values");
   }
   if( m_stringValues ) {
      throw std::runtime_error("adding key/value to configuration containing string values");
   }
   if(not m_keyValues) {
      if(not iDoSwap) {
         m_keyValues.reset(new KeyValues(1, std::make_pair(iKey,iConfig) ) );
      } else {
         m_keyValues.reset(new KeyValues(1, std::make_pair(iKey,FWConfiguration()) ) );
         m_keyValues->back().second.swap(iConfig);
      }
   } else {
      if(not iDoSwap) {
         m_keyValues->push_back(std::make_pair(iKey,iConfig));
      } else {
         m_keyValues->push_back(std::make_pair(iKey,FWConfiguration()));
         m_keyValues->back().second.swap(iConfig);
      }
   }
   return *this;
}

FWConfiguration&
FWConfiguration::addValue(const std::string& iValue)
{
   if( m_keyValues ) {
      throw std::runtime_error("adding string value to configuration containing key/value pairs");
   }
   if( not m_stringValues ) {
      m_stringValues.reset( new std::vector<std::string>(1,iValue) );
   } else {
      m_stringValues->push_back(iValue);
   }
   return *this;
}

void
FWConfiguration::swap(FWConfiguration& iRHS)
{
   std::swap(m_version, iRHS.m_version);
   m_stringValues.swap(iRHS.m_stringValues);
   m_keyValues.swap(iRHS.m_keyValues);
}

//
// const member functions
//
const std::string&
FWConfiguration::value(unsigned int iIndex) const
{
   if( not m_stringValues ) {
      throw std::runtime_error("no string values set");
   }
   return m_stringValues->at(iIndex);
}

const FWConfiguration*
FWConfiguration::valueForKey(const std::string& iKey) const
{
   if( m_stringValues ) {
      throw std::runtime_error("valueFoKey fails because configuration containing string values");
   }
   if(not m_keyValues) {
      throw std::runtime_error("valueForKey fails becuase no key/values set");
   }
   KeyValues::iterator itFind = std::find_if(m_keyValues->begin(),
                                             m_keyValues->end(),
                                             boost::bind(&std::pair<std::string,FWConfiguration>::first,_1) == iKey);
   if(itFind == m_keyValues->end()) {
      return 0;
   }
   return &(itFind->second);
}


/** Helper function to make sure we escape correctly xml entities embedded in
    the string @a value.
  */
std::string 
attrEscape(std::string value)
{
   std::string::size_type index=0;
   index = 0;
   while (std::string::npos != (index=value.find('&',index))){
      value.replace(index, 1,"&amp;", 5);
      // Do not check "&quot;" for quotes.
      index += 5;
   }

   while (std::string::npos != (index=value.find('"',index))){
      value.replace(index, 1,"&quot;", 6);
      // Do not check "&quot;" for quotes.
      index += 6;
   }
   
   index = 0;
   while (std::string::npos != (index=value.find('<',index))){
      value.replace(index, 1,"&lt;", 4);
      // Do not check "&quot;" for quotes.
      index += 4;
   }

   index = 0;
   while (std::string::npos != (index=value.find('>',index))){
      value.replace(index, 1,"&gt;", 4);
      // Do not check "&quot;" for quotes.
      index += 4;
   }

   return value;
}

/** Streaming FWConfiguration objects to xml.

    Example of dump is:
    
    <config name="top" version="1">
      <string value="S1">
      <string value="S2">
      ...
      <string value="SN">
      <config name="c1">
         ...
      </configuration>
      <config name="c2">
         ...
      </config>
      ...
    </config>
   
    Streaming the top level configuration item will stream all its children.
  */
void
streamTo(std::ostream& oTo, const FWConfiguration& iConfig, const std::string &name)
{
   static int recursionLevel = -1;
   recursionLevel += 1;
   std::string indentation(recursionLevel, ' ');
   oTo << indentation << "<config name=\"" << name 
                      << "\" version=\"" << iConfig.version() << "\">\n";
   if(iConfig.stringValues()) {
      for(FWConfiguration::StringValues::const_iterator it = iConfig.stringValues()->begin();
          it != iConfig.stringValues()->end();
          ++it) {
         oTo << indentation << "  <string>" << attrEscape(*it) << "</string>\n";
      }
   }
   if(iConfig.keyValues()) {
      for(FWConfiguration::KeyValues::const_iterator it = iConfig.keyValues()->begin();
          it != iConfig.keyValues()->end();
          ++it) {
         streamTo(oTo, it->second, attrEscape(it->first));
      }
   }
   oTo << indentation << "</config>" << std::endl;
   recursionLevel -= 1;
}

//
// static member functions
//
