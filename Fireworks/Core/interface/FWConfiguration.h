#ifndef Fireworks_Core_FWConfiguration_h
#define Fireworks_Core_FWConfiguration_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWConfiguration
//
/**\class FWConfiguration FWConfiguration.h Fireworks/Core/interface/FWConfiguration.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 22 15:54:22 EST 2008
// $Id: FWConfiguration.h,v 1.7 2011/11/18 02:57:07 amraktad Exp $
//

// system include files
#include <vector>
#include <boost/scoped_ptr.hpp>
#include <string>
#include <ostream>

// user include files

// forward declarations

class FWConfiguration
{

public:
   FWConfiguration(unsigned int iVersion=1) : m_version(iVersion) {
   }
   FWConfiguration(const std::string& iValue) : m_stringValues(new std::vector<std::string>( 1, iValue) ), m_version(0) {
   }
   virtual ~FWConfiguration();

   FWConfiguration(const FWConfiguration&);    // stop default


   FWConfiguration& operator=(const FWConfiguration&);    // stop default
   typedef std::vector<std::pair< std::string, FWConfiguration> > KeyValues;
   typedef KeyValues::const_iterator KeyValuesIt;

   typedef std::vector<std::string> StringValues;
   typedef StringValues::const_iterator StringValuesIt;

   // ---------- const member functions ---------------------
   const std::string& value(unsigned int iIndex=0) const;
   const FWConfiguration* valueForKey(const std::string& iKey) const;
   unsigned int version() const {
      return m_version;
   }

   const KeyValues* keyValues() const {
      return m_keyValues.get();
   }
   const StringValues* stringValues() const {
      return m_stringValues.get();
   }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   FWConfiguration& addKeyValue(const std::string&, const FWConfiguration&);
   FWConfiguration& addKeyValue(const std::string&, FWConfiguration&, bool iDoSwap=false);
   FWConfiguration& addValue(const std::string&);
   void swap(FWConfiguration& );


private:

   // ---------- member data --------------------------------

   boost::scoped_ptr<std::vector<std::string> > m_stringValues;
   boost::scoped_ptr<std::vector<std::pair< std::string, FWConfiguration> > > m_keyValues;
   unsigned int m_version;

};

void streamTo(std::ostream&, const FWConfiguration&, const std::string &name);

#endif
