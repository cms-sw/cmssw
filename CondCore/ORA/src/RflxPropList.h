#ifndef CONDCORE_ORA_RFLXPROPLIST
#define CONDCORE_ORA_RFLXPROPLIST 1

#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"

#include <string>
#include "TClassAttributeMap.h"

#include "CondProperties.h"

namespace Reflex {

    class PropertyList : public TClassAttributeMap {
    
      public:
          PropertyList(const edm::TypeWithDict   &dict) { m_className = dict.qualifiedName(); }
          PropertyList(const edm::MemberWithDict &dict) { m_className = dict.typeOf().qualifiedName(); }

          PropertyList(std::string className="") : m_className(className) { /* NOOP */ }
          PropertyList(const PropertyList &other) : m_className(other.m_className) { /* NOOP */ }
          PropertyList& operator=(const PropertyList &other) { m_className = other.getName(); return *this; }

          void setName(std::string className) { m_className = className; }
          std::string getName() const { return m_className; }

          bool HasProperty (const std::string& key) const { 
              // return ( std::string( GetPropertyAsString( key.c_str() ) ) != ""); 
              return _hasProperty(key);
          }
          
          std::string PropertyAsString(const std::string& key) const { 
              return std::string( GetPropertyAsString( key.c_str() ) ); 
          }

      protected:
          bool _hasProperty(const std::string &key) const {
            if ( _findProperty(key).second == key ) return true;
            return false;
          }
          std::pair<std::string, std::string> _findProperty(const std::string &key) const {
              auto prop = condProperties.find(m_className)->second; // get the vector of properties
              for ( auto itr = prop.begin(); itr != prop.end(); itr++) {
                  if ( (*itr).first == key ) return *itr;
              }
              return std::make_pair("","");
          }

      private:
          std::string m_className;

    }; // end class PropertyList

} // end namespace Reflex

#endif // CONDCORE_ORA_RFLXPROPLIST
