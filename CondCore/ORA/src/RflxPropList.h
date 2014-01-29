#ifndef CONDCORE_ORA_RFLXPROPLIST
#define CONDCORE_ORA_RFLXPROPLIST 1

#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"

#include <string>
#include "TDictAttributeMap.h"

namespace Reflex {

    class PropertyList {
    
      public:
          PropertyList(const edm::TypeWithDict   &dict) { 
              m_wp = dict.getClass()->GetAttributeMap();
          }
          PropertyList(const edm::MemberWithDict &dict) { 
              m_wp = dict.typeOf().getClass()->GetAttributeMap();
          }

          PropertyList() : m_wp(0) { /* NOOP */ }

          PropertyList(const PropertyList &other) : m_wp(other.m_wp) { /* NOOP */ }
          PropertyList& operator=(const PropertyList &other) { m_wp = other.getMap(); return *this; }

          TDictAttributeMap * getMap() const { return m_wp; }

          bool HasProperty (const std::string& key) const { 
              if (m_wp) {
                  return m_wp->HasKey( key.c_str() );
              }
              return false;
          }
          std::string PropertyAsString(const std::string& key) const { 
              if (m_wp) {
                  if   (m_wp->HasKey( key.c_str() ) ) {
                      // The class has an attribute named "persistent" specified,
                      // now get the value and reuturn it
                      return std::string( m_wp->GetPropertyAsString( key.c_str() ) );
                  }
              }
              return std::string("");
          }
      private:
          TDictAttributeMap *m_wp;

    }; // end class PropertyList

} // end namespace Reflex

#endif // CONDCORE_ORA_RFLXPROPLIST
