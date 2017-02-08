// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

#include <ostream>

#include "FWCore/ParameterSet/interface/Registry.h"

namespace edm {
  namespace pset {

    Registry*
    Registry::instance() {
      [[cms::thread_safe]] static Registry s_reg;
      return &s_reg;
    }
    
    bool
    Registry::getMapped(key_type const& k, value_type& result) const {
      auto it = m_map.find(k);
      bool found = it != m_map.end();
      if(found) {
        result = it->second;
      }
      return found;
    }
    
    Registry::value_type const*
    Registry::getMapped(key_type const& k) const {
      auto it = m_map.find(k);
      bool found = it != m_map.end();
      return found? &(it->second) : static_cast<value_type const*>(nullptr);
    }
  
    bool
    Registry::insertMapped(value_type const& v) {
      return m_map.insert(std::make_pair(v.id(),v)).second;
    }
    
    void
    Registry::clear() {
      m_map.clear();
    }

    void
    Registry::fillMap(regmap_type& fillme) const {
      fillme.clear();
      // Note: The tracked part is in the registry.
      for (auto const& item : m_map) {
        fillme[item.first].pset() = item.second.toString();
      }
    }

    void
    Registry::print(std::ostream& os) const {
      os << "Registry with " << size() << " entries\n";
      for(auto const& item : *this) {
        os << item.first << " " << item.second << '\n';
      }
    }
  } // namespace pset
} // namespace edm
