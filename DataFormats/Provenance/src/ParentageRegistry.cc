#include "DataFormats/Provenance/interface/ParentageRegistry.h"

namespace edm {
  ParentageRegistry*
  ParentageRegistry::instance() {
    static ParentageRegistry s_reg;
    return &s_reg;
  }
  
  bool
  ParentageRegistry::getMapped(key_type const& k, value_type& result) const
  {
    auto it = m_map.find(k);
    bool found = it != m_map.end();
    if(found) {
      result = it->second;
    }
    return found;
  }
  
  ParentageRegistry::value_type const*
  ParentageRegistry::getMapped(key_type const& k) const
  {
    auto it = m_map.find(k);
    bool found = it != m_map.end();
    return found? &(it->second) : static_cast<value_type const*>(nullptr);
  }

  bool
  ParentageRegistry::insertMapped(value_type const& v) {
    return m_map.insert(std::make_pair(v.id(),v)).second;
  }
  
  void
  ParentageRegistry::clear() {
    m_map.clear();
  }
}