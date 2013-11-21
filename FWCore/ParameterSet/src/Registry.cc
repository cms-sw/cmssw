// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

#include <ostream>

#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace pset {
    static ParameterSetID s_ProcessParameterSetID; 

    Registry*
    Registry::instance() {
      static Registry s_reg;
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

    ParameterSetID const&
    getProcessParameterSetID() {
      if (!s_ProcessParameterSetID.isValid()) {
        throw edm::Exception(errors::LogicError)
          << "Illegal attempt to access the process top level parameter set ID\n"
          << "before that parameter set has been frozen and registered.\n"
          << "The parameter set can be changed during module validation,\n"
          << "which occurs concurrently with module construction.\n"
          << "It is illegal to access the parameter set before it is frozen.\n";
      }
      return s_ProcessParameterSetID;
    }

    void setID(ParameterSetID const& id) {
      pset::s_ProcessParameterSetID = id;
    }

  } // namespace pset

  ParameterSet const& getProcessParameterSet() {

    if (!pset::s_ProcessParameterSetID.isValid()) {
      throw edm::Exception(errors::LogicError)
        << "Illegal attempt to access the process top level parameter set ID\n"
        << "before that parameter set has been frozen and registered.\n"
        << "The parameter set can be changed during module validation,\n"
        << "which occurs concurrently with module construction.\n"
        << "It is illegal to access the parameter set before it is frozen.\n";
    }

    pset::Registry const& reg = *pset::Registry::instance();
    ParameterSet const* result;
    if (nullptr == (result = reg.getMapped(pset::s_ProcessParameterSetID))) {
      throw edm::Exception(errors::EventCorruption, "Unknown ParameterSetID")
        << "Unable to find the ParameterSet for id: "
        << pset::s_ProcessParameterSetID
        << ";\nthis was supposed to be the process ParameterSet\n";
    }
    return *result;
  }

} // namespace edm
