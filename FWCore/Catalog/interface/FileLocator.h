#ifndef FWCore_Catalog_FileLocator_h
#define FWCore_Catalog_FileLocator_h

#include <map>
#include <regex>
#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>

namespace edm {

  struct CatalogAttributes;

  class FileLocator {
  public:
    explicit FileLocator(
        CatalogAttributes const& catalogAttributes,
        unsigned iCatalog = 0,
        //storageDescriptionPath is used to override path provided by SiteLocalConfig. This is used in FileLocator_t.cpp tests
        std::string const& storageDescriptionPath = std::string());

    std::string pfn(std::string const& ilfn) const;

  private:
    struct Rule {
      std::regex pathMatch;
      std::string result;
      std::string chain;
    };

    using Rules = std::vector<Rule>;
    using ProtocolRules = std::map<std::string, Rules>;

    void init(CatalogAttributes const& catalogAttributes, unsigned iCatalog, std::string const& storageDescriptionPath);

    void parseRule(boost::property_tree::ptree::value_type const& storageRule,
                   std::string const& protocol,
                   ProtocolRules& rules);

    std::string applyRules(ProtocolRules const& protocolRules, std::string const& protocol, std::string name) const;

    /** Direct rules are used to do the mapping from LFN to PFN taken from storage.json*/
    ProtocolRules m_directRules;
    std::string m_protocol;
  };
}  // namespace edm

#endif  //  FWCore_Catalog_FileLocator_h
