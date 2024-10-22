#ifndef FWCore_Catalog_FileLocator_h
#define FWCore_Catalog_FileLocator_h

#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include <string>
#include <list>
#include <map>
#include <utility>
#include <regex>
#include "tinyxml2.h"
#include <boost/property_tree/ptree.hpp>

namespace edm {

  class FileLocator {
  public:
    explicit FileLocator(
        edm::CatalogAttributes const& catAttr,
        unsigned iCatalog = 0,
        //storageDescriptionPath is used to override path provided by SiteLocalConfig. This is used in FileLocator_t.cpp tests
        std::string const& storageDescriptionPath = std::string());
    explicit FileLocator(std::string const& catUrl, unsigned iCatalog = 0);

    ~FileLocator();

    std::string pfn(std::string const& ilfn, edm::CatalogType catType) const;

  private:
    /** For the time being the only allowed configuration item is a
     *  prefix to be added to the GUID/LFN.
     */
    static int s_numberOfInstances;

    struct Rule {
      std::regex pathMatch;
      std::regex destinationMatch;
      std::string result;
      std::string chain;
    };

    typedef std::vector<Rule> Rules;
    typedef std::map<std::string, Rules> ProtocolRules;

    void init_trivialCatalog(std::string const& catUrl, unsigned iCatalog);

    void parseRuleTrivialCatalog(tinyxml2::XMLElement* ruleNode, ProtocolRules& rules);
    //using data-access
    void init(edm::CatalogAttributes const& input_dataCatalog,
              unsigned iCatalog,
              std::string const& storageDescriptionPath);
    void parseRule(boost::property_tree::ptree::value_type const& storageRule,
                   std::string const& protocol,
                   ProtocolRules& rules);

    std::string applyRules(ProtocolRules const& protocolRules,
                           std::string const& protocol,
                           std::string const& destination,
                           bool direct,
                           std::string name) const;

    std::string convert(std::string const& input, ProtocolRules const& rules, bool direct) const;

    /** Direct rules are used to do the mapping from LFN to PFN.*/
    ProtocolRules m_directRules_trivialCatalog;
    /** Inverse rules are used to do the mapping from PFN to LFN*/
    ProtocolRules m_inverseRules;
    /** Direct rules are used to do the mapping from LFN to PFN taken from storage.json*/
    ProtocolRules m_directRules;

    std::string m_fileType;
    std::string m_filename;
    //TFC allows more than one protocols provided in a catalog, separated by a comma
    //In new Rucio storage description, only one protocol is provided in a catalog
    //This variable can be simplified in the future
    std::vector<std::string> m_protocols;
    std::string m_destination;
  };
}  // namespace edm

#endif  //  FWCore_Catalog_FileLocator_h
