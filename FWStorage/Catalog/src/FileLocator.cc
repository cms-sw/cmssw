#include <algorithm>
#include <sstream>
#include <utility>

#include <boost/algorithm/string/replace.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {

  std::string replaceWithRegexp(std::smatch const& matches, std::string const& outputFormat) {
    std::string result = outputFormat;
    std::stringstream str;

    for (size_t i = 1; i < matches.size(); ++i) {
      str.str("");
      str << "$" << i;
      std::string const matchedString(matches[i].first, matches[i].second);
      if (!matchedString.empty())
        boost::algorithm::replace_all(result, str.str(), matchedString);
    }
    return result;
  }

  constexpr char const* const kEmptyString = "";
  constexpr char const* const kLFNPrefix = "/store/";

}  // namespace

namespace pt = boost::property_tree;

namespace edm {

  FileLocator::FileLocator(CatalogAttributes const& catalogAttributes, std::filesystem::path const& filename_storage) {
    //now read json
    pt::ptree json;
    try {
      boost::property_tree::read_json(filename_storage.string(), json);
    } catch (std::exception& e) {
      cms::Exception ex("FileCatalog");
      ex << "Can not open storage.json (" << filename_storage.string()
         << "). Check SITECONFIG_PATH and site-local-config.xml <data-access>";
      ex.addContext("edm::FileLocator:init()");
      throw ex;
    }
    auto found_site = std::find_if(json.begin(), json.end(), [&](pt::ptree::value_type const& site) {
      //get site name
      std::string siteName = site.second.get("site", kEmptyString);
      //get volume name
      std::string volName = site.second.get("volume", kEmptyString);
      return catalogAttributes.storageSite == siteName && catalogAttributes.volume == volName;
    });

    //let enforce that site-local-config.xml and storage.json contains valid catalogs in <data-access>, in which site defined in site-local-config.xml <data-access> should be found in storage.json
    if (found_site == json.end()) {
      cms::Exception ex("FileCatalog");
      ex << "Can not find storage site \"" << catalogAttributes.storageSite << "\" and volume \""
         << catalogAttributes.volume
         << "\" in storage.json. Check site-local-config.xml <data-access> and storage.json";
      ex.addContext("edm::FileLocator:init()");
      throw ex;
    }

    const pt::ptree& protocols = found_site->second.find("protocols")->second;
    auto found_protocol = std::find_if(protocols.begin(), protocols.end(), [&](pt::ptree::value_type const& protocol) {
      std::string protName = protocol.second.get("protocol", kEmptyString);
      return catalogAttributes.protocol == protName;
    });

    //let enforce that site-local-config.xml and storage.json contains valid catalogs, in which protocol defined in site-local-config.xml <data-access> should be found in storage.json
    if (found_protocol == protocols.end()) {
      cms::Exception ex("FileCatalog");
      ex << "Can not find protocol \"" << catalogAttributes.protocol << "\" for the storage site \""
         << catalogAttributes.storageSite << "\" and volume \"" << catalogAttributes.volume
         << "\" in storage.json. Check site-local-config.xml <data-access> and storage.json";
      ex.addContext("edm::FileLocator:init()");
      throw ex;
    }

    m_protocol = found_protocol->second.get("protocol", kEmptyString);

    //store all prefixes and rules to m_directRules. We need to do this so that "applyRules" can find the rule in case chaining is used
    //loop over protocols
    for (pt::ptree::value_type const& protocol : protocols) {
      std::string protName = protocol.second.get("protocol", kEmptyString);
      //loop over rules
      std::string prefixTmp = protocol.second.get("prefix", kEmptyString);
      if (prefixTmp == kEmptyString) {
        const pt::ptree& rules = protocol.second.find("rules")->second;
        for (pt::ptree::value_type const& storageRule : rules) {
          parseRule(storageRule, protName, m_directRules);
        }
      }
      //now convert prefix to a rule and save it
      else {
        Rule rule;
        rule.pathMatch.assign("/?(.*)");
        rule.result = prefixTmp + "/$1";
        rule.chain = kEmptyString;
        m_directRules[protName].emplace_back(std::move(rule));
      }
    }
  }

  std::string FileLocator::pfn(std::string const& ilfn) const {
    //check if ilfn is an authentic LFN
    if (ilfn.compare(0, 7, kLFNPrefix) != 0) {
      return "";
    }
    return applyRules(m_directRules, m_protocol, ilfn);
  }

  void FileLocator::parseRule(pt::ptree::value_type const& storageRule,
                              std::string const& protocol,
                              ProtocolRules& rules) {
    if (storageRule.second.empty()) {
      throw cms::Exception("RucioFileCatalog", "edm::FileLocator::parseRule Malformed storage rule");
    }
    auto const pathMatchRegexp = storageRule.second.get<std::string>("lfn");
    auto const result = storageRule.second.get<std::string>("pfn");
    auto const chain = storageRule.second.get("chain", kEmptyString);
    Rule rule;
    rule.pathMatch.assign(pathMatchRegexp);
    rule.result = result;
    rule.chain = chain;
    rules[protocol].emplace_back(std::move(rule));
  }

  std::string FileLocator::applyRules(ProtocolRules const& protocolRules,
                                      std::string const& protocol,
                                      std::string name) const {
    ProtocolRules::const_iterator const rulesIterator = protocolRules.find(protocol);
    if (rulesIterator == protocolRules.end()) {
      return "";
    }

    Rules const& rules = rulesIterator->second;

    std::smatch nameMatches;

    /* Look up for a matching rule*/
    for (auto const& rule : rules) {
      if (!std::regex_match(name, rule.pathMatch)) {
        continue;
      }

      std::string const& chain = rule.chain;
      if (!chain.empty()) {
        name = applyRules(protocolRules, chain, name);
        if (name.empty()) {
          return "";
        }
      }

      std::regex_match(name, nameMatches, rule.pathMatch);
      return replaceWithRegexp(nameMatches, rule.result);
    }
    return "";
  }
}  // namespace edm
