#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <filesystem>
#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace pt = boost::property_tree;

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

  const char* safe(const char* iCheck) {
    if (iCheck == nullptr) {
      return kEmptyString;
    }
    return iCheck;
  }

}  // namespace

namespace pt = boost::property_tree;

namespace edm {

  FileLocator::FileLocator(std::string const& catUrl, unsigned iCatalog) : m_destination("any") {
    init_trivialCatalog(catUrl, iCatalog);
  }

  FileLocator::FileLocator(edm::CatalogAttributes const& catAttr,
                           unsigned iCatalog,
                           std::string const& storageDescriptionPath)
      : m_destination("any") {
    init(catAttr, iCatalog, storageDescriptionPath);
  }

  FileLocator::~FileLocator() {}

  std::string FileLocator::pfn(std::string const& ilfn, edm::CatalogType catType) const {
    if (catType == edm::CatalogType::TrivialCatalog)
      return convert(ilfn, m_directRules_trivialCatalog, true);
    return convert(ilfn, m_directRules, true);
  }

  std::string FileLocator::convert(std::string const& input, ProtocolRules const& rules, bool direct) const {
    std::string out = "";
    //check if input is an authentic LFN
    if (input.compare(0, 7, kLFNPrefix) != 0)
      return out;
    for (size_t pi = 0, pe = m_protocols.size(); pi != pe; ++pi) {
      out = applyRules(rules, m_protocols[pi], m_destination, direct, input);
      if (!out.empty()) {
        return out;
      }
    }
    return out;
  }

  void FileLocator::parseRuleTrivialCatalog(tinyxml2::XMLElement* ruleElement, ProtocolRules& rules) {
    if (!ruleElement) {
      throw cms::Exception("TrivialFileCatalog", std::string("TrivialFileCatalog::connect: Malformed trivial catalog"));
    }

    auto const protocol = safe(ruleElement->Attribute("protocol"));
    auto destinationMatchRegexp = ruleElement->Attribute("destination-match");
    if (destinationMatchRegexp == nullptr or destinationMatchRegexp[0] == 0) {
      destinationMatchRegexp = ".*";
    }

    auto const pathMatchRegexp = safe(ruleElement->Attribute("path-match"));
    auto const result = safe(ruleElement->Attribute("result"));
    auto const chain = safe(ruleElement->Attribute("chain"));

    Rule rule;
    rule.pathMatch.assign(pathMatchRegexp);
    rule.destinationMatch.assign(destinationMatchRegexp);
    rule.result = result;
    rule.chain = chain;
    rules[protocol].emplace_back(std::move(rule));
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
    rule.destinationMatch.assign(".*");
    rule.result = result;
    rule.chain = chain;
    rules[protocol].emplace_back(std::move(rule));
  }

  void FileLocator::init_trivialCatalog(std::string const& catUrl, unsigned iCatalog) {
    std::string url = catUrl;
    if (url.empty()) {
      Service<SiteLocalConfig> localconfservice;
      if (!localconfservice.isAvailable())
        throw cms::Exception("TrivialFileCatalog", "edm::SiteLocalConfigService is not available");
      if (iCatalog >= localconfservice->trivialDataCatalogs().size())
        throw cms::Exception("TrivialFileCatalog", "edm::FileLocator: Request nonexistence data catalog");
      url = localconfservice->trivialDataCatalogs()[iCatalog];
    }

    if (url.find("file:") == std::string::npos) {
      throw cms::Exception("TrivialFileCatalog",
                           "TrivialFileCatalog::connect: Malformed url for file catalog configuration");
    }

    url = url.erase(0, url.find(':') + 1);

    std::vector<std::string> tokens;
    boost::algorithm::split(tokens, url, boost::is_any_of(std::string("?")));
    m_filename = tokens[0];

    if (tokens.size() == 2) {
      std::string const options = tokens[1];
      std::vector<std::string> optionTokens;
      boost::algorithm::split(optionTokens, options, boost::is_any_of(std::string("&")));

      std::string const equalSign("=");
      std::string const comma(",");

      for (size_t oi = 0, oe = optionTokens.size(); oi != oe; ++oi) {
        std::string const option = optionTokens[oi];
        std::vector<std::string> argTokens;
        boost::algorithm::split(argTokens, option, boost::is_any_of(equalSign));

        if (argTokens.size() != 2) {
          throw cms::Exception("TrivialFileCatalog",
                               "TrivialFileCatalog::connect: Malformed url for file catalog configuration");
        }

        if (argTokens[0] == "protocol") {
          boost::algorithm::split(m_protocols, argTokens[1], boost::is_any_of(comma));
        } else if (argTokens[0] == "destination") {
          m_destination = argTokens[1];
        }
      }
    }

    if (m_protocols.empty()) {
      throw cms::Exception("TrivialFileCatalog",
                           "TrivialFileCatalog::connect: protocol was not supplied in the contact string");
    }

    std::ifstream configFile;
    configFile.open(m_filename.c_str());

    if (!configFile.good() || !configFile.is_open()) {
      throw cms::Exception("TrivialFileCatalog",
                           "TrivialFileCatalog::connect: Unable to open trivial file catalog " + m_filename);
    }

    configFile.close();

    tinyxml2::XMLDocument doc;
    auto loadErr = doc.LoadFile(m_filename.c_str());
    if (loadErr != tinyxml2::XML_SUCCESS) {
      throw cms::Exception("TrivialFileCatalog")
          << "tinyxml file load failed with error : " << doc.ErrorStr() << std::endl;
    }
    /* trivialFileCatalog matches the following xml schema
	 FIXME: write a proper DTD
	 <storage-mapping>
	 <lfn-to-pfn protocol="direct" destination-match=".*"
	 path-match="lfn/guid match regular expression"
	 result="/castor/cern.ch/cms/$1"/>
	 <pfn-to-lfn protocol="srm"
	 path-match="lfn/guid match regular expression"
	 result="$1"/>
	 </storage-mapping>
    */
    auto rootElement = doc.RootElement();
    /*first of all do the lfn-to-pfn bit*/
    for (auto el = rootElement->FirstChildElement("lfn-to-pfn"); el != nullptr;
         el = el->NextSiblingElement("lfn-to-pfn")) {
      parseRuleTrivialCatalog(el, m_directRules_trivialCatalog);
    }

    /*Then we handle the pfn-to-lfn bit*/
    for (auto el = rootElement->FirstChildElement("pfn-to-lfn"); el != nullptr;
         el = el->NextSiblingElement("pfn-to-lfn")) {
      parseRuleTrivialCatalog(el, m_inverseRules);
    }
  }

  void FileLocator::init(edm::CatalogAttributes const& input_dataCatalog,
                         unsigned iCatalog,
                         std::string const& storageDescriptionPath) {
    Service<SiteLocalConfig> localconfservice;
    edm::CatalogAttributes aCatalog = input_dataCatalog;
    if (input_dataCatalog.empty()) {
      if (!localconfservice.isAvailable()) {
        cms::Exception ex("FileCatalog");
        ex << "edm::SiteLocalConfigService is not available";
        ex.addContext("Calling edm::FileLocator::init()");
        throw ex;
      }
      if (iCatalog >= localconfservice->dataCatalogs().size()) {
        cms::Exception ex("FileCatalog");
        ex << "Request nonexistence data catalog";
        ex.addContext("Calling edm::FileLocator::init()");
        throw ex;
      }
      aCatalog = localconfservice->dataCatalogs()[iCatalog];
    }

    std::filesystem::path filename_storage = localconfservice->storageDescriptionPath(aCatalog);

    //use path to storage description from input parameter
    if (!storageDescriptionPath.empty())
      filename_storage = storageDescriptionPath;

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
      return aCatalog.storageSite == siteName && aCatalog.volume == volName;
    });

    //let enforce that site-local-config.xml and storage.json contains valid catalogs in <data-access>, in which site defined in site-local-config.xml <data-access> should be found in storage.json
    if (found_site == json.end()) {
      cms::Exception ex("FileCatalog");
      ex << "Can not find storage site \"" << aCatalog.storageSite << "\" and volume \"" << aCatalog.volume
         << "\" in storage.json. Check site-local-config.xml <data-access> and storage.json";
      ex.addContext("edm::FileLocator:init()");
      throw ex;
    }

    const pt::ptree& protocols = found_site->second.find("protocols")->second;
    auto found_protocol = std::find_if(protocols.begin(), protocols.end(), [&](pt::ptree::value_type const& protocol) {
      std::string protName = protocol.second.get("protocol", kEmptyString);
      return aCatalog.protocol == protName;
    });

    //let enforce that site-local-config.xml and storage.json contains valid catalogs, in which protocol defined in site-local-config.xml <data-access> should be found in storage.json
    if (found_protocol == protocols.end()) {
      cms::Exception ex("FileCatalog");
      ex << "Can not find protocol \"" << aCatalog.protocol << "\" for the storage site \"" << aCatalog.storageSite
         << "\" and volume \"" << aCatalog.volume
         << "\" in storage.json. Check site-local-config.xml <data-access> and storage.json";
      ex.addContext("edm::FileLocator:init()");
      throw ex;
    }

    std::string protName = found_protocol->second.get("protocol", kEmptyString);
    m_protocols.push_back(protName);

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
        rule.destinationMatch.assign(".*");
        rule.result = prefixTmp + "/$1";
        rule.chain = kEmptyString;
        m_directRules[protName].emplace_back(std::move(rule));
      }
    }
  }

  std::string FileLocator::applyRules(ProtocolRules const& protocolRules,
                                      std::string const& protocol,
                                      std::string const& destination,
                                      bool direct,
                                      std::string name) const {
    ProtocolRules::const_iterator const rulesIterator = protocolRules.find(protocol);
    if (rulesIterator == protocolRules.end()) {
      return "";
    }

    Rules const& rules = (*(rulesIterator)).second;

    std::smatch destinationMatches;
    std::smatch nameMatches;

    /* Look up for a matching rule*/
    for (Rules::const_iterator i = rules.begin(); i != rules.end(); ++i) {
      if (!std::regex_match(destination, destinationMatches, i->destinationMatch)) {
        continue;
      }

      if (!std::regex_match(name, i->pathMatch)) {
        continue;
      }

      // std::cerr << "Rule " << i->pathMatch << "matched! " << std::endl;

      std::string const chain = i->chain;
      if ((direct == true) && (!chain.empty())) {
        name = applyRules(protocolRules, chain, destination, direct, name);
        if (name.empty()) {
          return "";
        }
      }

      std::regex_match(name, nameMatches, i->pathMatch);
      name = replaceWithRegexp(nameMatches, i->result);

      if ((direct == false) && (!chain.empty())) {
        name = applyRules(protocolRules, chain, destination, direct, name);
      }
      return name;
    }
    return "";
  }
}  // namespace edm
