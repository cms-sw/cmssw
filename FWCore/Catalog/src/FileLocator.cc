#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

  std::string replaceWithRegexp(std::smatch const& matches, std::string const& outputFormat) {
    std::string result = outputFormat;
    std::stringstream str;

    // std::cerr << "Output format: "<< outputFormat << std::endl;
    for (size_t i = 1; i < matches.size(); ++i) {
      str.str("");
      str << "$" << i;
      // std::cerr << "Current match: " << matches[i] << std::endl;
      std::string const matchedString(matches[i].first, matches[i].second);
      if (!matchedString.empty())
        boost::algorithm::replace_all(result, str.str(), matchedString);
    }
    // std::cerr << "Final string: " << result << std::endl;
    return result;
  }

  constexpr char const* const kEmptyString = "";

  const char* safe(const char* iCheck) {
    if (iCheck == nullptr) {
      return kEmptyString;
    }
    return iCheck;
  }

}  // namespace

namespace edm {
  FileLocator::FileLocator(std::string const& catUrl, unsigned iCatalog) : m_destination("any") {
    init(catUrl, iCatalog);

    // std::cout << m_protocols.size() << " protocols" << std::endl;
    // std::cout << m_directRules[m_protocols[0]].size() << " rules" << std::endl;
  }

  FileLocator::~FileLocator() {}

  std::string FileLocator::pfn(std::string const& ilfn) const { return convert(ilfn, m_directRules, true); }

  std::string FileLocator::lfn(std::string const& ipfn) const { return convert(ipfn, m_inverseRules, false); }

  std::string FileLocator::convert(std::string const& input, ProtocolRules const& rules, bool direct) const {
    std::string out = "";

    for (size_t pi = 0, pe = m_protocols.size(); pi != pe; ++pi) {
      out = applyRules(rules, m_protocols[pi], m_destination, direct, input);
      if (!out.empty())
        return out;
    }
    return out;
  }

  void FileLocator::parseRule(tinyxml2::XMLElement* ruleElement, ProtocolRules& rules) {
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

  void FileLocator::init(std::string const& catUrl, unsigned iCatalog) {
    std::string m_url = catUrl;

    if (m_url.empty()) {
      Service<SiteLocalConfig> localconfservice;
      if (!localconfservice.isAvailable())
        throw cms::Exception("TrivialFileCatalog", "edm::SiteLocalConfigService is not available");
      if (iCatalog >= localconfservice->dataCatalogs().size())
        throw cms::Exception("TrivialFileCatalog", "edm::FileLocator: Request nonexistence data catalog");
      m_url = localconfservice->dataCatalogs()[iCatalog];
    }

    if (m_url.find("file:") == std::string::npos) {
      throw cms::Exception("TrivialFileCatalog",
                           "TrivialFileCatalog::connect: Malformed url for file catalog configuration");
    }

    m_url = m_url.erase(0, m_url.find(':') + 1);

    std::vector<std::string> tokens;
    boost::algorithm::split(tokens, m_url, boost::is_any_of(std::string("?")));
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

    //
    // std::cout << "Using catalog configuration " << m_filename << std::endl;

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
      parseRule(el, m_directRules);
    }

    /*Then we handle the pfn-to-lfn bit*/
    for (auto el = rootElement->FirstChildElement("pfn-to-lfn"); el != nullptr;
         el = el->NextSiblingElement("pfn-to-lfn")) {
      parseRule(el, m_inverseRules);
    }
  }

  std::string FileLocator::applyRules(ProtocolRules const& protocolRules,
                                      std::string const& protocol,
                                      std::string const& destination,
                                      bool direct,
                                      std::string name) const {
    // std::cerr << "Calling apply rules with protocol: " << protocol << "\n destination: " << destination << "\n " << " on name " << name << std::endl;

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
