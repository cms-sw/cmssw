#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace xercesc;

namespace {

  inline std::string _toString(XMLCh const* toTranscode) {
    std::string tmp(XMLString::transcode(toTranscode));
    return tmp;
  }

  inline XMLCh*  _toDOMS(std::string temp) {
    XMLCh* buff = XMLString::transcode(temp.c_str());
    return  buff;
  }

  std::string
  replaceWithRegexp(boost::smatch const& matches,
                    std::string const& outputFormat) {
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
}

namespace edm {

  int FileLocator::s_numberOfInstances = 0;

  FileLocator::FileLocator(std::string const& catUrl, bool fallback)
    : m_destination("any") {
    try {
      //  << "Xerces-c initialization Number "
      //   << s_numberOfInstances <<
      if (s_numberOfInstances == 0) {
        cms::concurrency::xercesInitialize();
      }
    }
    catch (XMLException const& e) {
      // << "Xerces-c error in initialization \n"
      //      << "Exception message is:  \n"
      //      << _toString(e.getMessage()) <<
      throw
        cms::Exception("TrivialFileCatalog", std::string("Fatal Error on edm::FileLocator:")+ _toString(e.getMessage()));
    }
    ++s_numberOfInstances;

    init(catUrl, fallback);

    // std::cout << m_protocols.size() << " protocols" << std::endl;
    // std::cout << m_directRules[m_protocols[0]].size() << " rules" << std::endl;
  }

  FileLocator::~FileLocator()
  {}


  std::string
  FileLocator::pfn(std::string const& ilfn) const {
    return convert(ilfn, m_directRules, true);
  }

  std::string
  FileLocator::lfn(std::string const& ipfn) const {
    return convert(ipfn, m_inverseRules, false);
  }

  std::string
  FileLocator::convert(std::string const& input, ProtocolRules const& rules, bool direct) const {
    std::string out = "";

    for (size_t pi = 0, pe = m_protocols.size(); pi != pe; ++pi) {
      out = applyRules(rules, m_protocols[pi], m_destination, direct, input);
      if (!out.empty())
        return out;
    }
    return out;
  }

  void
  FileLocator::parseRule(DOMNode* ruleNode, ProtocolRules& rules) {
    if (!ruleNode) {
      throw cms::Exception("TrivialFileCatalog", std::string("TrivialFileCatalog::connect: Malformed trivial catalog"));
    }

    // ruleNode is actually always a DOMElement because it's the result of
    // a `getElementsByTagName()` in the calling method.
    DOMElement* ruleElement = static_cast<DOMElement *>(ruleNode);

    std::string const protocol = _toString(ruleElement->getAttribute(_toDOMS("protocol")));
    std::string destinationMatchRegexp = _toString(ruleElement->getAttribute(_toDOMS("destination-match")));

    if (destinationMatchRegexp.empty()) {
      destinationMatchRegexp = ".*";
    }

    std::string const pathMatchRegexp
      = _toString(ruleElement->getAttribute(_toDOMS("path-match")));
    std::string const result
      = _toString(ruleElement->getAttribute(_toDOMS("result")));
    std::string const chain
      = _toString(ruleElement->getAttribute(_toDOMS("chain")));

    Rule rule;
    rule.pathMatch.assign(pathMatchRegexp);
    rule.destinationMatch.assign(destinationMatchRegexp);
    rule.result = result;
    rule.chain = chain;
    rules[protocol].push_back(rule);
  }

  void
  FileLocator::init(std::string const& catUrl, bool fallback) {
    std::string m_url = catUrl;

    if (m_url.empty()) {
      Service<SiteLocalConfig> localconfservice;
      if (!localconfservice.isAvailable())
              throw cms::Exception("TrivialFileCatalog", "edm::SiteLocalConfigService is not available");

      m_url = (fallback ? localconfservice->fallbackDataCatalog() : localconfservice->dataCatalog());
    }

    // std::cout << "Connecting to the catalog " << m_url << std::endl;

    if (m_url.find("file:") == std::string::npos) {
      throw cms::Exception("TrivialFileCatalog", "TrivialFileCatalog::connect: Malformed url for file catalog configuration");
    }

    m_url = m_url.erase(0, m_url.find(":") + 1);

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
          throw  cms::Exception("TrivialFileCatalog", "TrivialFileCatalog::connect: Malformed url for file catalog configuration");
        }

        if (argTokens[0] == "protocol") {
          boost::algorithm::split(m_protocols, argTokens[1], boost::is_any_of(comma));
        } else if (argTokens[0] == "destination") {
          m_destination = argTokens[1];
        }
      }
    }

    if (m_protocols.empty()) {
      throw cms::Exception("TrivialFileCatalog", "TrivialFileCatalog::connect: protocol was not supplied in the contact string");
    }

    std::ifstream configFile;
    configFile.open(m_filename.c_str());

    //
    // std::cout << "Using catalog configuration " << m_filename << std::endl;

    if (!configFile.good() || !configFile.is_open()) {
      throw cms::Exception("TrivialFileCatalog", "TrivialFileCatalog::connect: Unable to open trivial file catalog " + m_filename);
    }

    configFile.close();

    XercesDOMParser* parser = new XercesDOMParser;
    parser->setValidationScheme(XercesDOMParser::Val_Auto);
    parser->setDoNamespaces(false);
    parser->parse(m_filename.c_str());
    DOMDocument* doc = parser->getDocument();
    assert(doc);

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

    /*first of all do the lfn-to-pfn bit*/
    {
      DOMNodeList* rules = doc->getElementsByTagName(_toDOMS("lfn-to-pfn"));
      unsigned int const ruleTagsNum = rules->getLength();

      // FIXME: we should probably use a DTD for checking validity

      for (unsigned int i = 0; i < ruleTagsNum; ++i) {
        DOMNode* ruleNode = rules->item(i);
        parseRule(ruleNode, m_directRules);
      }
    }
    /*Then we handle the pfn-to-lfn bit*/
    {
      DOMNodeList* rules = doc->getElementsByTagName(_toDOMS("pfn-to-lfn"));
      unsigned int ruleTagsNum = rules->getLength();

      for (unsigned int i = 0; i < ruleTagsNum; ++i) {
        DOMNode* ruleNode = rules->item(i);
        parseRule(ruleNode, m_inverseRules);
      }
    }
  }

  std::string
  FileLocator::applyRules(ProtocolRules const& protocolRules,
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

    boost::smatch destinationMatches;
    boost::smatch nameMatches;

    /* Look up for a matching rule*/
    for (Rules::const_iterator i = rules.begin(); i != rules.end(); ++i) {

      if (!boost::regex_match(destination, destinationMatches, i->destinationMatch)) {
        continue;
      }

      if (!boost::regex_match(name, i->pathMatch)) {
        continue;
      }

      // std::cerr << "Rule " << i->pathMatch << "matched! " << std::endl;

      std::string const chain = i->chain;
      if ((direct == true) && (chain != "")) {
        name = applyRules(protocolRules, chain, destination, direct, name);
        if (name.empty()) {
          return "";
        }
      }

      boost::regex_match(name, nameMatches, i->pathMatch);
      name = replaceWithRegexp(nameMatches, i->result);

      if ((direct == false) && (chain != "")) {
        name = applyRules(protocolRules, chain, destination, direct, name);
      }
      return name;
    }
    return "";
  }
}
