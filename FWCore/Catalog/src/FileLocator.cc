#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"


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
  replaceWithRegexp(lat::RegexpMatch const matches,
		    std::string const inputString,
		    std::string const outputFormat) {
    // std::cerr << "InputString:" << inputString << std::endl;

    char buffer[8];
    std::string result = outputFormat;

    for (int i = 1; i < matches.numMatches(); ++i) {
	// If this is not true, man, we are in trouble...
	assert(i < 1000000);
	sprintf(buffer, "%i", i);
	std::string variableRegexp = std::string("[$]") + buffer;
	std::string matchResult = matches.matchString(inputString, i);

	lat::Regexp sustitutionToken(variableRegexp);

	//std::cerr << "Current match: " << matchResult << std::endl;

	result = lat::StringOps::replace(result, sustitutionToken, matchResult);
      }
    return result;
  }

}

namespace edm {

  int FileLocator::s_numberOfInstances = 0;

  FileLocator::FileLocator(std::string const& catUrl, bool fallback) :
      m_destination("any") {
    try {
      //  << "Xerces-c initialization Number "
      //   << s_numberOfInstances <<
      if (s_numberOfInstances==0)
	XMLPlatformUtils::Initialize();
    }
    catch (XMLException const& e) {
      // << "Xerces-c error in initialization \n"
      //      << "Exception message is:  \n"
      //      << _toString(e.getMessage()) <<
      throw
	cms::Exception(std::string("Fatal Error on edm::FileLocator:")+ _toString(e.getMessage()));
    }
    ++s_numberOfInstances;

    init(catUrl, fallback);

    // std::cout << m_protocols.size() << " protocols" << std::endl;
    // std::cout << m_directRules[m_protocols[0]].size() << " rules" << std::endl;
  }

  FileLocator::~FileLocator() {
  }


  std::string FileLocator::pfn(std::string const& ilfn) const {
    std::string pfn = "";

    for (lat::StringList::const_iterator protocol = m_protocols.begin();
	 protocol != m_protocols.end();
	 ++protocol) {
      pfn = applyRules(m_directRules,
			*protocol,
			m_destination,
			true,
			ilfn);
      if (!pfn.empty()) {
	return pfn;
      }
    }
    return pfn;
  }

  std::string FileLocator::lfn(std::string const& ipfn) const {
    std::string fid;
    std::string tmpPfn = ipfn;

    for (lat::StringList::const_iterator protocol = m_protocols.begin(); protocol != m_protocols.end(); ++protocol) {
      fid = applyRules(m_inverseRules, *protocol, m_destination, false, tmpPfn);
      if (!fid.empty()) {
        return fid;	
      }
    }
    return fid;
  }

  void
  FileLocator::parseRule(DOMNode *ruleNode, ProtocolRules &rules) {
    if (!ruleNode) {
      throw cms::Exception(std::string("TrivialFileCatalog::connect: Malformed trivial catalog"));
    }

    DOMElement* ruleElement = static_cast<DOMElement *>(ruleNode);	

    if (!ruleElement) {
      throw cms::Exception(std::string("TrivialFileCatalog::connect: Malformed trivial catalog"));
    }

    std::string protocol = _toString(ruleElement->getAttribute(_toDOMS("protocol")));	
    std::string destinationMatchRegexp = _toString(ruleElement->getAttribute(_toDOMS("destination-match")));

    if (destinationMatchRegexp.empty()) {
      destinationMatchRegexp = ".*";
    }

    std::string pathMatchRegexp
      = _toString(ruleElement->getAttribute(_toDOMS("path-match")));
    std::string result
      = _toString(ruleElement->getAttribute(_toDOMS("result")));
    std::string chain
      = _toString(ruleElement->getAttribute(_toDOMS("chain")));

    Rule rule;
    rule.pathMatch.setPattern(pathMatchRegexp);
    rule.pathMatch.compile();
    rule.destinationMatch.setPattern(destinationMatchRegexp);
    rule.destinationMatch.compile();
    rule.result = result;
    rule.chain = chain;
    rules[protocol].push_back(rule);
  }

  void
  FileLocator::init(std::string const& catUrl, bool fallback) {
    std::string m_url = catUrl;

    if (m_url.empty()) {
      Service<SiteLocalConfig> localconfservice;
      if (!localconfservice.isAvailable()) {
	throw cms::Exception("edm::SiteLocalConfigService is not available");
      }
      m_url = (fallback ? localconfservice->fallbackDataCatalog() : localconfservice->dataCatalog());
    }

    // std::cout << "Connecting to the catalog " << m_url << std::endl;


    if (m_url.find("file:") != std::string::npos) {
      m_url = m_url.erase(0, m_url.find(":") + 1);
    } else {
      throw cms::Exception("TrivialFileCatalog::connect: Malformed url for file catalog configuration");
    }

    lat::StringList tokens = lat::StringOps::split(m_url, "?");
    m_filename = tokens[0];

    if (tokens.size() == 2) {
      std::string options = tokens[1];
      lat::StringList optionTokens = lat::StringOps::split(options, "&");

      for (lat::StringList::iterator option = optionTokens.begin(); option != optionTokens.end(); ++option) {
	lat::StringList argTokens = lat::StringOps::split(*option, "=");
	if (argTokens.size() != 2) {
	  throw  cms::Exception("TrivialFileCatalog::connect: Malformed url for file catalog configuration");
	}
	
	std::string key = argTokens[0];
	std::string value = argTokens[1];
	
	if (key == "protocol") {
	  m_protocols = lat::StringOps::split(value, ",");
	} else if (key == "destination") {
	  m_destination = value;
	}
      }
    }

    if (m_protocols.empty()) {
      throw cms::Exception("TrivialFileCatalog::connect: protocol was not supplied in the contact string");
    }

    std::ifstream configFile;
    configFile.open(m_filename.c_str());

    //
    // std::cout << "Using catalog configuration " << m_filename << std::endl;

    if (!configFile.good() || !configFile.is_open()) {
      throw cms::Exception("TrivialFileCatalog::connect: Unable to open trivial file catalog " + m_filename);
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
      DOMNodeList *rules =doc->getElementsByTagName(_toDOMS("lfn-to-pfn"));
      unsigned int ruleTagsNum  =
	rules->getLength();

      // FIXME: we should probably use a DTD for checking validity

      for (unsigned int i = 0; i < ruleTagsNum; ++i) {
	DOMNode* ruleNode = rules->item(i);
	parseRule(ruleNode, m_directRules);
      }
    }
    /*Then we handle the pfn-to-lfn bit*/
    {
      DOMNodeList *rules =doc->getElementsByTagName(_toDOMS("pfn-to-lfn"));
      unsigned int ruleTagsNum  =
	rules->getLength();

      for (unsigned int i = 0; i < ruleTagsNum; ++i){
	DOMNode* ruleNode = rules->item(i);
	parseRule(ruleNode, m_inverseRules);
      }	
    }
  }

  std::string
  replaceWithRegexp(lat::RegexpMatch const matches,
  		   std::string const inputString,
  		   std::string const outputFormat) {
      //std::cerr << "InputString:" << inputString << std::endl;

      char buffer[8];
      std::string result = outputFormat;

      for (int i = 1; i < matches.numMatches(); ++i) {
  	// If this is not true, man, we are in trouble...
  	assert(i < 1000000);
  	sprintf(buffer, "%i", i);
  	std::string variableRegexp = std::string("[$]") + buffer;
  	std::string matchResult = matches.matchString(inputString, i);

  	lat::Regexp sustitutionToken(variableRegexp);

  	//std::cerr << "Current match: " << matchResult << std::endl;

  	result = lat::StringOps::replace(result, sustitutionToken, matchResult);
      }
      return result;
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

    Rules const& rules=(*(rulesIterator)).second;

    /* Look up for a matching rule*/
    for (Rules::const_iterator i = rules.begin(); i != rules.end(); ++i) {
      if (!i->destinationMatch.exactMatch(destination)) {
  	continue;
      }

      if (!i->pathMatch.exactMatch(name)) {
  	continue;
      }

      //std::cerr << "Rule " << i->pathMatch.pattern() << "matched! " << std::endl;

      std::string chain = i->chain;
      if ((direct == true) && (chain != "")) {
        name = applyRules(protocolRules, chain, destination, direct, name);
      }

      lat::RegexpMatch matches;
      i->pathMatch.match(name, 0, 0, &matches);

      name = replaceWithRegexp(matches, name, i->result);

      if ((direct == false) && (chain !="")) {
  	name = applyRules (protocolRules, chain, destination, direct, name);
      }
      return name;
    }
    return "";
  }
}

