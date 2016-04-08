#ifndef __l1t_XmlConfigReader_h__
#define __l1t_XmlConfigReader_h__

#include <string>
#include <vector>

#include "L1Trigger/L1TCommon/interface/trigSystem.h"

#include "xercesc/util/XercesDefs.hpp"
#include "xercesc/parsers/XercesDOMParser.hpp"
#include "xercesc/dom/DOM.hpp"

namespace XERCES_CPP_NAMESPACE {

class DOMNode;
class XercesDOMParser;

}

namespace l1t {

class XmlConfigReader{

 public:

  const std::string kModuleNameAlgo;
  const std::string kModuleNameRunSettings;

  XmlConfigReader();
  XmlConfigReader(xercesc::DOMDocument* doc);
  void readDOMFromFile(const std::string& fName);
  void readDOMFromFile(const std::string& fName, xercesc::DOMDocument*& doc);
  void readContext(const xercesc::DOMElement* element, const std::string& sysId, const std::string& contextId);
  void readContexts(const std::string& key, const std::string& sysId, const std::string& contextId);
  xercesc::DOMElement* getKeyElement(const std::string& key);
  void buildGlobalDoc(const std::string& key);

 private:

  xercesc::XercesDOMParser* parser_;
  xercesc::DOMDocument* doc_;

  void appendNodesFromSubDoc(xercesc::DOMNode* parentNode, xercesc::DOMDocument* subDoc);
};

}
#endif
