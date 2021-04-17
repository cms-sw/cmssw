#include <iostream>
//#include <algorithm>
//#include <utility>
using namespace std;

#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"

#include "xercesc/util/PlatformUtils.hpp"

XERCES_CPP_NAMESPACE_USE

using namespace l1t;

//inline std::string _toString(XMLCh const* toTranscode) {
//    char *c = xercesc::XMLString::transcode(toTranscode);
//    std::string retval = c;
//    xercesc::XMLString::release(&c);
//    return retval;
//}
//
//inline XMLCh* _toDOMS(std::string temp) {
//  XMLCh* buff = XMLString::transcode(temp.c_str());
//  return  buff;
//}

XmlConfigParser::XmlConfigParser()
    : kTagHw(nullptr),
      kTagAlgo(nullptr),
      kTagRunSettings(nullptr),
      kTagDb(nullptr),
      kTagKey(nullptr),
      kTagLoad(nullptr),
      kTagContext(nullptr),
      kTagParam(nullptr),
      kTagMask(nullptr),
      kTagDisable(nullptr),
      kTagExclBoards(nullptr),
      kTagExclude(nullptr),
      kTagColumns(nullptr),
      kTagTypes(nullptr),
      kTagRow(nullptr),
      kTagProcessor(nullptr),
      kTagRole(nullptr),
      kTagCrate(nullptr),
      kTagSlot(nullptr),
      kTagDaqTtc(nullptr),
      kAttrId(nullptr),
      kAttrType(nullptr),
      kAttrDelim(nullptr),
      kAttrModule(nullptr),
      kTypeTable("table") {
  XMLPlatformUtils::Initialize();

  kTagHw = XMLString::transcode("system");
  kTagAlgo = XMLString::transcode("algo");
  kTagRunSettings = XMLString::transcode("run-settings");
  kTagDb = XMLString::transcode("db");
  kTagKey = XMLString::transcode("key");
  kTagLoad = XMLString::transcode("load");
  kTagContext = XMLString::transcode("context");
  kTagParam = XMLString::transcode("param");
  kTagMask = XMLString::transcode("mask");
  kTagDisable = XMLString::transcode("disable");
  kTagExclBoards = XMLString::transcode("excluded-boards");
  kTagExclude = XMLString::transcode("exclude");
  kTagColumns = XMLString::transcode("columns");
  kTagTypes = XMLString::transcode("types");
  kTagRow = XMLString::transcode("row");
  kTagProcessor = XMLString::transcode("processor");
  kTagRole = XMLString::transcode("role");
  kTagCrate = XMLString::transcode("crate");
  kTagSlot = XMLString::transcode("slot");
  kTagDaqTtc = XMLString::transcode("daqttc-mgr");
  kAttrId = XMLString::transcode("id");
  kAttrType = XMLString::transcode("type");
  kAttrDelim = XMLString::transcode("delimiter");
  kAttrModule = XMLString::transcode("module");

  ///Initialise XML parser
  parser_ = new XercesDOMParser();
  parser_->setValidationScheme(XercesDOMParser::Val_Auto);
  parser_->setDoNamespaces(false);

  doc_ = nullptr;
}

XmlConfigParser::XmlConfigParser(DOMDocument* doc)
    : kTagHw(nullptr),
      kTagAlgo(nullptr),
      kTagRunSettings(nullptr),
      kTagDb(nullptr),
      kTagKey(nullptr),
      kTagLoad(nullptr),
      kTagContext(nullptr),
      kTagParam(nullptr),
      kTagMask(nullptr),
      kTagDisable(nullptr),
      kTagExclBoards(nullptr),
      kTagExclude(nullptr),
      kTagColumns(nullptr),
      kTagTypes(nullptr),
      kTagRow(nullptr),
      kTagProcessor(nullptr),
      kTagRole(nullptr),
      kTagCrate(nullptr),
      kTagSlot(nullptr),
      kTagDaqTtc(nullptr),
      kAttrId(nullptr),
      kAttrType(nullptr),
      kAttrDelim(nullptr),
      kAttrModule(nullptr),
      kTypeTable("table") {
  XMLPlatformUtils::Initialize();

  kTagHw = XMLString::transcode("system");
  kTagAlgo = XMLString::transcode("algo");
  kTagRunSettings = XMLString::transcode("run-settings");
  kTagDb = XMLString::transcode("db");
  kTagKey = XMLString::transcode("key");
  kTagLoad = XMLString::transcode("load");
  kTagContext = XMLString::transcode("context");
  kTagParam = XMLString::transcode("param");
  kTagMask = XMLString::transcode("mask");
  kTagDisable = XMLString::transcode("disable");
  kTagExclBoards = XMLString::transcode("excluded-boards");
  kTagExclude = XMLString::transcode("exclude");
  kTagColumns = XMLString::transcode("columns");
  kTagTypes = XMLString::transcode("types");
  kTagRow = XMLString::transcode("row");
  kTagProcessor = XMLString::transcode("processor");
  kTagRole = XMLString::transcode("role");
  kTagCrate = XMLString::transcode("crate");
  kTagSlot = XMLString::transcode("slot");
  kTagDaqTtc = XMLString::transcode("daqttc-mgr");
  kAttrId = XMLString::transcode("id");
  kAttrType = XMLString::transcode("type");
  kAttrDelim = XMLString::transcode("delimiter");
  kAttrModule = XMLString::transcode("module");

  parser_ = nullptr;
  doc_ = doc;
}

XmlConfigParser::~XmlConfigParser() {
  delete parser_;
  XMLPlatformUtils::Terminate();
}

void XmlConfigParser::readDOMFromString(const std::string& str, DOMDocument*& doc) {
  MemBufInputSource xmlstr_buf((const XMLByte*)(str.c_str()), str.size(), "xmlstrbuf");
  parser_->parse(xmlstr_buf);
  doc = parser_->getDocument();
  assert(doc);
}

void XmlConfigParser::readDOMFromString(const std::string& str) {
  MemBufInputSource xmlstr_buf((const XMLByte*)(str.c_str()), str.size(), "xmlstrbuf");
  parser_->parse(xmlstr_buf);
  doc_ = parser_->getDocument();
  assert(doc_);
}

void XmlConfigParser::readDOMFromFile(const std::string& fName, DOMDocument*& doc) {
  parser_->parse(fName.c_str());
  doc = parser_->getDocument();

  if (!doc) {
    edm::LogError("XmlConfigParser") << "Could not parse file " << fName << "\n";
  }

  assert(doc);
}

void XmlConfigParser::readDOMFromFile(const std::string& fName) {
  parser_->parse(fName.c_str());
  doc_ = parser_->getDocument();

  if (!doc_) {
    edm::LogError("XmlConfigParser") << "Could not parse file " << fName << "\n";
  }

  assert(doc_);
}

void XmlConfigParser::readRootElement(TriggerSystem& aTriggerSystem, const std::string& sysId) {
  DOMElement* rootElement = doc_->getDocumentElement();
  if (rootElement) {
    if (rootElement->getNodeType() == DOMNode::ELEMENT_NODE) {
      readElement(rootElement, aTriggerSystem, sysId);
    }
  } else {
    throw std::runtime_error("No xml root element found");
  }
}

void XmlConfigParser::readElement(const DOMElement* element, TriggerSystem& aTriggerSystem, const std::string& sysId) {
  if (XMLString::equals(element->getTagName(), kTagHw)) {
    // in case this is a HW description
    readHwDescription(element, aTriggerSystem, sysId);
  } else if (XMLString::equals(element->getTagName(), kTagAlgo) ||
             XMLString::equals(element->getTagName(), kTagRunSettings)) {
    // in case this is a configuration snippet
    readContext(element, sysId, aTriggerSystem);
  }
}

void XmlConfigParser::readHwDescription(const DOMElement* element,
                                        TriggerSystem& aTriggerSystem,
                                        const std::string& sysId) {
  // if sysId == "" set the systemId of the trigsystem from the xml sytem id
  if (!sysId.empty()) {
    std::string sId = cms::xerces::toString(element->getAttribute(kAttrId));
    pruneString(sId);
    if (sId != sysId)
      return;
  }

  char* sysId_ = xercesc::XMLString::transcode(element->getAttribute(kAttrId));
  aTriggerSystem.setSystemId(pruneString(sysId_));
  xercesc::XMLString::release(&sysId_);

  // handle processors
  DOMNodeList* processors = element->getElementsByTagName(kTagProcessor);
  XMLSize_t nodeCount = processors->getLength();
  for (XMLSize_t xx = 0; xx < nodeCount; ++xx) {
    DOMNode* currentNode = processors->item(xx);
    if (currentNode->getNodeType() && currentNode->getNodeType() == DOMNode::ELEMENT_NODE) {  //no null and is element
      DOMElement* currentElement = static_cast<DOMElement*>(currentNode);

      string proc = cms::xerces::toString(currentElement->getAttribute(kAttrId));
      pruneString(proc);
      string role, crate, slot;

      DOMNodeList* roles = currentElement->getElementsByTagName(kTagRole);
      // roles of this processor (should be only one)
      if (roles->getLength() != 1)
        throw std::runtime_error(string("Configutration specify not exactly one role"));
      //      for (XMLSize_t i = 0; i < roles->getLength(); ++i) {
      {
        XMLSize_t i = 0;
        DOMNodeList* roleChilds = roles->item(i)->getChildNodes();
        if (roleChilds->getLength() != 1)
          throw std::runtime_error(string("Configutration specify not exactly one role"));
        {
          //        for (XMLSize_t j = 0; j < roleChilds->getLength(); ++j) {
          XMLSize_t j = 0;
          if (roleChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            role = cms::xerces::toString(roleChilds->item(j)->getNodeValue());
            pruneString(role);
          }
        }
      }

      DOMNodeList* crates = currentElement->getElementsByTagName(kTagCrate);
      // crates of this processor (should be only one)
      if (crates->getLength() != 1)
        throw std::runtime_error(string("Configutration specify not exactly one crate"));
      //      for (XMLSize_t i = 0; i < crates->getLength(); ++i) {
      {
        XMLSize_t i = 0;
        DOMNodeList* crateChilds = crates->item(i)->getChildNodes();
        if (crateChilds->getLength() != 1)
          throw std::runtime_error(string("Configutration specify not exactly one crate"));
        {
          XMLSize_t j = 0;
          //        for (XMLSize_t j = 0; j < crateChilds->getLength(); ++j) {
          if (crateChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            crate = cms::xerces::toString(crateChilds->item(j)->getNodeValue());
            pruneString(crate);
          }
        }
      }

      DOMNodeList* slots = currentElement->getElementsByTagName(kTagSlot);
      // slots of this processor (should be only one)
      if (slots->getLength() != 1)
        throw std::runtime_error(string("Configutration specify not exactly one slot"));
      //      for (XMLSize_t i = 0; i < slots->getLength(); ++i) {
      {
        XMLSize_t i = 0;
        DOMNodeList* slotChilds = slots->item(i)->getChildNodes();
        if (slotChilds->getLength() != 1)
          throw std::runtime_error(string("Configutration specify not exactly one slot"));
        //        for (XMLSize_t j = 0; j < slotChilds->getLength(); ++j) {
        {
          XMLSize_t j = 0;
          if (slotChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            slot = cms::xerces::toString(slotChilds->item(j)->getNodeValue());
            pruneString(slot);
          }
        }
      }

      aTriggerSystem.addProcessor(proc.c_str(), role.c_str(), crate.c_str(), slot.c_str());
    }
  }

  // handle DAQ TTC managers
  DOMNodeList* daqttcs = element->getElementsByTagName(kTagDaqTtc);
  nodeCount = daqttcs->getLength();
  for (XMLSize_t xx = 0; xx < nodeCount; ++xx) {
    DOMNode* currentNode = daqttcs->item(xx);
    if (currentNode->getNodeType() && currentNode->getNodeType() == DOMNode::ELEMENT_NODE) {  //no null and is element
      DOMElement* currentElement = static_cast<DOMElement*>(currentNode);
      string daqttc = cms::xerces::toString(currentElement->getAttribute(kAttrId));
      string role, crate;

      DOMNodeList* roles = currentElement->getElementsByTagName(kTagRole);
      // roles of this DAQ TTC manager (should be only one)
      if (roles->getLength() != 1)
        throw std::runtime_error(string("Configutration specify not exactly one role"));
      //      for (XMLSize_t i = 0; i < roles->getLength(); ++i) {
      {
        XMLSize_t i = 0;
        DOMNodeList* roleChilds = roles->item(i)->getChildNodes();
        if (roleChilds->getLength() != 1)
          throw std::runtime_error(string("Configutration specify not exactly one role"));
        //        for (XMLSize_t j = 0; j < roleChilds->getLength(); ++j) {
        {
          XMLSize_t j = 0;
          if (roleChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            role = cms::xerces::toString(roleChilds->item(j)->getNodeValue());
            pruneString(role);
          }
        }
      }

      DOMNodeList* crates = currentElement->getElementsByTagName(kTagCrate);
      // crates of this DAQ TTC manager (should be only one)
      if (crates->getLength() != 1)
        throw std::runtime_error(string("Configutration specify not exactly one crate"));
      //      for (XMLSize_t i = 0; i < crates->getLength(); ++i) {
      {
        XMLSize_t i = 0;
        DOMNodeList* crateChilds = crates->item(i)->getChildNodes();
        if (crateChilds->getLength() != 1)
          throw std::runtime_error(string("Configutration specify not exactly one crate"));
        //        for (XMLSize_t j = 0; j < crateChilds->getLength(); ++j) {
        {
          XMLSize_t j = 0;
          if (crateChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            crate = cms::xerces::toString(crateChilds->item(j)->getNodeValue());
            pruneString(crate);
          }
        }
      }

      aTriggerSystem.addDaq(daqttc.c_str(), role.c_str(), crate.c_str());
    }
  }

  // handle excluded boards
  DOMNodeList* exclBoards = element->getElementsByTagName(kTagExclBoards);
  for (XMLSize_t xx = 0; xx < exclBoards->getLength(); ++xx) {
    DOMElement* exclBoardElem = static_cast<DOMElement*>(exclBoards->item(xx));

    for (DOMElement* elem = static_cast<DOMElement*>(exclBoardElem->getFirstChild()); elem;
         elem = static_cast<DOMElement*>(elem->getNextSibling())) {
      if (elem->getNodeType() == DOMNode::ELEMENT_NODE) {
        if (XMLString::equals(elem->getTagName(), kTagExclude)) {
          // found an excluded board
          std::string id = cms::xerces::toString(elem->getAttribute(kAttrId));
          pruneString(id);
          aTriggerSystem.disableProcOrRoleOrDaq(
              id.c_str());  // handle in the same way as disabled ids in the run-settings
        }
      }
    }
  }
}

void XmlConfigParser::readContext(const DOMElement* element, const std::string& sysId, TriggerSystem& aTriggerSystem) {
  std::string systemId = sysId;
  if (systemId.empty()) {
    systemId = aTriggerSystem.systemId();
  }
  if (cms::xerces::toString(element->getAttribute(kAttrId)) == systemId) {
    DOMNodeList* contextElements = element->getElementsByTagName(kTagContext);

    for (XMLSize_t i = 0; i < contextElements->getLength(); ++i) {
      DOMElement* contextElement = static_cast<DOMElement*>(contextElements->item(i));
      std::string contextId = cms::xerces::toString(contextElement->getAttribute(kAttrId));
      pruneString(contextId);

      for (DOMElement* elem = static_cast<DOMElement*>(contextElement->getFirstChild()); elem;
           elem = static_cast<DOMElement*>(elem->getNextSibling())) {
        if (elem->getNodeType() == DOMNode::ELEMENT_NODE) {
          if (XMLString::equals(elem->getTagName(), kTagParam)) {
            // found a parameter
            string id = cms::xerces::toString(elem->getAttribute(kAttrId));
            string type = cms::xerces::toString(elem->getAttribute(kAttrType));
            string delim = cms::xerces::toString(elem->getAttribute(kAttrDelim));
            pruneString(id);
            pruneString(type);
            pruneString(delim);

            // the type table needs special treatment since it consists of child nodes
            if (type == kTypeTable) {
              // get the columns string
              std::string columnsStr = "";
              DOMNodeList* colElements = elem->getElementsByTagName(kTagColumns);
              for (XMLSize_t j = 0; j < colElements->getLength(); ++j) {
                DOMNodeList* colChilds = colElements->item(j)->getChildNodes();
                for (XMLSize_t k = 0; k < colChilds->getLength(); ++k) {
                  if (colChilds->item(k)->getNodeType() == DOMNode::TEXT_NODE) {
                    ///                    columnsStr = cms::xerces::toString(colChilds->item(k)->getNodeValue());
                    ///                    pruneString(columnsStr);
                    char* cStr = xercesc::XMLString::transcode(colChilds->item(k)->getNodeValue());
                    char *saveptr, first = 1;
                    for (char* item = strtok_r(cStr, delim.c_str(), &saveptr); item != nullptr;
                         item = strtok_r(nullptr, delim.c_str(), &saveptr), first = 0)
                      columnsStr += (first ? std::string("") : delim) + pruneString(item);
                    xercesc::XMLString::release(&cStr);
                  }
                }
              }

              // get the column type string
              std::string typesStr = "";
              DOMNodeList* colTypesElements = elem->getElementsByTagName(kTagTypes);
              for (XMLSize_t j = 0; j < colTypesElements->getLength(); ++j) {
                DOMNodeList* colTypesChilds = colTypesElements->item(j)->getChildNodes();
                for (XMLSize_t k = 0; k < colTypesChilds->getLength(); ++k) {
                  if (colTypesChilds->item(k)->getNodeType() == DOMNode::TEXT_NODE) {
                    ///                    typesStr = cms::xerces::toString(colTypesChilds->item(k)->getNodeValue());
                    ///                    pruneString(typesStr);
                    char* tStr = xercesc::XMLString::transcode(colTypesChilds->item(k)->getNodeValue());
                    char *saveptr, first = 1;
                    for (char* item = strtok_r(tStr, delim.c_str(), &saveptr); item != nullptr;
                         item = strtok_r(nullptr, delim.c_str(), &saveptr), first = 0)
                      typesStr += (first ? std::string("") : delim) + pruneString(item);
                    xercesc::XMLString::release(&tStr);
                  }
                }
              }

              // get the rows
              std::vector<std::string> rowStrs;
              DOMNodeList* rowElements = elem->getElementsByTagName(kTagRow);
              for (XMLSize_t j = 0; j < rowElements->getLength(); ++j) {
                DOMNodeList* rowChilds = rowElements->item(j)->getChildNodes();
                for (XMLSize_t k = 0; k < rowChilds->getLength(); ++k) {
                  if (rowChilds->item(k)->getNodeType() == DOMNode::TEXT_NODE) {
                    ///                    std::string rowStr = cms::xerces::toString(rowChilds->item(k)->getNodeValue());
                    ///                    pruneString(rowStr);
                    ///                    rowStrs.push_back(rowStr);
                    char* rStr = xercesc::XMLString::transcode(rowChilds->item(k)->getNodeValue());
                    char *saveptr, first = 1;
                    std::string row;
                    for (char* item = strtok_r(rStr, delim.c_str(), &saveptr); item != nullptr;
                         item = strtok_r(nullptr, delim.c_str(), &saveptr), first = 0)
                      row += (first ? std::string("") : delim) + pruneString(item);

                    rowStrs.push_back(row);
                    xercesc::XMLString::release(&rStr);
                  }
                }
              }
              //		 std::cout << "param element node with id attribute " << id << " and type attribute " << typesStr << std::endl;
              aTriggerSystem.addTable(
                  id.c_str(), contextId.c_str(), columnsStr.c_str(), typesStr.c_str(), rowStrs, delim.c_str());

            } else {  // all types other than table
              std::string value = "";
              DOMNodeList* valNodes = elem->getChildNodes();
              for (XMLSize_t j = 0; j < valNodes->getLength(); ++j) {
                if (valNodes->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
                  value += cms::xerces::toString(valNodes->item(j)->getNodeValue());  // should we prune?
                }
              }

              // strip leading and trailing line breaks and spaces
              pruneString(value);

              //              std::cout << "param element node with id attribute " << id << " and type attribute " << type << " with value: [" << value << "]" << std::endl;
              aTriggerSystem.addParameter(id.c_str(), contextId.c_str(), type.c_str(), value.c_str(), delim.c_str());
            }

          } else if (XMLString::equals(elem->getTagName(), kTagMask)) {
            // found a Mask
            std::string id = cms::xerces::toString(elem->getAttribute(kAttrId));
            //std::cout << "Mask element node with id attribute " << id << std::endl;
            pruneString(id);
            aTriggerSystem.addMask(id.c_str(), contextId.c_str());

          } else if (XMLString::equals(elem->getTagName(), kTagDisable)) {
            // found a disable
            std::string id = cms::xerces::toString(elem->getAttribute(kAttrId));
            pruneString(id);
            aTriggerSystem.disableProcOrRoleOrDaq(id.c_str());
          }
        }
      }
    }
  }
}

void XmlConfigParser::readContexts(const std::string& key, const std::string& sysId, TriggerSystem& aTriggerSystem) {
  DOMElement* keyElement = getKeyElement(key);
  if (keyElement) {
    for (DOMElement* elem = static_cast<DOMElement*>(keyElement->getFirstChild()); elem;
         elem = static_cast<DOMElement*>(elem->getNextSibling())) {
      if (elem->getNodeType() == DOMNode::ELEMENT_NODE) {
        if (XMLString::equals(elem->getTagName(), kTagAlgo) || XMLString::equals(elem->getTagName(), kTagRunSettings)) {
          readContext(elem, sysId, aTriggerSystem);
        }
      }
    }
  } else {
    throw std::runtime_error("Key not found: " + key);
  }
}

DOMElement* XmlConfigParser::getKeyElement(const std::string& key) {
  DOMElement* rootElement = doc_->getDocumentElement();
  if (XMLString::equals(rootElement->getTagName(), kTagDb)) {
    DOMNodeList* keyElements = rootElement->getElementsByTagName(kTagKey);

    for (XMLSize_t i = 0; i < keyElements->getLength(); ++i) {
      DOMElement* keyElement = static_cast<DOMElement*>(keyElements->item(i));
      if (cms::xerces::toString(keyElement->getAttribute(kAttrId)) == key) {  // we found the key we were looking for
        return keyElement;
      }
    }
  }
  return nullptr;
}

void XmlConfigParser::buildGlobalDoc(const std::string& key, const std::string& topPath) {
  DOMElement* keyElement = getKeyElement(key);
  if (keyElement) {
    DOMNodeList* loadElements = keyElement->getElementsByTagName(kTagLoad);
    for (XMLSize_t i = 0; i < loadElements->getLength(); ++i) {
      DOMElement* loadElement = static_cast<DOMElement*>(loadElements->item(i));
      std::string fileName = cms::xerces::toString(loadElement->getAttribute(kAttrModule));
      if (fileName.find('/') != 0) {  // load element has a relative path
        // build an absolute path with directory of top xml file
        size_t pos;
        std::string topDir = "";
        pos = topPath.find_last_of('/');
        if (pos != std::string::npos) {
          topDir = topPath.substr(0, pos + 1);
        }
        fileName = topDir + fileName;
      }
      //std::cout << "loading file " << fileName << std::endl;
      DOMDocument* subDoc = nullptr;
      readDOMFromFile(fileName, subDoc);
      if (subDoc) {
        appendNodesFromSubDoc(keyElement, subDoc);
      }
    }
  }
}

void XmlConfigParser::appendNodesFromSubDoc(DOMNode* parentNode, DOMDocument* subDoc) {
  DOMElement* subDocRootElement = subDoc->getDocumentElement();
  //std::cout << "root element tag: " << cms::xerces::toString(subDocRootElement->getTagName()) << std::endl;
  if (XMLString::equals(subDocRootElement->getTagName(), kTagAlgo) ||
      XMLString::equals(subDocRootElement->getTagName(), kTagRunSettings)) {
    DOMNode* importedNode = doc_->importNode(subDocRootElement, true);
    parentNode->appendChild(importedNode);
  }
}

// the small static look up table below indicates in O(1) time if the symbol is '\n', (ascii code 10) '\t' (code 9), or ' ' (code 32)
const char XmlConfigParser_reject_symbols[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,                                // '\n'
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                // '\t'
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  // ' '
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

char* XmlConfigParser::pruneString(char*& str) {
  size_t alphanumBegin = 0, alphanumEnd = strlen(str) - 1;

  while (str[alphanumBegin] && XmlConfigParser_reject_symbols[unsigned(str[alphanumBegin])])
    alphanumBegin++;
  while (alphanumEnd >= alphanumBegin && XmlConfigParser_reject_symbols[unsigned(str[alphanumEnd])])
    alphanumEnd--;

  str[alphanumEnd + 1] = '\0';
  return str + alphanumBegin;
}

void XmlConfigParser::pruneString(std::string& str) {
  std::size_t alphanumBegin = str.find_first_not_of("\n\t ");
  std::size_t alphanumEnd = str.find_last_not_of("\n\t ");
  if (alphanumBegin != std::string::npos) {
    if (alphanumEnd != std::string::npos) {
      str = str.substr(alphanumBegin, alphanumEnd - alphanumBegin + 1);
    } else {
      str = str.substr(alphanumBegin);
    }
  }
}
