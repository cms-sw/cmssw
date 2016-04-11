#include <iostream>
//#include <algorithm>
//#include <utility>

#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"
#include "L1Trigger/L1TCommon/interface/trigSystem.h"

#include "xercesc/util/PlatformUtils.hpp"

XERCES_CPP_NAMESPACE_USE

using namespace l1t;

inline std::string _toString(XMLCh const* toTranscode) {
std::string tmp(xercesc::XMLString::transcode(toTranscode));
return tmp;
}


inline XMLCh* _toDOMS(std::string temp) {
  XMLCh* buff = XMLString::transcode(temp.c_str());
  return  buff;
}


XmlConfigReader::XmlConfigReader() : kModuleNameAlgo("algo"), kModuleNameRunSettings("run-settings")
{
  XMLPlatformUtils::Initialize();
 
  ///Initialise XML parser  
  parser_ = new XercesDOMParser(); 
  parser_->setValidationScheme(XercesDOMParser::Val_Auto);
  parser_->setDoNamespaces(false);

  doc_ = nullptr;
}


XmlConfigReader::XmlConfigReader(DOMDocument* doc) : kModuleNameAlgo("algo"), kModuleNameRunSettings("run-settings")
{
  XMLPlatformUtils::Initialize();
 
  parser_ = nullptr; 
  doc_ = doc;
}


void XmlConfigReader::readDOMFromFile(const std::string& fName, DOMDocument*& doc)
{
  parser_->parse(fName.c_str()); 
  doc = parser_->getDocument();
  assert(doc);
}


void XmlConfigReader::readDOMFromFile(const std::string& fName)
{
  parser_->parse(fName.c_str()); 
  doc_ = parser_->getDocument();
  assert(doc_);
}


void XmlConfigReader::readContext(const DOMElement* element, const std::string& sysId, trigSystem& aTrigSystem)
{
  if (_toString(element->getAttribute(_toDOMS("id"))) == sysId) {
    DOMNodeList* contextElements = element->getElementsByTagName(_toDOMS("context"));

    for (XMLSize_t i = 0; i < contextElements->getLength(); ++i) {
      DOMElement* contextElement = static_cast<DOMElement*>(contextElements->item(i));
      std::string contextId = _toString(contextElement->getAttribute(_toDOMS("id")));

      for (DOMElement* elem = static_cast<DOMElement*>(contextElement->getFirstChild()); elem; elem = static_cast<DOMElement*>(elem->getNextSibling())) {
        if (elem->getNodeType() == DOMNode::ELEMENT_NODE) {
          if (_toString(elem->getTagName()) == "param") {
            // found a parameter
            std::string id = _toString(elem->getAttribute(_toDOMS("id")));
            std::string type = _toString(elem->getAttribute(_toDOMS("type")));
            std::string value = "";
            DOMNodeList* valNodes = elem->getChildNodes();
            for (XMLSize_t j = 0; j < valNodes->getLength(); ++j) {
              if (valNodes->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
                value += _toString(valNodes->item(j)->getNodeValue());
              }
            }
            // strip leading and trailing line breaks and spaces
            std::size_t alphanumBegin = value.find_first_not_of("\n ");
            std::size_t alphanumEnd = value.find_last_not_of("\n ");
            if (alphanumBegin != std::string::npos) {
              if (alphanumEnd != std::string::npos) {
                value = value.substr(alphanumBegin, alphanumEnd - alphanumBegin + 1);
              } else {
                value = value.substr(alphanumBegin);
              }
            }
            //std::cout << "param element node with id attribute " << id << " and type attribute " << type << " with value: [" << value << "]" << std::endl;
            aTrigSystem.addSetting(type, id, value, contextId);
          } else if (_toString(elem->getTagName()) == "mask") {
            // found a mask
            std::string id = _toString(elem->getAttribute(_toDOMS("id")));
            //std::cout << "mask element node with id attribute " << id << std::endl;
            aTrigSystem.addMask(id, contextId);
          }
        }
      }
    }
  }
}


void XmlConfigReader::readContexts(const std::string& key, const std::string& sysId, trigSystem& aTrigSystem)
{
  DOMElement* keyElement = getKeyElement(key);
  if (keyElement) {
    for (DOMElement* elem = static_cast<DOMElement*>(keyElement->getFirstChild()); elem; elem = static_cast<DOMElement*>(elem->getNextSibling())) {
      if (elem->getNodeType() == DOMNode::ELEMENT_NODE) {
        if (_toString(elem->getTagName()) == kModuleNameAlgo || _toString(elem->getTagName()) == kModuleNameRunSettings) {
          readContext(elem, sysId, aTrigSystem);
        }
      }
    }
  } else {
    std::cout << "Key not found: " << key << std::endl;
  }
}


DOMElement* XmlConfigReader::getKeyElement(const std::string& key)
{
  DOMElement* rootElement = doc_->getDocumentElement();
  if (_toString(rootElement->getTagName()) == "db") {
    DOMNodeList* keyElements = rootElement->getElementsByTagName(_toDOMS("key"));

    for (XMLSize_t i = 0; i < keyElements->getLength(); ++i) {
      DOMElement* keyElement = static_cast<DOMElement*>(keyElements->item(i));
      if (_toString(keyElement->getAttribute(_toDOMS("id"))) == key) { // we found the key we were looking for
        return keyElement;
      }
    }
  }
  return nullptr;
}

void XmlConfigReader::buildGlobalDoc(const std::string& key)
{
  DOMElement* keyElement = getKeyElement(key);
  if (keyElement) {
    DOMNodeList* loadElements = keyElement->getElementsByTagName(_toDOMS("load"));
    for (XMLSize_t i = 0; i < loadElements->getLength(); ++i) {
      DOMElement* loadElement = static_cast<DOMElement*>(loadElements->item(i));
      std::string fileName = _toString(loadElement->getAttribute(_toDOMS("module")));
      std::cout << "loading file " << fileName << std::endl;
      DOMDocument* subDoc = nullptr;
      readDOMFromFile(fileName, subDoc);
      if (subDoc) {
        appendNodesFromSubDoc(keyElement, subDoc);
      }
    }
  }
}


void XmlConfigReader::appendNodesFromSubDoc(DOMNode* parentNode, DOMDocument* subDoc)
{
  DOMElement* subDocRootElement = subDoc->getDocumentElement();
  //std::cout << "root element tag: " << _toString(subDocRootElement->getTagName()) << std::endl;
  if (_toString(subDocRootElement->getTagName()) == kModuleNameAlgo || _toString(subDocRootElement->getTagName()) == kModuleNameRunSettings) {
    DOMNode* importedNode = doc_->importNode(subDocRootElement, true);
    parentNode->appendChild(importedNode);
  }
}


