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


XmlConfigReader::XmlConfigReader() :
  kTagHw(         XMLString::transcode("system")),
  kTagAlgo(       XMLString::transcode("algo")),
  kTagRunSettings(XMLString::transcode("run-settings")),
  kTagDb(         XMLString::transcode("db")),
  kTagKey(        XMLString::transcode("key")),
  kTagLoad(       XMLString::transcode("load")),
  kTagContext(    XMLString::transcode("context")),
  kTagParam(      XMLString::transcode("param")),
  kTagMask(       XMLString::transcode("mask")),
  kTagDisable(    XMLString::transcode("disable")),
  kAttrProcessor( XMLString::transcode("processor")),
  kAttrId(        XMLString::transcode("id")),
  kAttrRole(      XMLString::transcode("role")),
  kAttrType(      XMLString::transcode("type")),
  kAttrModule(    XMLString::transcode("module"))
{
  XMLPlatformUtils::Initialize();
 
  ///Initialise XML parser  
  parser_ = new XercesDOMParser(); 
  parser_->setValidationScheme(XercesDOMParser::Val_Auto);
  parser_->setDoNamespaces(false);

  doc_ = nullptr;
}


XmlConfigReader::XmlConfigReader(DOMDocument* doc) :
  kTagHw(         XMLString::transcode("system")),
  kTagAlgo(       XMLString::transcode("algo")),
  kTagRunSettings(XMLString::transcode("run-settings")),
  kTagDb(         XMLString::transcode("db")),
  kTagKey(        XMLString::transcode("key")),
  kTagLoad(       XMLString::transcode("load")),
  kTagContext(    XMLString::transcode("context")),
  kTagParam(      XMLString::transcode("param")),
  kTagMask(       XMLString::transcode("mask")),
  kTagDisable(    XMLString::transcode("disable")),
  kAttrProcessor( XMLString::transcode("processor")),
  kAttrId(        XMLString::transcode("id")),
  kAttrRole(      XMLString::transcode("role")),
  kAttrType(      XMLString::transcode("type")),
  kAttrModule(    XMLString::transcode("module"))
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


void XmlConfigReader::readHwDescription(const DOMElement* element, const std::string& sysId, trigSystem& aTrigSystem)
{
  if (_toString(element->getAttribute(kAttrId)) == sysId) {
    DOMNodeList* processors = element->getElementsByTagName(kAttrProcessor);
    const  XMLSize_t nodeCount = processors->getLength();

    for (XMLSize_t xx = 0; xx < nodeCount; ++xx) {
      DOMNode* currentNode = processors->item(xx);
      if (currentNode->getNodeType() &&  currentNode->getNodeType() == DOMNode::ELEMENT_NODE) { //no null and is element 
        DOMElement* currentElement = static_cast<DOMElement*>( currentNode );
        aTrigSystem.addProcRole(_toString(currentElement->getAttribute(kAttrRole)), _toString(currentElement->getAttribute(kAttrId)));
      }
    }
  }
}


void XmlConfigReader::readContext(const DOMElement* element, const std::string& sysId, trigSystem& aTrigSystem)
{
  if (XMLString::equals(element->getTagName(), kTagHw)) { // in case this is a HW description
    readHwDescription(element, sysId, aTrigSystem);
  } else {
    if (_toString(element->getAttribute(kAttrId)) == sysId) {
      DOMNodeList* contextElements = element->getElementsByTagName(kTagContext);

      for (XMLSize_t i = 0; i < contextElements->getLength(); ++i) {
        DOMElement* contextElement = static_cast<DOMElement*>(contextElements->item(i));
        std::string contextId = _toString(contextElement->getAttribute(kAttrId));

        for (DOMElement* elem = static_cast<DOMElement*>(contextElement->getFirstChild()); elem; elem = static_cast<DOMElement*>(elem->getNextSibling())) {
          if (elem->getNodeType() == DOMNode::ELEMENT_NODE) {
            if (XMLString::equals(elem->getTagName(), kTagParam)) {
              // found a parameter
              std::string id = _toString(elem->getAttribute(kAttrId));
              std::string type = _toString(elem->getAttribute(kAttrType));
              std::string value = "";
              DOMNodeList* valNodes = elem->getChildNodes();
              // the type table needs special treatment since it consists of child nodes
              if (type == "table") {
                // TODO: handle table type
                aTrigSystem.addSetting(type, id, value, contextId);
              } else { // all other types
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
              }
            } else if (XMLString::equals(elem->getTagName(), kTagMask)) {
              // found a mask
              std::string id = _toString(elem->getAttribute(kAttrId));
              //std::cout << "mask element node with id attribute " << id << std::endl;
              aTrigSystem.addMask(id, contextId);
            } else if (XMLString::equals(elem->getTagName(), kTagDisable)) {
              std::string id = _toString(elem->getAttribute(kAttrId));
              aTrigSystem.disableDaqProc(id);
            }
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
        if (XMLString::equals(elem->getTagName(), kTagAlgo) || XMLString::equals(elem->getTagName(), kTagRunSettings)) {
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
  if (XMLString::equals(rootElement->getTagName(), kTagDb)) {
    DOMNodeList* keyElements = rootElement->getElementsByTagName(kTagKey);

    for (XMLSize_t i = 0; i < keyElements->getLength(); ++i) {
      DOMElement* keyElement = static_cast<DOMElement*>(keyElements->item(i));
      if (_toString(keyElement->getAttribute(kAttrId)) == key) { // we found the key we were looking for
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
    DOMNodeList* loadElements = keyElement->getElementsByTagName(kTagLoad);
    for (XMLSize_t i = 0; i < loadElements->getLength(); ++i) {
      DOMElement* loadElement = static_cast<DOMElement*>(loadElements->item(i));
      std::string fileName = _toString(loadElement->getAttribute(kAttrModule));
      //std::cout << "loading file " << fileName << std::endl;
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
  if (XMLString::equals(subDocRootElement->getTagName(), kTagAlgo) || XMLString::equals(subDocRootElement->getTagName(), kTagRunSettings)) {
    DOMNode* importedNode = doc_->importNode(subDocRootElement, true);
    parentNode->appendChild(importedNode);
  }
}


