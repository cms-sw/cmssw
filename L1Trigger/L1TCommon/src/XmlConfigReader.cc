#include <iostream>
//#include <algorithm>
//#include <utility>

#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"
#include "L1Trigger/L1TCommon/interface/TrigSystem.h"

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
  kTagHw(nullptr),
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
  kTagProcessor(  nullptr),
  kTagRole(nullptr),
  kTagCrate(nullptr),
  kTagSlot(nullptr),
  kTagDaqTtc(nullptr),
  kAttrId(nullptr),
  kAttrType(nullptr),
  kAttrDelim(nullptr),
  kAttrModule(nullptr),
  kTypeTable("table")
{
  XMLPlatformUtils::Initialize();

  kTagHw          = XMLString::transcode("system");
  kTagAlgo        = XMLString::transcode("algo");
  kTagRunSettings = XMLString::transcode("run-settings");
  kTagDb          = XMLString::transcode("db");
  kTagKey         = XMLString::transcode("key");
  kTagLoad        = XMLString::transcode("load");
  kTagContext     = XMLString::transcode("context");
  kTagParam       = XMLString::transcode("param");
  kTagMask        = XMLString::transcode("mask");
  kTagDisable     = XMLString::transcode("disable");
  kTagExclBoards  = XMLString::transcode("excluded-boards");
  kTagExclude     = XMLString::transcode("exclude");
  kTagColumns     = XMLString::transcode("columns");
  kTagTypes       = XMLString::transcode("types");
  kTagRow         = XMLString::transcode("row");
  kTagProcessor   = XMLString::transcode("processor");
  kTagRole        = XMLString::transcode("role");
  kTagCrate       = XMLString::transcode("crate");
  kTagSlot        = XMLString::transcode("slot");
  kTagDaqTtc      = XMLString::transcode("daqttc-mgr");
  kAttrId         = XMLString::transcode("id");
  kAttrType       = XMLString::transcode("type");
  kAttrDelim      = XMLString::transcode("delimiter");
  kAttrModule     = XMLString::transcode("module");
 
  ///Initialise XML parser  
  parser_ = new XercesDOMParser(); 
  parser_->setValidationScheme(XercesDOMParser::Val_Auto);
  parser_->setDoNamespaces(false);

  doc_ = nullptr;
}


XmlConfigReader::XmlConfigReader(DOMDocument* doc) :
  kTagHw(nullptr),
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
  kTypeTable("table")
{
  XMLPlatformUtils::Initialize();

  kTagHw          = XMLString::transcode("system");
  kTagAlgo        = XMLString::transcode("algo");
  kTagRunSettings = XMLString::transcode("run-settings");
  kTagDb          = XMLString::transcode("db");
  kTagKey         = XMLString::transcode("key");
  kTagLoad        = XMLString::transcode("load");
  kTagContext     = XMLString::transcode("context");
  kTagParam       = XMLString::transcode("param");
  kTagMask        = XMLString::transcode("mask");
  kTagDisable     = XMLString::transcode("disable");
  kTagExclBoards  = XMLString::transcode("excluded-boards");
  kTagExclude     = XMLString::transcode("exclude");
  kTagColumns     = XMLString::transcode("columns");
  kTagTypes       = XMLString::transcode("types");
  kTagRow         = XMLString::transcode("row");
  kTagProcessor   = XMLString::transcode("processor");
  kTagRole        = XMLString::transcode("role");
  kTagCrate       = XMLString::transcode("crate");
  kTagSlot        = XMLString::transcode("slot");
  kTagDaqTtc      = XMLString::transcode("daqttc-mgr");
  kAttrId         = XMLString::transcode("id");
  kAttrType       = XMLString::transcode("type");
  kAttrDelim      = XMLString::transcode("delimiter");
  kAttrModule     = XMLString::transcode("module");
 
  parser_ = nullptr; 
  doc_ = doc;
}


XmlConfigReader::~XmlConfigReader()
{
  delete parser_;
  XMLPlatformUtils::Terminate();
}


void XmlConfigReader::readDOMFromString(const std::string& str, DOMDocument*& doc)
{
  MemBufInputSource xmlstr_buf((const XMLByte*)(str.c_str()), str.size(), "xmlstrbuf");
  parser_->parse(xmlstr_buf);
  doc = parser_->getDocument();
  assert(doc);
}


void XmlConfigReader::readDOMFromString(const std::string& str)
{
  MemBufInputSource xmlstr_buf((const XMLByte*)(str.c_str()), str.size(), "xmlstrbuf");
  parser_->parse(xmlstr_buf);
  doc_ = parser_->getDocument();
  assert(doc_);
}


void XmlConfigReader::readDOMFromFile(const std::string& fName, DOMDocument*& doc)
{
  parser_->parse(fName.c_str()); 
  doc = parser_->getDocument();

  if (! doc) {
    edm::LogError("XmlConfigReader") << "Could not parse file " << fName << "\n";
  }

  assert(doc);
}


void XmlConfigReader::readDOMFromFile(const std::string& fName)
{
  parser_->parse(fName.c_str()); 
  doc_ = parser_->getDocument();

  if (! doc_) {
    edm::LogError("XmlConfigReader") << "Could not parse file " << fName << "\n";
  }

  assert(doc_);
}


void XmlConfigReader::readRootElement(TrigSystem& aTrigSystem, const std::string& sysId)
{
  DOMElement* rootElement = doc_->getDocumentElement();
  if (rootElement) {
    if (rootElement->getNodeType() == DOMNode::ELEMENT_NODE) {
      readElement(rootElement, aTrigSystem, sysId);
    }
  } else {
    std::cout << "No xml root element found" << std::endl;
  }
}


void XmlConfigReader::readElement(const DOMElement* element, TrigSystem& aTrigSystem, const std::string& sysId)
{
  if (XMLString::equals(element->getTagName(), kTagHw)) {
    // in case this is a HW description
    readHwDescription(element, aTrigSystem, sysId);
  } else if (XMLString::equals(element->getTagName(), kTagAlgo) || XMLString::equals(element->getTagName(), kTagRunSettings)) {
    // in case this is a configuration snippet
    readContext(element, sysId, aTrigSystem);
  }
}


void XmlConfigReader::readHwDescription(const DOMElement* element, TrigSystem& aTrigSystem, const std::string& sysId)
{
  // if sysId == "" set the systemId of the trigsystem from the xml sytem id
  if (sysId != "" && sysId != _toString(element->getAttribute(kAttrId))) {
    return;
  }
  aTrigSystem.setSystemId(_toString(element->getAttribute(kAttrId)));

  // handle processors
  DOMNodeList* processors = element->getElementsByTagName(kTagProcessor);
  XMLSize_t nodeCount = processors->getLength();
  for (XMLSize_t xx = 0; xx < nodeCount; ++xx) {
    DOMNode* currentNode = processors->item(xx);
    if (currentNode->getNodeType() &&  currentNode->getNodeType() == DOMNode::ELEMENT_NODE) { //no null and is element
      DOMElement* currentElement = static_cast<DOMElement*>( currentNode );
      std::string procStr = _toString(currentElement->getAttribute(kAttrId));

      DOMNodeList* roles = currentElement->getElementsByTagName(kTagRole);
      // roles of this processor (should be only one)
      for (XMLSize_t i = 0; i < roles->getLength(); ++i) {
        DOMNodeList* roleChilds = roles->item(i)->getChildNodes();
        for (XMLSize_t j = 0; j < roleChilds->getLength(); ++j) {
          if (roleChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            aTrigSystem.addProcRole(procStr, _toString(roleChilds->item(j)->getNodeValue()));
          }
        }
      }

      DOMNodeList* crates = currentElement->getElementsByTagName(kTagCrate);
      // crates of this processor (should be only one)
      for (XMLSize_t i = 0; i < crates->getLength(); ++i) {
        DOMNodeList* crateChilds = crates->item(i)->getChildNodes();
        for (XMLSize_t j = 0; j < crateChilds->getLength(); ++j) {
          if (crateChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            aTrigSystem.addProcCrate(procStr, _toString(crateChilds->item(j)->getNodeValue()));
          }
        }
      }

      DOMNodeList* slots = currentElement->getElementsByTagName(kTagSlot);
      // slots of this processor (should be only one)
      for (XMLSize_t i = 0; i < slots->getLength(); ++i) {
        DOMNodeList* slotChilds = slots->item(i)->getChildNodes();
        for (XMLSize_t j = 0; j < slotChilds->getLength(); ++j) {
          if (slotChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            aTrigSystem.addProcSlot(procStr, _toString(slotChilds->item(j)->getNodeValue()));
          }
        }
      }

    }
  }

  // handle DAQ TTC managers
  DOMNodeList* daqttcs = element->getElementsByTagName(kTagDaqTtc);
  nodeCount = daqttcs->getLength();
  for (XMLSize_t xx = 0; xx < nodeCount; ++xx) {
    DOMNode* currentNode = daqttcs->item(xx);
    if (currentNode->getNodeType() &&  currentNode->getNodeType() == DOMNode::ELEMENT_NODE) { //no null and is element
      DOMElement* currentElement = static_cast<DOMElement*>( currentNode );
      std::string daqttcStr = _toString(currentElement->getAttribute(kAttrId));

      DOMNodeList* roles = currentElement->getElementsByTagName(kTagRole);
      // roles of this DAQ TTC manager (should be only one)
      for (XMLSize_t i = 0; i < roles->getLength(); ++i) {
        DOMNodeList* roleChilds = roles->item(i)->getChildNodes();
        for (XMLSize_t j = 0; j < roleChilds->getLength(); ++j) {
          if (roleChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            aTrigSystem.addDaqRole(daqttcStr, _toString(roleChilds->item(j)->getNodeValue()));
          }
        }
      }

      DOMNodeList* crates = currentElement->getElementsByTagName(kTagCrate);
      // crates of this DAQ TTC manager (should be only one)
      for (XMLSize_t i = 0; i < crates->getLength(); ++i) {
        DOMNodeList* crateChilds = crates->item(i)->getChildNodes();
        for (XMLSize_t j = 0; j < crateChilds->getLength(); ++j) {
          if (crateChilds->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
            aTrigSystem.addDaqCrate(daqttcStr, _toString(crateChilds->item(j)->getNodeValue()));
          }
        }
      }
    }
  }

  // handle excluded boards
  DOMNodeList* exclBoards = element->getElementsByTagName(kTagExclBoards);
  for (XMLSize_t xx = 0; xx < exclBoards->getLength(); ++xx) {
    DOMElement* exclBoardElem = static_cast<DOMElement*>(exclBoards->item(xx));

    for (DOMElement* elem = static_cast<DOMElement*>(exclBoardElem->getFirstChild()); elem; elem = static_cast<DOMElement*>(elem->getNextSibling())) {
      if (elem->getNodeType() == DOMNode::ELEMENT_NODE) {
        if (XMLString::equals(elem->getTagName(), kTagExclude)) {
          // found an excluded board
          std::string id = _toString(elem->getAttribute(kAttrId));
          aTrigSystem.disableDaqProc(id); // handle in the same way as disabled ids in the run-settings
        }
      }
    }
  }
}


void XmlConfigReader::readContext(const DOMElement* element, const std::string& sysId, TrigSystem& aTrigSystem)
{
  std::string systemId = sysId;
  if (systemId == "") {
    systemId = aTrigSystem.systemId();
  }
  if (_toString(element->getAttribute(kAttrId)) == systemId) {
    DOMNodeList* contextElements = element->getElementsByTagName(kTagContext);

    for (XMLSize_t i = 0; i < contextElements->getLength(); ++i) {
      DOMElement* contextElement = static_cast<DOMElement*>(contextElements->item(i));
      std::string contextId = _toString(contextElement->getAttribute(kAttrId));

      for (DOMElement* elem = static_cast<DOMElement*>(contextElement->getFirstChild()); elem; elem = static_cast<DOMElement*>(elem->getNextSibling())) {
        if (elem->getNodeType() == DOMNode::ELEMENT_NODE) {
          if (XMLString::equals(elem->getTagName(), kTagParam)) {
            // found a parameter
            char *id    = xercesc::XMLString::transcode( elem->getAttribute(kAttrId) );
            char *type  = xercesc::XMLString::transcode( elem->getAttribute(kAttrType) );
            char *delim = xercesc::XMLString::transcode( elem->getAttribute(kAttrDelim) );

            // the type table needs special treatment since it consists of child nodes
            if (type == kTypeTable) {

              // get the columns string
              std::string columnsStr = "";
              DOMNodeList* colElements = elem->getElementsByTagName(kTagColumns);
              for (XMLSize_t j = 0; j < colElements->getLength(); ++j) {
                DOMNodeList* colChilds = colElements->item(j)->getChildNodes();
                for (XMLSize_t k = 0; k < colChilds->getLength(); ++k) {
                  if (colChilds->item(k)->getNodeType() == DOMNode::TEXT_NODE) {
///                    columnsStr = _toString(colChilds->item(k)->getNodeValue());
///                    pruneString(columnsStr);
                    char *cStr = xercesc::XMLString::transcode( colChilds->item(k)->getNodeValue() );
                    columnsStr = pruneString( cStr );
                    xercesc::XMLString::release( &cStr );
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
///                    typesStr = _toString(colTypesChilds->item(k)->getNodeValue());
///                    pruneString(typesStr);
                    char *tStr = xercesc::XMLString::transcode( colTypesChilds->item(k)->getNodeValue() );
                    typesStr = pruneString( tStr );
                    xercesc::XMLString::release( &tStr );
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
///                    std::string rowStr = _toString(rowChilds->item(k)->getNodeValue());
///                    pruneString(rowStr);
///                    rowStrs.push_back(rowStr);
                    char *rStr = xercesc::XMLString::transcode( rowChilds->item(k)->getNodeValue() );
                    rowStrs.push_back( pruneString(rStr) );
                    xercesc::XMLString::release( &rStr );
                  }
                }
              }
//		 std::cout << "param element node with id attribute " << id << " and type attribute " << typesStr << std::endl;
              aTrigSystem.addSettingTable(std::string(id), columnsStr, typesStr, rowStrs, contextId, std::string(delim));

            } else { // all types other than table
              std::string value = "";
              DOMNodeList* valNodes = elem->getChildNodes();
              for (XMLSize_t j = 0; j < valNodes->getLength(); ++j) {
                if (valNodes->item(j)->getNodeType() == DOMNode::TEXT_NODE) {
                  value += _toString(valNodes->item(j)->getNodeValue());
                }
              }

              // strip leading and trailing line breaks and spaces
              pruneString(value);

//              std::cout << "param element node with id attribute " << id << " and type attribute " << type << " with value: [" << value << "]" << std::endl;
              aTrigSystem.addSetting(std::string(type), std::string(id), value, contextId, std::string(delim));
            }

            xercesc::XMLString::release(&id);
            xercesc::XMLString::release(&type);
            xercesc::XMLString::release(&delim);

          } else if (XMLString::equals(elem->getTagName(), kTagMask)) {
            // found a Mask
            std::string id = _toString(elem->getAttribute(kAttrId));
            //std::cout << "Mask element node with id attribute " << id << std::endl;
            aTrigSystem.addMask(id, contextId);

          } else if (XMLString::equals(elem->getTagName(), kTagDisable)) {
            // found a disable
            std::string id = _toString(elem->getAttribute(kAttrId));
            aTrigSystem.disableDaqProc(id);
          }
        }
      }
    }
  }
}


void XmlConfigReader::readContexts(const std::string& key, const std::string& sysId, TrigSystem& aTrigSystem)
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


void XmlConfigReader::buildGlobalDoc(const std::string& key, const std::string& topPath)
{
  DOMElement* keyElement = getKeyElement(key);
  if (keyElement) {
    DOMNodeList* loadElements = keyElement->getElementsByTagName(kTagLoad);
    for (XMLSize_t i = 0; i < loadElements->getLength(); ++i) {
      DOMElement* loadElement = static_cast<DOMElement*>(loadElements->item(i));
      std::string fileName = _toString(loadElement->getAttribute(kAttrModule));
      if (fileName.find("/") != 0) { // load element has a relative path
        // build an absolute path with directory of top xml file
        size_t pos;
        std::string topDir = "";
        pos = topPath.find_last_of("/");
        if (pos != std::string::npos) {
          topDir = topPath.substr(0, pos+1);
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


void XmlConfigReader::appendNodesFromSubDoc(DOMNode* parentNode, DOMDocument* subDoc)
{
  DOMElement* subDocRootElement = subDoc->getDocumentElement();
  //std::cout << "root element tag: " << _toString(subDocRootElement->getTagName()) << std::endl;
  if (XMLString::equals(subDocRootElement->getTagName(), kTagAlgo) || XMLString::equals(subDocRootElement->getTagName(), kTagRunSettings)) {
    DOMNode* importedNode = doc_->importNode(subDocRootElement, true);
    parentNode->appendChild(importedNode);
  }
}

// the small static look up table below indicates in O(1) time if the symbol is '\n', (ascii code 10) '\t' (code 9), or ' ' (code 32)
const char XmlConfigReader_reject_symbols[256] = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // '\n'
1, 0, 0, 0, 0, 0, 0, 0, 0, 0, // '\t'
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, // ' '
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0};

char* XmlConfigReader::pruneString(char* &str)
{
  size_t alphanumBegin = 0, alphanumEnd = strlen(str)-1; 

  while( str[alphanumBegin]           && XmlConfigReader_reject_symbols[ unsigned(str[alphanumBegin]) ] ) alphanumBegin++;
  while( alphanumEnd >= alphanumBegin && XmlConfigReader_reject_symbols[ unsigned(str[alphanumEnd]  ) ] ) alphanumEnd--;

  str[alphanumEnd+1] = '\0';
  return str + alphanumBegin;
}

void XmlConfigReader::pruneString(std::string& str)
{
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

