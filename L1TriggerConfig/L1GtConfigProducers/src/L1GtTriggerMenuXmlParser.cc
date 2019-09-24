/**
 * \class L1GtTriggerMenuXmlParser
 *
 *
 * Description: Xerces-C XML parser for the L1 Trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuXmlParser.h"

// system include files
#include <string>
#include <vector>

#include <iostream>
#include <fstream>
#include <iomanip>

// user include files
// base class
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"

#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "Utilities/Xerces/interface/Xerces.h"
#include <cstdint>

// constructor
L1GtTriggerMenuXmlParser::L1GtTriggerMenuXmlParser()
    : L1GtXmlParserTags(),
      m_xmlErrHandler(nullptr),
      m_triggerMenuInterface("NULL"),
      m_triggerMenuName("NULL"),
      m_triggerMenuImplementation("NULL"),
      m_scaleDbKey("NULL")

{
  // menu names, scale key initialized to NULL due to ORACLE treatment of strings

  // empty
}

// destructor
L1GtTriggerMenuXmlParser::~L1GtTriggerMenuXmlParser() { clearMaps(); }

// set the number of condition chips in GTL
void L1GtTriggerMenuXmlParser::setGtNumberConditionChips(const unsigned int& numberConditionChipsValue) {
  m_numberConditionChips = numberConditionChipsValue;
}

// set the number of pins on the GTL condition chips
void L1GtTriggerMenuXmlParser::setGtPinsOnConditionChip(const unsigned int& pinsOnConditionChipValue) {
  m_pinsOnConditionChip = pinsOnConditionChipValue;
}

// set the correspondence "condition chip - GTL algorithm word"
// in the hardware
void L1GtTriggerMenuXmlParser::setGtOrderConditionChip(const std::vector<int>& orderConditionChipValue) {
  m_orderConditionChip = orderConditionChipValue;
}

// set the number of physics trigger algorithms
void L1GtTriggerMenuXmlParser::setGtNumberPhysTriggers(const unsigned int& numberPhysTriggersValue) {
  m_numberPhysTriggers = numberPhysTriggersValue;
}

// set the number of technical triggers
void L1GtTriggerMenuXmlParser::setGtNumberTechTriggers(const unsigned int& numberTechTriggersValue) {
  m_numberTechTriggers = numberTechTriggersValue;
}

// set the number of L1 jet counts received by GT
void L1GtTriggerMenuXmlParser::setGtNumberL1JetCounts(const unsigned int& numberL1JetCountsValue) {
  m_numberL1JetCounts = numberL1JetCountsValue;
}

// set the condition maps
void L1GtTriggerMenuXmlParser::setGtConditionMap(const std::vector<ConditionMap>& condMap) { m_conditionMap = condMap; }

// set the trigger menu name
void L1GtTriggerMenuXmlParser::setGtTriggerMenuInterface(const std::string& menuInterface) {
  m_triggerMenuInterface = menuInterface;
}

void L1GtTriggerMenuXmlParser::setGtTriggerMenuName(const std::string& menuName) { m_triggerMenuName = menuName; }

void L1GtTriggerMenuXmlParser::setGtTriggerMenuImplementation(const std::string& menuImplementation) {
  m_triggerMenuImplementation = menuImplementation;
}

// set menu associated scale key
void L1GtTriggerMenuXmlParser::setGtScaleDbKey(const std::string& scaleKey) { m_scaleDbKey = scaleKey; }

// set the vectors containing the conditions
void L1GtTriggerMenuXmlParser::setVecMuonTemplate(const std::vector<std::vector<L1GtMuonTemplate> >& vecMuonTempl) {
  m_vecMuonTemplate = vecMuonTempl;
}

void L1GtTriggerMenuXmlParser::setVecCaloTemplate(const std::vector<std::vector<L1GtCaloTemplate> >& vecCaloTempl) {
  m_vecCaloTemplate = vecCaloTempl;
}

void L1GtTriggerMenuXmlParser::setVecEnergySumTemplate(
    const std::vector<std::vector<L1GtEnergySumTemplate> >& vecEnergySumTempl) {
  m_vecEnergySumTemplate = vecEnergySumTempl;
}

void L1GtTriggerMenuXmlParser::setVecJetCountsTemplate(
    const std::vector<std::vector<L1GtJetCountsTemplate> >& vecJetCountsTempl) {
  m_vecJetCountsTemplate = vecJetCountsTempl;
}

void L1GtTriggerMenuXmlParser::setVecCastorTemplate(
    const std::vector<std::vector<L1GtCastorTemplate> >& vecCastorTempl) {
  m_vecCastorTemplate = vecCastorTempl;
}

void L1GtTriggerMenuXmlParser::setVecHfBitCountsTemplate(
    const std::vector<std::vector<L1GtHfBitCountsTemplate> >& vecHfBitCountsTempl) {
  m_vecHfBitCountsTemplate = vecHfBitCountsTempl;
}

void L1GtTriggerMenuXmlParser::setVecHfRingEtSumsTemplate(
    const std::vector<std::vector<L1GtHfRingEtSumsTemplate> >& vecHfRingEtSumsTempl) {
  m_vecHfRingEtSumsTemplate = vecHfRingEtSumsTempl;
}

void L1GtTriggerMenuXmlParser::setVecBptxTemplate(const std::vector<std::vector<L1GtBptxTemplate> >& vecBptxTempl) {
  m_vecBptxTemplate = vecBptxTempl;
}

void L1GtTriggerMenuXmlParser::setVecExternalTemplate(
    const std::vector<std::vector<L1GtExternalTemplate> >& vecExternalTempl) {
  m_vecExternalTemplate = vecExternalTempl;
}

void L1GtTriggerMenuXmlParser::setVecCorrelationTemplate(
    const std::vector<std::vector<L1GtCorrelationTemplate> >& vecCorrelationTempl) {
  m_vecCorrelationTemplate = vecCorrelationTempl;
}

// set the vectors containing the conditions for correlation templates
//
void L1GtTriggerMenuXmlParser::setCorMuonTemplate(const std::vector<std::vector<L1GtMuonTemplate> >& corMuonTempl) {
  m_corMuonTemplate = corMuonTempl;
}

void L1GtTriggerMenuXmlParser::setCorCaloTemplate(const std::vector<std::vector<L1GtCaloTemplate> >& corCaloTempl) {
  m_corCaloTemplate = corCaloTempl;
}

void L1GtTriggerMenuXmlParser::setCorEnergySumTemplate(
    const std::vector<std::vector<L1GtEnergySumTemplate> >& corEnergySumTempl) {
  m_corEnergySumTemplate = corEnergySumTempl;
}

// set the algorithm map (by algorithm names)
void L1GtTriggerMenuXmlParser::setGtAlgorithmMap(const AlgorithmMap& algoMap) { m_algorithmMap = algoMap; }

// set the algorithm map (by algorithm aliases)
void L1GtTriggerMenuXmlParser::setGtAlgorithmAliasMap(const AlgorithmMap& algoMap) { m_algorithmAliasMap = algoMap; }

// set the technical trigger map
void L1GtTriggerMenuXmlParser::setGtTechnicalTriggerMap(const AlgorithmMap& ttMap) { m_technicalTriggerMap = ttMap; }

//

// parse def.xml and vme.xml files
void L1GtTriggerMenuXmlParser::parseXmlFile(const std::string& defXmlFile, const std::string& vmeXmlFile) {
  XERCES_CPP_NAMESPACE_USE

  // resize the vector of condition maps
  // the number of condition chips should be correctly set before calling parseXmlFile
  m_conditionMap.resize(m_numberConditionChips);

  m_vecMuonTemplate.resize(m_numberConditionChips);
  m_vecCaloTemplate.resize(m_numberConditionChips);
  m_vecEnergySumTemplate.resize(m_numberConditionChips);
  m_vecJetCountsTemplate.resize(m_numberConditionChips);
  m_vecCastorTemplate.resize(m_numberConditionChips);
  m_vecHfBitCountsTemplate.resize(m_numberConditionChips);
  m_vecHfRingEtSumsTemplate.resize(m_numberConditionChips);
  m_vecBptxTemplate.resize(m_numberConditionChips);
  m_vecExternalTemplate.resize(m_numberConditionChips);

  m_vecCorrelationTemplate.resize(m_numberConditionChips);
  m_corMuonTemplate.resize(m_numberConditionChips);
  m_corCaloTemplate.resize(m_numberConditionChips);
  m_corEnergySumTemplate.resize(m_numberConditionChips);

  // set the name of the trigger menu name:
  //     defXmlFile, stripped of absolute path and .xml
  // will be overwritten by the value read from the xml file, with a warning if
  // they are not the same
  m_triggerMenuName = defXmlFile;
  size_t xmlPos = m_triggerMenuName.find_last_of("/");
  m_triggerMenuName.erase(m_triggerMenuName.begin(), m_triggerMenuName.begin() + xmlPos + 1);

  xmlPos = m_triggerMenuName.find_last_of(".");
  m_triggerMenuName.erase(m_triggerMenuName.begin() + xmlPos, m_triggerMenuName.end());

  // error handler for xml-parser
  m_xmlErrHandler = nullptr;

  XercesDOMParser* parser;

  LogTrace("L1GtTriggerMenuXmlParser") << "\nOpening XML-File: \n  " << defXmlFile << std::endl;

  if ((parser = initXML(defXmlFile)) != nullptr) {
    workXML(parser);
  }
  cleanupXML(parser);
}

//

void L1GtTriggerMenuXmlParser::setGtTriggerMenuInterfaceDate(const std::string& val) {
  m_triggerMenuInterfaceDate = val;
}

void L1GtTriggerMenuXmlParser::setGtTriggerMenuInterfaceAuthor(const std::string& val) {
  m_triggerMenuInterfaceAuthor = val;
}

void L1GtTriggerMenuXmlParser::setGtTriggerMenuInterfaceDescription(const std::string& val) {
  m_triggerMenuInterfaceDescription = val;
}

void L1GtTriggerMenuXmlParser::setGtTriggerMenuDate(const std::string& val) { m_triggerMenuDate = val; }

void L1GtTriggerMenuXmlParser::setGtTriggerMenuAuthor(const std::string& val) { m_triggerMenuAuthor = val; }

void L1GtTriggerMenuXmlParser::setGtTriggerMenuDescription(const std::string& val) { m_triggerMenuDescription = val; }

void L1GtTriggerMenuXmlParser::setGtAlgorithmImplementation(const std::string& val) { m_algorithmImplementation = val; }

// private methods

/**
 * initXML - Initialize XML-utilities and try to create a parser for the specified file.
 *
 * @param xmlFile Filename of the XML-File
 *
 * @return A reference to a XercesDOMParser object if succeeded. 0 if an error occurred.
 *
 */

XERCES_CPP_NAMESPACE::XercesDOMParser* L1GtTriggerMenuXmlParser::initXML(const std::string& xmlFile) {
  XERCES_CPP_NAMESPACE_USE

  // try to initialize
  try {
    cms::concurrency::xercesInitialize();
  } catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());

    edm::LogError("L1GtTriggerMenuXmlParser") << "Error during Xerces-c initialization! :" << message << std::endl;

    XMLString::release(&message);
    return nullptr;
  }

  XercesDOMParser* parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Always);
  parser->setDoNamespaces(false);  // we got no dtd

  if (m_xmlErrHandler == nullptr) {  // redundant check
    m_xmlErrHandler = (ErrorHandler*)new HandlerBase();
  } else {
    // TODO ASSERTION
  }
  parser->setErrorHandler(m_xmlErrHandler);

  // try to parse the file
  try {
    parser->parse(xmlFile.c_str());
  } catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());

    edm::LogError("L1GtTriggerMenuXmlParser") << "Exception while parsing XML: \n" << message << std::endl;

    XMLString::release(&message);
    delete parser;
    delete m_xmlErrHandler;
    m_xmlErrHandler = nullptr;
    return nullptr;
  } catch (const DOMException& toCatch) {
    char* message = XMLString::transcode(toCatch.msg);

    edm::LogError("L1GtTriggerMenuXmlParser") << "DOM-Exception while parsing XML: \n" << message << std::endl;

    XMLString::release(&message);
    delete parser;
    delete m_xmlErrHandler;
    m_xmlErrHandler = nullptr;
    return nullptr;
  } catch (...) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Unexpected Exception while parsing XML!" << std::endl;

    delete parser;
    delete m_xmlErrHandler;
    m_xmlErrHandler = nullptr;
    return nullptr;
  }

  return parser;
}

// find a named child of a xml node
XERCES_CPP_NAMESPACE::DOMNode* L1GtTriggerMenuXmlParser::findXMLChild(XERCES_CPP_NAMESPACE::DOMNode* startChild,
                                                                      const std::string& tagName,
                                                                      bool beginOnly,
                                                                      std::string* rest) {
  XERCES_CPP_NAMESPACE_USE

  char* nodeName = nullptr;

  DOMNode* n1 = startChild;
  if (n1 == nullptr) {
    return nullptr;
  }

  if (!tagName.empty()) {
    nodeName = XMLString::transcode(n1->getNodeName());

    if (!beginOnly) {
      //match the whole tag
      while (XMLString::compareIString(nodeName, tagName.c_str())) {
        XMLString::release(&nodeName);
        n1 = n1->getNextSibling();
        if (n1 == nullptr) {
          break;
        }

        nodeName = XMLString::transcode(n1->getNodeName());
      }
    } else {
      // match only the beginning
      while (XMLString::compareNIString(nodeName, tagName.c_str(), tagName.length())) {
        XMLString::release(&nodeName);
        n1 = n1->getNextSibling();
        if (n1 == nullptr) {
          break;
        }

        nodeName = XMLString::transcode(n1->getNodeName());
      }
      if (n1 != nullptr && rest != nullptr) {
        *rest = std::string(nodeName).substr(tagName.length(), strlen(nodeName) - tagName.length());
      }
    }
  } else {  // empty string given
    while (n1->getNodeType() != DOMNode::ELEMENT_NODE) {
      n1 = n1->getNextSibling();
      if (n1 == nullptr) {
        break;
      }
    }
    if (n1 != nullptr && rest != nullptr) {
      nodeName = XMLString::transcode(n1->getNodeName());
      *rest = std::string(nodeName);
    }
  }

  XMLString::release(&nodeName);

  return n1;
}

/**
 * getXMLAttribute - get a named attribute from a node
 *
 * @param node The node to get the attribute from
 * @param name The name of the attribut to get
 *
 * @return The value of the attribute or empty string if an error occurred.
 */

std::string L1GtTriggerMenuXmlParser::getXMLAttribute(const XERCES_CPP_NAMESPACE::DOMNode* node,
                                                      const std::string& name) {
  XERCES_CPP_NAMESPACE_USE

  std::string ret;

  // get attributes list
  DOMNamedNodeMap* attributes = node->getAttributes();
  if (attributes == nullptr) {
    return ret;
  }

  // get attribute node
  XMLCh* attrName = XMLString::transcode(name.c_str());
  DOMNode* attribNode = attributes->getNamedItem(attrName);

  XMLString::release(&attrName);
  if (attribNode == nullptr) {
    return ret;
  }

  char* retCstr = XMLString::transcode(attribNode->getNodeValue());
  ret = retCstr;
  XMLString::release(&retCstr);

  return ret;
}

/**
 * getXMLTextValue - get the textvalue from a specified node
 *
 * @param node The reference to the node.
 * @return The textvalue of the node or an empty std::string if no textvalue is available.
 *
 */

std::string L1GtTriggerMenuXmlParser::getXMLTextValue(XERCES_CPP_NAMESPACE::DOMNode* node) {
  XERCES_CPP_NAMESPACE_USE

  std::string ret;

  DOMNode* n1 = node;
  if (n1 == nullptr) {
    return ret;
  }

  const XMLCh* retXmlCh = n1->getTextContent();
  if (retXmlCh == nullptr) {
    return ret;
  }

  char* retCstr = XMLString::transcode(retXmlCh);
  XMLString::trim(retCstr);  // trim spaces

  ret = retCstr;
  XMLString::release(&retCstr);

  return ret;
}

/**
 * hexString2UInt128 converts an up to 128 bit hexadecimal string to two uint64_t
 *
 * @param hex The string to be converted.
 * @param dstL The target for the lower 64 bit.
 * @param dstH The target for the upper 64 bit.
 *
 * @return true if conversion succeeded, false if an error occurred.
 */

bool L1GtTriggerMenuXmlParser::hexString2UInt128(const std::string& hexString, uint64_t& dstL, uint64_t& dstH) {
  // string to determine start of hex value, do not ignore leading zeros
  static const std::string valid_hex_start("0123456789ABCDEFabcdef");

  // string to determine end of hex value
  static const std::string valid_hex_end("0123456789ABCDEFabcdef");

  std::string tempStr = hexString;

  // start / end position of the hex value in the string
  unsigned int hexStart = tempStr.find_first_of(valid_hex_start);
  unsigned int hexEnd = tempStr.find_first_not_of(valid_hex_end, hexStart);

  if (hexStart == hexEnd) {
    LogDebug("L1GtTriggerMenuXmlParser") << "No hex value found in: " << tempStr << std::endl;

    return false;
  }

  tempStr = tempStr.substr(hexStart, hexEnd - hexStart);

  if (tempStr.empty()) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Empty value in " << __PRETTY_FUNCTION__ << std::endl;

    return false;
  }

  // split the string
  std::string tempStrH, tempStrL;

  if (tempStr.length() > 16) {  // more than 64 bit
    tempStrL = tempStr.substr(tempStr.length() - 16, 16);
    tempStrH = tempStr.substr(0, tempStr.length() - 16);
  } else {
    tempStrL = tempStr;
    tempStrH = "0";
  }

  // convert lower 64bit
  char* endPtr = (char*)tempStrL.c_str();
  uint64_t tempUIntL = strtoull(tempStrL.c_str(), &endPtr, 16);

  if (*endPtr != 0) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Unable to convert " << tempStr << " to hex." << std::endl;

    return false;
  }

  // convert higher64 bit
  endPtr = (char*)tempStrH.c_str();
  uint64_t tempUIntH = strtoull(tempStrH.c_str(), &endPtr, 16);

  if (*endPtr != 0) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Unable to convert " << tempStr << " to hex." << std::endl;

    return false;
  }

  dstL = tempUIntL;
  dstH = tempUIntH;

  return true;
}

/**
 * getXMLHexTextValue128 Get the integer representation of a xml-node
 *     containing a hexadecimal value. The value may contain up to 128 bits.
 *
 * node - The reference to the node to get the value from.
 * dstL - The destination for the lower 64bit
 * dstH - The destination for the higher 64bit
 *
 */

bool L1GtTriggerMenuXmlParser::getXMLHexTextValue128(XERCES_CPP_NAMESPACE::DOMNode* node,
                                                     uint64_t& dstL,
                                                     uint64_t& dstH) {
  if (node == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "node == 0 in " << __PRETTY_FUNCTION__ << std::endl;

    return false;
  }

  uint64_t tempUIntH, tempUIntL;

  std::string tempStr = getXMLTextValue(node);
  if (!hexString2UInt128(tempStr, tempUIntL, tempUIntH)) {
    return false;
  }

  dstL = tempUIntL;
  dstH = tempUIntH;

  return true;
}

/**
 * getXMLHexTextValue Get the integer representation of a xml text child
 *     representing a hex value
 *
 * @param node The xml node to get the value from.
 * @param dst The destination the value is written to.
 *
 * @return true if succeeded, false if an error occurred
 *
 */

bool L1GtTriggerMenuXmlParser::getXMLHexTextValue(XERCES_CPP_NAMESPACE::DOMNode* node, uint64_t& dst) {
  uint64_t dummyH;    // dummy for eventual higher 64bit
  uint64_t tempUInt;  // temporary unsigned integer

  if (!getXMLHexTextValue128(node, tempUInt, dummyH)) {
    return false;
  }

  if (dummyH != 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Too large hex-value!" << std::endl;
    return false;
  }

  dst = tempUInt;

  return true;
}

/**
 * countConditionChildMaxBits Count the set bits in the max attribute.
 *     Needed for the wsc-values to determine 180 degree.
 *
 * @param node The xml node of the condition.
 * @param childName The name of the child
 * @param dst The destination to write the number of bits.
 *
 * @return true if the bits could be determined, otherwise false.
 */

bool L1GtTriggerMenuXmlParser::countConditionChildMaxBits(XERCES_CPP_NAMESPACE::DOMNode* node,
                                                          const std::string& childName,
                                                          unsigned int& dst) {
  XERCES_CPP_NAMESPACE_USE

  // should never happen...
  if (node == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "node == 0 in " << __PRETTY_FUNCTION__ << std::endl;

    return false;
  }

  DOMNode* n1 = findXMLChild(node->getFirstChild(), childName);

  if (n1 == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Child of condition not found ( " << childName << ")" << std::endl;

    return false;
  }

  DOMNode* n2 = findXMLChild(n1->getFirstChild(), m_xmlTagValue);

  if (n2 == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "No value tag found for child " << childName << " in "
                                         << __PRETTY_FUNCTION__ << std::endl;

    return false;
  }

  // first try direct
  std::string maxString = getXMLAttribute(n1, m_xmlAttrMax);  // string for the maxbits

  if (maxString.empty()) {
    maxString = getXMLAttribute(n2, m_xmlAttrMax);  // try next value tag
    // if no max was found neither in value nor in the childName tag
    if (maxString.empty()) {
      LogDebug("L1GtTriggerMenuXmlParser") << "No Max value found for " << childName << std::endl;

      return false;
    }
  }

  // do the hex conversion

  uint64_t maxBitsL, maxBitsH;
  if (!hexString2UInt128(maxString, maxBitsL, maxBitsH)) {
    return false;
  }

  // count the bits
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< std::dec
  //<< "        words: dec: high (MSB) word = " << maxBitsH << " low word = " << maxBitsL
  //<< std::hex << "\n"
  //<< "        words: hex: high (MSB) word = " << maxBitsH << " low word = " << maxBitsL
  //<< std::dec
  //<< std::endl;

  unsigned int counter = 0;

  while (maxBitsL != 0) {
    // check if bits set countinously
    if ((maxBitsL & 1) == 0) {
      edm::LogError("L1GtTriggerMenuXmlParser") << "      Confused by not continous set bits for max value "
                                                << maxString << "(child=" << childName << ")" << std::endl;

      return false;
    }

    maxBitsL >>= 1;
    counter++;
  }

  if ((maxBitsH != 0) && (counter != 64)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "      Confused by not continous set bits for max value " << maxString
                                              << "(child=" << childName << ")" << std::endl;

    return false;
  }

  while (maxBitsH != 0) {
    //check if bits set countinously
    if ((maxBitsH & 1) == 0) {
      edm::LogError("L1GtTriggerMenuXmlParser") << "      Confused by not continous set bits for max value "
                                                << maxString << "(child=" << childName << ")" << std::endl;

      return false;
    }

    maxBitsH >>= 1;
    counter++;
  }

  dst = counter;
  return true;
}

/**
 * getConditionChildValues - Get values from a child of a condition.
 *
 * @param node The xml node of the condition.
 * @param childName The name of the child the values should be extracted from.
 * @param num The number of values needed.
 * @param dst A pointer to a vector of uint64_t where the results are written.
 *
 * @return true if succeeded. false if an error occurred or not enough values found.
 */

bool L1GtTriggerMenuXmlParser::getConditionChildValues(XERCES_CPP_NAMESPACE::DOMNode* node,
                                                       const std::string& childName,
                                                       unsigned int num,
                                                       std::vector<uint64_t>& dst) {
  XERCES_CPP_NAMESPACE_USE

  if (node == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "node == 0 in " << __PRETTY_FUNCTION__ << std::endl;

    return false;
  }

  DOMNode* n1 = findXMLChild(node->getFirstChild(), childName);

  // if child not found
  if (n1 == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Child of condition not found ( " << childName << ")" << std::endl;

    return false;
  }

  // no values are sucessfull
  if (num == 0) {
    return true;
  }

  //
  dst.reserve(num);

  //
  n1 = findXMLChild(n1->getFirstChild(), m_xmlTagValue);
  for (unsigned int i = 0; i < num; i++) {
    if (n1 == nullptr) {
      LogDebug("L1GtTriggerMenuXmlParser")
          << "Not enough values in condition child ( " << childName << ")" << std::endl;

      return false;
    }

    if (!getXMLHexTextValue(n1, dst[i])) {
      edm::LogError("L1GtTriggerMenuXmlParser")
          << "Error converting condition child ( " << childName << ") value." << std::endl;

      return false;
    }

    n1 = findXMLChild(n1->getNextSibling(), m_xmlTagValue);  // next child
  }

  return true;
}

/**
 * cleanupXML - Delete parser and error handler. Shutdown XMLPlatformUtils.
 *
 * @param parser A reference to the parser to be deleted.
 *
 */

void L1GtTriggerMenuXmlParser::cleanupXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {
  XERCES_CPP_NAMESPACE_USE

  if (parser != nullptr) {
    delete parser;
  }

  if (m_xmlErrHandler != nullptr) {
    delete m_xmlErrHandler;
    m_xmlErrHandler = nullptr;
  }

  cms::concurrency::xercesTerminate();
}

// methods for the VME file

/**
 * parseVmeXML parse a xml file
 *
 * @param parser The parser to use for parsing the file.
 *
 * @return true if succeeded, false if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseVmeXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {
  XERCES_CPP_NAMESPACE_USE

  DOMDocument* doc = parser->getDocument();
  DOMNode* n1 = doc->getFirstChild();

  if (n1 == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error: Found no XML child" << std::endl;

    return false;
  }

  // find "vme"-tag
  n1 = findXMLChild(n1, m_xmlTagVme);
  if (n1 == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error: No vme tag found." << std::endl;
    return false;
  }

  n1 = n1->getFirstChild();

  unsigned int chipCounter = 0;  // count chips

  while (chipCounter < m_numberConditionChips) {
    n1 = findXMLChild(n1, m_xmlTagChip, true);
    if (n1 == nullptr) {
      // just break if no more chips found
      break;
    }

    // node for a particle
    //DOMNode* particleNode = n1->getFirstChild(); // FIXME un-comment

    // FIXME parse vme.xml, modify the menu

    n1 = n1->getNextSibling();
    chipCounter++;
  }  // end while chipCounter

  return true;
}

// methods for conditions and algorithms

// clearMaps - delete all conditions and algorithms in
// the maps and clear the maps.
void L1GtTriggerMenuXmlParser::clearMaps() {
  // loop over condition maps (one map per condition chip)
  // then loop over conditions in the map
  for (std::vector<ConditionMap>::iterator itCondOnChip = m_conditionMap.begin(); itCondOnChip != m_conditionMap.end();
       itCondOnChip++) {
    // the conditions in the maps are deleted in L1GtTriggerMenu, not here

    itCondOnChip->clear();
  }

  // the algorithms in the maps are deleted in L1GtTriggerMenu, not here
  m_algorithmMap.clear();
}

// insertConditionIntoMap - safe insert of condition into condition map.
// if the condition name already exists, do not insert it and return false
bool L1GtTriggerMenuXmlParser::insertConditionIntoMap(L1GtCondition& cond, const int chipNr) {
  std::string cName = cond.condName();
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "    Trying to insert condition \"" << cName << "\" in the condition map." ;

  // no condition name has to appear twice!
  if ((m_conditionMap[chipNr]).count(cName) != 0) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      Condition " << cName << " already exists - not inserted!"
                                         << std::endl;
    return false;
  }

  (m_conditionMap[chipNr])[cName] = &cond;
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      OK - condition inserted!"
  //<< std::endl;

  return true;
}

// insert an algorithm into algorithm map
bool L1GtTriggerMenuXmlParser::insertAlgorithmIntoMap(const L1GtAlgorithm& alg) {
  std::string algName = alg.algoName();
  const std::string& algAlias = alg.algoAlias();
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "    Trying to insert algorithm \"" << algName << "\" in the algorithm map." ;

  // no algorithm name has to appear twice!
  if (m_algorithmMap.count(algName) != 0) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      Algorithm \"" << algName
                                         << "\"already exists in the algorithm map- not inserted!" << std::endl;
    return false;
  }

  if (m_algorithmAliasMap.count(algAlias) != 0) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      Algorithm alias \"" << algAlias
                                         << "\"already exists in the algorithm alias map- not inserted!" << std::endl;
    return false;
  }

  // bit number less than zero or greater than maximum number of algorithms
  int bitNumber = alg.algoBitNumber();
  if ((bitNumber < 0) || (bitNumber >= static_cast<int>(m_numberPhysTriggers))) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      Bit number " << bitNumber << " outside allowed range [0, "
                                         << m_numberPhysTriggers << ") - algorithm not inserted!" << std::endl;
    return false;
  }

  // maximum number of algorithms
  if (m_algorithmMap.size() >= m_numberPhysTriggers) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      More than maximum allowed " << m_numberPhysTriggers
                                         << " algorithms in the algorithm map - not inserted!" << std::endl;
    return false;
  }

  // chip number outside allowed values
  int chipNr = alg.algoChipNumber(
      static_cast<int>(m_numberConditionChips), static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

  if ((chipNr < 0) || (chipNr > static_cast<int>(m_numberConditionChips))) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      Chip number " << chipNr << " outside allowed range [0, "
                                         << m_numberConditionChips << ") - algorithm not inserted!" << std::endl;
    return false;
  }

  // output pin outside allowed values
  int outputPin = alg.algoOutputPin(
      static_cast<int>(m_numberConditionChips), static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

  if ((outputPin < 0) || (outputPin > static_cast<int>(m_pinsOnConditionChip))) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      Output pin " << outputPin << " outside allowed range [0, "
                                         << m_pinsOnConditionChip << "] - algorithm not inserted!" << std::endl;
    return false;
  }

  // no two algorithms on the same chip can have the same output pin
  for (CItAlgo itAlgo = m_algorithmMap.begin(); itAlgo != m_algorithmMap.end(); itAlgo++) {
    int iPin = (itAlgo->second)
                   .algoOutputPin(static_cast<int>(m_numberConditionChips),
                                  static_cast<int>(m_pinsOnConditionChip),
                                  m_orderConditionChip);
    std::string iName = itAlgo->first;
    int iChip = (itAlgo->second)
                    .algoChipNumber(static_cast<int>(m_numberConditionChips),
                                    static_cast<int>(m_pinsOnConditionChip),
                                    m_orderConditionChip);

    if ((outputPin == iPin) && (chipNr == iChip)) {
      LogTrace("L1GtTriggerMenuXmlParser")
          << "      Output pin " << outputPin << " is the same as for algorithm " << iName
          << "\n      from the same chip number " << chipNr << " - algorithm not inserted!" << std::endl;
      return false;
    }
  }

  // insert algorithm
  m_algorithmMap[algName] = alg;
  m_algorithmAliasMap[algAlias] = alg;

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      OK - algorithm inserted!"
  //<< std::endl;

  return true;
}

// insert a technical trigger into technical trigger map
bool L1GtTriggerMenuXmlParser::insertTechTriggerIntoMap(const L1GtAlgorithm& alg) {
  std::string algName = alg.algoName();
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "    Trying to insert technical trigger \"" << algName
  //<< "\" in the technical trigger map." ;

  // no technical trigger name has to appear twice!
  if (m_technicalTriggerMap.count(algName) != 0) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      Technical trigger \"" << algName
                                         << "\"already exists in the technical trigger map- not inserted!" << std::endl;
    return false;
  }

  // bit number less than zero or greater than maximum number of technical triggers
  int bitNumber = alg.algoBitNumber();
  if ((bitNumber < 0) || (bitNumber >= static_cast<int>(m_numberTechTriggers))) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      Bit number " << bitNumber << " outside allowed range [0, "
                                         << m_numberTechTriggers << ") - technical trigger not inserted!" << std::endl;
    return false;
  }

  // no two technical triggers can have the same bit number
  for (CItAlgo itAlgo = m_technicalTriggerMap.begin(); itAlgo != m_technicalTriggerMap.end(); itAlgo++) {
    int iBitNumber = (itAlgo->second).algoBitNumber();
    std::string iName = itAlgo->first;

    if (iBitNumber == bitNumber) {
      LogTrace("L1GtTriggerMenuXmlParser")
          << "      Bit number " << iBitNumber << " is the same as for technical trigger " << iName
          << " - technical trigger not inserted!" << std::endl;
      return false;
    }
  }

  // maximum number of technical triggers
  if (m_technicalTriggerMap.size() >= m_numberTechTriggers) {
    LogTrace("L1GtTriggerMenuXmlParser") << "      More than maximum allowed " << m_numberTechTriggers
                                         << " technical triggers in the technical trigger map - not inserted!"
                                         << std::endl;
    return false;
  }

  // insert technical trigger
  m_technicalTriggerMap[algName] = alg;

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      OK - technical trigger inserted!"
  //<< std::endl;

  return true;
}

// get the type of the condition, as defined in enum, from the condition type
// as defined in the XML file
L1GtConditionType L1GtTriggerMenuXmlParser::getTypeFromType(const std::string& type) {
  if (type == m_xmlConditionAttrType1s) {
    return Type1s;
  }

  if (type == m_xmlConditionAttrType2s) {
    return Type2s;
  }

  if (type == m_xmlConditionAttrType3s) {
    return Type3s;
  }

  if (type == m_xmlConditionAttrType4s) {
    return Type4s;
  }

  if (type == m_xmlConditionAttrType2wsc) {
    return Type2wsc;
  }

  if (type == m_xmlConditionAttrType2cor) {
    return Type2cor;
  }

  return TypeNull;
}

/**
 * getNumFromType - get the number of particles from a specified type name
 * (for calorimeter objects and muons)
 *
 * @param type The name of the type
 *
 * @return The number of particles in this condition. -1 if type not found.
 */

int L1GtTriggerMenuXmlParser::getNumFromType(const std::string& type) {
  if (type == m_xmlConditionAttrType1s) {
    return 1;
  }

  if (type == m_xmlConditionAttrType2s) {
    return 2;
  }

  if (type == m_xmlConditionAttrType3s) {
    return 3;
  }

  if (type == m_xmlConditionAttrType4s) {
    return 4;
  }

  if (type == m_xmlConditionAttrType2wsc) {
    return 2;
  }

  if (type == m_xmlConditionAttrType2cor) {
    return 2;
  }

  return -1;
}

/**
 * getBitFromNode Get a bit from a specified bitvalue node.
 *
 * @param node The xml node.
 *
 * @return The value of the bit or -1 if an error occurred.
 */

int L1GtTriggerMenuXmlParser::getBitFromNode(XERCES_CPP_NAMESPACE::DOMNode* node) {
  if (getXMLAttribute(node, m_xmlAttrMode) != m_xmlAttrModeBit) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Invalid mode for single bit" << std::endl;

    return -1;
  }

  std::string tmpStr = getXMLTextValue(node);
  if (tmpStr == "0") {
    return 0;
  } else if (tmpStr == "1") {
    return 1;
  } else {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Bad bit value (" << tmpStr << ")" << std::endl;
    return -1;
  }
}

/**
 * getGEqFlag - get the "greater or equal flag" from a condition
 *
 * @param node The xml node of the condition.
 * @nodeName The name of the node from which the flag is a subchild.
 *
 * @return The value of the flag or -1 if no flag was found.
 */

int L1GtTriggerMenuXmlParser::getGEqFlag(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& nodeName) {
  XERCES_CPP_NAMESPACE_USE

  if (node == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "node == 0 in " << __PRETTY_FUNCTION__ << std::endl;

    return -1;
  }

  // usually the GEq flag is a child of the first child (the first element node)
  DOMNode* n1 = node->getFirstChild();
  n1 = findXMLChild(n1, nodeName);

  if (n1 != nullptr) {
    n1 = findXMLChild(n1->getFirstChild(), m_xmlTagGEq);
    if (n1 == nullptr) {
      LogDebug("L1GtTriggerMenuXmlParser") << "No \"greater or equal\" tag found" << std::endl;

      return -1;
    }

    return getBitFromNode(n1);
  } else {
    return -1;
  }
}

/**
 * getMuonMipIsoBits - get MIP and Isolation bits from a muon.
 *
 * @param node The node of the condition.
 * @param num The number of bits required.
 * @param mipDst A pointer to the vector of the MIP bits.
 * @param isoEnDst A pointer to the vector of the "enable isolation" bits.
 * @param isoReqDst A pointer to the vector of the "request isolation" bits.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 */

bool L1GtTriggerMenuXmlParser::getMuonMipIsoBits(XERCES_CPP_NAMESPACE::DOMNode* node,
                                                 unsigned int num,
                                                 std::vector<bool>& mipDst,
                                                 std::vector<bool>& isoEnDst,
                                                 std::vector<bool>& isoReqDst) {
  XERCES_CPP_NAMESPACE_USE

  if (node == nullptr) {
    return false;
  }

  // find ptLowThreshold child
  DOMNode* n1 = findXMLChild(node->getFirstChild(), m_xmlTagPtLowThreshold);

  if (n1 == nullptr) {
    return false;
  }

  // get first value tag
  n1 = findXMLChild(n1->getFirstChild(), m_xmlTagValue);

  for (unsigned int i = 0; i < num; i++) {
    if (n1 == nullptr) {
      return false;
    }

    // MIP bit

    DOMNode* bitnode = findXMLChild(n1->getFirstChild(), m_xmlTagEnableMip);
    if (bitnode == nullptr) {
      return true;
    }

    int tmpint = getBitFromNode(bitnode);
    if (tmpint < 0) {
      return false;
    }

    mipDst[i] = (tmpint != 0);

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      MIP bit value for muon " << i << " = " << mipDst[i]
    //<< std::endl;

    // enable iso bit
    bitnode = findXMLChild(n1->getFirstChild(), m_xmlTagEnableIso);
    if (bitnode == nullptr) {
      return true;
    }

    tmpint = getBitFromNode(bitnode);
    if (tmpint < 0) {
      return false;
    }

    isoEnDst[i] = (tmpint != 0);

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      Enabled iso bit value for muon " << i << " = " << isoEnDst[i]
    //<< std::endl;

    // request iso bit
    bitnode = findXMLChild(n1->getFirstChild(), m_xmlTagRequestIso);
    if (bitnode == nullptr) {
      return true;
    }

    tmpint = getBitFromNode(bitnode);
    if (tmpint < 0) {
      return false;
    }

    isoReqDst[i] = (tmpint != 0);

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      Request iso bit value for muon " << i << " = " << isoReqDst[i]
    //<< std::endl;

    //
    n1 = findXMLChild(n1->getNextSibling(), m_xmlTagValue);  // next value
  }

  return true;
}

/**
 * parseMuon Parse a muon condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseMuon(XERCES_CPP_NAMESPACE::DOMNode* node,
                                         const std::string& name,
                                         unsigned int chipNr,
                                         const bool corrFlag) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name (must be muon) and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  if (particle != m_xmlConditionAttrObjectMu) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Wrong particle for muon-condition (" << particle << ")" << std::endl;
    return false;
  }

  int nrObj = getNumFromType(type);
  if (nrObj < 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Unknown type for muon-condition (" << type << ")"
                                              << "\nCan not determine number of trigger objects. " << std::endl;
    return false;
  }

  // get greater equal flag

  int intGEq = getGEqFlag(node, m_xmlTagPtHighThreshold);
  if (intGEq < 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error getting \"greater or equal\" flag" << std::endl;
    return false;
  }
  // set the boolean value for the ge_eq mode
  bool gEq = (intGEq != 0);

  // get values

  // temporary storage of the parameters
  std::vector<L1GtMuonTemplate::ObjectParameter> objParameter(nrObj);
  L1GtMuonTemplate::CorrelationParameter corrParameter;

  // need at least two values for deltaPhi
  std::vector<uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);

  // get ptHighThreshold values and fill into structure
  if (!getConditionChildValues(node, m_xmlTagPtHighThreshold, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].ptHighThreshold = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      Muon pT high threshold (hex) for muon " << i << " = "
    //<< std::hex << objParameter[i].ptHighThreshold << std::dec
    //<< std::endl;
  }

  // get ptLowThreshold values  and fill into structure
  if (!getConditionChildValues(node, m_xmlTagPtLowThreshold, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "        Muon pT low threshold word (hex) for muon " << i << " = "
    //<< std::hex << tmpValues[i] << std::dec
    //<< std::endl;

    // TODO FIXME stupid format in def.xml...
    // one takes mip bit also, therefore one divide by 16
    tmpValues[i] = (tmpValues[i]) / 16;

    objParameter[i].ptLowThreshold = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      Muon pT low threshold (hex) for muon  " << i << " = "
    //<< std::hex << objParameter[i].ptLowThreshold << std::dec
    //<< std::endl;
  }

  // get qualityRange and fill into structure
  if (!getConditionChildValues(node, m_xmlTagQuality, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].qualityRange = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      qualityRange mask (hex) for muon " << i << " = "
    //<< std::hex << objParameter[i].qualityRange << std::dec
    //<< std::endl;
  }

  // get etaRange and fill into structure
  if (!getConditionChildValues(node, m_xmlTagEta, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].etaRange = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      etaRange (hex) for muon " << i << " = "
    //<< std::hex << objParameter[i].etaRange << std::dec
    //<< std::endl;
  }

  // get phiHigh values and fill into structure
  if (!getConditionChildValues(node, m_xmlTagPhiHigh, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].phiHigh = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      phiHigh (hex) for muon " << i << " = "
    //<< std::hex << objParameter[i].phiHigh << std::dec
    //<< std::endl;
  }

  // get phiLow values and fill into structure
  if (!getConditionChildValues(node, m_xmlTagPhiLow, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].phiLow = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      phiLow (hex) for muon " << i << " = "
    //<< std::hex << objParameter[i].phiLow << std::dec
    //<< std::endl;
  }

  // get charge correlation and fill into structure
  if (!getXMLHexTextValue(findXMLChild(node->getFirstChild(), m_xmlTagChargeCorrelation), tmpValues[0])) {
    LogDebug("L1GtTriggerMenuXmlParser") << "    Error getting charge correlation from muon condition (" << name << ")"
                                         << std::endl;
    return false;
  }

  corrParameter.chargeCorrelation = tmpValues[0];

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      charge correlation" << " = "
  //<< std::hex << corrParameter.chargeCorrelation << std::dec
  //<< std::endl;

  // get mip and iso bits and fill into structure

  std::vector<bool> tmpMip(nrObj);
  std::vector<bool> tmpEnableIso(nrObj);
  std::vector<bool> tmpRequestIso(nrObj);

  if (!getMuonMipIsoBits(node, nrObj, tmpMip, tmpEnableIso, tmpRequestIso)) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "    Could not get mip and iso bits from muon condition (" << name << ")" << std::endl;
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].enableMip = tmpMip[i];
    objParameter[i].enableIso = tmpEnableIso[i];
    objParameter[i].requestIso = tmpRequestIso[i];
  }

  // indicates if a correlation is used
  bool wscVal = (type == m_xmlConditionAttrType2wsc);

  if (wscVal) {
    // get deltaEtaRange
    if (!getConditionChildValues(node, m_xmlTagDeltaEta, 1, tmpValues)) {
      return false;
    }

    corrParameter.deltaEtaRange = tmpValues[0];

    // deltaPhi is larger than 64bit
    if (!getXMLHexTextValue128(findXMLChild(node->getFirstChild(), m_xmlTagDeltaPhi), tmpValues[0], tmpValues[1])) {
      edm::LogError("L1GtTriggerMenuXmlParser")
          << "    Could not get deltaPhi for muon condition with wsc (" << name << ")" << std::endl;
      return false;
    }

    corrParameter.deltaPhiRange0Word = tmpValues[0];
    corrParameter.deltaPhiRange1Word = tmpValues[1];

    // get maximum number of bits for delta phi
    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      Counting deltaPhiMaxbits"
    //<< std::endl;

    unsigned int maxbits;

    if (!countConditionChildMaxBits(node, m_xmlTagDeltaPhi, maxbits)) {
      return false;
    }

    corrParameter.deltaPhiMaxbits = maxbits;
    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "        deltaPhiMaxbits (dec) = " << maxbits
    //<< std::endl;
  }

  // get the type of the condition, as defined in enum, from the condition type
  // as defined in the XML file
  L1GtConditionType cType = getTypeFromType(type);
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Condition type (enum value) = " << cType
  //<< std::endl;

  if (cType == TypeNull) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "Type for muon condition id TypeNull - it means not defined in the XML file."
        << "\nNumber of trigger objects is set to zero. " << std::endl;
    return false;
  }

  // object types - all muons
  std::vector<L1GtObject> objType(nrObj, Mu);

  // now create a new CondMuonition

  L1GtMuonTemplate muonCond(name);

  muonCond.setCondType(cType);
  muonCond.setObjectType(objType);
  muonCond.setCondGEq(gEq);
  muonCond.setCondChipNr(chipNr);

  muonCond.setConditionParameter(objParameter, corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    muonCond.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map and into muon template vector
  if (!insertConditionIntoMap(muonCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;
    return false;
  } else {
    if (corrFlag) {
      (m_corMuonTemplate[chipNr]).push_back(muonCond);

    } else {
      (m_vecMuonTemplate[chipNr]).push_back(muonCond);
    }
  }

  //
  return true;
}

/**
 * parseCalo Parse a calo condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseCalo(XERCES_CPP_NAMESPACE::DOMNode* node,
                                         const std::string& name,
                                         unsigned int chipNr,
                                         const bool corrFlag) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  // determine object type type
  L1GtObject caloObjType;

  if (particle == m_xmlConditionAttrObjectNoIsoEG) {
    caloObjType = NoIsoEG;
  } else if (particle == m_xmlConditionAttrObjectIsoEG) {
    caloObjType = IsoEG;
  } else if (particle == m_xmlConditionAttrObjectCenJet) {
    caloObjType = CenJet;
  } else if (particle == m_xmlConditionAttrObjectTauJet) {
    caloObjType = TauJet;
  } else if (particle == m_xmlConditionAttrObjectForJet) {
    caloObjType = ForJet;
  } else {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Wrong particle for calo-condition (" << particle << ")" << std::endl;
    return false;
  }

  int nrObj = getNumFromType(type);
  if (nrObj < 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Unknown type for calo-condition (" << type << ")"
                                              << "\nCan not determine number of trigger objects. " << std::endl;
    return false;
  }

  // get greater equal flag

  int intGEq = getGEqFlag(node, m_xmlTagEtThreshold);
  if (intGEq < 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error getting \"greater or equal\" flag" << std::endl;
    return false;
  }
  // set the boolean value for the ge_eq mode
  bool gEq = (intGEq != 0);

  // get values

  // temporary storage of the parameters
  std::vector<L1GtCaloTemplate::ObjectParameter> objParameter(nrObj);
  L1GtCaloTemplate::CorrelationParameter corrParameter;

  // need at least one value for deltaPhiRange
  std::vector<uint64_t> tmpValues((nrObj > 1) ? nrObj : 1);

  // get etThreshold values and fill into structure
  if (!getConditionChildValues(node, m_xmlTagEtThreshold, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].etThreshold = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      Calo ET high threshold (hex) for calo object " << i << " = "
    //<< std::hex << objParameter[i].etThreshold << std::dec
    //<< std::endl;
  }

  // get etaRange and fill into structure
  if (!getConditionChildValues(node, m_xmlTagEta, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].etaRange = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      etaRange (hex) for calo object " << i << " = "
    //<< std::hex << objParameter[i].etaRange << std::dec
    //<< std::endl;
  }

  // get phiRange values and fill into structure
  if (!getConditionChildValues(node, m_xmlTagPhi, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].phiRange = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      phiRange (hex) for calo object " << i << " = "
    //<< std::hex << objParameter[i].phiRange << std::dec
    //<< std::endl;
  }

  // indicates if a correlation is used
  bool wscVal = (type == m_xmlConditionAttrType2wsc);

  if (wscVal) {
    // get deltaEtaRange
    if (!getConditionChildValues(node, m_xmlTagDeltaEta, 1, tmpValues)) {
      return false;
    }

    corrParameter.deltaEtaRange = tmpValues[0];

    // get deltaPhiRange
    if (!getConditionChildValues(node, m_xmlTagDeltaPhi, 1, tmpValues)) {
      return false;
    }

    corrParameter.deltaPhiRange = tmpValues[0];

    // get maximum number of bits for delta phi
    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      Counting deltaPhiMaxbits"
    //<< std::endl;

    unsigned int maxbits;

    if (!countConditionChildMaxBits(node, m_xmlTagDeltaPhi, maxbits)) {
      return false;
    }

    corrParameter.deltaPhiMaxbits = maxbits;
    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "        deltaPhiMaxbits (dec) = " << maxbits
    //<< std::endl;
  }

  // get the type of the condition, as defined in enum, from the condition type
  // as defined in the XML file
  L1GtConditionType cType = getTypeFromType(type);
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Condition type (enum value) = " << cType
  //<< std::endl;

  if (cType == TypeNull) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "Type for calo condition id TypeNull - it means not defined in the XML file."
        << "\nNumber of trigger objects is set to zero. " << std::endl;
    return false;
  }

  // object types - all same caloObjType
  std::vector<L1GtObject> objType(nrObj, caloObjType);

  // now create a new calo condition

  L1GtCaloTemplate caloCond(name);

  caloCond.setCondType(cType);
  caloCond.setObjectType(objType);
  caloCond.setCondGEq(gEq);
  caloCond.setCondChipNr(chipNr);

  caloCond.setConditionParameter(objParameter, corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    caloCond.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(caloCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    if (corrFlag) {
      (m_corCaloTemplate[chipNr]).push_back(caloCond);
    } else {
      (m_vecCaloTemplate[chipNr]).push_back(caloCond);
    }
  }

  //
  return true;
}

/**
 * parseEnergySum Parse an "energy sum" condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseEnergySum(XERCES_CPP_NAMESPACE::DOMNode* node,
                                              const std::string& name,
                                              unsigned int chipNr,
                                              const bool corrFlag) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  // determine object type type
  L1GtObject energySumObjType;
  L1GtConditionType cType;

  if ((particle == m_xmlConditionAttrObjectETM) && (type == m_xmlConditionAttrObjectETM)) {
    energySumObjType = ETM;
    cType = TypeETM;

  } else if ((particle == m_xmlConditionAttrObjectETT) && (type == m_xmlConditionAttrObjectETT)) {
    energySumObjType = ETT;
    cType = TypeETT;

  } else if ((particle == m_xmlConditionAttrObjectHTT) && (type == m_xmlConditionAttrObjectHTT)) {
    energySumObjType = HTT;
    cType = TypeHTT;

  } else if ((particle == m_xmlConditionAttrObjectHTM) && (type == m_xmlConditionAttrObjectHTM)) {
    energySumObjType = HTM;
    cType = TypeHTM;

  } else {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "Wrong particle or type for energy-sum condition (" << particle << ", " << type << ")" << std::endl;
    return false;
  }

  // global object
  int nrObj = 1;

  // get greater equal flag

  int intGEq = getGEqFlag(node, m_xmlTagEtThreshold);
  if (intGEq < 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error getting \"greater or equal\" flag" << std::endl;
    return false;
  }
  // set the boolean value for the ge_eq mode
  bool gEq = (intGEq != 0);

  // get values

  // temporary storage of the parameters
  std::vector<L1GtEnergySumTemplate::ObjectParameter> objParameter(nrObj);

  // need at least two values for phi
  std::vector<uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);

  // get etThreshold values and fill into structure
  if (!getConditionChildValues(node, m_xmlTagEtThreshold, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].etThreshold = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      EnergySum ET high threshold (hex) for energySum object " << i << " = "
    //<< std::hex << objParameter[i].etThreshold << std::dec
    //<< std::endl;

    // for ETM and HTM read phi value
    // phi is larger than 64 bits for ETM - it needs two 64bits words
    // phi is less than 64 bits for HTM   - it needs one 64bits word
    if (energySumObjType == ETM) {
      if (!getXMLHexTextValue128(findXMLChild(node->getFirstChild(), m_xmlTagPhi), tmpValues[0], tmpValues[1])) {
        edm::LogError("L1GtTriggerMenuXmlParser")
            << "    Could not get phi for ETM condition (" << name << ")" << std::endl;
        return false;
      }

      objParameter[i].phiRange0Word = tmpValues[0];
      objParameter[i].phiRange1Word = tmpValues[1];

    } else if (energySumObjType == HTM) {
      if (!getXMLHexTextValue(findXMLChild(node->getFirstChild(), m_xmlTagPhi), tmpValues[0])) {
        edm::LogError("L1GtTriggerMenuXmlParser")
            << "    Could not get phi for HTM condition (" << name << ")" << std::endl;
        return false;
      }

      objParameter[i].phiRange0Word = tmpValues[0];
    }

    // get energyOverflow logical flag and fill into structure
    DOMNode* n1;
    if ((n1 = findXMLChild(node->getFirstChild(), m_xmlTagEtThreshold)) == nullptr) {
      edm::LogError("L1GtTriggerMenuXmlParser")
          << "    Could not get energyOverflow for EnergySum condition (" << name << ")" << std::endl;
      return false;
    }
    if ((n1 = findXMLChild(n1->getFirstChild(), m_xmlTagEnergyOverflow)) == nullptr) {
      edm::LogError("L1GtTriggerMenuXmlParser")
          << "    Could not get energyOverflow for EnergySum condition (" << name << ")" << std::endl;
      return false;
    }

    int tmpInt = getBitFromNode(n1);
    if (tmpInt == 0) {
      objParameter[i].energyOverflow = false;

      //LogTrace("L1GtTriggerMenuXmlParser")
      //<< "      EnergySum energyOverflow logical flag (hex) = "
      //<< std::hex << objParameter[i].energyOverflow << std::dec
      //<< std::endl;
    } else if (tmpInt == 1) {
      objParameter[i].energyOverflow = true;

      //LogTrace("L1GtTriggerMenuXmlParser")
      //<< "      EnergySum energyOverflow logical flag (hex) = "
      //<< std::hex << objParameter[i].energyOverflow << std::dec
      //<< std::endl;
    } else {
      LogTrace("L1GtTriggerMenuXmlParser") << "      EnergySum energyOverflow logical flag (hex) = " << std::hex
                                           << tmpInt << std::dec << " - wrong value! " << std::endl;
      return false;
    }
  }

  // object types - all same energySumObjType
  std::vector<L1GtObject> objType(nrObj, energySumObjType);

  // now create a new energySum condition

  L1GtEnergySumTemplate energySumCond(name);

  energySumCond.setCondType(cType);
  energySumCond.setObjectType(objType);
  energySumCond.setCondGEq(gEq);
  energySumCond.setCondChipNr(chipNr);

  energySumCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    energySumCond.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(energySumCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    if (corrFlag) {
      (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);

    } else {
      (m_vecEnergySumTemplate[chipNr]).push_back(energySumCond);
    }
  }

  //
  return true;
}

/**
 * parseJetCounts Parse a "jet counts" condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseJetCounts(XERCES_CPP_NAMESPACE::DOMNode* node,
                                              const std::string& name,
                                              unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  if (particle != m_xmlConditionAttrObjectJetCounts) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "Wrong particle for JetCounts condition (" << particle << ")" << std::endl;
    return false;
  }

  // object type and condition type
  L1GtObject jetCountsObjType = JetCounts;
  L1GtConditionType cType = TypeJetCounts;

  // global object
  int nrObj = 1;

  // get greater equal flag

  int intGEq = getGEqFlag(node, m_xmlTagCountThreshold);
  if (intGEq < 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error getting \"greater or equal\" flag" << std::endl;
    return false;
  }
  // set the boolean value for the ge_eq mode
  bool gEq = (intGEq != 0);

  // get values

  // temporary storage of the parameters
  std::vector<L1GtJetCountsTemplate::ObjectParameter> objParameter(nrObj);

  // get countIndex value and fill into structure
  // they are expressed in  base 10  (values: 0 - m_numberL1JetCounts)
  char* endPtr = nullptr;
  long int typeInt = strtol(type.c_str(), &endPtr, 10);  // base = 10

  if (*endPtr != 0) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Unable to convert " << type << " to dec." << std::endl;

    return false;
  }

  // test if count index is out of range
  if ((typeInt < 0) || (typeInt > static_cast<int>(m_numberL1JetCounts))) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Count index " << typeInt << " outside range [0, " << m_numberL1JetCounts
                                         << "]" << std::endl;

    return false;
  }

  objParameter[0].countIndex = static_cast<unsigned int>(typeInt);

  // get count threshold values and fill into structure
  std::vector<uint64_t> tmpValues(nrObj);

  if (!getConditionChildValues(node, m_xmlTagCountThreshold, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].countThreshold = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      JetCounts count threshold (hex) for JetCounts object " << i << " = "
    //<< std::hex << objParameter[i].countThreshold << std::dec
    //<< std::endl;

    // TODO FIXME un-comment when tag available in XML file

    //        // get countOverflow logical flag and fill into structure
    //        DOMNode* n1;
    //        if ( (n1 = findXMLChild(node->getFirstChild(), m_xmlTagCountThreshold)) == 0) {
    //            edm::LogError("L1GtTriggerMenuXmlParser")
    //            << "    Could not get countOverflow for JetCounts condition ("
    //            << name << ")"
    //            << std::endl;
    //            return false;
    //        }
    //        if ( (n1 = findXMLChild(n1->getFirstChild(), m_xmlTagCountThreshold)) == 0) {
    //            edm::LogError("L1GtTriggerMenuXmlParser")
    //            << "    Could not get countOverflow for JetCounts condition ("
    //            << name << ")"
    //            << std::endl;
    //            return false;
    //        }
    //
    //        int tmpInt = getBitFromNode(n1);
    //        if (tmpInt == 0) {
    //            objParameter[i].countOverflow = false;
    //
    //            LogTrace("L1GtTriggerMenuXmlParser")
    //            << "      JetCounts countOverflow logical flag (hex) = "
    //            << std::hex << objParameter[i].countOverflow << std::dec
    //            << std::endl;
    //        } else if (tmpInt == 1) {
    //            objParameter[i].countOverflow = true;
    //
    //            LogTrace("L1GtTriggerMenuXmlParser")
    //            << "      JetCounts countOverflow logical flag (hex) = "
    //            << std::hex << objParameter[i].countOverflow << std::dec
    //            << std::endl;
    //        } else {
    //            LogTrace("L1GtTriggerMenuXmlParser")
    //            << "      JetCounts countOverflow logical flag (hex) = "
    //            << std::hex << tmpInt << std::dec << " - wrong value! "
    //            << std::endl;
    //            return false;
    //        }
  }

  // object types - all same objType
  std::vector<L1GtObject> objType(nrObj, jetCountsObjType);

  // now create a new JetCounts condition

  L1GtJetCountsTemplate jetCountsCond(name);

  jetCountsCond.setCondType(cType);
  jetCountsCond.setObjectType(objType);
  jetCountsCond.setCondGEq(gEq);
  jetCountsCond.setCondChipNr(chipNr);

  jetCountsCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    jetCountsCond.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(jetCountsCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    (m_vecJetCountsTemplate[chipNr]).push_back(jetCountsCond);
  }

  //
  return true;
}

/**
 * parseCastor Parse a CASTOR condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseCastor(XERCES_CPP_NAMESPACE::DOMNode* node,
                                           const std::string& name,
                                           unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  if (particle != m_xmlConditionAttrObjectCastor) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "\nError: wrong particle for Castor condition (" << particle << ")" << std::endl;
    return false;
  }

  // object type and condition type
  // object type - irrelevant for CASTOR conditions
  L1GtConditionType cType = TypeCastor;

  // no objects for CASTOR conditions

  // set the boolean value for the ge_eq mode - irrelevant for CASTOR conditions
  bool gEq = false;

  // now create a new CASTOR condition

  L1GtCastorTemplate castorCond(name);

  castorCond.setCondType(cType);
  castorCond.setCondGEq(gEq);
  castorCond.setCondChipNr(chipNr);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    castorCond.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(castorCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    (m_vecCastorTemplate[chipNr]).push_back(castorCond);
  }

  //
  return true;
}

/**
 * parseHfBitCounts Parse a "HF bit counts" condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseHfBitCounts(XERCES_CPP_NAMESPACE::DOMNode* node,
                                                const std::string& name,
                                                unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  if (particle != m_xmlConditionAttrObjectHfBitCounts) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "Wrong particle for HfBitCounts condition (" << particle << ")" << std::endl;
    return false;
  }

  // object type and condition type
  L1GtObject hfBitCountsObjType = HfBitCounts;
  L1GtConditionType cType = TypeHfBitCounts;

  // global object
  int nrObj = 1;

  // get greater equal flag

  int intGEq = getGEqFlag(node, m_xmlTagCountThreshold);
  if (intGEq < 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error getting \"greater or equal\" flag" << std::endl;
    return false;
  }
  // set the boolean value for the ge_eq mode
  bool gEq = (intGEq != 0);

  // get values

  // temporary storage of the parameters
  std::vector<L1GtHfBitCountsTemplate::ObjectParameter> objParameter(nrObj);

  // get countIndex value and fill into structure
  // they are expressed in  base 10
  char* endPtr = nullptr;
  long int typeInt = strtol(type.c_str(), &endPtr, 10);  // base = 10

  if (*endPtr != 0) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Unable to convert " << type << " to dec." << std::endl;

    return false;
  }

  // test if count index is out of range FIXME introduce m_numberL1HfBitCounts?
  //if ((typeInt < 0) || (typeInt > static_cast<int>(m_numberL1HfBitCounts))) {
  //    LogDebug("L1GtTriggerMenuXmlParser") << "Count index " << typeInt
  //        << " outside range [0, " << m_numberL1HfBitCounts << "]" << std::endl;
  //
  //    return false;
  //}

  objParameter[0].countIndex = static_cast<unsigned int>(typeInt);

  // get count threshold values and fill into structure
  std::vector<uint64_t> tmpValues(nrObj);

  if (!getConditionChildValues(node, m_xmlTagCountThreshold, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].countThreshold = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      HfBitCounts count threshold (hex) for HfBitCounts object " << i << " = "
    //<< std::hex << objParameter[i].countThreshold << std::dec
    //<< std::endl;
  }

  // object types - all same objType
  std::vector<L1GtObject> objType(nrObj, hfBitCountsObjType);

  // now create a new HfBitCounts condition

  L1GtHfBitCountsTemplate hfBitCountsCond(name);

  hfBitCountsCond.setCondType(cType);
  hfBitCountsCond.setObjectType(objType);
  hfBitCountsCond.setCondGEq(gEq);
  hfBitCountsCond.setCondChipNr(chipNr);

  hfBitCountsCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    hfBitCountsCond.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(hfBitCountsCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    (m_vecHfBitCountsTemplate[chipNr]).push_back(hfBitCountsCond);
  }

  //
  return true;
}

/**
 * parseHfRingEtSums Parse a "HF Ring ET sums" condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseHfRingEtSums(XERCES_CPP_NAMESPACE::DOMNode* node,
                                                 const std::string& name,
                                                 unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  if (particle != m_xmlConditionAttrObjectHfRingEtSums) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "Wrong particle for HfRingEtSums condition (" << particle << ")" << std::endl;
    return false;
  }

  // object type and condition type
  L1GtObject hfRingEtSumsObjType = HfRingEtSums;
  L1GtConditionType cType = TypeHfRingEtSums;

  // global object
  int nrObj = 1;

  // get greater equal flag

  int intGEq = getGEqFlag(node, m_xmlTagEtThreshold);
  if (intGEq < 0) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error getting \"greater or equal\" flag" << std::endl;
    return false;
  }
  // set the boolean value for the ge_eq mode
  bool gEq = (intGEq != 0);

  // get values

  // temporary storage of the parameters
  std::vector<L1GtHfRingEtSumsTemplate::ObjectParameter> objParameter(nrObj);

  // get etSumIndex value and fill into structure
  // they are expressed in  base 10
  char* endPtr = nullptr;
  long int typeInt = strtol(type.c_str(), &endPtr, 10);  // base = 10

  if (*endPtr != 0) {
    LogDebug("L1GtTriggerMenuXmlParser") << "Unable to convert " << type << " to dec." << std::endl;

    return false;
  }

  // test if ET sum index is out of range FIXME introduce m_numberL1HfRingEtSums?
  //if ((typeInt < 0) || (typeInt > static_cast<int>(m_numberL1HfRingEtSums))) {
  //    LogDebug("L1GtTriggerMenuXmlParser") << "Count index " << typeInt
  //        << " outside range [0, " << m_numberL1HfRingEtSums << "]" << std::endl;
  //
  //    return false;
  //}

  objParameter[0].etSumIndex = static_cast<unsigned int>(typeInt);

  // get ET sum threshold values and fill into structure
  std::vector<uint64_t> tmpValues(nrObj);

  if (!getConditionChildValues(node, m_xmlTagEtThreshold, nrObj, tmpValues)) {
    return false;
  }

  for (int i = 0; i < nrObj; i++) {
    objParameter[i].etSumThreshold = tmpValues[i];

    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "      HfRingEtSums count threshold (hex) for HfRingEtSums object " << i << " = "
    //<< std::hex << objParameter[i].etSumThreshold << std::dec
    //<< std::endl;
  }

  // object types - all same objType
  std::vector<L1GtObject> objType(nrObj, hfRingEtSumsObjType);

  // now create a new HfRingEtSums condition

  L1GtHfRingEtSumsTemplate hfRingEtSumsCond(name);

  hfRingEtSumsCond.setCondType(cType);
  hfRingEtSumsCond.setObjectType(objType);
  hfRingEtSumsCond.setCondGEq(gEq);
  hfRingEtSumsCond.setCondChipNr(chipNr);

  hfRingEtSumsCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    hfRingEtSumsCond.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(hfRingEtSumsCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    (m_vecHfRingEtSumsTemplate[chipNr]).push_back(hfRingEtSumsCond);
  }

  //
  return true;
}

/**
 * parseBptx Parse a BPTX condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseBptx(XERCES_CPP_NAMESPACE::DOMNode* node,
                                         const std::string& name,
                                         unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  if (particle != m_xmlConditionAttrObjectBptx) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "\nError: wrong particle for Bptx condition (" << particle << ")" << std::endl;
    return false;
  }

  // object type and condition type
  // object type - irrelevant for BPTX conditions
  L1GtConditionType cType = TypeBptx;

  // no objects for BPTX conditions

  // set the boolean value for the ge_eq mode - irrelevant for BPTX conditions
  bool gEq = false;

  // now create a new BPTX condition

  L1GtBptxTemplate bptxCond(name);

  bptxCond.setCondType(cType);
  bptxCond.setCondGEq(gEq);
  bptxCond.setCondChipNr(chipNr);

  LogTrace("L1GtTriggerMenuXmlParser") << bptxCond << "\n" << std::endl;

  // insert condition into the map
  if (!insertConditionIntoMap(bptxCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    (m_vecBptxTemplate[chipNr]).push_back(bptxCond);
  }

  //
  return true;
}

/**
 * parseExternal Parse an External condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseExternal(XERCES_CPP_NAMESPACE::DOMNode* node,
                                             const std::string& name,
                                             unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  if (particle != m_xmlConditionAttrObjectGtExternal) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "\nError: wrong particle for External condition (" << particle << ")" << std::endl;
    return false;
  }

  // object type and condition type
  // object type - irrelevant for External conditions
  L1GtConditionType cType = TypeExternal;

  // no objects for External conditions

  // set the boolean value for the ge_eq mode - irrelevant for External conditions
  bool gEq = false;

  // now create a new External condition

  L1GtExternalTemplate externalCond(name);

  externalCond.setCondType(cType);
  externalCond.setCondGEq(gEq);
  externalCond.setCondChipNr(chipNr);

  LogTrace("L1GtTriggerMenuXmlParser") << externalCond << "\n" << std::endl;

  // insert condition into the map
  if (!insertConditionIntoMap(externalCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    (m_vecExternalTemplate[chipNr]).push_back(externalCond);
  }

  //
  return true;
}

/**
 * parseCorrelation Parse a correlation condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseCorrelation(XERCES_CPP_NAMESPACE::DOMNode* node,
                                                const std::string& name,
                                                unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  // create a new correlation condition
  L1GtCorrelationTemplate correlationCond(name);

  // check that the condition does not exist already in the map
  if (!insertConditionIntoMap(correlationCond, chipNr)) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "    Error: duplicate correlation condition (" << name << ")" << std::endl;

    return false;
  }

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  LogTrace("L1GtTriggerMenuXmlParser") << "    Condition category: " << condition << ", particle: " << particle
                                       << ", type: " << type << "\n"
                                       << std::endl;

  // condition type
  L1GtConditionType cType = Type2cor;

  // two objects (for sure)
  const int nrObj = 2;

  // object types and greater equal flag - filled in the loop
  int intGEq[nrObj] = {-1, -1};
  std::vector<L1GtObject> objType(nrObj);
  std::vector<L1GtConditionCategory> condCateg(nrObj);

  // correlation flag and index in the cor*vector
  const bool corrFlag = true;
  int corrIndexVal[nrObj] = {-1, -1};

  // get the subconditions

  DOMNode* conditionsNode = node->getFirstChild();
  std::string conditionNameNodeName;
  conditionsNode = findXMLChild(conditionsNode, "", true, &conditionNameNodeName);

  for (int iSubCond = 0; iSubCond < nrObj; ++iSubCond) {
    // get for sub-condition:  category, object name and type name and condition name
    condition = getXMLAttribute(conditionsNode, m_xmlConditionAttrCondition);
    particle = getXMLAttribute(conditionsNode, m_xmlConditionAttrObject);
    type = getXMLAttribute(conditionsNode, m_xmlConditionAttrType);

    LogTrace("L1GtTriggerMenuXmlParser") << "    Sub-condition category: " << condition << ", particle: " << particle
                                         << ", type: " << type << ", name: " << conditionNameNodeName << "\n"
                                         << std::endl;

    // call the appropriate function for this condition
    if (condition == m_xmlConditionAttrConditionMuon) {
      if (!parseMuon(conditionsNode, conditionNameNodeName, chipNr, corrFlag)) {
        edm::LogError("L1GtTriggerMenuXmlParser") << "Error parsing sub-condition " << condition << ")"
                                                  << " with name " << conditionNameNodeName << std::endl;
      }

      // get greater equal flag
      intGEq[iSubCond] = getGEqFlag(conditionsNode, m_xmlTagPtHighThreshold);
      if (intGEq[iSubCond] < 0) {
        edm::LogError("L1GtTriggerMenuXmlParser")
            << "Error getting \"greater or equal\" flag"
            << " for sub-condition " << conditionNameNodeName << " for the correlation condition " << name << std::endl;
        return false;
      }

      // set object type and sub-condition category
      objType[iSubCond] = Mu;
      condCateg[iSubCond] = CondMuon;
      corrIndexVal[iSubCond] = (m_corMuonTemplate[chipNr]).size() - 1;

    } else if (condition == m_xmlConditionAttrConditionCalo) {
      if (!parseCalo(conditionsNode, conditionNameNodeName, chipNr, corrFlag)) {
        edm::LogError("L1GtTriggerMenuXmlParser") << "Error parsing sub-condition " << condition << ")"
                                                  << " with name " << conditionNameNodeName << std::endl;
      }

      // get greater equal flag
      intGEq[iSubCond] = getGEqFlag(conditionsNode, m_xmlTagEtThreshold);
      if (intGEq[iSubCond] < 0) {
        edm::LogError("L1GtTriggerMenuXmlParser")
            << "Error getting \"greater or equal\" flag"
            << " for sub-condition " << conditionNameNodeName << " for the correlation condition " << name << std::endl;
        return false;
      }

      // set object type and sub-condition category
      if (particle == m_xmlConditionAttrObjectNoIsoEG) {
        objType[iSubCond] = NoIsoEG;
      } else if (particle == m_xmlConditionAttrObjectIsoEG) {
        objType[iSubCond] = IsoEG;
      } else if (particle == m_xmlConditionAttrObjectCenJet) {
        objType[iSubCond] = CenJet;
      } else if (particle == m_xmlConditionAttrObjectTauJet) {
        objType[iSubCond] = TauJet;
      } else if (particle == m_xmlConditionAttrObjectForJet) {
        objType[iSubCond] = ForJet;
      } else {
        edm::LogError("L1GtTriggerMenuXmlParser")
            << "Wrong object type " << particle << " for sub-condition " << conditionNameNodeName
            << " from the correlation condition " << name << std::endl;
        return false;
      }

      condCateg[iSubCond] = CondCalo;
      corrIndexVal[iSubCond] = (m_corCaloTemplate[chipNr]).size() - 1;

    } else if (condition == m_xmlConditionAttrConditionEnergySum) {
      if (!parseEnergySum(conditionsNode, conditionNameNodeName, chipNr, corrFlag)) {
        edm::LogError("L1GtTriggerMenuXmlParser") << "Error parsing sub-condition " << condition << ")"
                                                  << " with name " << conditionNameNodeName << std::endl;
      }

      // get greater equal flag
      intGEq[iSubCond] = getGEqFlag(conditionsNode, m_xmlTagEtThreshold);
      if (intGEq[iSubCond] < 0) {
        edm::LogError("L1GtTriggerMenuXmlParser")
            << "Error getting \"greater or equal\" flag"
            << " for sub-condition " << conditionNameNodeName << " for the correlation condition " << name << std::endl;
        return false;
      }

      // set object type and sub-condition category
      if (particle == m_xmlConditionAttrObjectETM) {
        objType[iSubCond] = ETM;
      } else if (particle == m_xmlConditionAttrObjectETT) {
        objType[iSubCond] = ETT;
      } else if (particle == m_xmlConditionAttrObjectHTT) {
        objType[iSubCond] = HTT;
      } else if (particle == m_xmlConditionAttrObjectHTM) {
        objType[iSubCond] = HTM;
      } else {
        edm::LogError("L1GtTriggerMenuXmlParser")
            << "Wrong object type " << particle << " for sub-condition " << conditionNameNodeName
            << " from the correlation condition " << name << std::endl;
        return false;
      }

      condCateg[iSubCond] = CondEnergySum;
      corrIndexVal[iSubCond] = (m_corEnergySumTemplate[chipNr]).size() - 1;

    } else {
      edm::LogError("L1GtTriggerMenuXmlParser")
          << "Unknown or un-adequate sub-condition (" << condition << ")"
          << " with name " << conditionNameNodeName << " for the correlation condition " << name << std::endl;

      return false;
    }

    conditionsNode = findXMLChild(conditionsNode->getNextSibling(), "", true, &conditionNameNodeName);
  }

  // get greater equal flag for the correlation condition
  bool gEq = true;
  if (intGEq[0] != intGEq[1]) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "Inconsistent GEq flags for sub-conditions (" << condition << ")"
        << " with name " << conditionNameNodeName << " for the correlation condition " << name << std::endl;
    return false;

  } else {
    gEq = (intGEq[0] != 0);
  }

  // correlation parameters

  // temporary storage of the parameters
  L1GtCorrelationTemplate::CorrelationParameter corrParameter;
  std::vector<uint64_t> tmpValues(nrObj);

  // get deltaEtaRange
  //    if ( !getConditionChildValues(node, m_xmlTagDeltaEta, 1, tmpValues) ) {
  //        return false;
  //    }
  //
  //    corrParameter.deltaEtaRange = tmpValues[0];

  XERCES_CPP_NAMESPACE::DOMNode* node1 = findXMLChild(node->getFirstChild(), m_xmlTagDeltaEta);

  std::string valString;

  if (node1 == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "    Could not get deltaEta for correlation condition " << name << ". " << std::endl;
    return false;
  } else {
    valString = getXMLTextValue(node1);
  }

  corrParameter.deltaEtaRange = valString;

  //    // deltaPhi is larger than 64bit
  //    if ( !getXMLHexTextValue128(findXMLChild(node->getFirstChild(), m_xmlTagDeltaPhi),
  //        tmpValues[0], tmpValues[1])) {
  //        edm::LogError("L1GtTriggerMenuXmlParser")
  //            << "    Could not get deltaPhi for correlation condition " << name << ". "
  //            << std::endl;
  //        return false;
  //    }
  //
  //    corrParameter.deltaPhiRange = tmpValues[0];

  node1 = findXMLChild(node->getFirstChild(), m_xmlTagDeltaPhi);

  if (node1 == nullptr) {
    return false;
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "    Could not get deltaPhi for correlation condition " << name << ". " << std::endl;
  } else {
    valString = getXMLTextValue(node1);
  }

  corrParameter.deltaPhiRange = valString;

  // get maximum number of bits for delta phi
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Counting deltaPhiMaxbits"
  //<< std::endl;

  unsigned int maxbits;

  if (!countConditionChildMaxBits(node, m_xmlTagDeltaPhi, maxbits)) {
    return false;
  }

  corrParameter.deltaPhiMaxbits = maxbits;
  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "        deltaPhiMaxbits (dec) = " << maxbits
  //<< std::endl;

  // fill the correlation condition
  correlationCond.setCondType(cType);
  correlationCond.setObjectType(objType);
  correlationCond.setCondGEq(gEq);
  correlationCond.setCondChipNr(chipNr);

  correlationCond.setCond0Category(condCateg[0]);
  correlationCond.setCond1Category(condCateg[1]);

  correlationCond.setCond0Index(corrIndexVal[0]);
  correlationCond.setCond1Index(corrIndexVal[1]);

  correlationCond.setCorrelationParameter(corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    correlationCond.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  // condition is not duplicate, check was done at the beginning

  (m_vecCorrelationTemplate[chipNr]).push_back(correlationCond);

  //
  return true;
}

/**
 * parseId - parse all identification attributes (trigger menu names, scale DB key, etc)
 *
 * @param parser The parser to parse the XML file with.
 *
 * @return "true" if succeeded. "false" if an error occurred.
 *
 */
bool L1GtTriggerMenuXmlParser::parseId(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {
  XERCES_CPP_NAMESPACE_USE

  DOMNode* doc = parser->getDocument();
  DOMNode* n1 = doc->getFirstChild();

  // we assume that the first child is m_xmlTagDef because it was checked in workXML

  DOMNode* headerNode = n1->getFirstChild();
  if (headerNode == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error: No child of <" << m_xmlTagDef << "> tag found." << std::endl;
    return false;
  }

  headerNode = findXMLChild(headerNode, m_xmlTagHeader);
  if (headerNode == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagHeader << "> tag"
                                         << "\n   - No header information." << std::endl;

  } else {
    DOMNode* idNode = headerNode->getFirstChild();

    // find menu interface name
    idNode = findXMLChild(idNode, m_xmlTagMenuInterface);
    if (idNode == nullptr) {
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagMenuInterface << "> tag"
                                           << "\n   - Trigger menu interface name derived from file name." << std::endl;

      // set the name of the trigger menu interface: from beginning of file names
      // until beginning of "_L1T_Scales"
      size_t xmlPos = m_triggerMenuName.find("_L1T_Scales", 0);
      if (xmlPos == std::string::npos) {
        LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find \"_L1T_Scales\" "
                                             << "string in file name"
                                             << "\n   - Trigger menu interface name set to file name." << std::endl;
        m_triggerMenuInterface = m_triggerMenuName;

      } else {
        m_triggerMenuInterface = m_triggerMenuName;
        m_triggerMenuInterface.erase(m_triggerMenuInterface.begin(), m_triggerMenuInterface.begin() + xmlPos);
      }

    } else {
      m_triggerMenuInterface = getXMLTextValue(idNode);
    }

    // find menu interface creation date
    idNode = headerNode->getFirstChild();
    idNode = findXMLChild(idNode, m_xmlTagMenuInterfaceDate);

    if (idNode == nullptr) {
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagMenuInterfaceDate << "> tag"
                                           << "\n   - No creation date." << m_triggerMenuInterfaceDate << std::endl;

    } else {
      m_triggerMenuInterfaceDate = getXMLTextValue(idNode);
    }

    // find menu interface creation author
    idNode = headerNode->getFirstChild();
    idNode = findXMLChild(idNode, m_xmlTagMenuInterfaceAuthor);

    if (idNode == nullptr) {
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagMenuInterfaceAuthor << "> tag"
                                           << "\n   - No creation author." << m_triggerMenuInterfaceAuthor << std::endl;

    } else {
      m_triggerMenuInterfaceAuthor = getXMLTextValue(idNode);
    }

    // find menu interface description
    idNode = headerNode->getFirstChild();
    idNode = findXMLChild(idNode, m_xmlTagMenuInterfaceDescription);

    if (idNode == nullptr) {
      LogTrace("L1GtTriggerMenuXmlParser")
          << "\n  Warning: Could not find <" << m_xmlTagMenuInterfaceDescription << "> tag"
          << "\n   - No description." << m_triggerMenuInterfaceDescription << std::endl;

    } else {
      m_triggerMenuInterfaceDescription = getXMLTextValue(idNode);
    }

    // find menu creation date
    idNode = headerNode->getFirstChild();
    idNode = findXMLChild(idNode, m_xmlTagMenuDate);

    if (idNode == nullptr) {
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagMenuDate << "> tag"
                                           << "\n   - No creation date." << m_triggerMenuDate << std::endl;

    } else {
      m_triggerMenuDate = getXMLTextValue(idNode);
    }

    // find menu creation author
    idNode = headerNode->getFirstChild();
    idNode = findXMLChild(idNode, m_xmlTagMenuAuthor);

    if (idNode == nullptr) {
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagMenuAuthor << "> tag"
                                           << "\n   - No creation author." << m_triggerMenuAuthor << std::endl;

    } else {
      m_triggerMenuAuthor = getXMLTextValue(idNode);
    }

    // find menu description
    idNode = headerNode->getFirstChild();
    idNode = findXMLChild(idNode, m_xmlTagMenuDescription);

    if (idNode == nullptr) {
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagMenuDescription << "> tag"
                                           << "\n   - No description." << m_triggerMenuDescription << std::endl;

    } else {
      m_triggerMenuDescription = getXMLTextValue(idNode);
    }

    // find algorithm implementation tag

    idNode = headerNode->getFirstChild();

    idNode = findXMLChild(idNode, m_xmlTagMenuAlgImpl);
    if (idNode == nullptr) {
      m_algorithmImplementation = "";
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagMenuAlgImpl << "> tag"
                                           << "\n   - Algorithm implementation tag set to empty string." << std::endl;

    } else {
      m_algorithmImplementation = getXMLTextValue(idNode);
    }

    // find DB key for L1 scales

    idNode = headerNode->getFirstChild();

    idNode = findXMLChild(idNode, m_xmlTagScaleDbKey);
    if (idNode == nullptr) {
      m_scaleDbKey = "NULL";
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagScaleDbKey << "> tag"
                                           << "\n   - Scale key set to " << m_scaleDbKey << " string." << std::endl;

    } else {
      m_scaleDbKey = getXMLTextValue(idNode);
    }
  }

  LogDebug("L1GtTriggerMenuXmlParser") << "\n  Parsed values from XML file"
                                       << "\nL1 MenuInterface:                   " << m_triggerMenuInterface
                                       << "\nL1 MenuInterface - Creation date:   " << m_triggerMenuInterfaceDate
                                       << "\nL1 MenuInterface - Creation author: " << m_triggerMenuInterfaceAuthor
                                       << "\nL1 MenuInterface - Description:     " << m_triggerMenuInterfaceDescription
                                       << "\n"
                                       << "\nAlgorithm implementation tag:       " << m_algorithmImplementation << "\n"
                                       << "\nL1 Menu - Creation date:            " << m_triggerMenuDate
                                       << "\nL1 Menu - Creation author:          " << m_triggerMenuAuthor
                                       << "\nL1 Menu - Description:              " << m_triggerMenuDescription
                                       << std::endl;

  // set the trigger menu name
  // format:
  // L1MenuInterface/ScaleDbKey/AlgorithmImplementationTag

  std::string menuName = m_triggerMenuInterface + "/" + m_scaleDbKey + "/" + m_algorithmImplementation;

  if (menuName != m_triggerMenuName) {
    LogDebug("L1GtTriggerMenuXmlParser") << "\n  Warning: Inconsistent L1 menu name:"
                                         << "\n    from XML file name: " << m_triggerMenuName
                                         << "\n    from XML tag:       " << menuName << std::endl;

    if (!m_triggerMenuInterface.empty()) {
      if (m_scaleDbKey == "NULL") {
        m_triggerMenuName = m_triggerMenuInterface;
      } else {
        m_triggerMenuName = menuName;
      }

      LogTrace("L1GtTriggerMenuXmlParser") << "\n  L1 menu name set to value from XML tag!"
                                           << "\n  L1 Menu name: " << m_triggerMenuName << std::endl;

    } else {
      LogTrace("L1GtTriggerMenuXmlParser") << "\n  L1 menu name set to file name!"
                                           << "\n  L1 Menu name: " << m_triggerMenuName << std::endl;
    }
  }

  //
  return true;
}

/**
 * workCondition - call the appropriate function to parse this condition.
 *
 * @param node The corresponding node to the condition.
 * @param name The name of the condition.
 * @param chipNr The number of the chip the condition is located on.
 *
 * @return "true" on success, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::workCondition(XERCES_CPP_NAMESPACE::DOMNode* node,
                                             const std::string& name,
                                             unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  // get condition, particle name and type name
  std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
  std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
  std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

  if (condition.empty() || particle.empty() || type.empty()) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Missing attributes for condition " << name << std::endl;

    return false;
  }

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "    condition: " << condition << ", particle: " << particle
  //<< ", type: " << type << std::endl;

  // call the appropiate function for this condition

  if (condition == m_xmlConditionAttrConditionMuon) {
    return parseMuon(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionCalo) {
    return parseCalo(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionEnergySum) {
    return parseEnergySum(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionJetCounts) {
    return parseJetCounts(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionCastor) {
    return parseCastor(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionHfBitCounts) {
    return parseHfBitCounts(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionHfRingEtSums) {
    return parseHfRingEtSums(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionBptx) {
    return parseBptx(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionExternal) {
    return parseExternal(node, name, chipNr);
  } else if (condition == m_xmlConditionAttrConditionCorrelation) {
    return parseCorrelation(node, name, chipNr);
  } else {
    edm::LogError("L1GtTriggerMenuXmlParser") << "\n Error: unknown condition (" << condition << ")" << std::endl;

    return false;
  }

  return true;
}

/**
 * parseConditions - look for conditions and call the workCondition
 *                   function for each node
 *
 * @param parser The parser to parse the XML file with.
 *
 * @return "true" if succeeded. "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseConditions(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {
  XERCES_CPP_NAMESPACE_USE

  LogTrace("L1GtTriggerMenuXmlParser") << "\nParsing conditions" << std::endl;

  DOMNode* doc = parser->getDocument();
  DOMNode* n1 = doc->getFirstChild();

  // we assume that the first child is m_xmlTagDef because it was checked in workXML

  DOMNode* chipNode = n1->getFirstChild();
  if (chipNode == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error: No child of <" << m_xmlTagDef << "> tag found." << std::endl;

    return false;
  }

  // find chip

  std::string chipName;  // name of the actual chip
  chipNode = findXMLChild(chipNode, m_xmlTagChip, true, &chipName);
  if (chipNode == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "  Error: Could not find <" << m_xmlTagChip << "> tag" << std::endl;

    return false;
  }

  unsigned int chipNr = 0;
  do {
    // find conditions
    DOMNode* conditionsNode = chipNode->getFirstChild();
    conditionsNode = findXMLChild(conditionsNode, m_xmlTagConditions);
    if (conditionsNode == nullptr) {
      edm::LogError("L1GtTriggerMenuXmlParser")
          << "Error: No <" << m_xmlTagConditions << "> child found in Chip " << chipName << std::endl;

      return false;
    }

    char* nodeName = XMLString::transcode(chipNode->getNodeName());
    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "\n  Found Chip: " << nodeName << " Name: " << chipName
    //<< std::endl;

    XMLString::release(&nodeName);

    // walk through conditions
    DOMNode* conditionNameNode = conditionsNode->getFirstChild();
    std::string conditionNameNodeName;
    conditionNameNode = findXMLChild(conditionNameNode, "", true, &conditionNameNodeName);
    while (conditionNameNode != nullptr) {
      LogTrace("L1GtTriggerMenuXmlParser")
          << "\n    Found a condition with name: " << conditionNameNodeName << std::endl;

      if (!workCondition(conditionNameNode, conditionNameNodeName, chipNr)) {
        return false;
      }
      conditionNameNode = findXMLChild(conditionNameNode->getNextSibling(), "", true, &conditionNameNodeName);
    }
    // next chip
    chipNode = findXMLChild(chipNode->getNextSibling(), m_xmlTagChip, true, &chipName);
    chipNr++;

  } while (chipNode != nullptr && chipNr < m_numberConditionChips);

  return true;
}

/**
 * workAlgorithm - parse the algorithm and insert it into algorithm map.
 *
 * @param node The corresponding node to the algorithm.
 * @param name The name of the algorithm.
 * @param chipNr The number of the chip the conditions for that algorithm are located on.
 *
 * @return "true" on success, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::workAlgorithm(XERCES_CPP_NAMESPACE::DOMNode* node,
                                             const std::string& algName,
                                             unsigned int chipNr) {
  XERCES_CPP_NAMESPACE_USE

  if (node == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "    Node is 0 in " << __PRETTY_FUNCTION__
                                         << " can not parse the algorithm " << algName << std::endl;
    return false;
  }

  // get alias
  std::string algAlias = getXMLAttribute(node, m_xmlAlgorithmAttrAlias);

  if (algAlias.empty()) {
    algAlias = algName;
    LogDebug("L1GtTriggerMenuXmlParser") << "\n    No alias defined for algorithm. Alias set to algorithm name."
                                         << "\n    Algorithm name:  " << algName
                                         << "\n    Algorithm alias: " << algAlias << std::endl;
  } else {
    LogDebug("L1GtTriggerMenuXmlParser") << "\n    Alias defined for algorithm."
                                         << "\n    Algorithm name:  " << algName
                                         << "\n    Algorithm alias: " << algAlias << std::endl;
  }

  // get the logical expression from the node
  std::string logExpression = getXMLTextValue(node);

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Logical expression: " << logExpression
  //<< std::endl;

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Chip number:        " << chipNr
  //<< std::endl;

  // determine output pin
  DOMNode* pinNode = findXMLChild(node->getFirstChild(), m_xmlTagOutput);
  std::string pinString;
  int outputPin = 0;

  pinNode = node->getFirstChild();
  if ((pinNode = findXMLChild(pinNode, m_xmlTagOutputPin)) != nullptr) {
    pinString = getXMLAttribute(pinNode, m_xmlAttrNr);

    // convert pinString to integer
    std::istringstream opStream(pinString);

    if ((opStream >> outputPin).fail()) {
      LogDebug("L1GtTriggerMenuXmlParser")
          << "    Unable to convert pin string " << pinString << " to int for algorithm : " << algName << std::endl;

      return false;
    }
  }

  if (pinNode == nullptr) {
    LogTrace("L1GtTriggerMenuXmlParser") << "    Warning: No pin number found for algorithm: " << algName << std::endl;

    return false;
  }

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Output pin:         " << outputPin
  //<< std::endl;

  // compute the bit number from chip number, output pin and order of the chips
  // pin numbering start with 1, bit numbers with 0
  int bitNumber = outputPin + (m_orderConditionChip[chipNr] - 1) * m_pinsOnConditionChip - 1;

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Bit number:         " << bitNumber
  //<< std::endl;

  // create a new algorithm and insert it into algorithm map
  L1GtAlgorithm alg(algName, logExpression, bitNumber);
  alg.setAlgoChipNumber(static_cast<int>(chipNr));
  alg.setAlgoAlias(algAlias);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    alg.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert algorithm into the map
  if (!insertAlgorithmIntoMap(alg)) {
    return false;
  }

  return true;
}

/*
 * parseAlgorithms Parse the algorithms
 * NOTE: the algorithms used here are equivalent to "prealgo" from XML file
 *       for the VERSION_FINAL
 *       The "VERSION_PROTOTYPE algo" will be phased out later in the XML file
 *       See L1GlobalTriggerConfig.h (in the attic)
 *
 * @param parser A reference to the XercesDOMParser to use.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseAlgorithms(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {
  XERCES_CPP_NAMESPACE_USE

  //LogTrace("L1GtTriggerMenuXmlParser") << "\nParsing algorithms" << std::endl;

  DOMNode* doc = parser->getDocument();
  DOMNode* node = doc->getFirstChild();

  DOMNode* chipNode = node->getFirstChild();
  if (chipNode == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "  Error: No child found for " << m_xmlTagDef << std::endl;
    return false;
  }

  // find first chip
  std::string chipName;
  chipNode = findXMLChild(chipNode, m_xmlTagChip, true, &chipName);
  if (chipNode == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "  Error: Could not find <" << m_xmlTagChip << std::endl;
    return false;
  }

  unsigned int chipNr = 0;
  do {
    //LogTrace("L1GtTriggerMenuXmlParser") << std::endl;

    std::string nodeName = m_xmlTagChip + chipName;
    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "  Chip: " << nodeName << " Name: " << chipName
    //<< std::endl;

    // find algorithms
    DOMNode* algNode = chipNode->getFirstChild();
    algNode = findXMLChild(algNode, m_xmlTagAlgorithms);
    if (algNode == nullptr) {
      edm::LogError("L1GtTriggerMenuXmlParser")
          << "    Error: No <" << m_xmlTagAlgorithms << "> child found in chip " << chipName << std::endl;
      return false;
    }

    // walk through algorithms
    DOMNode* algNameNode = algNode->getFirstChild();
    std::string algNameNodeName;
    algNameNode = findXMLChild(algNameNode, "", true, &algNameNodeName);

    while (algNameNode != nullptr) {
      //LogTrace("L1GtTriggerMenuXmlParser")
      //<< "    Found an algorithm with name: " << algNameNodeName
      //<< std::endl;

      if (!workAlgorithm(algNameNode, algNameNodeName, chipNr)) {
        return false;
      }

      algNameNode = findXMLChild(algNameNode->getNextSibling(), "", true, &algNameNodeName);
    }

    // next chip
    chipNode = findXMLChild(chipNode->getNextSibling(), m_xmlTagChip, true, &chipName);
    chipNr++;

  } while (chipNode != nullptr && chipNr < m_numberConditionChips);

  return true;
}

/**
 * workTechTrigger - parse the technical trigger and insert it into technical trigger map.
 *
 * @param node The corresponding node to the technical trigger.
 * @param name The name of the technical trigger.
 *
 * @return "true" on success, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::workTechTrigger(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& algName) {
  XERCES_CPP_NAMESPACE_USE

  if (node == nullptr) {
    LogDebug("L1GtTriggerMenuXmlParser") << "    Node is 0 in " << __PRETTY_FUNCTION__
                                         << " can not parse the technical trigger " << algName << std::endl;
    return false;
  }

  // get the logical expression from the node
  std::string logExpression = getXMLTextValue(node);

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Logical expression: " << logExpression
  //<< std::endl;

  // determine bit number (use output pin tag)
  DOMNode* pinNode = findXMLChild(node->getFirstChild(), m_xmlTagOutput);
  std::string pinString;
  int outputPin = 0;

  pinNode = node->getFirstChild();
  if ((pinNode = findXMLChild(pinNode, m_xmlTagOutputPin)) != nullptr) {
    pinString = getXMLAttribute(pinNode, m_xmlAttrNr);

    // convert pinString to integer
    std::istringstream opStream(pinString);

    if ((opStream >> outputPin).fail()) {
      LogDebug("L1GtTriggerMenuXmlParser") << "    Unable to convert pin string " << pinString
                                           << " to int for technical trigger : " << algName << std::endl;

      return false;
    }
  }

  if (pinNode == nullptr) {
    LogTrace("L1GtTriggerMenuXmlParser") << "    Warning: No pin number found for technical trigger: " << algName
                                         << std::endl;

    return false;
  }

  // set the bit number
  int bitNumber = outputPin;

  //LogTrace("L1GtTriggerMenuXmlParser")
  //<< "      Bit number:         " << bitNumber
  //<< std::endl;

  // create a new technical trigger and insert it into technical trigger map
  // alias set automatically to name
  L1GtAlgorithm alg(algName, logExpression, bitNumber);
  alg.setAlgoAlias(algName);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    alg.print(myCoutStream);
    LogTrace("L1GtTriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert technical trigger into the map
  if (!insertTechTriggerIntoMap(alg)) {
    return false;
  }

  return true;
}

/*
 * parseTechTriggers Parse the technical triggers
 *
 * @param parser A reference to the XercesDOMParser to use.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool L1GtTriggerMenuXmlParser::parseTechTriggers(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {
  XERCES_CPP_NAMESPACE_USE

  //LogTrace("L1GtTriggerMenuXmlParser") << "\nParsing technical triggers" << std::endl;

  DOMNode* doc = parser->getDocument();
  DOMNode* node = doc->getFirstChild();

  DOMNode* algNode = node->getFirstChild();
  if (algNode == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "  Error: No child found for " << m_xmlTagDef << std::endl;
    return false;
  }

  algNode = findXMLChild(algNode, m_xmlTagTechTriggers);
  if (algNode == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "    Error: No <" << m_xmlTagTechTriggers << "> child found." << std::endl;
    return false;
  }

  // walk through technical triggers
  DOMNode* algNameNode = algNode->getFirstChild();
  std::string algNameNodeName;
  algNameNode = findXMLChild(algNameNode, "", true, &algNameNodeName);

  while (algNameNode != nullptr) {
    //LogTrace("L1GtTriggerMenuXmlParser")
    //<< "    Found an technical trigger with name: " << algNameNodeName
    //<< std::endl;

    if (!workTechTrigger(algNameNode, algNameNodeName)) {
      return false;
    }

    algNameNode = findXMLChild(algNameNode->getNextSibling(), "", true, &algNameNodeName);
  }

  return true;
}

/**
 * workXML parse the XML-File
 *
 * @param parser The parser to use for parsing the XML-File
 *
 * @return "true" if succeeded, "false" if an error occurred.
 */

bool L1GtTriggerMenuXmlParser::workXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {
  XERCES_CPP_NAMESPACE_USE

  DOMDocument* doc = parser->getDocument();
  DOMNode* n1 = doc->getFirstChild();

  if (n1 == nullptr) {
    edm::LogError("L1GtTriggerMenuXmlParser") << "Error: Found no XML child" << std::endl;

    return false;
  }

  char* nodeName = XMLString::transcode(n1->getNodeName());

  if (XMLString::compareIString(nodeName, m_xmlTagDef.c_str())) {
    edm::LogError("L1GtTriggerMenuXmlParser")
        << "Error: First XML child is not \" " << m_xmlTagDef << "\" " << std::endl;

    return false;
  }

  LogTrace("L1GtTriggerMenuXmlParser") << "\nFirst node name is: " << nodeName << std::endl;
  XMLString::release(&nodeName);

  // clear possible old maps
  clearMaps();

  if (!parseId(parser)) {
    clearMaps();
    return false;
  }

  if (!parseConditions(parser)) {
    clearMaps();
    return false;
  }

  if (!parseAlgorithms(parser)) {
    clearMaps();
    return false;
  }

  if (!parseTechTriggers(parser)) {
    clearMaps();
    return false;
  }

  return true;
}

// static class members
