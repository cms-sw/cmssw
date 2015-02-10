/**
 * \class TriggerMenuXmlParser
 *
 *
 * Description: Xerces-C XML parser for the L1 Trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "TriggerMenuXmlParser.h"

// system include files
#include <string>
#include <vector>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/cstdint.hpp>

// user include files
// base class
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"

#include "L1Trigger/L1TGlobal/interface/GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "L1Trigger/L1TGlobal/src/L1TMenuEditor/L1TriggerMenu.hxx"

// constructor
l1t::TriggerMenuXmlParser::TriggerMenuXmlParser() :
    L1GtXmlParserTags(), m_xmlErrHandler(0), m_triggerMenuInterface("NULL"),
    m_triggerMenuName("NULL"), m_triggerMenuImplementation("NULL"), m_scaleDbKey("NULL")

{

    // menu names, scale key initialized to NULL due to ORACLE treatment of strings

    // empty

}

// destructor
l1t::TriggerMenuXmlParser::~TriggerMenuXmlParser() {

    clearMaps();

}

// set the number of condition chips in GTL
void l1t::TriggerMenuXmlParser::setGtNumberConditionChips(
    const unsigned int& numberConditionChipsValue) {

    m_numberConditionChips = numberConditionChipsValue;

}

// set the number of pins on the GTL condition chips
void l1t::TriggerMenuXmlParser::setGtPinsOnConditionChip(const unsigned int& pinsOnConditionChipValue) {

    m_pinsOnConditionChip = pinsOnConditionChipValue;

}

// set the correspondence "condition chip - GTL algorithm word"
// in the hardware
void l1t::TriggerMenuXmlParser::setGtOrderConditionChip(
    const std::vector<int>& orderConditionChipValue) {

    m_orderConditionChip = orderConditionChipValue;

}

// set the number of physics trigger algorithms
void l1t::TriggerMenuXmlParser::setGtNumberPhysTriggers(
        const unsigned int& numberPhysTriggersValue) {

    m_numberPhysTriggers = numberPhysTriggersValue;

}

// set the number of technical triggers
void l1t::TriggerMenuXmlParser::setGtNumberTechTriggers(
        const unsigned int& numberTechTriggersValue) {

    m_numberTechTriggers = numberTechTriggersValue;

}

// set the number of L1 jet counts received by GT
void l1t::TriggerMenuXmlParser::setGtNumberL1JetCounts(const unsigned int& numberL1JetCountsValue) {

    m_numberL1JetCounts = numberL1JetCountsValue;

}


// set the condition maps
void l1t::TriggerMenuXmlParser::setGtConditionMap(const std::vector<ConditionMap>& condMap) {
    m_conditionMap = condMap;
}

// set the trigger menu name
void l1t::TriggerMenuXmlParser::setGtTriggerMenuInterface(const std::string& menuInterface) {
    m_triggerMenuInterface = menuInterface;
}

void l1t::TriggerMenuXmlParser::setGtTriggerMenuName(const std::string& menuName) {
    m_triggerMenuName = menuName;
}

void l1t::TriggerMenuXmlParser::setGtTriggerMenuImplementation(const std::string& menuImplementation) {
    m_triggerMenuImplementation = menuImplementation;
}

// set menu associated scale key
void l1t::TriggerMenuXmlParser::setGtScaleDbKey(const std::string& scaleKey) {
    m_scaleDbKey = scaleKey;
}

// set the vectors containing the conditions
void l1t::TriggerMenuXmlParser::setVecMuonTemplate(
        const std::vector<std::vector<MuonTemplate> >& vecMuonTempl) {

    m_vecMuonTemplate = vecMuonTempl;
}

void l1t::TriggerMenuXmlParser::setVecCaloTemplate(
        const std::vector<std::vector<CaloTemplate> >& vecCaloTempl) {

    m_vecCaloTemplate = vecCaloTempl;
}

void l1t::TriggerMenuXmlParser::setVecEnergySumTemplate(
        const std::vector<std::vector<L1GtEnergySumTemplate> >& vecEnergySumTempl) {

    m_vecEnergySumTemplate = vecEnergySumTempl;
}

void l1t::TriggerMenuXmlParser::setVecJetCountsTemplate(
        const std::vector<std::vector<L1GtJetCountsTemplate> >& vecJetCountsTempl) {

    m_vecJetCountsTemplate = vecJetCountsTempl;
}

void l1t::TriggerMenuXmlParser::setVecCastorTemplate(
        const std::vector<std::vector<L1GtCastorTemplate> >& vecCastorTempl) {

    m_vecCastorTemplate = vecCastorTempl;
}

void l1t::TriggerMenuXmlParser::setVecHfBitCountsTemplate(
        const std::vector<std::vector<L1GtHfBitCountsTemplate> >& vecHfBitCountsTempl) {

    m_vecHfBitCountsTemplate = vecHfBitCountsTempl;
}

void l1t::TriggerMenuXmlParser::setVecHfRingEtSumsTemplate(
        const std::vector<std::vector<L1GtHfRingEtSumsTemplate> >& vecHfRingEtSumsTempl) {

    m_vecHfRingEtSumsTemplate = vecHfRingEtSumsTempl;
}

void l1t::TriggerMenuXmlParser::setVecBptxTemplate(
        const std::vector<std::vector<L1GtBptxTemplate> >& vecBptxTempl) {

    m_vecBptxTemplate = vecBptxTempl;
}

void l1t::TriggerMenuXmlParser::setVecExternalTemplate(
        const std::vector<std::vector<L1GtExternalTemplate> >& vecExternalTempl) {

    m_vecExternalTemplate = vecExternalTempl;
}

void l1t::TriggerMenuXmlParser::setVecCorrelationTemplate(
        const std::vector<std::vector<CorrelationTemplate> >& vecCorrelationTempl) {

    m_vecCorrelationTemplate = vecCorrelationTempl;
}

// set the vectors containing the conditions for correlation templates
//
void l1t::TriggerMenuXmlParser::setCorMuonTemplate(
        const std::vector<std::vector<MuonTemplate> >& corMuonTempl) {

    m_corMuonTemplate = corMuonTempl;
}

void l1t::TriggerMenuXmlParser::setCorCaloTemplate(
        const std::vector<std::vector<CaloTemplate> >& corCaloTempl) {

    m_corCaloTemplate = corCaloTempl;
}

void l1t::TriggerMenuXmlParser::setCorEnergySumTemplate(
        const std::vector<std::vector<L1GtEnergySumTemplate> >& corEnergySumTempl) {

    m_corEnergySumTemplate = corEnergySumTempl;
}




// set the algorithm map (by algorithm names)
void l1t::TriggerMenuXmlParser::setGtAlgorithmMap(const AlgorithmMap& algoMap) {
    m_algorithmMap = algoMap;
}

// set the algorithm map (by algorithm aliases)
void l1t::TriggerMenuXmlParser::setGtAlgorithmAliasMap(const AlgorithmMap& algoMap) {
    m_algorithmAliasMap = algoMap;
}

// set the technical trigger map
void l1t::TriggerMenuXmlParser::setGtTechnicalTriggerMap(const AlgorithmMap& ttMap) {
    m_technicalTriggerMap = ttMap;
}

//


// parse def.xml and vme.xml files
void l1t::TriggerMenuXmlParser::parseXmlFile(const std::string& defXmlFile,
    const std::string& vmeXmlFile) {

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
    m_triggerMenuName.erase(m_triggerMenuName.begin(), m_triggerMenuName.begin()
            + xmlPos + 1);

    xmlPos = m_triggerMenuName.find_last_of(".");
    m_triggerMenuName.erase(m_triggerMenuName.begin() + xmlPos, m_triggerMenuName.end());

    // error handler for xml-parser
    m_xmlErrHandler = 0;

    std::auto_ptr<l1t::L1TriggerMenu> tm(l1t::l1TriggerMenu(defXmlFile));

    LogTrace("TriggerMenuXmlParser") << "\nOpening XML-File: \n  " << defXmlFile << std::endl;

    l1t::ConditionList conditions = tm->conditions();

    workXML( tm );

//     if ((parser = initXML(defXmlFile)) != 0) {
//         workXML(parser);
//     }
//     cleanupXML(parser);

}

//

void l1t::TriggerMenuXmlParser::setGtTriggerMenuInterfaceDate(const std::string& val) {

    m_triggerMenuInterfaceDate = val;

}

void l1t::TriggerMenuXmlParser::setGtTriggerMenuInterfaceAuthor(const std::string& val) {

    m_triggerMenuInterfaceAuthor = val;

}

void l1t::TriggerMenuXmlParser::setGtTriggerMenuInterfaceDescription(const std::string& val) {

    m_triggerMenuInterfaceDescription = val;

}


void l1t::TriggerMenuXmlParser::setGtTriggerMenuDate(const std::string& val) {

    m_triggerMenuDate = val;

}

void l1t::TriggerMenuXmlParser::setGtTriggerMenuAuthor(const std::string& val) {

    m_triggerMenuAuthor = val;

}

void l1t::TriggerMenuXmlParser::setGtTriggerMenuDescription(const std::string& val) {

    m_triggerMenuDescription = val;

}

void l1t::TriggerMenuXmlParser::setGtAlgorithmImplementation(const std::string& val) {

    m_algorithmImplementation = val;

}

// private methods



/**
 * initXML - Initialize XML-utilities and try to create a parser for the specified file.
 *
 * @param xmlFile Filename of the XML-File
 *
 * @return A reference to a XercesDOMParser object if succeeded. 0 if an error occurred.
 *
 */

XERCES_CPP_NAMESPACE::XercesDOMParser* l1t::TriggerMenuXmlParser::initXML(const std::string &xmlFile) {

    XERCES_CPP_NAMESPACE_USE

    // try to initialize
    try {
        XMLPlatformUtils::Initialize();
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());

        edm::LogError("TriggerMenuXmlParser")
        << "Error during Xerces-c initialization! :"
        << message << std::endl;

        XMLString::release(&message);
        return 0;
    }

    XercesDOMParser* parser = new XercesDOMParser();
    parser->setValidationScheme(XercesDOMParser::Val_Always);
    parser->setDoNamespaces(false); // we got no dtd

    if (m_xmlErrHandler == 0) { // redundant check
        m_xmlErrHandler = (ErrorHandler*) new HandlerBase();
    }
    else {
        // TODO ASSERTION
    }
    parser->setErrorHandler(m_xmlErrHandler);

    // try to parse the file
    try {
        parser->parse(xmlFile.c_str());
    }
    catch(const XMLException &toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());

        edm::LogError("TriggerMenuXmlParser")
        << "Exception while parsing XML: \n"
        << message << std::endl;

        XMLString::release(&message);
        delete parser;
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
        return 0;
    }
    catch (const DOMException &toCatch) {
        char *message = XMLString::transcode(toCatch.msg);

        edm::LogError("TriggerMenuXmlParser")
        << "DOM-Exception while parsing XML: \n"
        << message << std::endl;

        XMLString::release(&message);
        delete parser;
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
        return 0;
    }
    catch (...) {

        edm::LogError("TriggerMenuXmlParser")
        << "Unexpected Exception while parsing XML!"
        << std::endl;

        delete parser;
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
        return 0;
    }

    return parser;
}

// find a named child of a xml node
XERCES_CPP_NAMESPACE::DOMNode* l1t::TriggerMenuXmlParser::findXMLChild(
    XERCES_CPP_NAMESPACE::DOMNode* startChild, const std::string& tagName, bool beginOnly,
    std::string* rest) {

    XERCES_CPP_NAMESPACE_USE

    char* nodeName = 0;

    DOMNode *n1 = startChild;
    if (n1 == 0) {
        return 0;
    }

    if ( !tagName.empty() ) {
        nodeName = XMLString::transcode(n1->getNodeName());

        if (!beginOnly) {
            //match the whole tag
            while (XMLString::compareIString(nodeName, tagName.c_str())) {

                XMLString::release(&nodeName);
                n1 = n1->getNextSibling();
                if (n1 == 0) {
                    break;
                }

                nodeName = XMLString::transcode(n1->getNodeName());
            }
        }
        else {
            // match only the beginning
            while (XMLString::compareNIString(nodeName, tagName.c_str(), tagName.length())) {
                XMLString::release(&nodeName);
                n1 = n1->getNextSibling();
                if (n1 == 0) {
                    break;
                }

                nodeName = XMLString::transcode(n1->getNodeName());
            }
            if (n1 != 0 && rest != 0) {
                *rest = std::string(nodeName).substr(tagName.length(), strlen(nodeName) - tagName.length());
            }
        }
    }
    else { // empty string given
        while (n1->getNodeType() != DOMNode::ELEMENT_NODE) {
            n1 = n1->getNextSibling();
            if (n1 == 0) {
                break;
            }

        }
        if (n1 != 0 && rest != 0) {
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

std::string l1t::TriggerMenuXmlParser::getXMLAttribute(const XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name) {

    XERCES_CPP_NAMESPACE_USE

    std::string ret;

    // get attributes list
    DOMNamedNodeMap* attributes = node->getAttributes();
    if (attributes == 0) {
        return ret;
    }

    // get attribute node
    XMLCh* attrName = XMLString::transcode(name.c_str());
    DOMNode* attribNode = attributes->getNamedItem(attrName);

    XMLString::release(&attrName);
    if (attribNode == 0) {
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

std::string l1t::TriggerMenuXmlParser::getXMLTextValue(XERCES_CPP_NAMESPACE::DOMNode* node) {

    XERCES_CPP_NAMESPACE_USE

    std::string ret;

    DOMNode* n1 = node;
    if (n1 == 0) {
        return ret;
    }

    const XMLCh* retXmlCh = n1->getTextContent();
    if (retXmlCh == 0) {
        return ret;
    }

    char* retCstr = XMLString::transcode(retXmlCh);
    XMLString::trim(retCstr); // trim spaces

    ret = retCstr;
    XMLString::release(&retCstr);

    return ret;
}

/**
 * hexString2UInt128 converts an up to 128 bit hexadecimal string to two boost::uint64_t
 *
 * @param hex The string to be converted.
 * @param dstL The target for the lower 64 bit.
 * @param dstH The target for the upper 64 bit.
 *
 * @return true if conversion succeeded, false if an error occurred.
 */

bool l1t::TriggerMenuXmlParser::hexString2UInt128(const std::string& hexString,
    boost::uint64_t& dstL, boost::uint64_t& dstH) {

    // string to determine start of hex value, do not ignore leading zeros
    static const std::string valid_hex_start("0123456789ABCDEFabcdef");

    // string to determine end of hex value
    static const std::string valid_hex_end("0123456789ABCDEFabcdef");

    std::string tempStr = hexString;

    // start / end position of the hex value in the string
    unsigned int hexStart = tempStr.find_first_of(valid_hex_start);
    unsigned int hexEnd = tempStr.find_first_not_of(valid_hex_end, hexStart);

    if (hexStart == hexEnd) {

        LogDebug("TriggerMenuXmlParser") << "No hex value found in: " << tempStr << std::endl;

        return false;
    }

    tempStr = tempStr.substr(hexStart, hexEnd - hexStart);

    if (tempStr.empty() ) {

        LogDebug("TriggerMenuXmlParser") << "Empty value in " << __PRETTY_FUNCTION__
            << std::endl;

        return false;
    }

    // split the string
    std::string tempStrH, tempStrL;

    if (tempStr.length() > 16) { // more than 64 bit
        tempStrL = tempStr.substr(tempStr.length()-16, 16);
        tempStrH = tempStr.substr(0, tempStr.length()-16);
    }
    else {
        tempStrL = tempStr;
        tempStrH = "0";
    }

    // convert lower 64bit
    char* endPtr = (char*) tempStrL.c_str();
    boost::uint64_t tempUIntL = strtoull(tempStrL.c_str(), &endPtr, 16);

    if (*endPtr != 0) {

        LogDebug("TriggerMenuXmlParser") << "Unable to convert " << tempStr << " to hex."
            << std::endl;

        return false;
    }

    // convert higher64 bit
    endPtr = (char*) tempStrH.c_str();
    boost::uint64_t tempUIntH = strtoull(tempStrH.c_str(), &endPtr, 16);

    if (*endPtr != 0) {

        LogDebug("TriggerMenuXmlParser") << "Unable to convert " << tempStr << " to hex."
            << std::endl;

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

bool l1t::TriggerMenuXmlParser::getXMLHexTextValue128Old(XERCES_CPP_NAMESPACE::DOMNode* node,
    boost::uint64_t& dstL, boost::uint64_t& dstH) {

    if (node == 0) {

        LogDebug("TriggerMenuXmlParser") << "node == 0 in " << __PRETTY_FUNCTION__ << std::endl;

        return false;
    }

    boost::uint64_t tempUIntH, tempUIntL;

    std::string tempStr = getXMLTextValue(node);
    if ( !hexString2UInt128(tempStr, tempUIntL, tempUIntH) ) {
        return false;
    }

    dstL = tempUIntL;
    dstH = tempUIntH;

    return true;
}

bool l1t::TriggerMenuXmlParser::getXMLHexTextValue128(const std::string& childName,
    boost::uint64_t& dstL, boost::uint64_t& dstH) {

    boost::uint64_t tempUIntH, tempUIntL;

    std::string tempStr = childName;
    if ( !hexString2UInt128(tempStr, tempUIntL, tempUIntH) ) {
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

bool l1t::TriggerMenuXmlParser::getXMLHexTextValueOld(XERCES_CPP_NAMESPACE::DOMNode* node,
    boost::uint64_t& dst) {

    boost::uint64_t dummyH; // dummy for eventual higher 64bit
    boost::uint64_t tempUInt; // temporary unsigned integer

    if ( !getXMLHexTextValue128Old(node, tempUInt, dummyH) ) {
        return false;
    }

    if (dummyH != 0) {
        edm::LogError("TriggerMenuXmlParser") << "Too large hex-value!" << std::endl;
        return false;
    }

    dst = tempUInt;

    return true;
}

bool l1t::TriggerMenuXmlParser::getXMLHexTextValue(const std::string& childName,
    boost::uint64_t& dst) {

    boost::uint64_t dummyH; // dummy for eventual higher 64bit
    boost::uint64_t tempUInt; // temporary unsigned integer

    if ( !getXMLHexTextValue128( childName, tempUInt, dummyH) ) {
        return false;
    }

    if (dummyH != 0) {
        edm::LogError("TriggerMenuXmlParser") << "Too large hex-value!" << std::endl;
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
bool l1t::TriggerMenuXmlParser::countConditionChildMaxBits( const std::string& childName, 
    unsigned int& dst) {

    XERCES_CPP_NAMESPACE_USE

    // should never happen...
    // first try direct
    std::string maxString = childName; // string for the maxbits

    // do the hex conversion

    boost::uint64_t maxBitsL, maxBitsH;
    if ( !hexString2UInt128(maxString, maxBitsL, maxBitsH) ) {
        return false;
    }

    // count the bits
    //LogTrace("TriggerMenuXmlParser")
    //<< std::dec
    //<< "        words: dec: high (MSB) word = " << maxBitsH << " low word = " << maxBitsL
    //<< std::hex << "\n"
    //<< "        words: hex: high (MSB) word = " << maxBitsH << " low word = " << maxBitsL
    //<< std::dec
    //<< std::endl;

    unsigned int counter = 0;

    while (maxBitsL != 0) {
        // check if bits set countinously
        if ( (maxBitsL & 1) == 0) {

            edm::LogError("TriggerMenuXmlParser")
	      << "      Confused by not continous set bits for max value " << maxString
	      << std::endl;

            return false;
        }

        maxBitsL >>= 1;
        counter++;
    }

    if ( (maxBitsH != 0) && (counter != 64)) {

        edm::LogError("TriggerMenuXmlParser")
	  << "      Confused by not continous set bits for max value " << maxString 
	  << std::endl;

        return false;
    }

    while (maxBitsH != 0) {
        //check if bits set countinously
        if ( (maxBitsH & 1) == 0) {

            edm::LogError("TriggerMenuXmlParser")
	      << "      Confused by not continous set bits for max value " << maxString
	      << std::endl;

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
 * @param dst A pointer to a vector of boost::uint64_t where the results are written.
 *
 * @return true if succeeded. false if an error occurred or not enough values found.
 */

bool l1t::TriggerMenuXmlParser::getConditionChildValuesOld(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& childName, unsigned int num, std::vector<boost::uint64_t>& dst) {

    XERCES_CPP_NAMESPACE_USE

    if (node == 0) {

        LogDebug("TriggerMenuXmlParser")
        << "node == 0 in " << __PRETTY_FUNCTION__
        << std::endl;

        return false;
    }

    DOMNode* n1 = findXMLChild(node->getFirstChild(), childName);

    // if child not found
    if (n1 == 0) {

        LogDebug("TriggerMenuXmlParser") << "Child of condition not found ( " << childName
            << ")" << std::endl;

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
        if (n1 == 0) {

            LogDebug("TriggerMenuXmlParser") << "Not enough values in condition child ( "
                << childName << ")" << std::endl;

            return false;
        }

        if ( !getXMLHexTextValueOld(n1, dst[i]) ) {

            edm::LogError("TriggerMenuXmlParser") << "Error converting condition child ( "
                << childName << ") value." << std::endl;

            return false;
        }

        n1 = findXMLChild(n1->getNextSibling(), m_xmlTagValue); // next child
    }

    return true;
}

/**
 * cleanupXML - Delete parser and error handler. Shutdown XMLPlatformUtils.
 *
 * @param parser A reference to the parser to be deleted.
 *
 */

void l1t::TriggerMenuXmlParser::cleanupXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {

    XERCES_CPP_NAMESPACE_USE

    if (parser != 0) {
        delete parser;
    }

    if (m_xmlErrHandler != 0) {
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
    }

    XMLPlatformUtils::Terminate();

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

bool l1t::TriggerMenuXmlParser::parseVmeXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {

    XERCES_CPP_NAMESPACE_USE

    DOMDocument* doc = parser->getDocument();
    DOMNode* n1 = doc->getFirstChild();

    if (n1 == 0) {

        edm::LogError("TriggerMenuXmlParser") << "Error: Found no XML child" << std::endl;

        return false;
    }

    // find "vme"-tag
    n1 = findXMLChild(n1, m_xmlTagVme);
    if (n1 == 0) {

        edm::LogError("TriggerMenuXmlParser") << "Error: No vme tag found." << std::endl;
        return false;
    }

    n1 = n1->getFirstChild();

    unsigned int chipCounter = 0; // count chips

    while (chipCounter < m_numberConditionChips) {

        n1 = findXMLChild(n1, m_xmlTagChip, true);
        if (n1 == 0) {
            // just break if no more chips found
            break;
        }

        // node for a particle
        //DOMNode* particleNode = n1->getFirstChild(); // FIXME un-comment

        // FIXME parse vme.xml, modify the menu

        n1 = n1->getNextSibling();
        chipCounter++;
    } // end while chipCounter

    return true;

}

// methods for conditions and algorithms

// clearMaps - delete all conditions and algorithms in
// the maps and clear the maps.
void l1t::TriggerMenuXmlParser::clearMaps() {

    // loop over condition maps (one map per condition chip)
    // then loop over conditions in the map
    for (std::vector<ConditionMap>::iterator itCondOnChip = m_conditionMap.begin(); itCondOnChip
        != m_conditionMap.end(); itCondOnChip++) {

        // the conditions in the maps are deleted in L1uGtTriggerMenu, not here

        itCondOnChip->clear();

    }

    // the algorithms in the maps are deleted in L1uGtTriggerMenu, not here
    m_algorithmMap.clear();

}

// insertConditionIntoMap - safe insert of condition into condition map.
// if the condition name already exists, do not insert it and return false
bool l1t::TriggerMenuXmlParser::insertConditionIntoMap(GtCondition& cond, const int chipNr) {

    std::string cName = cond.condName();
    //LogTrace("TriggerMenuXmlParser")
    //<< "    Trying to insert condition \"" << cName << "\" in the condition map." ;

    // no condition name has to appear twice!
    if ((m_conditionMap[chipNr]).count(cName) != 0) {
        LogTrace("TriggerMenuXmlParser") << "      Condition " << cName
            << " already exists - not inserted!" << std::endl;
        return false;
    }

    (m_conditionMap[chipNr])[cName] = &cond;
    //LogTrace("TriggerMenuXmlParser")
    //<< "      OK - condition inserted!"
    //<< std::endl;


    return true;

}

// insert an algorithm into algorithm map
bool l1t::TriggerMenuXmlParser::insertAlgorithmIntoMap(const L1GtAlgorithm& alg) {

    std::string algName = alg.algoName();
    std::string algAlias = alg.algoAlias();
    //LogTrace("TriggerMenuXmlParser")
    //<< "    Trying to insert algorithm \"" << algName << "\" in the algorithm map." ;

    // no algorithm name has to appear twice!
    if (m_algorithmMap.count(algName) != 0) {
        LogTrace("TriggerMenuXmlParser") << "      Algorithm \"" << algName
            << "\"already exists in the algorithm map- not inserted!" << std::endl;
        return false;
    }

    if (m_algorithmAliasMap.count(algAlias) != 0) {
        LogTrace("TriggerMenuXmlParser") << "      Algorithm alias \"" << algAlias
            << "\"already exists in the algorithm alias map- not inserted!" << std::endl;
        return false;
    }

    // bit number less than zero or greater than maximum number of algorithms
    int bitNumber = alg.algoBitNumber();
    if ((bitNumber < 0) || (bitNumber >= static_cast<int>(m_numberPhysTriggers))) {
        LogTrace("TriggerMenuXmlParser") << "      Bit number " << bitNumber
            << " outside allowed range [0, " << m_numberPhysTriggers
            << ") - algorithm not inserted!" << std::endl;
        return false;
    }

    // maximum number of algorithms
    if (m_algorithmMap.size() >= m_numberPhysTriggers) {
        LogTrace("TriggerMenuXmlParser") << "      More than maximum allowed "
            << m_numberPhysTriggers << " algorithms in the algorithm map - not inserted!"
            << std::endl;
        return false;
    }

    // chip number outside allowed values
    int chipNr = alg.algoChipNumber(static_cast<int>(m_numberConditionChips),
        static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

    if ((chipNr < 0) || (chipNr > static_cast<int>(m_numberConditionChips))) {
        LogTrace("TriggerMenuXmlParser") << "      Chip number " << chipNr
            << " outside allowed range [0, " << m_numberConditionChips
            << ") - algorithm not inserted!" << std::endl;
        return false;
    }

    // output pin outside allowed values
    int outputPin = alg.algoOutputPin(static_cast<int>(m_numberConditionChips),
        static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

    if ((outputPin < 0) || (outputPin > static_cast<int>(m_pinsOnConditionChip))) {
        LogTrace("TriggerMenuXmlParser") << "      Output pin " << outputPin
            << " outside allowed range [0, " << m_pinsOnConditionChip
            << "] - algorithm not inserted!" << std::endl;
        return false;
    }

    // no two algorithms on the same chip can have the same output pin
    for (CItAlgo itAlgo = m_algorithmMap.begin(); itAlgo != m_algorithmMap.end(); itAlgo++) {

        int iPin = (itAlgo->second).algoOutputPin( static_cast<int>(m_numberConditionChips),
            static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);
        std::string iName = itAlgo->first;
        int iChip = (itAlgo->second).algoChipNumber(static_cast<int>(m_numberConditionChips),
            static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

        if ( (outputPin == iPin) && (chipNr == iChip)) {
            LogTrace("TriggerMenuXmlParser") << "      Output pin " << outputPin
                << " is the same as for algorithm " << iName
                << "\n      from the same chip number " << chipNr << " - algorithm not inserted!"
                << std::endl;
            return false;
        }

    }

    // insert algorithm
    m_algorithmMap[algName] = alg;
    m_algorithmAliasMap[algAlias] = alg;

    //LogTrace("TriggerMenuXmlParser")
    //<< "      OK - algorithm inserted!"
    //<< std::endl;

    return true;

}

// insert a technical trigger into technical trigger map
bool l1t::TriggerMenuXmlParser::insertTechTriggerIntoMap(const L1GtAlgorithm& alg) {

    std::string algName = alg.algoName();
    //LogTrace("TriggerMenuXmlParser")
    //<< "    Trying to insert technical trigger \"" << algName
    //<< "\" in the technical trigger map." ;

    // no technical trigger name has to appear twice!
    if (m_technicalTriggerMap.count(algName) != 0) {
        LogTrace("TriggerMenuXmlParser") << "      Technical trigger \""
                << algName
                << "\"already exists in the technical trigger map- not inserted!"
                << std::endl;
        return false;
    }

    // bit number less than zero or greater than maximum number of technical triggers
    int bitNumber = alg.algoBitNumber();
    if ((bitNumber < 0)
            || (bitNumber >= static_cast<int>(m_numberTechTriggers))) {
        LogTrace("TriggerMenuXmlParser") << "      Bit number "
                << bitNumber << " outside allowed range [0, "
                << m_numberTechTriggers
                << ") - technical trigger not inserted!" << std::endl;
        return false;
    }

    // no two technical triggers can have the same bit number
    for (CItAlgo itAlgo = m_technicalTriggerMap.begin(); itAlgo
            != m_technicalTriggerMap.end(); itAlgo++) {

        int iBitNumber = (itAlgo->second).algoBitNumber();
        std::string iName = itAlgo->first;

        if ((iBitNumber == bitNumber)) {
            LogTrace("TriggerMenuXmlParser") << "      Bit number "
                    << iBitNumber << " is the same as for technical trigger "
                    << iName << " - technical trigger not inserted!"
                    << std::endl;
            return false;
        }

    }

    // maximum number of technical triggers
    if (m_technicalTriggerMap.size() >= m_numberTechTriggers) {
        LogTrace("TriggerMenuXmlParser")
                << "      More than maximum allowed " << m_numberTechTriggers
                << " technical triggers in the technical trigger map - not inserted!"
                << std::endl;
        return false;
    }

    // insert technical trigger
    m_technicalTriggerMap[algName] = alg;

    //LogTrace("TriggerMenuXmlParser")
    //<< "      OK - technical trigger inserted!"
    //<< std::endl;

    return true;

}



// get the type of the condition, as defined in enum, from the condition type
// as defined in the XML file
l1t::GtConditionType l1t::TriggerMenuXmlParser::getTypeFromType(const std::string& type) {

    if (type == m_xmlConditionAttrType1s) {
        return l1t::Type1s;
    }

    if (type == m_xmlConditionAttrType2s) {
        return l1t::Type2s;
    }

    if (type == m_xmlConditionAttrType3s) {
        return l1t::Type3s;
    }

    if (type == m_xmlConditionAttrType4s) {
        return l1t::Type4s;
    }

    if (type == m_xmlConditionAttrType2wsc) {
        return l1t::Type2wsc;
    }

    if (type == m_xmlConditionAttrType2cor) {
        return l1t::Type2cor;
    }

    return l1t::TypeNull;
}

/**
 * getNumFromType - get the number of particles from a specified type name
 * (for calorimeter objects and muons)
 *
 * @param type The name of the type
 *
 * @return The number of particles in this condition. -1 if type not found.
 */

int l1t::TriggerMenuXmlParser::getNumFromType(const std::string &type) {

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

int l1t::TriggerMenuXmlParser::getBitFromNode(XERCES_CPP_NAMESPACE::DOMNode* node) {

    if (getXMLAttribute(node, m_xmlAttrMode) != m_xmlAttrModeBit) {

        edm::LogError("TriggerMenuXmlParser") << "Invalid mode for single bit" << std::endl;

        return -1;
    }

    std::string tmpStr = getXMLTextValue(node);
    if (tmpStr == "0") {
        return 0;
    }
    else if (tmpStr == "1") {
        return 1;
    }
    else {
        edm::LogError("TriggerMenuXmlParser") << "Bad bit value (" << tmpStr << ")"
            << std::endl;
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

int l1t::TriggerMenuXmlParser::getGEqFlag(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& nodeName) {

    XERCES_CPP_NAMESPACE_USE

    if (node == 0) {

        LogDebug("TriggerMenuXmlParser")
        << "node == 0 in " << __PRETTY_FUNCTION__
        << std::endl;

        return -1;
    }

    // usually the GEq flag is a child of the first child (the first element node)
    DOMNode* n1 = node->getFirstChild();
    n1 = findXMLChild(n1, nodeName);

    if (n1 != 0) {
        n1 = findXMLChild(n1->getFirstChild(), m_xmlTagGEq);
        if (n1 == 0) {

            LogDebug("TriggerMenuXmlParser") << "No \"greater or equal\" tag found"
                << std::endl;

            return -1;
        }

        return getBitFromNode(n1);
    }
    else {

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

bool l1t::TriggerMenuXmlParser::getMuonMipIsoBits(XERCES_CPP_NAMESPACE::DOMNode* node,
    unsigned int num, std::vector<bool>& mipDst, std::vector<bool>& isoEnDst,
    std::vector<bool>& isoReqDst) {

    XERCES_CPP_NAMESPACE_USE

    if (node == 0) {
        return false;
    }

    // find ptLowThreshold child
    DOMNode* n1 = findXMLChild(node->getFirstChild(), m_xmlTagPtLowThreshold);

    if (n1 == 0) {
        return false;
    }

    // get first value tag
    n1 = findXMLChild(n1->getFirstChild(), m_xmlTagValue);

    for (unsigned int i = 0; i < num; i++) {

        if (n1 == 0) {
            return false;
        }

        // MIP bit

        DOMNode* bitnode = findXMLChild(n1->getFirstChild(), m_xmlTagEnableMip);
        if (bitnode == 0) {
            return true;
        }

        int tmpint = getBitFromNode(bitnode);
        if (tmpint < 0) {
            return false;
        }

        mipDst[i] = (tmpint != 0);

        //LogTrace("TriggerMenuXmlParser")
        //<< "      MIP bit value for muon " << i << " = " << mipDst[i]
        //<< std::endl;


        // enable iso bit
        bitnode = findXMLChild(n1->getFirstChild(), m_xmlTagEnableIso);
        if (bitnode == 0) {
            return true;
        }

        tmpint = getBitFromNode(bitnode);
        if (tmpint < 0) {
            return false;
        }

        isoEnDst[i] = (tmpint != 0);

        //LogTrace("TriggerMenuXmlParser")
        //<< "      Enabled iso bit value for muon " << i << " = " << isoEnDst[i]
        //<< std::endl;

        // request iso bit
        bitnode = findXMLChild(n1->getFirstChild(), m_xmlTagRequestIso);
        if (bitnode == 0) {
            return true;
        }

        tmpint = getBitFromNode(bitnode);
        if (tmpint < 0) {
            return false;
        }

        isoReqDst[i] = (tmpint != 0);

        //LogTrace("TriggerMenuXmlParser")
        //<< "      Request iso bit value for muon " << i << " = " << isoReqDst[i]
        //<< std::endl;

        //
        n1 = findXMLChild(n1->getNextSibling(), m_xmlTagValue); // next value
    }

    return true;
}


template <typename T> std::string l1t::TriggerMenuXmlParser::l1t2string( T data ){
  std::stringstream ss;
  ss << data;
  return ss.str();
}
std::string l1t::TriggerMenuXmlParser::l1tDateTime2string( l1t::DateTime date ){
  std::stringstream ss;
  ss << std::setfill('0');
  ss << std::setw(4) << date.year() << "-" << std::setw(2) << date.month() << "-" << std::setw(2) << date.day() << "T";
  ss << std::setw(2) << date.hours() << ":" << std::setw(2) << date.minutes() << ":" << std::setw(2) << date.seconds();
  //ss << data;
  return ss.str();
}
int l1t::TriggerMenuXmlParser::l1t2int( l1t::RelativeBx data ){
  std::stringstream ss;
  ss << data;
  int value;
  ss >> value;
  return value;
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

bool l1t::TriggerMenuXmlParser::parseMuon(l1t::MuonCondition condMu,
        unsigned int chipNr, const bool corrFlag) {

    XERCES_CPP_NAMESPACE_USE

    // get condition, particle name (must be muon) and type name
    std::string condition = "muon";
    std::string particle = "muon";//l1t2string( condMu.objectType() );
    std::string type = l1t2string( condMu.type() );
    std::string name = l1t2string( condMu.name() );

    if( particle=="mu" ) particle = "muon";

    if( type=="double_wsc" )  type = "2_wsc";
    else if( type=="single" ) type = "1_s";
    else if( type=="double" ) type = "2_s";
    else if( type=="triple" ) type = "3";
    else if( type=="quad"   ) type = "4";

    LogDebug("l1t|Global")
      << "\n ****************************************** "
      << "\n      parseMuon  "
      << "\n condition = " << condition
      << "\n particle  = " << particle
      << "\n type      = " << type
      << "\n name      = " << name
      << std::endl;

    if (particle != m_xmlConditionAttrObjectMu) {
        edm::LogError("TriggerMenuXmlParser") << "Wrong particle for muon-condition ("
            << particle << ")" << std::endl;
        return false;
    }

    // get greater equal flag
    std::string str_etComparison = l1t2string( condMu.comparison_operator() );

    int nrObj = getNumFromType(type);
    if (nrObj < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Unknown type for calo-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    // get greater equal flag
    int intGEq = ( str_etComparison=="ge" ) ? 1 : 0;
    if (intGEq < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Error getting \"greater or equal\" flag"
            << std::endl;
        return false;
    }
    // set the boolean value for the ge_eq mode
    bool gEq = (intGEq != 0);

//     // get values

    // temporary storage of the parameters
    std::vector<MuonTemplate::ObjectParameter> objParameter(nrObj);
    MuonTemplate::CorrelationParameter corrParameter;

    // need at least two values for deltaPhi
    std::vector<boost::uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);
    tmpValues.reserve( nrObj );

    if( int(condMu.objectRequirements().objectRequirement().size())!=nrObj ){
      edm::LogError("TriggerMenuXmlParser") << " condMu objects: nrObj = " << nrObj
					    << "condMu.objectRequirements().objectRequirement().size() = " 
					    << condMu.objectRequirements().objectRequirement().size()
					    << std::endl;
      return false;
    }

    
    std::string str_chargeCorrelation = l1t2string( condMu.requestedChargeCorr() );

    unsigned int chargeCorrelation = 0;
    if( str_chargeCorrelation=="ig" )      chargeCorrelation = 1;
    else if( str_chargeCorrelation=="ls" ) chargeCorrelation = 2;
    else if( str_chargeCorrelation=="os" ) chargeCorrelation = 4;

    //getXMLHexTextValue("1", dst);
    corrParameter.chargeCorrelation = chargeCorrelation;//tmpValues[0];

    std::string str_condMu = "";
    boost::uint64_t tempUIntH, tempUIntL;
    boost::uint64_t dst;
    int cnt = 0;
    for( l1t::MuonObjectRequirementList::objectRequirement_const_iterator objPar = condMu.objectRequirements().objectRequirement().begin();
	 objPar != condMu.objectRequirements().objectRequirement().end(); ++objPar ){

      // ET Threshold
      str_condMu = l1t2string( objPar->ptThreshold() );
      if( !getXMLHexTextValue(str_condMu, dst) ) return false;
      //if( cnt<nrObj ) objParameter[cnt].etThreshold = dst;
      /// DMP: Use dec instead of hex
      if( cnt<nrObj ){
	objParameter[cnt].ptHighThreshold = objPar->ptThreshold();
	objParameter[cnt].ptLowThreshold  = objPar->ptThreshold();
      }

      // Eta Range
      //str_condMu = "ffff";
      str_condMu = "7f7f";
      //str_condMu = "0f0f";
      if( !getXMLHexTextValue(str_condMu, dst) ) return false;
      if( cnt<nrObj ) objParameter[cnt].etaRange = dst;

      // Phi Range
      str_condMu = "3ffff";
      if( !getXMLHexTextValue(str_condMu, dst) ) return false;
      //if( cnt<nrObj ) objParameter[cnt].phiRange = dst;
      getXMLHexTextValue("8f", dst);
      objParameter[cnt].phiHigh = dst;//tmpValues[i];
      objParameter[cnt].phiLow  = dst;//tmpValues[i];

      objParameter[cnt].enableMip = false;//tmpMip[i];
      objParameter[cnt].enableIso = false;//tmpEnableIso[i];
      objParameter[cnt].requestIso = false;//tmpRequestIso[i];

      std::string str_charge = l1t2string( objPar->requestedCharge() );
      int charge = 0;
      if( str_charge=="ign" )       charge = -1;
      else if( str_charge=="pos" ) charge = 0;
      else if( str_charge=="neg" ) charge = 1;

      objParameter[cnt].charge = charge;

      int cntQual=0;
      int qualityLUT = 0;
      for( l1t::MuonQualityLUT::quality_const_iterator iQualFlag = objPar->qualityLut().quality().begin();
	   iQualFlag != objPar->qualityLut().quality().end(); ++iQualFlag ){
	
	bool flag = (*iQualFlag);

	qualityLUT |= (flag << cntQual);

	LogDebug("l1t|Global")
	  << "\n quality flag " << cntQual << " = " << flag
	  << "\n qualityLUT = " << qualityLUT 
	  << std::endl;

	cntQual++;
      }

      objParameter[cnt].qualityLUT = qualityLUT;


      int cntIso=0;
      int isolationLUT = 0;
      for( l1t::MuonIsolationLUT::isolation_const_iterator iIsoFlag = objPar->isolationLut().isolation().begin();
	   iIsoFlag != objPar->isolationLut().isolation().end(); ++iIsoFlag ){
	
	bool flag = (*iIsoFlag);

	isolationLUT |= (flag << cntIso);

	LogDebug("l1t|Global")
	  << "\n isolation flag " << cntIso << " = " << flag
	  << "\n isolationLUT = " << isolationLUT 
	  << std::endl;

	cntIso++;
      }

      objParameter[cnt].isolationLUT = isolationLUT;


      int cntEta=0;
      unsigned int etaWindowLower=-1, etaWindowUpper=-1, etaWindowVetoLower=-1, etaWindowVetoUpper=-1;
      // Temporary before translation
      for( l1t::MuonObjectRequirement::etaWindow_const_iterator etaWindow =objPar->etaWindow().begin();
	   etaWindow != objPar->etaWindow().end(); ++etaWindow ){
	
	LogDebug("l1t|Global")
	  << "\n etaWindow lower = " << etaWindow->lower()
	  << "\n etaWindow upper = " << etaWindow->upper() 
	  << std::endl;
	if( cntEta==0 ){      etaWindowLower = etaWindow->lower(); etaWindowUpper = etaWindow->upper(); }
	else if( cntEta==1 ){ etaWindowVetoLower = etaWindow->lower(); etaWindowVetoUpper = etaWindow->upper(); }
	cntEta++;
      }

      int cntPhi=0;
      unsigned int phiWindowLower=-1, phiWindowUpper=-1, phiWindowVetoLower=-1, phiWindowVetoUpper=-1;
      for( l1t::MuonObjectRequirement::phiWindow_const_iterator phiWindow =objPar->phiWindow().begin();
	   phiWindow != objPar->phiWindow().end(); ++phiWindow ){
 
	LogDebug("l1t|Global")
	  << "\n phiWindow begin = " << phiWindow->lower()
	  << "\n phiWindow end   = " << phiWindow->upper() 
	  << std::endl;

	if( cntPhi==0 ){      phiWindowLower = phiWindow->lower(); phiWindowUpper = phiWindow->upper(); }
	else if( cntPhi==1 ){ phiWindowVetoLower = phiWindow->lower(); phiWindowVetoUpper = phiWindow->upper(); }
	cntPhi++;
      }

      objParameter[cnt].etaWindowLower     = etaWindowLower;
      objParameter[cnt].etaWindowUpper     = etaWindowUpper;
      objParameter[cnt].etaWindowVetoLower = etaWindowVetoLower;
      objParameter[cnt].etaWindowVetoUpper = etaWindowVetoUpper;

      objParameter[cnt].phiWindowLower     = phiWindowLower;
      objParameter[cnt].phiWindowUpper     = phiWindowUpper;
      objParameter[cnt].phiWindowVetoLower = phiWindowVetoLower;
      objParameter[cnt].phiWindowVetoUpper = phiWindowVetoUpper;

      
      // Output for debugging
      LogDebug("l1t|Global") 
	<< "\n      Muon PT high threshold (hex) for muon object " << cnt << " = "
	<< std::hex << objParameter[cnt].ptHighThreshold << std::dec
	<< "\n      etaWindow (hex) for muon object " << cnt << " = "
	<< std::hex << objParameter[cnt].etaRange << std::dec
	// << "\n      phiRange (hex) for muon object " << cnt << " = "
	// << std::hex << objParameter[cnt].phiRange << std::dec
	<< "\n      etaWindow Lower / Upper for muon object " << cnt << " = "
	<< objParameter[cnt].etaWindowLower << " / " << objParameter[cnt].etaWindowUpper
	<< "\n      etaWindowVeto Lower / Upper for muon object " << cnt << " = "
	<< objParameter[cnt].etaWindowVetoLower << " / " << objParameter[cnt].etaWindowVetoUpper
	<< "\n      phiWindow Lower / Upper for muon object " << cnt << " = "
	<< objParameter[cnt].phiWindowLower << " / " << objParameter[cnt].phiWindowUpper
	<< "\n      phiWindowVeto Lower / Upper for muon object " << cnt << " = "
	<< objParameter[cnt].phiWindowVetoLower << " / " << objParameter[cnt].phiWindowVetoUpper
	<< std::endl;

      cnt++;
    }



    // indicates if a correlation is used
    bool wscVal = (type == m_xmlConditionAttrType2wsc );

    if( wscVal ){

      xsd::cxx::tree::optional<l1t::DeltaRequirement> condRanges = condMu.deltaRequirement();
      LogDebug("l1t|Global") 
	<< "\t condRanges->deltaEtaRange().lower() = " << condRanges->deltaEtaRange().lower()
	<< "\n\t condRanges->deltaEtaRange().upper()   = " << condRanges->deltaEtaRange().upper()
	<< "\n\t condRanges->deltaPhiRange().lower() = " << condRanges->deltaPhiRange().lower()
	<< "\n\t condRanges->deltaPhiRange().upper() = " << condRanges->deltaPhiRange().upper() 
	<< std::endl;

      corrParameter.deltaEtaRangeLower = condRanges->deltaEtaRange().lower();
      corrParameter.deltaEtaRangeUpper = condRanges->deltaEtaRange().upper();

      corrParameter.deltaPhiRangeLower = condRanges->deltaPhiRange().lower();
      corrParameter.deltaPhiRangeUpper = condRanges->deltaPhiRange().upper();

      //
      /// Temporary
      //

      // Eta Range
      str_condMu = "0003";
      if ( !hexString2UInt128(str_condMu, tempUIntL, tempUIntH) ) {
	return false;
      }
      if( tempUIntH != 0 ){
        edm::LogError("TriggerMenuXmlParser") << "Too large hex-value!" << std::endl;
        return false;
      }
      corrParameter.deltaEtaRange = tempUIntL;

      // Phi Range
      str_condMu = "003";
      if ( !hexString2UInt128(str_condMu, tempUIntL, tempUIntH) ) {
	return false;
      }
      if( tempUIntH != 0 ){
        edm::LogError("TriggerMenuXmlParser") << "Too large hex-value!" << std::endl;
        return false;
      }
      corrParameter.deltaPhiRange = tempUIntL;



      // Max Phi Range
      std::string maxString = "3FF";

      unsigned int maxbits = 0;

      if ( !countConditionChildMaxBits(maxString, maxbits) ) {
	return false;
      }

      corrParameter.deltaPhiMaxbits = maxbits;
      LogTrace("TriggerMenuXmlParser")
        << "        deltaPhiMaxbits (dec) = " << maxbits
        << std::endl;
    }


    // get the type of the condition, as defined in enum, from the condition type
    // as defined in the XML file
    GtConditionType cType = getTypeFromType(type);
    //LogTrace("TriggerMenuXmlParser")
    //<< "      Condition type (enum value) = " << cType
    //<< std::endl;

    if (cType == l1t::TypeNull) {
        edm::LogError("TriggerMenuXmlParser")
            << "Type for muon condition id l1t::TypeNull - it means not defined in the XML file."
            << "\nNumber of trigger objects is set to zero. " << std::endl;
        return false;
    }

    // object types - all muons
    std::vector<L1GtObject> objType(nrObj, Mu);

    //////

    int relativeBx = l1t2int( condMu.relativeBx() );

    //////
    // now create a new CondMuonition

    MuonTemplate muonCond(name);

    muonCond.setCondType(cType);
    muonCond.setObjectType(objType);
    muonCond.setCondGEq(gEq);
    muonCond.setCondChipNr(chipNr);
    muonCond.setCondRelativeBx(relativeBx);

    muonCond.setConditionParameter(objParameter, corrParameter);

    if (edm::isDebugEnabled()) {
        std::ostringstream myCoutStream;
        muonCond.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;
    }

    // insert condition into the map and into muon template vector
    if ( !insertConditionIntoMap(muonCond, chipNr)) {
        edm::LogError("TriggerMenuXmlParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;
        return false;
    }
    else {
        if (corrFlag) {
            (m_corMuonTemplate[chipNr]).push_back(muonCond);
        }
        else {
            (m_vecMuonTemplate[chipNr]).push_back(muonCond);
        }

    }


    LogDebug("l1t|Global") 
      << "\n intGEq  = " << intGEq
      << " nrObj   = " << nrObj 
      << "\n ****************************************** " 
      << std::endl;

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

bool l1t::TriggerMenuXmlParser::parseCalo(l1t::CalorimeterCondition condCalo,
        unsigned int chipNr, const bool corrFlag) {

    XERCES_CPP_NAMESPACE_USE

    // get condition, particle name and type name

    std::string condition = "calo";
    std::string particle = l1t2string( condCalo.objectType() );
    std::string type = l1t2string( condCalo.type() );
    std::string name = l1t2string( condCalo.name() );

    LogDebug("l1t|Global")
      << "\n ****************************************** " 
      << "\n      DARRENS TEST OUTPUT (in parseCalo) " 
      << "\n condition = " << condition 
      << "\n particle  = " << particle 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;


    // determine object type type
    L1GtObject caloObjType;

    if (particle == m_xmlConditionAttrObjectNoIsoEG) {
        caloObjType = NoIsoEG;
    }
    else if (particle == m_xmlConditionAttrObjectIsoEG) {
        caloObjType = IsoEG;
    }
    else if (particle == m_xmlConditionAttrObjectCenJet) {
        caloObjType = CenJet;
    }
    else if (particle == m_xmlConditionAttrObjectTauJet) {
        caloObjType = TauJet;
    }
    else if (particle == m_xmlConditionAttrObjectForJet) {
        caloObjType = ForJet;
    }
    else {
        edm::LogError("TriggerMenuXmlParser") << "Wrong particle for calo-condition ("
            << particle << ")" << std::endl;
        return false;
    }

    if( type=="double_wsc" )  type = "2_wsc";
    else if( type=="single" ) type = "1_s";
    else if( type=="double" ) type = "2_s";
    else if( type=="triple" ) type = "3";
    else if( type=="quad"   ) type = "4";


    std::string str_etComparison = l1t2string( condCalo.comparison_operator() );

    int nrObj = getNumFromType(type);
    if (nrObj < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Unknown type for calo-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    // get greater equal flag
    int intGEq = ( str_etComparison=="ge" ) ? 1 : 0;
    if (intGEq < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Error getting \"greater or equal\" flag"
            << std::endl;
        return false;
    }
    // set the boolean value for the ge_eq mode
    bool gEq = (intGEq != 0);

    // get values

    // temporary storage of the parameters
    std::vector<CaloTemplate::ObjectParameter> objParameter(nrObj);
    CaloTemplate::CorrelationParameter corrParameter;

    // need at least one value for deltaPhiRange
    std::vector<boost::uint64_t> tmpValues((nrObj > 1) ? nrObj : 1);
    tmpValues.reserve( nrObj );

    if( int(condCalo.objectRequirements().objectRequirement().size())!=nrObj ){
      edm::LogError("TriggerMenuXmlParser") << " condCalo objects: nrObj = " << nrObj
						    << "condCalo.objectRequirements().objectRequirement().size() = " 
						    << condCalo.objectRequirements().objectRequirement().size()
						    << std::endl;
      return false;
    }


    std::string str_condCalo = "";
    boost::uint64_t tempUIntH, tempUIntL;
    boost::uint64_t dst;
    int cnt = 0;
    for( l1t::CalorimeterObjectRequirements::objectRequirement_const_iterator objPar = condCalo.objectRequirements().objectRequirement().begin();
	 objPar != condCalo.objectRequirements().objectRequirement().end(); ++objPar ){

      // ET Threshold
      str_condCalo = l1t2string( objPar->etThreshold() );
      if( !getXMLHexTextValue(str_condCalo, dst) ) return false;
      //if( cnt<nrObj ) objParameter[cnt].etThreshold = dst;
      /// DMP: Use dec instead of hex
      if( cnt<nrObj ) objParameter[cnt].etThreshold = objPar->etThreshold();

      // Eta Range
      //str_condCalo = "ffff";
      str_condCalo = "7f7f";
      //str_condCalo = "0f0f";
      if( !getXMLHexTextValue(str_condCalo, dst) ) return false;
      if( cnt<nrObj ) objParameter[cnt].etaRange = dst;

      // Phi Range
      str_condCalo = "3ffff";
      if( !getXMLHexTextValue(str_condCalo, dst) ) return false;
      if( cnt<nrObj ) objParameter[cnt].phiRange = dst;



      int cntEta=0;
      unsigned int etaWindowLower=-1, etaWindowUpper=-1, etaWindowVetoLower=-1, etaWindowVetoUpper=-1;
      // Temporary before translation
      for( l1t::CalorimeterObjectRequirement::etaWindow_const_iterator etaWindow =objPar->etaWindow().begin();
	   etaWindow != objPar->etaWindow().end(); ++etaWindow ){
	
	LogDebug("l1t|Global")
	  << "\n etaWindow lower = " << etaWindow->lower()
	  << "\n etaWindow upper = " << etaWindow->upper() 
	  << std::endl;
	if( cntEta==0 ){      etaWindowLower = etaWindow->lower(); etaWindowUpper = etaWindow->upper(); }
	else if( cntEta==1 ){ etaWindowVetoLower = etaWindow->lower(); etaWindowVetoUpper = etaWindow->upper(); }
	cntEta++;
      }

      int cntPhi=0;
      unsigned int phiWindowLower=-1, phiWindowUpper=-1, phiWindowVetoLower=-1, phiWindowVetoUpper=-1;
      for( l1t::CalorimeterObjectRequirement::phiWindow_const_iterator phiWindow =objPar->phiWindow().begin();
	   phiWindow != objPar->phiWindow().end(); ++phiWindow ){
 
	LogDebug("l1t|Global")
	  << "\n phiWindow begin = " << phiWindow->lower()
	  << "\n phiWindow end   = " << phiWindow->upper() 
	  << std::endl;

	if( cntPhi==0 ){      phiWindowLower = phiWindow->lower(); phiWindowUpper = phiWindow->upper(); }
	else if( cntPhi==1 ){ phiWindowVetoLower = phiWindow->lower(); phiWindowVetoUpper = phiWindow->upper(); }
	cntPhi++;
      }

      objParameter[cnt].etaWindowLower     = etaWindowLower;
      objParameter[cnt].etaWindowUpper     = etaWindowUpper;
      objParameter[cnt].etaWindowVetoLower = etaWindowVetoLower;
      objParameter[cnt].etaWindowVetoUpper = etaWindowVetoUpper;

      objParameter[cnt].phiWindowLower     = phiWindowLower;
      objParameter[cnt].phiWindowUpper     = phiWindowUpper;
      objParameter[cnt].phiWindowVetoLower = phiWindowVetoLower;
      objParameter[cnt].phiWindowVetoUpper = phiWindowVetoUpper;

      
      // Output for debugging
      LogDebug("l1t|Global") 
	<< "\n      Calo ET high threshold (hex) for calo object " << cnt << " = "
	<< std::hex << objParameter[cnt].etThreshold << std::dec
	<< "\n      etaWindow (hex) for calo object " << cnt << " = "
	<< std::hex << objParameter[cnt].etaRange << std::dec
	<< "\n      phiRange (hex) for calo object " << cnt << " = "
	<< std::hex << objParameter[cnt].phiRange << std::dec
	<< "\n      etaWindow Lower / Upper for calo object " << cnt << " = "
	<< objParameter[cnt].etaWindowLower << " / " << objParameter[cnt].etaWindowUpper
	<< "\n      etaWindowVeto Lower / Upper for calo object " << cnt << " = "
	<< objParameter[cnt].etaWindowVetoLower << " / " << objParameter[cnt].etaWindowVetoUpper
	<< "\n      phiWindow Lower / Upper for calo object " << cnt << " = "
	<< objParameter[cnt].phiWindowLower << " / " << objParameter[cnt].phiWindowUpper
	<< "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = "
	<< objParameter[cnt].phiWindowVetoLower << " / " << objParameter[cnt].phiWindowVetoUpper
	<< std::endl;

      cnt++;
    }



    // indicates if a correlation is used
    bool wscVal = (type == m_xmlConditionAttrType2wsc );

    if( wscVal ){

      xsd::cxx::tree::optional<l1t::DeltaRequirement> condRanges = condCalo.deltaRequirement();
      LogDebug("l1t|Global") 
	<< "\t condRanges->deltaEtaRange().lower() = " << condRanges->deltaEtaRange().lower()
	<< "\n\t condRanges->deltaEtaRange().upper()   = " << condRanges->deltaEtaRange().upper()
	<< "\n\t condRanges->deltaPhiRange().lower() = " << condRanges->deltaPhiRange().lower()
	<< "\n\t condRanges->deltaPhiRange().upper() = " << condRanges->deltaPhiRange().upper() 
	<< std::endl;

      corrParameter.deltaEtaRangeLower = condRanges->deltaEtaRange().lower();
      corrParameter.deltaEtaRangeUpper = condRanges->deltaEtaRange().upper();

      corrParameter.deltaPhiRangeLower = condRanges->deltaPhiRange().lower();
      corrParameter.deltaPhiRangeUpper = condRanges->deltaPhiRange().upper();

      //
      /// Temporary
      //

      // Eta Range
      str_condCalo = "0003";
      if ( !hexString2UInt128(str_condCalo, tempUIntL, tempUIntH) ) {
	return false;
      }
      if( tempUIntH != 0 ){
        edm::LogError("TriggerMenuXmlParser") << "Too large hex-value!" << std::endl;
        return false;
      }
      corrParameter.deltaEtaRange = tempUIntL;

      // Phi Range
      str_condCalo = "003";
      if ( !hexString2UInt128(str_condCalo, tempUIntL, tempUIntH) ) {
	return false;
      }
      if( tempUIntH != 0 ){
        edm::LogError("TriggerMenuXmlParser") << "Too large hex-value!" << std::endl;
        return false;
      }
      corrParameter.deltaPhiRange = tempUIntL;



      // Max Phi Range
      std::string maxString = "3FF";

      unsigned int maxbits = 0;

      if ( !countConditionChildMaxBits(maxString, maxbits) ) {
	return false;
      }

      corrParameter.deltaPhiMaxbits = maxbits;
      LogTrace("TriggerMenuXmlParser")
        << "        deltaPhiMaxbits (dec) = " << maxbits
        << std::endl;
    }



    // get the type of the condition, as defined in enum, from the condition type
    // as defined in the XML file
    GtConditionType cType = getTypeFromType(type);
    LogTrace("TriggerMenuXmlParser")
      << "      Condition type (enum value) = " << cType
      << std::endl;

    if (cType == l1t::TypeNull) {
        edm::LogError("TriggerMenuXmlParser")
            << "Type for calo condition id l1t::TypeNull - it means not defined in the XML file."
            << "\nNumber of trigger objects is set to zero. " << std::endl;
        return false;
    }

    // object types - all same caloObjType
    std::vector<L1GtObject> objType(nrObj, caloObjType);


    int relativeBx = l1t2int( condCalo.relativeBx() );

    // now create a new calo condition
    CaloTemplate caloCond(name);

    caloCond.setCondType(cType);
    caloCond.setObjectType(objType);
    caloCond.setCondGEq(gEq);
    caloCond.setCondChipNr(chipNr);
    caloCond.setCondRelativeBx(relativeBx);

    caloCond.setConditionParameter(objParameter, corrParameter);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        caloCond.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert condition into the map
    if ( !insertConditionIntoMap(caloCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;

        return false;
    }
    else {

        if (corrFlag) {
            (m_corCaloTemplate[chipNr]).push_back(caloCond);
       }
        else {
            (m_vecCaloTemplate[chipNr]).push_back(caloCond);
        }

    }

    LogDebug("l1t|Global") 
      << "\n intGEq  = " << intGEq
      << " nrObj   = " << nrObj 
      << "\n ****************************************** " 
      << std::endl;


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

bool l1t::TriggerMenuXmlParser::parseEnergySum(
        XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name,
        unsigned int chipNr, const bool corrFlag) {

    XERCES_CPP_NAMESPACE_USE

      /*
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    // determine object type type
    L1GtObject energySumObjType;
    GtConditionType cType;

    if ((particle == m_xmlConditionAttrObjectETM) && (type == m_xmlConditionAttrObjectETM)) {

        energySumObjType = ETM;
        cType = TypeETM;

    }
    else if ((particle == m_xmlConditionAttrObjectETT) && (type == m_xmlConditionAttrObjectETT)) {

        energySumObjType = ETT;
        cType = TypeETT;

    }
    else if ((particle == m_xmlConditionAttrObjectHTT) && (type == m_xmlConditionAttrObjectHTT)) {

        energySumObjType = HTT;
        cType = TypeHTT;

    }
    else if ((particle == m_xmlConditionAttrObjectHTM) && (type == m_xmlConditionAttrObjectHTM)) {

        energySumObjType = HTM;
        cType = TypeHTM;

    }
    else {
        edm::LogError("TriggerMenuXmlParser")
            << "Wrong particle or type for energy-sum condition (" << particle << ", " << type
            << ")" << std::endl;
        return false;
    }

    // global object
    int nrObj = 1;

    // get greater equal flag

    int intGEq = getGEqFlag(node, m_xmlTagEtThreshold);
    if (intGEq < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Error getting \"greater or equal\" flag"
            << std::endl;
        return false;
    }
    // set the boolean value for the ge_eq mode
    bool gEq = (intGEq != 0);

    // get values

    // temporary storage of the parameters
    std::vector<L1GtEnergySumTemplate::ObjectParameter> objParameter(nrObj);

    // need at least two values for phi
    std::vector<boost::uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);

    // get etThreshold values and fill into structure
    if ( !getConditionChildValuesOld(node, m_xmlTagEtThreshold, nrObj, tmpValues) ) {
        return false;
    }

    for (int i = 0; i < nrObj; i++) {
        objParameter[i].etThreshold = tmpValues[i];

        //LogTrace("TriggerMenuXmlParser")
        //<< "      EnergySum ET high threshold (hex) for energySum object " << i << " = "
        //<< std::hex << objParameter[i].etThreshold << std::dec
        //<< std::endl;

        // for ETM and HTM read phi value
        // phi is larger than 64 bits for ETM - it needs two 64bits words
        // phi is less than 64 bits for HTM   - it needs one 64bits word
        if (energySumObjType == ETM) {

            if (!getXMLHexTextValue128Old(
                    findXMLChild(node->getFirstChild(), m_xmlTagPhi), tmpValues[0], tmpValues[1])) {
                edm::LogError("TriggerMenuXmlParser")
                        << "    Could not get phi for ETM condition (" << name << ")" << std::endl;
                return false;
            }

            objParameter[i].phiRange0Word = tmpValues[0];
            objParameter[i].phiRange1Word = tmpValues[1];

        } else if (energySumObjType == HTM) {

            if (!getXMLHexTextValueOld(findXMLChild(node->getFirstChild(), m_xmlTagPhi), tmpValues[0])) {
                edm::LogError("TriggerMenuXmlParser")
                        << "    Could not get phi for HTM condition (" << name << ")" << std::endl;
                return false;
            }

            objParameter[i].phiRange0Word = tmpValues[0];

        }

        // get energyOverflow logical flag and fill into structure
        DOMNode* n1;
        if ( (n1 = findXMLChild(node->getFirstChild(), m_xmlTagEtThreshold)) == 0) {
            edm::LogError("TriggerMenuXmlParser")
                << "    Could not get energyOverflow for EnergySum condition (" << name << ")"
                << std::endl;
            return false;
        }
        if ( (n1 = findXMLChild(n1->getFirstChild(), m_xmlTagEnergyOverflow)) == 0) {
            edm::LogError("TriggerMenuXmlParser")
                << "    Could not get energyOverflow for EnergySum condition (" << name << ")"
                << std::endl;
            return false;
        }

        int tmpInt = getBitFromNode(n1);
        if (tmpInt == 0) {
            objParameter[i].energyOverflow = false;

            //LogTrace("TriggerMenuXmlParser")
            //<< "      EnergySum energyOverflow logical flag (hex) = "
            //<< std::hex << objParameter[i].energyOverflow << std::dec
            //<< std::endl;
        }
        else if (tmpInt == 1) {
            objParameter[i].energyOverflow = true;

            //LogTrace("TriggerMenuXmlParser")
            //<< "      EnergySum energyOverflow logical flag (hex) = "
            //<< std::hex << objParameter[i].energyOverflow << std::dec
            //<< std::endl;
        }
        else {
            LogTrace("TriggerMenuXmlParser")
                << "      EnergySum energyOverflow logical flag (hex) = " << std::hex << tmpInt
                << std::dec << " - wrong value! " << std::endl;
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

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        energySumCond.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert condition into the map
    if ( !insertConditionIntoMap(energySumCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;

        return false;
    }
    else {

        if (corrFlag) {
            (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);

        }
        else {
            (m_vecEnergySumTemplate[chipNr]).push_back(energySumCond);
        }

    }
      */

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

bool l1t::TriggerMenuXmlParser::parseJetCounts(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

      /*
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    if (particle != m_xmlConditionAttrObjectJetCounts) {
        edm::LogError("TriggerMenuXmlParser") << "Wrong particle for JetCounts condition ("
            << particle << ")" << std::endl;
        return false;
    }

    // object type and condition type
    L1GtObject jetCountsObjType = JetCounts;
    GtConditionType cType = TypeJetCounts;

    // global object
    int nrObj = 1;

    // get greater equal flag

    int intGEq = getGEqFlag(node, m_xmlTagCountThreshold);
    if (intGEq < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Error getting \"greater or equal\" flag"
            << std::endl;
        return false;
    }
    // set the boolean value for the ge_eq mode
    bool gEq = (intGEq != 0);

    // get values

    // temporary storage of the parameters
    std::vector<L1GtJetCountsTemplate::ObjectParameter> objParameter(nrObj);

    // get countIndex value and fill into structure
    // they are expressed in  base 10  (values: 0 - m_numberL1JetCounts)
    char* endPtr = const_cast<char*>(type.c_str());
    long int typeInt = strtol(type.c_str(), &endPtr, 10); // base = 10

    if (*endPtr != 0) {

        LogDebug("TriggerMenuXmlParser") << "Unable to convert " << type << " to dec."
            << std::endl;

        return false;
    }

    // test if count index is out of range
    if ((typeInt < 0) || (typeInt > static_cast<int>(m_numberL1JetCounts))) {
        LogDebug("TriggerMenuXmlParser") << "Count index " << typeInt
            << " outside range [0, " << m_numberL1JetCounts << "]" << std::endl;

        return false;
    }

    objParameter[0].countIndex = static_cast<unsigned int>(typeInt);

    // get count threshold values and fill into structure
    std::vector<boost::uint64_t> tmpValues(nrObj);

    if ( !getConditionChildValuesOld(node, m_xmlTagCountThreshold, nrObj, tmpValues) ) {
        return false;
    }

    for (int i = 0; i < nrObj; i++) {
        objParameter[i].countThreshold = tmpValues[i];

        //LogTrace("TriggerMenuXmlParser")
        //<< "      JetCounts count threshold (hex) for JetCounts object " << i << " = "
        //<< std::hex << objParameter[i].countThreshold << std::dec
        //<< std::endl;

        // TODO FIXME un-comment when tag available in XML file

        //        // get countOverflow logical flag and fill into structure
        //        DOMNode* n1;
        //        if ( (n1 = findXMLChild(node->getFirstChild(), m_xmlTagCountThreshold)) == 0) {
        //            edm::LogError("TriggerMenuXmlParser")
        //            << "    Could not get countOverflow for JetCounts condition ("
        //            << name << ")"
        //            << std::endl;
        //            return false;
        //        }
        //        if ( (n1 = findXMLChild(n1->getFirstChild(), m_xmlTagCountThreshold)) == 0) {
        //            edm::LogError("TriggerMenuXmlParser")
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
        //            LogTrace("TriggerMenuXmlParser")
        //            << "      JetCounts countOverflow logical flag (hex) = "
        //            << std::hex << objParameter[i].countOverflow << std::dec
        //            << std::endl;
        //        } else if (tmpInt == 1) {
        //            objParameter[i].countOverflow = true;
        //
        //            LogTrace("TriggerMenuXmlParser")
        //            << "      JetCounts countOverflow logical flag (hex) = "
        //            << std::hex << objParameter[i].countOverflow << std::dec
        //            << std::endl;
        //        } else {
        //            LogTrace("TriggerMenuXmlParser")
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

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        jetCountsCond.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert condition into the map
    if ( !insertConditionIntoMap(jetCountsCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser") << "    Error: duplicate condition (" << name
            << ")" << std::endl;

        return false;
    } else {

        (m_vecJetCountsTemplate[chipNr]).push_back(jetCountsCond);

    }

      */
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

bool l1t::TriggerMenuXmlParser::parseCastor(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

      /*
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    if (particle != m_xmlConditionAttrObjectCastor) {
        edm::LogError("TriggerMenuXmlParser")
            << "\nError: wrong particle for Castor condition ("
            << particle << ")" << std::endl;
        return false;
    }

    // object type and condition type
    // object type - irrelevant for CASTOR conditions
    GtConditionType cType = TypeCastor;

    // no objects for CASTOR conditions

    // set the boolean value for the ge_eq mode - irrelevant for CASTOR conditions
    bool gEq = false;

    // now create a new CASTOR condition

    L1GtCastorTemplate castorCond(name);

    castorCond.setCondType(cType);
    castorCond.setCondGEq(gEq);
    castorCond.setCondChipNr(chipNr);


    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        castorCond.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert condition into the map
    if ( !insertConditionIntoMap(castorCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser")
            << "    Error: duplicate condition (" << name
            << ")" << std::endl;

        return false;
    } else {

        (m_vecCastorTemplate[chipNr]).push_back(castorCond);

    }
      */

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

bool l1t::TriggerMenuXmlParser::parseHfBitCounts(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

      /*
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    if (particle != m_xmlConditionAttrObjectHfBitCounts) {
        edm::LogError("TriggerMenuXmlParser") << "Wrong particle for HfBitCounts condition ("
            << particle << ")" << std::endl;
        return false;
    }

    // object type and condition type
    L1GtObject hfBitCountsObjType = HfBitCounts;
    GtConditionType cType = TypeHfBitCounts;

    // global object
    int nrObj = 1;

    // get greater equal flag

    int intGEq = getGEqFlag(node, m_xmlTagCountThreshold);
    if (intGEq < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Error getting \"greater or equal\" flag"
            << std::endl;
        return false;
    }
    // set the boolean value for the ge_eq mode
    bool gEq = (intGEq != 0);

    // get values

    // temporary storage of the parameters
    std::vector<L1GtHfBitCountsTemplate::ObjectParameter> objParameter(nrObj);

    // get countIndex value and fill into structure
    // they are expressed in  base 10
    char* endPtr = const_cast<char*>(type.c_str());
    long int typeInt = strtol(type.c_str(), &endPtr, 10); // base = 10

    if (*endPtr != 0) {

        LogDebug("TriggerMenuXmlParser") << "Unable to convert " << type << " to dec."
            << std::endl;

        return false;
    }

    // test if count index is out of range FIXME introduce m_numberL1HfBitCounts?
    //if ((typeInt < 0) || (typeInt > static_cast<int>(m_numberL1HfBitCounts))) {
    //    LogDebug("TriggerMenuXmlParser") << "Count index " << typeInt
    //        << " outside range [0, " << m_numberL1HfBitCounts << "]" << std::endl;
    //
    //    return false;
    //}

    objParameter[0].countIndex = static_cast<unsigned int>(typeInt);

    // get count threshold values and fill into structure
    std::vector<boost::uint64_t> tmpValues(nrObj);

    if ( !getConditionChildValuesOld(node, m_xmlTagCountThreshold, nrObj, tmpValues) ) {
        return false;
    }

    for (int i = 0; i < nrObj; i++) {
        objParameter[i].countThreshold = tmpValues[i];

        //LogTrace("TriggerMenuXmlParser")
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

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        hfBitCountsCond.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert condition into the map
    if ( !insertConditionIntoMap(hfBitCountsCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser") << "    Error: duplicate condition (" << name
            << ")" << std::endl;

        return false;
    } else {

        (m_vecHfBitCountsTemplate[chipNr]).push_back(hfBitCountsCond);

    }

      */
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

bool l1t::TriggerMenuXmlParser::parseHfRingEtSums(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

      /*
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    if (particle != m_xmlConditionAttrObjectHfRingEtSums) {
        edm::LogError("TriggerMenuXmlParser") << "Wrong particle for HfRingEtSums condition ("
            << particle << ")" << std::endl;
        return false;
    }

    // object type and condition type
    L1GtObject hfRingEtSumsObjType = HfRingEtSums;
    GtConditionType cType = TypeHfRingEtSums;

    // global object
    int nrObj = 1;

    // get greater equal flag

    int intGEq = getGEqFlag(node, m_xmlTagEtThreshold);
    if (intGEq < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Error getting \"greater or equal\" flag"
            << std::endl;
        return false;
    }
    // set the boolean value for the ge_eq mode
    bool gEq = (intGEq != 0);

    // get values

    // temporary storage of the parameters
    std::vector<L1GtHfRingEtSumsTemplate::ObjectParameter> objParameter(nrObj);

    // get etSumIndex value and fill into structure
    // they are expressed in  base 10
    char* endPtr = const_cast<char*>(type.c_str());
    long int typeInt = strtol(type.c_str(), &endPtr, 10); // base = 10

    if (*endPtr != 0) {

        LogDebug("TriggerMenuXmlParser") << "Unable to convert " << type << " to dec."
            << std::endl;

        return false;
    }

    // test if ET sum index is out of range FIXME introduce m_numberL1HfRingEtSums?
    //if ((typeInt < 0) || (typeInt > static_cast<int>(m_numberL1HfRingEtSums))) {
    //    LogDebug("TriggerMenuXmlParser") << "Count index " << typeInt
    //        << " outside range [0, " << m_numberL1HfRingEtSums << "]" << std::endl;
    //
    //    return false;
    //}

    objParameter[0].etSumIndex = static_cast<unsigned int>(typeInt);

    // get ET sum threshold values and fill into structure
    std::vector<boost::uint64_t> tmpValues(nrObj);

    if ( !getConditionChildValuesOld(node, m_xmlTagEtThreshold, nrObj, tmpValues) ) {
        return false;
    }

    for (int i = 0; i < nrObj; i++) {
        objParameter[i].etSumThreshold = tmpValues[i];

        //LogTrace("TriggerMenuXmlParser")
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

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        hfRingEtSumsCond.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert condition into the map
    if ( !insertConditionIntoMap(hfRingEtSumsCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser") << "    Error: duplicate condition (" << name
            << ")" << std::endl;

        return false;
    } else {

        (m_vecHfRingEtSumsTemplate[chipNr]).push_back(hfRingEtSumsCond);

    }
      */

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

bool l1t::TriggerMenuXmlParser::parseBptx(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

      /*
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    if (particle != m_xmlConditionAttrObjectBptx) {
        edm::LogError("TriggerMenuXmlParser")
            << "\nError: wrong particle for Bptx condition ("
            << particle << ")" << std::endl;
        return false;
    }

    // object type and condition type
    // object type - irrelevant for BPTX conditions
    GtConditionType cType = TypeBptx;

    // no objects for BPTX conditions

    // set the boolean value for the ge_eq mode - irrelevant for BPTX conditions
    bool gEq = false;

    // now create a new BPTX condition

    L1GtBptxTemplate bptxCond(name);

    bptxCond.setCondType(cType);
    bptxCond.setCondGEq(gEq);
    bptxCond.setCondChipNr(chipNr);

    LogTrace("TriggerMenuXmlParser") << bptxCond << "\n" << std::endl;

    // insert condition into the map
    if ( !insertConditionIntoMap(bptxCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser")
            << "    Error: duplicate condition (" << name
            << ")" << std::endl;

        return false;
    } else {

        (m_vecBptxTemplate[chipNr]).push_back(bptxCond);

    }
      */

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

bool l1t::TriggerMenuXmlParser::parseExternal(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

      /*
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    if (particle != m_xmlConditionAttrObjectGtExternal) {
        edm::LogError("TriggerMenuXmlParser")
            << "\nError: wrong particle for External condition ("
            << particle << ")" << std::endl;
        return false;
    }

    // object type and condition type
    // object type - irrelevant for External conditions
    GtConditionType cType = TypeExternal;

    // no objects for External conditions

    // set the boolean value for the ge_eq mode - irrelevant for External conditions
    bool gEq = false;

    // now create a new External condition

    L1GtExternalTemplate externalCond(name);

    externalCond.setCondType(cType);
    externalCond.setCondGEq(gEq);
    externalCond.setCondChipNr(chipNr);

    LogTrace("TriggerMenuXmlParser") << externalCond << "\n" << std::endl;

    // insert condition into the map
    if ( !insertConditionIntoMap(externalCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser")
            << "    Error: duplicate condition (" << name
            << ")" << std::endl;

        return false;
    } else {

        (m_vecExternalTemplate[chipNr]).push_back(externalCond);

    }
      */

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

bool l1t::TriggerMenuXmlParser::parseCorrelation(
        XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name,
        unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

    // create a new correlation condition
    CorrelationTemplate correlationCond(name);

    // check that the condition does not exist already in the map
    if ( !insertConditionIntoMap(correlationCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser")
                << "    Error: duplicate correlation condition (" << name << ")"
                << std::endl;

        return false;
    }

    /*
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    std::cout << " ****************************************** " << std::endl;
    std::cout << "      DARRENS TEST OUTPUT (in parseCorrelation) " << std::endl;
    std::cout << " condition = " << condition << std::endl;
    std::cout << " particle  = " << particle << std::endl;
    std::cout << " type      = " << type << std::endl;
    std::cout << " name      = " << name << std::endl;

    LogTrace("TriggerMenuXmlParser") << "    Condition category: "
            << condition << ", particle: " << particle << ", type: " << type
            << "\n" << std::endl;

    // condition type
    GtConditionType cType = l1t::Type2cor;

    // two objects (for sure)
    const int nrObj = 2;

    // object types and greater equal flag - filled in the loop
    int intGEq[nrObj] = { -1, -1 };
    std::vector<L1GtObject> objType(nrObj);
    std::vector<GtConditionCategory> condCateg(nrObj);

    // correlation flag and index in the cor*vector
    const bool corrFlag = true;
    int corrIndexVal[nrObj] = { -1, -1 };

    // get the subconditions

    DOMNode* conditionsNode = node->getFirstChild();
    std::string conditionNameNodeName;
    conditionsNode = findXMLChild(conditionsNode, "", true,
            &conditionNameNodeName);


    for (int iSubCond = 0; iSubCond < nrObj; ++iSubCond) {

        // get for sub-condition:  category, object name and type name and condition name
        condition = getXMLAttribute(conditionsNode, m_xmlConditionAttrCondition);
        particle = getXMLAttribute(conditionsNode, m_xmlConditionAttrObject);
        type = getXMLAttribute(conditionsNode, m_xmlConditionAttrType);

        LogTrace("TriggerMenuXmlParser") << "    Sub-condition category: "
                << condition << ", particle: " << particle << ", type: "
                << type << ", name: " << conditionNameNodeName << "\n"
                << std::endl;

        // call the appropriate function for this condition
        if (condition == m_xmlConditionAttrConditionMuon) {
            if (!parseMuon(conditionsNode, conditionNameNodeName, chipNr,
                    corrFlag)) {
                edm::LogError("TriggerMenuXmlParser")
                        << "Error parsing sub-condition " << condition << ")"
                        << " with name " << conditionNameNodeName << std::endl;

            }

            // get greater equal flag
            intGEq[iSubCond] = getGEqFlag(conditionsNode,
                    m_xmlTagPtHighThreshold);
            if (intGEq[iSubCond] < 0) {
                edm::LogError("TriggerMenuXmlParser")
                        << "Error getting \"greater or equal\" flag"
                        << " for sub-condition " << conditionNameNodeName
                        << " for the correlation condition " << name
                        << std::endl;
                return false;
            }

            // set object type and sub-condition category
            objType[iSubCond] = Mu;
            condCateg[iSubCond] = CondMuon;
            corrIndexVal[iSubCond] = (m_corMuonTemplate[chipNr]).size() - 1;

        }
        else if (condition == m_xmlConditionAttrConditionCalo) {
            if (!parseCalo(conditionsNode, conditionNameNodeName, chipNr,
                    corrFlag)) {
                edm::LogError("TriggerMenuXmlParser")
                        << "Error parsing sub-condition " << condition << ")"
                        << " with name " << conditionNameNodeName << std::endl;

            }

            // get greater equal flag
            intGEq[iSubCond] = getGEqFlag(conditionsNode, m_xmlTagEtThreshold);
            if (intGEq[iSubCond] < 0) {
                edm::LogError("TriggerMenuXmlParser")
                        << "Error getting \"greater or equal\" flag"
                        << " for sub-condition " << conditionNameNodeName
                        << " for the correlation condition " << name
                        << std::endl;
                return false;
            }

            // set object type and sub-condition category
            if (particle == m_xmlConditionAttrObjectNoIsoEG) {
                objType[iSubCond] = NoIsoEG;
            }
            else if (particle == m_xmlConditionAttrObjectIsoEG) {
                objType[iSubCond] = IsoEG;
            }
            else if (particle == m_xmlConditionAttrObjectCenJet) {
                objType[iSubCond] = CenJet;
            }
            else if (particle == m_xmlConditionAttrObjectTauJet) {
                objType[iSubCond] = TauJet;
            }
            else if (particle == m_xmlConditionAttrObjectForJet) {
                objType[iSubCond] = ForJet;
            }
            else {
                edm::LogError("TriggerMenuXmlParser")
                        << "Wrong object type " << particle
                        << " for sub-condition " << conditionNameNodeName
                        << " from the correlation condition " << name
                        << std::endl;
                return false;
            }

            condCateg[iSubCond] = CondCalo;
            corrIndexVal[iSubCond] = (m_corCaloTemplate[chipNr]).size() - 1;

        }
        else if (condition == m_xmlConditionAttrConditionEnergySum) {
            if (!parseEnergySum(conditionsNode, conditionNameNodeName, chipNr,
                    corrFlag)) {
                edm::LogError("TriggerMenuXmlParser")
                        << "Error parsing sub-condition " << condition << ")"
                        << " with name " << conditionNameNodeName << std::endl;

            }

            // get greater equal flag
            intGEq[iSubCond] = getGEqFlag(conditionsNode, m_xmlTagEtThreshold);
            if (intGEq[iSubCond] < 0) {
                edm::LogError("TriggerMenuXmlParser")
                        << "Error getting \"greater or equal\" flag"
                        << " for sub-condition " << conditionNameNodeName
                        << " for the correlation condition " << name
                        << std::endl;
                return false;
            }

            // set object type and sub-condition category
            if (particle == m_xmlConditionAttrObjectETM) {
                objType[iSubCond] = ETM;
            }
            else if (particle == m_xmlConditionAttrObjectETT) {
                objType[iSubCond] = ETT;
            }
            else if (particle == m_xmlConditionAttrObjectHTT) {
                objType[iSubCond] = HTT;
            }
            else if (particle == m_xmlConditionAttrObjectHTM) {
                objType[iSubCond] = HTM;
            }
            else {
                edm::LogError("TriggerMenuXmlParser")
                        << "Wrong object type " << particle
                        << " for sub-condition " << conditionNameNodeName
                        << " from the correlation condition " << name
                        << std::endl;
                return false;
            }

            condCateg[iSubCond] = CondEnergySum;
            corrIndexVal[iSubCond] = (m_corEnergySumTemplate[chipNr]).size() - 1;

        }
        else {
            edm::LogError("TriggerMenuXmlParser")
                    << "Unknown or un-adequate sub-condition (" << condition
                    << ")" << " with name " << conditionNameNodeName
                    << " for the correlation condition " << name << std::endl;

            return false;
        }

        conditionsNode = findXMLChild(conditionsNode->getNextSibling(), "",
                true, &conditionNameNodeName);

    }

    // get greater equal flag for the correlation condition
    bool gEq = true;
    if (intGEq[0] != intGEq[1]) {
        edm::LogError("TriggerMenuXmlParser")
                << "Inconsistent GEq flags for sub-conditions (" << condition
                << ")" << " with name " << conditionNameNodeName
                << " for the correlation condition " << name << std::endl;
        return false;

    }
    else {
        gEq = (intGEq[0] != 0);

    }

    // correlation parameters

    // temporary storage of the parameters
    CorrelationTemplate::CorrelationParameter corrParameter;
    std::vector<boost::uint64_t> tmpValues(nrObj);

    // get deltaEtaRange
//    if ( !getConditionChildValuesOld(node, m_xmlTagDeltaEta, 1, tmpValues) ) {
//        return false;
//    }
//
//    corrParameter.deltaEtaRange = tmpValues[0];

    XERCES_CPP_NAMESPACE::DOMNode* node1 = findXMLChild(node->getFirstChild(),
            m_xmlTagDeltaEta);

    std::string valString;

    if (node1 == 0) {
        edm::LogError("TriggerMenuXmlParser")
                << "    Could not get deltaEta for correlation condition "
                << name << ". " << std::endl;
        return false;
    }
    else {
        valString = getXMLTextValue(node1);
    }

    corrParameter.deltaEtaRange = valString;

//    // deltaPhi is larger than 64bit
//    if ( !getXMLHexTextValue128Old(findXMLChild(node->getFirstChild(), m_xmlTagDeltaPhi),
//        tmpValues[0], tmpValues[1])) {
//        edm::LogError("TriggerMenuXmlParser")
//            << "    Could not get deltaPhi for correlation condition " << name << ". "
//            << std::endl;
//        return false;
//    }
//
//    corrParameter.deltaPhiRange = tmpValues[0];

   node1 = findXMLChild(node->getFirstChild(), m_xmlTagDeltaPhi);

    if (node1 == 0) {
        return false;
        edm::LogError("TriggerMenuXmlParser")
                << "    Could not get deltaPhi for correlation condition "
                << name << ". " << std::endl;
    }
    else {
        valString = getXMLTextValue(node1);
    }

    corrParameter.deltaPhiRange = valString;

    // get maximum number of bits for delta phi
    //LogTrace("TriggerMenuXmlParser")
    //<< "      Counting deltaPhiMaxbits"
    //<< std::endl;

    unsigned int maxbits;

    if ( !countConditionChildMaxBits(node, m_xmlTagDeltaPhi, maxbits) ) {
        return false;
    }

    corrParameter.deltaPhiMaxbits = maxbits;
    //LogTrace("TriggerMenuXmlParser")
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

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        correlationCond.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n"
                << std::endl;

    }

    // insert condition into the map
    // condition is not duplicate, check was done at the beginning

    (m_vecCorrelationTemplate[chipNr]).push_back(correlationCond);

    */
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
bool l1t::TriggerMenuXmlParser::parseId( l1t::Meta meta ) {

    XERCES_CPP_NAMESPACE_USE

//     DOMNode* doc = parser->getDocument();
//     DOMNode* n1 = doc->getFirstChild();

//     // we assume that the first child is m_xmlTagDef because it was checked in workXML

//     DOMNode* headerNode = n1->getFirstChild();
//     if (headerNode == 0) {
//         edm::LogError("TriggerMenuXmlParser") << "Error: No child of <" << m_xmlTagDef
//                 << "> tag found." << std::endl;
//         return false;
//     }

//     headerNode = findXMLChild(headerNode, m_xmlTagHeader);
//     if (headerNode == 0) {

//         LogDebug("TriggerMenuXmlParser") << "\n  Warning: Could not find <" << m_xmlTagHeader
//                 << "> tag" << "\n   - No header information." << std::endl;

//     } else {

//         DOMNode* idNode = headerNode->getFirstChild();

//         // find menu interface name
//         idNode = findXMLChild(idNode, m_xmlTagMenuInterface);
//         if (idNode == 0) {

//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagMenuInterface << "> tag"
//                     << "\n   - Trigger menu interface name derived from file name." << std::endl;

//             // set the name of the trigger menu interface: from beginning of file names
//             // until beginning of "_L1T_Scales"
//             size_t xmlPos = m_triggerMenuName.find("_L1T_Scales", 0);
//             if (xmlPos == std::string::npos) {
//                 LogTrace("TriggerMenuXmlParser")
//                         << "\n  Warning: Could not find \"_L1T_Scales\" " << "string in file name"
//                         << "\n   - Trigger menu interface name set to file name." << std::endl;
//                 m_triggerMenuInterface = m_triggerMenuName;

//             } else {
//                 m_triggerMenuInterface = m_triggerMenuName;
//                 m_triggerMenuInterface.erase(
//                         m_triggerMenuInterface.begin(), m_triggerMenuInterface.begin() + xmlPos);
//             }

//         } else {
//             m_triggerMenuInterface = getXMLTextValue(idNode);
//         }

//         // find menu interface creation date
//         idNode = headerNode->getFirstChild();
//         idNode = findXMLChild(idNode, m_xmlTagMenuInterfaceDate);

//         if (idNode == 0) {

//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagMenuInterfaceDate << "> tag" << "\n   - No creation date."
//                     << m_triggerMenuInterfaceDate << std::endl;

//         } else {

//             m_triggerMenuInterfaceDate = getXMLTextValue(idNode);
//         }

//         // find menu interface creation author
//         idNode = headerNode->getFirstChild();
//         idNode = findXMLChild(idNode, m_xmlTagMenuInterfaceAuthor);

//         if (idNode == 0) {

//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagMenuInterfaceAuthor << "> tag" << "\n   - No creation author."
//                     << m_triggerMenuInterfaceAuthor << std::endl;

//         } else {

//             m_triggerMenuInterfaceAuthor = getXMLTextValue(idNode);
//         }

//         // find menu interface description
//         idNode = headerNode->getFirstChild();
//         idNode = findXMLChild(idNode, m_xmlTagMenuInterfaceDescription);

//         if (idNode == 0) {

//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagMenuInterfaceDescription << "> tag" << "\n   - No description."
//                     << m_triggerMenuInterfaceDescription << std::endl;

//         } else {

//             m_triggerMenuInterfaceDescription = getXMLTextValue(idNode);
//         }

//         // find menu creation date
//         idNode = headerNode->getFirstChild();
//         idNode = findXMLChild(idNode, m_xmlTagMenuDate);

//         if (idNode == 0) {

//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagMenuDate << "> tag" << "\n   - No creation date."
//                     << m_triggerMenuDate << std::endl;

//         } else {

//             m_triggerMenuDate = getXMLTextValue(idNode);
//         }

//         // find menu creation author
//         idNode = headerNode->getFirstChild();
//         idNode = findXMLChild(idNode, m_xmlTagMenuAuthor);

//         if (idNode == 0) {

//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagMenuAuthor << "> tag" << "\n   - No creation author."
//                     << m_triggerMenuAuthor << std::endl;

//         } else {

//             m_triggerMenuAuthor = getXMLTextValue(idNode);
//         }

//         // find menu description
//         idNode = headerNode->getFirstChild();
//         idNode = findXMLChild(idNode, m_xmlTagMenuDescription);

//         if (idNode == 0) {

//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagMenuDescription << "> tag" << "\n   - No description."
//                     << m_triggerMenuDescription << std::endl;

//         } else {

//             m_triggerMenuDescription = getXMLTextValue(idNode);
//         }

//         // find algorithm implementation tag

//         idNode = headerNode->getFirstChild();

//         idNode = findXMLChild(idNode, m_xmlTagMenuAlgImpl);
//         if (idNode == 0) {

//             m_algorithmImplementation = "";
//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagMenuAlgImpl << "> tag"
//                     << "\n   - Algorithm implementation tag set to empty string." << std::endl;

//         } else {

//             m_algorithmImplementation = getXMLTextValue(idNode);
//         }

//         // find DB key for L1 scales

//         idNode = headerNode->getFirstChild();

//         idNode = findXMLChild(idNode, m_xmlTagScaleDbKey);
//         if (idNode == 0) {

//             m_scaleDbKey = "NULL";
//             LogTrace("TriggerMenuXmlParser") << "\n  Warning: Could not find <"
//                     << m_xmlTagScaleDbKey << "> tag" << "\n   - Scale key set to " << m_scaleDbKey
//                     << " string." << std::endl;

//         } else {
//             m_scaleDbKey = getXMLTextValue(idNode);
//         }

//     }



    m_triggerMenuInterface = l1t2string( meta.name() );
    m_triggerMenuInterfaceDate = "2013-010-24T15:33:24";
    m_triggerMenuInterfaceAuthor = "Darren Puigh";
    m_triggerMenuInterfaceDescription = l1t2string( meta.comment() );
//     m_algorithmImplementation = l1t2string( meta.firmwareVersion() );
//     m_triggerMenuDate = l1t2string( meta.changesDate() );
//     m_triggerMenuAuthor = l1t2string( meta.changesAuthor() );
    m_triggerMenuDescription = l1t2string( meta.comment() );
    m_scaleDbKey = l1t2string( meta.scale_set() );


    int cnt = 0;
    for( l1t::RevisionList::revision_const_iterator revision = meta.revisions().revision().begin();
	 revision != meta.revisions().revision().end(); ++revision ){

      LogDebug("l1t|Global")
	<< "\t Revision " << cnt
	<< "\t\t author = " << l1t2string( revision->author() )
	<< "\t\t datetime = " << l1tDateTime2string( revision->datetime() ) 
	<< std::endl;

      if( cnt==0 ){
	m_triggerMenuDate = l1tDateTime2string( revision->datetime() );
	m_triggerMenuAuthor = l1t2string( revision->author() );
      }
      cnt++;
    }

    //LogDebug("TriggerMenuXmlParser")
    LogDebug("l1t|Global")
      << "\n  Parsed values from XML file DRULES"
      << "\nL1 MenuInterface:                   " << m_triggerMenuInterface
      << "\nL1 MenuInterface - Creation date:   " << m_triggerMenuInterfaceDate
      << "\nL1 MenuInterface - Creation author: " << m_triggerMenuInterfaceAuthor
      << "\nL1 MenuInterface - Description:     " << m_triggerMenuInterfaceDescription
      << "\n"
      << "\nAlgorithm implementation tag:       " << m_algorithmImplementation
      << "\n"
      << "\nL1 Menu - Creation date:            " << m_triggerMenuDate
      << "\nL1 Menu - Creation author:          " << m_triggerMenuAuthor
      << "\nL1 Menu - Description:              " << m_triggerMenuDescription
      << std::endl;


    // set the trigger menu name
    // format:
    // L1MenuInterface/ScaleDbKey/AlgorithmImplementationTag

    std::string menuName = m_triggerMenuInterface + "/" + m_scaleDbKey + "/" + m_algorithmImplementation;

    if (menuName != m_triggerMenuName) {

        LogDebug("TriggerMenuXmlParser") << "\n  Warning: Inconsistent L1 menu name:"
                << "\n    from XML file name: " << m_triggerMenuName
                << "\n    from XML tag:       " << menuName << std::endl;

        if (m_triggerMenuInterface != "") {
            if (m_scaleDbKey == "NULL") {
                m_triggerMenuName = m_triggerMenuInterface;
            } else {
                m_triggerMenuName = menuName;
            }

            LogTrace("TriggerMenuXmlParser") << "\n  L1 menu name set to value from XML tag!"
                    << "\n  L1 Menu name: " << m_triggerMenuName << std::endl;

        } else {
            LogTrace("TriggerMenuXmlParser") << "\n  L1 menu name set to file name!"
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

bool l1t::TriggerMenuXmlParser::workCondition(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, m_xmlConditionAttrCondition);
    std::string particle = getXMLAttribute(node, m_xmlConditionAttrObject);
    std::string type = getXMLAttribute(node, m_xmlConditionAttrType);

    LogDebug("l1t|Global")
      << "\n ****************************************** "
      << "\n      workCondition "
      << "\n condition = " << condition
      << "\n particle  = " << particle
      << "\n type      = " << type
      << "\n name      = " << name 
      << std::endl;

    if (condition.empty() || particle.empty() || type.empty() ) {

        edm::LogError("TriggerMenuXmlParser") << "Missing attributes for condition " << name
            << std::endl;

        return false;
    }

    //LogTrace("TriggerMenuXmlParser")
    //<< "    condition: " << condition << ", particle: " << particle
    //<< ", type: " << type << std::endl;

    // call the appropiate function for this condition

    /*
    if (condition == m_xmlConditionAttrConditionMuon) {
        return parseMuon(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionCalo) {
        return parseCalo(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionEnergySum) {
        return parseEnergySum(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionJetCounts) {
        return parseJetCounts(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionCastor) {
        return parseCastor(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionHfBitCounts) {
        return parseHfBitCounts(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionHfRingEtSums) {
        return parseHfRingEtSums(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionBptx) {
        return parseBptx(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionExternal) {
        return parseExternal(node, name, chipNr);
    }
    else if (condition == m_xmlConditionAttrConditionCorrelation) {
        return parseCorrelation(node, name, chipNr);
    }
    else {
        edm::LogError("TriggerMenuXmlParser")
            << "\n Error: unknown condition (" << condition << ")"
            << std::endl;

        return false;
    }

    */
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

bool l1t::TriggerMenuXmlParser::parseConditions( l1t::ConditionList conditions ){

    XERCES_CPP_NAMESPACE_USE

    LogTrace("TriggerMenuXmlParser") << "\nParsing conditions" << std::endl;

    int chipNr = 1;
    LogDebug("l1t|Global") << " ====> condCalorimeter" << std::endl;
    for (l1t::ConditionList::condCalorimeter_const_iterator condCalo = conditions.condCalorimeter().begin();
	 condCalo != conditions.condCalorimeter().end(); ++condCalo ){

      LogDebug("l1t|Global")
	<< condCalo->name()  << " {"                    
	<< "  comment: " << condCalo->comment()
	<< "  locked: "      << condCalo->locked()     
	<< "}" 
	<< std::endl;

      l1t::CalorimeterCondition condition = (*condCalo);

      parseCalo( condition, chipNr );
    }

    LogDebug("l1t|Global")  << " ====> condMuon " << std::endl;
    for (l1t::ConditionList::condMuon_const_iterator condMu = conditions.condMuon().begin();
	 condMu != conditions.condMuon().end(); ++condMu ){

      LogDebug("l1t|Global")
	<< condMu->name()  << " {"                    
	<< "  comment: " << condMu->comment()
	<< "  locked: "      << condMu->locked()     
	<< "}" 
	<< std::endl;

      l1t::MuonCondition condition = (*condMu);

      parseMuon( condition, chipNr );
    }


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

bool l1t::TriggerMenuXmlParser::workAlgorithm( l1t::Algorithm algorithm,
    unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE

//     if (node == 0) {
//         LogDebug("TriggerMenuXmlParser")
//         << "    Node is 0 in " << __PRETTY_FUNCTION__
//         << " can not parse the algorithm " << algName
//         << std::endl;
//         return false;
//     }

    // get alias
    std::string algAlias = l1t2string( algorithm.name() );
    std::string algName  = l1t2string( algorithm.name() );

    if (algAlias == "") {
        algAlias = algName;
        LogDebug("TriggerMenuXmlParser")
                << "\n    No alias defined for algorithm. Alias set to algorithm name."
                << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
                << std::endl;
    } else {
      //LogDebug("TriggerMenuXmlParser") 
      LogDebug("l1t|Global")  << "\n    Alias defined for algorithm."
			      << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
			      << std::endl;
    }

    // get the logical expression
    std::string logExpression = l1t2string( algorithm.logical_expression() );

    LogDebug("l1t|Global")
      << "      Logical expression: " << logExpression
      << "      Chip number:        " << chipNr
      << std::endl;

    // determine output pin
    std::string pinString = l1t2string( algorithm.index() );
    int outputPin = 0;

    std::istringstream opStream(pinString);

    if ((opStream >> outputPin).fail()) {
      LogDebug("TriggerMenuXmlParser")
	<< "    Unable to convert pin string " << pinString
	<< " to int for algorithm : " << algName
	<< std::endl;
      
      return false;
    }


    //LogTrace("TriggerMenuXmlParser")
    LogDebug("l1t|Global")  << "      Output pin:         " << outputPin
			    << std::endl;


    // compute the bit number from chip number, output pin and order of the chips
    // pin numbering start with 1, bit numbers with 0
    int bitNumber = outputPin;// + (m_orderConditionChip[chipNr] -1)*m_pinsOnConditionChip -1;

    //LogTrace("TriggerMenuXmlParser")
    LogDebug("l1t|Global")  << "      Bit number:         " << bitNumber
			    << std::endl;

    // create a new algorithm and insert it into algorithm map
    L1GtAlgorithm alg(algName, logExpression, bitNumber);
    alg.setAlgoChipNumber(static_cast<int>(chipNr));
    alg.setAlgoAlias(algAlias);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        alg.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert algorithm into the map
    if ( !insertAlgorithmIntoMap(alg)) {

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

bool l1t::TriggerMenuXmlParser::parseAlgorithms( l1t::AlgorithmList algorithms ) {

    XERCES_CPP_NAMESPACE_USE

    LogTrace("TriggerMenuXmlParser") << "\nParsing algorithms" << std::endl;

    int chipNr = 1;
    LogDebug("l1t|Global")  << " ====> algorithms " << std::endl;
    for( l1t::AlgorithmList::algorithm_const_iterator i = algorithms.algorithm().begin();
         i != algorithms.algorithm().end(); ++i ){

      l1t::Algorithm algorithm = (*i);
      LogDebug("l1t|Global") 
	<< algorithm.name()  << " {"                    
	<< "  index: "       << algorithm.index()       
	<< "  equation: "    << algorithm.logical_expression()    
	<< "  comment: "     << algorithm.comment() 
	<< "  locked: "      << algorithm.locked()      
	<< "}" 
	<< std::endl;


      workAlgorithm( algorithm, chipNr );
    }


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

bool l1t::TriggerMenuXmlParser::workTechTrigger(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& algName) {

    XERCES_CPP_NAMESPACE_USE

    if (node == 0) {
        LogDebug("TriggerMenuXmlParser")
        << "    Node is 0 in " << __PRETTY_FUNCTION__
        << " can not parse the technical trigger " << algName
        << std::endl;
        return false;
    }

    // get the logical expression from the node
    std::string logExpression = getXMLTextValue(node);

    //LogTrace("TriggerMenuXmlParser")
    //<< "      Logical expression: " << logExpression
    //<< std::endl;

    // determine bit number (use output pin tag)
    DOMNode* pinNode = findXMLChild(node->getFirstChild(), m_xmlTagOutput);
    std::string pinString;
    int outputPin = 0;

    pinNode = node->getFirstChild();
    if ( (pinNode = findXMLChild(pinNode, m_xmlTagOutputPin) ) != 0) {
        pinString = getXMLAttribute(pinNode, m_xmlAttrNr);

        // convert pinString to integer
        std::istringstream opStream(pinString);

        if ((opStream >> outputPin).fail()) {
            LogDebug("TriggerMenuXmlParser")
                    << "    Unable to convert pin string " << pinString
                    << " to int for technical trigger : " << algName
                    << std::endl;

            return false;
        }

    }

    if (pinNode == 0) {
        LogTrace("TriggerMenuXmlParser")
            << "    Warning: No pin number found for technical trigger: "
            << algName << std::endl;

        return false;
    }

    // set the bit number
    int bitNumber = outputPin;

    //LogTrace("TriggerMenuXmlParser")
    //<< "      Bit number:         " << bitNumber
    //<< std::endl;

    // create a new technical trigger and insert it into technical trigger map
    // alias set automatically to name
    L1GtAlgorithm alg(algName, logExpression, bitNumber);
    alg.setAlgoAlias(algName);

    if (edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        alg.print(myCoutStream);
        LogTrace("TriggerMenuXmlParser") << myCoutStream.str() << "\n" << std::endl;

    }

    // insert technical trigger into the map
    if ( !insertTechTriggerIntoMap(alg)) {

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

bool l1t::TriggerMenuXmlParser::parseTechTriggers(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {

    XERCES_CPP_NAMESPACE_USE

    //LogTrace("TriggerMenuXmlParser") << "\nParsing technical triggers" << std::endl;

    DOMNode* doc = parser->getDocument();
    DOMNode* node = doc->getFirstChild();

    DOMNode* algNode = node->getFirstChild();
    if (algNode == 0) {
        edm::LogError("TriggerMenuXmlParser")
                << "  Error: No child found for " << m_xmlTagDef << std::endl;
        return false;
    }

    algNode = findXMLChild(algNode, m_xmlTagTechTriggers);
    if (algNode == 0) {
        edm::LogError("TriggerMenuXmlParser") << "    Error: No <"
                << m_xmlTagTechTriggers << "> child found."
                << std::endl;
        return false;
    }

    // walk through technical triggers
    DOMNode* algNameNode = algNode->getFirstChild();
    std::string algNameNodeName;
    algNameNode = findXMLChild(algNameNode, "", true, &algNameNodeName);

    while (algNameNode != 0) {
        //LogTrace("TriggerMenuXmlParser")
        //<< "    Found an technical trigger with name: " << algNameNodeName
        //<< std::endl;

        if ( !workTechTrigger(algNameNode, algNameNodeName)) {
            return false;
        }

        algNameNode = findXMLChild(algNameNode->getNextSibling(), "", true,
                &algNameNodeName);

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

//bool TriggerMenuXmlParser::workXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser) {
bool l1t::TriggerMenuXmlParser::workXML( std::auto_ptr<l1t::L1TriggerMenu> tm ) {

    XERCES_CPP_NAMESPACE_USE


    // clear possible old maps
    clearMaps();

    l1t::Meta meta = tm->meta();
    l1t::ConditionList conditions = tm->conditions();
    l1t::AlgorithmList algorithms = tm->algorithms();

    if ( !parseId( meta ) ) {
      clearMaps();
      return false;
    }

    if ( !parseConditions( conditions ) ) {
        clearMaps();
        return false;
    }

    if ( !parseAlgorithms( algorithms ) ) {
        clearMaps();
        return false;
    }

//     if ( !parseTechTriggers(parser) ) {
//         clearMaps();
//         return false;
//     }

    return true;

}


// static class members

