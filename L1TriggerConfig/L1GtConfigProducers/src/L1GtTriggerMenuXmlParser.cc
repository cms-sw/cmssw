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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuXmlParser.h"

// system include files
#include <string>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/cstdint.hpp>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtTriggerMenuXmlParser::L1GtTriggerMenuXmlParser()
{
    // error handler for xml-parser
    m_xmlErrHandler = 0;


}

// destructor
L1GtTriggerMenuXmlParser::~L1GtTriggerMenuXmlParser()
{
    // empty
}


/**
 * initXML - Initialize XML-utilities and try to create a parser for the specified file.
 *
 * @param xmlFile Filename of the XML-File
 *
 * @return A reference to a XercesDOMParser object if suceeded. 0 if an error occured.
 *
 */

XERCES_CPP_NAMESPACE::XercesDOMParser* L1GtTriggerMenuXmlParser::initXML(
    const std::string &xmlFile)
{

    XERCES_CPP_NAMESPACE_USE

    // try to initialize
    try {
        XMLPlatformUtils::Initialize();
    } catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());

        edm::LogError("L1GtTriggerMenuXmlParser")
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
    } else {
        // TODO ASSERTION
    }
    parser->setErrorHandler(m_xmlErrHandler);

    // try to parse the file
    try {
        parser->parse(xmlFile.c_str());
    } catch(const XMLException &toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());

        edm::LogError("L1GtTriggerMenuXmlParser")
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

        edm::LogError("L1GtTriggerMenuXmlParser")
        << "DOM-Exception while parsing XML: \n"
        << message << std::endl;

        XMLString::release(&message);
        delete parser;
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
        return 0;
    }
    catch (...) {

        edm::LogError("L1GtTriggerMenuXmlParser")
        << "Unexpected Exception while parsing XML!"
        << std::endl;

        delete parser;
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
        return 0;
    }

    return parser;
}


/// find a named child of a xml node
XERCES_CPP_NAMESPACE::DOMNode* L1GtTriggerMenuXmlParser::findXMLChild(
    XERCES_CPP_NAMESPACE::DOMNode* startChild, const std::string& tagName,
    bool beginOnly = false, std::string* rest = 0)
{

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
        } else {
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
                *rest = std::string(nodeName).substr(
                            tagName.length(), strlen(nodeName) - tagName.length());
            }
        }
    } else { // empty string given
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
 * @return The value of the attribute or empty string if an error occured.
 */

std::string L1GtTriggerMenuXmlParser::getXMLAttribute(
    const XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name)
{

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

std::string L1GtTriggerMenuXmlParser::getXMLTextValue(XERCES_CPP_NAMESPACE::DOMNode* node)
{

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
 * @return true if conversion suceeded, false if an error occured.
 */


bool L1GtTriggerMenuXmlParser::hexString2UInt128(const std::string& hexString,
        boost::uint64_t& dstL, boost::uint64_t& dstH)
{

    // string to determine start of hex value, do not ignore leading zeros
    static const std::string valid_hex_start("0123456789ABCDEFabcdef");

    // string to determine end of hex value
    static const std::string valid_hex_end("0123456789ABCDEFabcdef");

    std::string tempStr = hexString;

    // start / end position of the hex value in the string
    unsigned int hexStart = tempStr.find_first_of(valid_hex_start);
    unsigned int hexEnd = tempStr.find_first_not_of(valid_hex_end, hexStart);

    if (hexStart == hexEnd) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "No hex value found in: " << tempStr << std::endl;

        return false;
    }

    tempStr = tempStr.substr(hexStart, hexEnd - hexStart);

    if ( tempStr.empty() ) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "Empty value in " << __PRETTY_FUNCTION__ << std::endl;

        return false;
    }

    // split the string
    std::string tempStrH, tempStrL;

    if (tempStr.length() > 16) { // more than 64 bit
        tempStrL = tempStr.substr(tempStr.length()-16, 16);
        tempStrH = tempStr.substr(0,tempStr.length()-16);
    } else {
        tempStrL = tempStr;
        tempStrH = "0";
    }

    // convert lower 64bit
    char* endPtr = (char*) tempStrL.c_str();
    if (*endPtr != 0) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "Unable to convert " << tempStr << " to hex." << std::endl;

        return false;
    }

    boost::uint64_t tempUIntL = strtoull(tempStrL.c_str(), &endPtr, 16);

    // convert higher64 bit
    endPtr = (char*) tempStrH.c_str();
    if (*endPtr != 0) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "Unable to convert " << tempStr << " to hex." << std::endl;

        return false;
    }

    boost::uint64_t tempUIntH = strtoull(tempStrH.c_str(), &endPtr, 16);

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
        boost::uint64_t& dstL, boost::uint64_t& dstH)
{

    if (node == 0) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "node == 0 in " << __PRETTY_FUNCTION__
        << std::endl;

        return false;
    }

    boost::uint64_t tempUIntH, tempUIntL;

    std::string tempStr = getXMLTextValue(node);
    if ( ! hexString2UInt128(tempStr, tempUIntL, tempUIntH) ) {
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
 * @return true if suceeded, false if an error occured
 *
 */

bool L1GtTriggerMenuXmlParser::getXMLHexTextValue(
    XERCES_CPP_NAMESPACE::DOMNode* node, boost::uint64_t& dst)
{

    boost::uint64_t dummyH; // dummy for eventual higher 64bit
    boost::uint64_t tempUInt; // temporary unsigned integer

    if ( ! getXMLHexTextValue128(node, tempUInt, dummyH) ) {
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
        const std::string& childName, unsigned int& dst)
{

    XERCES_CPP_NAMESPACE_USE

    // should never happen...
    if (node == 0) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "node == 0 in " << __PRETTY_FUNCTION__ << std::endl;

        return false;
    }

    DOMNode* n1 = findXMLChild(node->getFirstChild(), childName);

    if (n1 == 0) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "Child of condition not found ( " << childName << ")"
        << std::endl;

        return false;
    }

    DOMNode* n2 = findXMLChild(n1->getFirstChild(), xml_value_tag);

    if (n2 == 0) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "No value tag found for child " << childName << " in " << __PRETTY_FUNCTION__
        << std::endl;

        return false;
    }

    // first try direct
    std::string maxString = getXMLAttribute(n1, xml_attr_max); // string for the maxbits

    if ( maxString.empty() ) {
        maxString = getXMLAttribute(n2, xml_attr_max); // try next value tag
        // if no max was found neither in value nor in the childName tag
        if ( maxString.empty() ) {

            LogTrace("L1GtTriggerMenuXmlParser")
            << "No Max value found for " << childName
            << std::endl;

            return false;
        }
    }

    // do the hex conversion

    boost::uint64_t maxBitsL, maxBitsH;
    if ( ! hexString2UInt128(maxString, maxBitsL, maxBitsH) ) {
        return false;
    }

    // count the bits
    LogTrace("L1GtTriggerMenuXmlParser")
    << std::dec
    << "        words: dec: high (MSB) word = " << maxBitsH << " low word = " << maxBitsL
    << std::hex << "\n"
    << "        words: hex: high (MSB) word = " << maxBitsH << " low word = " << maxBitsL
    << std::dec
    << std::endl;

    unsigned int counter = 0;

    while (maxBitsL != 0) {
        // check if bits set countinously
        if ( (maxBitsL & 1) == 0 ) {

            edm::LogError("L1GtTriggerMenuXmlParser")
            << "      Confused by not continous set bits for max value "
            << maxString << "(child=" << childName << ")"
            << std::endl;

            return false;
        }

        maxBitsL >>= 1;
        counter++;
    }

    if ( (maxBitsH != 0) && (counter != 64) ) {

        edm::LogError("L1GtTriggerMenuXmlParser")
        << "      Confused by not continous set bits for max value "
        << maxString << "(child=" << childName << ")"
        << std::endl;

        return false;
    }


    while (maxBitsH != 0) {
        //check if bits set countinously
        if ( (maxBitsH & 1) == 0 ) {

            edm::LogError("L1GtTriggerMenuXmlParser")
            << "      Confused by not continous set bits for max value "
            << maxString << "(child=" << childName << ")"
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
 * @param dst A pointer to a array of boost::uint64_t where the results are written.
 *
 * @return true if suceeded. false if an error occured or not enough values found. 
 */

bool L1GtTriggerMenuXmlParser::getConditionChildValues(XERCES_CPP_NAMESPACE::DOMNode* node,
        const std::string& childName,
        unsigned int num, boost::uint64_t* dst)
{

    XERCES_CPP_NAMESPACE_USE

    if (node == 0) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "node == 0 in " << __PRETTY_FUNCTION__
        << std::endl;

        return false;
    }

    DOMNode* n1 = findXMLChild(node->getFirstChild(), childName);

    // if child not found
    if (n1 == 0) {

        LogTrace("L1GtTriggerMenuXmlParser")
        << "Child of condition not found ( " << childName << ")"
        << std::endl;

        return false;
    }

    // no values are sucessfull
    if (num == 0) {
        return true;
    }

    n1 = findXMLChild(n1->getFirstChild(), xml_value_tag);
    for (unsigned int i = 0; i < num; i++) {
        if (n1 == 0) {

            LogTrace("L1GtTriggerMenuXmlParser")
            << "Not enough values in condition child ( " << childName << ")"
            << std::endl;

            return false; // too less values
        }

        if ( ! getXMLHexTextValue(n1, dst[i]) ) {

            edm::LogError("L1GtTriggerMenuXmlParser")
            << "Error converting condition child ( " << childName << ") value."
            << std::endl;

            return false;
        }

        n1 = findXMLChild(n1->getNextSibling(), xml_value_tag); // next child
    }

    return true;
}


/**
 * cleanupXML - Delete parser and error handler. Shutdown XMLPlatformUtils.
 *
 * @param parser A reference to the parser to be deleted.
 *
 */

void L1GtTriggerMenuXmlParser::cleanupXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser)
{

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
 * writeVmeLine Write a line of the vme bus preamble to the output file
 *
 * @param clkcond
 * @param address The address to be written.
 * @param value The value to be written.
 * @param ofs The output stream where the line is written to.
 *
 */


void L1GtTriggerMenuXmlParser::writeVmeLine(unsigned int clkcond,
        unsigned long int address, unsigned int value, std::ofstream& ofs)
{

    ofs << " "; // begin with a space
    ofs << std::fixed << std::setprecision(1) << std::setw(5); // 1 digit after dot for Time
    ofs << std::dec << m_vmePreambleTime;
    ofs << "> ";
    ofs << std::setw(1); // width 1 for clkcond
    ofs << clkcond << " ";
    ofs << std::setw(6); // width 6 for address
    ofs << std::setfill('0'); // leading zeros
    ofs << std::hex << std::uppercase << address << " ";    // switch to hexadecimal uppercase and write address
    ofs << std::setw(2); // width 2 for value
    ofs << std::setfill(' '); // no leading zeros for value
    ofs << value << std::dec << std::nouppercase;
    ofs << m_vmePreambleLineRest; // write the rest
    ofs << std::endl; // end of line

    m_vmePreambleTime += m_vmePreambleTimeTick;
}


/**
 * addVmeAddress add two lines to the preamble and increase the preamble time
 *
 * @param node The node to add to the preamble.
 * @param ofs The filestream for writing out the lines.
 * @return "true" if succeeded, "false" if an error occured.
 *
 */

bool L1GtTriggerMenuXmlParser::addVmeAddress(XERCES_CPP_NAMESPACE::DOMNode* node,
        std::ofstream& ofs)
{

    XERCES_CPP_NAMESPACE_USE

    std::string addrSrc = getXMLTextValue(node); // source string for the address
    std::string binaryNumbers = "01";

    unsigned int startPos = addrSrc.find_first_of(binaryNumbers);
    unsigned int endPos = addrSrc.find_first_not_of(binaryNumbers,startPos);

    if (startPos == endPos) {
        edm::LogError("L1GtTriggerMenuXmlParser") << "Error: No address value found."
        << std::endl;
        return false;
    }

    if (startPos < endPos - 1) {
        endPos = endPos - 1; // the last digit is ignored
    }

    addrSrc = addrSrc.substr(startPos, endPos - startPos);
    char* endPtr = (char*) addrSrc.c_str(); // end pointer for conversion

    if (*endPtr != 0) {
        edm::LogError("L1GtTriggerMenuXmlParser") << "Error converting binary address."
        << std::endl;

        return false;
    }

    // integer value of address
    unsigned long int address = strtoul(addrSrc.c_str(), &endPtr, 2);

    // look for the value

    DOMNode* valueNode = findXMLChild(node->getFirstChild(), vmexml_value_tag);

    if (valueNode == 0) {
        edm::LogError("L1GtTriggerMenuXmlParser") << "Found no value node for address."
        << std::hex << address
        << std::dec << std::endl;

        return false;
    }

    std::string valueSrc = getXMLTextValue(valueNode); // source string for the value

    startPos = valueSrc.find_first_of(binaryNumbers);
    endPos = valueSrc.find_first_not_of(binaryNumbers, startPos);

    if (startPos == endPos) {
        edm::LogError("L1GtTriggerMenuXmlParser") << "Error: No binary value found."
        << std::endl;

        return false;
    }

    valueSrc = valueSrc.substr(startPos , endPos - startPos);
    endPtr = (char*) valueSrc.c_str();

    if (*endPtr != 0) {
        edm::LogError("L1GtTriggerMenuXmlParser") << "Error converting binary value."
        << std::endl;

        return false;
    }

    unsigned long int value = strtoul(valueSrc.c_str(), &endPtr, 2);

    writeVmeLine(1, address, value, ofs);
    writeVmeLine(0, address, value, ofs);

    return true;

}



/**
 * parseVmeXML parse a xml file for vme bus preamble specification, 
 *     write it to a file and store the time
 *
 * @param parser The parser to use for parsing the file.
 *
 * @return true if succeeded, false if an error occured.
 *
 */

bool L1GtTriggerMenuXmlParser::parseVmeXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser)
{

    XERCES_CPP_NAMESPACE_USE

    // simply search for address tags within the chips and write them to the file

    DOMDocument* doc = parser->getDocument();
    DOMNode* n1 = doc->getFirstChild();

    if (n1 == 0) {

        edm::LogError("L1GtTriggerMenuXmlParser") << "Error: Found no XML child" 
        << std::endl;

        return false;
    }

    // find "vme"-tag
    n1 = findXMLChild(n1, vmexml_vme_tag);
    if (n1 == 0) {

        edm::LogError("L1GtTriggerMenuXmlParser") << "Error: No vme tag found." 
        << std::endl;

        return false;
    }

    n1 = n1->getFirstChild();

    // open the file
    std::ofstream ofs(m_vmePreambleFileName);

    // reset the time
    m_vmePreambleTime = 0.0;

    unsigned int chipCounter = 0; // count chips

    // TODO FIXME get it from EventSetup 
    const unsigned int NumberConditionChips = 2;
    
    while (chipCounter < NumberConditionChips) {

        n1 = findXMLChild(n1, vmexml_condchip_tag, true);
        if (n1 == 0) {
            // just break if no more chips found
            break;
        }


        // node for a particle
        DOMNode* particleNode = n1->getFirstChild();

        DOMNode* addressNode;   // an adress node

        while ((particleNode = findXMLChild(particleNode, "")) != 0) {

            // check if muon
            if (getXMLAttribute(particleNode, vmexml_attr_particle)
                    == vmexml_attr_particle_muon) {

                // node for walking through a particle
                DOMNode* walkNode = particleNode->getFirstChild();

                while ((walkNode = findXMLChild(walkNode, "")) != 0) {
                    addressNode = walkNode->getFirstChild();
                    while ((addressNode = findXMLChild(addressNode, vmexml_address_tag)) != 0) {
                        // LogTrace("L1GtTriggerMenuXmlParser") << getXMLTextValue(addressNode);
                        addVmeAddress(addressNode, ofs);
                        addressNode = addressNode->getNextSibling();
                    }
                    walkNode = walkNode->getNextSibling();
                }

            } else { // other particles than muon just contain adress nodes

                addressNode = particleNode->getFirstChild();
                while ((addressNode = findXMLChild(addressNode, vmexml_address_tag)) != 0) {
                    // LogTrace("L1GtTriggerMenuXmlParser") << getXMLTextValue(addressNode);
                    addVmeAddress(addressNode, ofs);
                    addressNode = addressNode->getNextSibling();
                }
            }

            particleNode = particleNode->getNextSibling();
        } // end while particle

        n1 = n1->getNextSibling();
        chipCounter++;
    } // end while chipCounter

    return true;

}

    // methods for conditions and algorithms

/**
 * getNumFromType - get the number of particles from a specified type name 
 * (for calorimeter objects and muons)
 *
 * @param type The name of the type
 *
 * @return The number of particles in this condition. -1 if type not found.
 */

int L1GtTriggerMenuXmlParser::getNumFromType(const std::string &type)
{

    if (type == xml_condition_attr_type_1) {
        return 1;
    }

    if (type == xml_condition_attr_type_2) {
        return 2;
    }

    if (type == xml_condition_attr_type_3) {
        return 3;
    }

    if (type == xml_condition_attr_type_4) {
        return 4;
    }

    if (type == xml_condition_attr_type_2wsc) {
        return 2;
    }

    return -1;
}

/**
 * getBitFromNode Get a bit from a specified bitvalue node.
 * 
 * @param node The xml node.
 *
 * @return The value of the bit or -1 if an error occured.
 */

int L1GtTriggerMenuXmlParser::getBitFromNode(XERCES_CPP_NAMESPACE::DOMNode* node) {

        if (getXMLAttribute(node, xml_attr_mode) != xml_attr_mode_bit) {
            
            edm::LogError("L1GtTriggerMenuXmlParser") 
                << "Invalid mode for single bit" 
                << std::endl;
            
            return -1;
        }        

        std::string tmpStr = getXMLTextValue(node);
        if (tmpStr == "0") {
            return 0;
        } else if (tmpStr == "1") {
            return 1;
        } else {
            edm::LogError("L1GtTriggerMenuXmlParser") 
                << "Bad bit value (" << tmpStr << ")" 
                << std::endl;
            return -1;
        }
}

/**
 * getGeEqFlag - get the "greater or equal flag" from a condition
 *
 * @param node The xml node of the condition.
 * @nodeName The name of the node from which the flag is a subchild.
 *
 * @return The value of the flag or -1 if no flag was found.
 */


int L1GtTriggerMenuXmlParser::getGeEqFlag(XERCES_CPP_NAMESPACE::DOMNode* node,
 const std::string& nodeName) {
        
    XERCES_CPP_NAMESPACE_USE

    if (node == 0) {
        
        LogTrace("L1GtTriggerMenuXmlParser") 
            << "node == 0 in " << __PRETTY_FUNCTION__ 
            << std::endl;
        
        return -1;
    }
    
    // usually the geeq flag is a child of the first child (the first element node)
    DOMNode* n1 = node->getFirstChild();
    n1 = findXMLChild(n1, nodeName);

    if (n1 != 0) {
        n1 = findXMLChild( n1->getFirstChild(), xml_geeq_tag);
        if (n1 == 0) {
            
            LogTrace("L1GtTriggerMenuXmlParser") 
                << "No greater equal tag found" 
                << std::endl;
            
            return -1;
        }
        
        return getBitFromNode(n1);
    } else {
        
        return -1;
    
    }
        
}

///**
// * workCondition - call the apropiate function to parse this condition.
// *
// * @param node The corresponding node to the condition.
// * @param name The name of the condition.
// * @param chipNr The number of the chip the condition is located on.
// *
// * @return "true" on success, "false" if an error occured.
// *
// */
//
//bool L1GtTriggerMenuXmlParser::workCondition(DOMNode* node,
//    const std::string& name, unsigned int chipNr) {
//
//    // get condition, particle name and type name
//    std::string condition = getXMLAttribute(node, xml_condition_attr_condition);
//    std::string particle  = getXMLAttribute(node, xml_condition_attr_particle);
//    std::string type      = getXMLAttribute(node, xml_condition_attr_type);
//
//    if ( condition.empty() || particle.empty() || type.empty() ) {
//        
//        edm::LogError("L1GtTriggerMenuXmlParser") 
//            << "Missing attributes for condition " << name 
//            << std::endl;
//        
//        return false;
//    }
//
//    LogTrace("L1GtTriggerMenuXmlParser") 
//        << "    condition: " << condition << ", particle: " << particle 
//        << ", type: " << type << std::endl;
//
//    // call the appropiate function for this condition
//
//    if (condition == xml_condition_attr_condition_muon) {
//        return parseMuon(node, name, chipNr);
//    } else if (condition == xml_condition_attr_condition_calo) {
//        return parseCalo(node, name, chipNr);
//    } else if (condition == xml_condition_attr_condition_esums) {
//        return parseESums(node, name, chipNr);
//    } else if (condition == xml_condition_attr_condition_jetcnts) {
//        return parseJetCounts(node, name, chipNr);
//    } else {
//        edm::LogError("L1GtTriggerMenuXmlParser") 
//        << "Unknown condition (" << condition << ")" 
//        << std::endl;
//        
//        return false;
//    }
//     
//    return true;
//
//}
//
//
//
///**
// * parseConditions - look for conditions and call the workCondition 
// *     function for each node
// *
// * @param parser The parser to parse the XML file with.
// *
// * @return "true" if succeeded. "false" if an error occured.
// *
// */
//
//
//bool L1GtTriggerMenuXmlParser::parseConditions(XERCES_CPP_NAMESPACE::XercesDOMParser* parser)
//{
//
//    XERCES_CPP_NAMESPACE_USE
//
//    LogTrace("L1GtTriggerMenuXmlParser") << "\nParsing conditions" << std::endl;
//
//    DOMNode* doc = parser->getDocument();
//    DOMNode* n1 = doc->getFirstChild();
//
//    // we assume that the first child is "def" because it was checked in workXML
//
//    // TODO def tag
//    DOMNode* chipNode = n1->getFirstChild();
//    if (chipNode == 0) {
//
//        edm::LogError("L1GtTriggerMenuXmlParser")
//        << "Error: No child of <def> found"
//        << std::endl;
//
//        return false;
//    }
//
//    // find chip
//
//    std::string chipName;        // name of the actual chip
//    chipNode = findXMLChild(chipNode, xml_chip_tag, true, &chipName);
//    if (chipNode == 0) {
//
//        edm::LogError("L1GtTriggerMenuXmlParser")
//        << "  Error: Could not find <" << xml_chip_tag
//        << std::endl;
//
//        return false;
//    }
//
//    unsigned int chipNr = 0;
//    do {
//
//        // find conditions
//        DOMNode* conditionsNode = chipNode->getFirstChild();
//        conditionsNode = findXMLChild(conditionsNode, xml_conditions_tag);
//        if (conditionsNode == 0) {
//
//            edm::LogError("L1GtTriggerMenuXmlParser")
//            << "Error: No <" << xml_conditions_tag << "> child found in Chip "
//            << chipName << std::endl;
//
//            return false;
//        }
//
//        char* nodeName = XMLString::transcode(chipNode->getNodeName());
//        LogTrace("L1GtTriggerMenuXmlParser")
//        << "\n  Found Chip: " << nodeName << " Name: " << chipName
//        << std::endl;
//
//        XMLString::release(&nodeName);
//
//        // walk through conditions
//        DOMNode* conditionNameNode = conditionsNode->getFirstChild();
//        std::string conditionNameNodeName;
//        conditionNameNode = findXMLChild(conditionNameNode, "", true, &conditionNameNodeName);
//        while (conditionNameNode != 0) {
//
//            LogTrace("L1GtTriggerMenuXmlParser")
//            << "\n    Found a condition with name: " << conditionNameNodeName
//            << std::endl;
//
//            if (workCondition(conditionNameNode, conditionNameNodeName, chipNr) != 0) {
//                return false;
//            }
//            conditionNameNode = findXMLChild(conditionNameNode->getNextSibling(), "",
//                                             true, &conditionNameNodeName);
//
//        }
//        // next chip
//        chipNode = findXMLChild(chipNode->getNextSibling(), xml_chip_tag, true, &chipName);
//        chipNr++;
//
//    } while (chipNode != 0 && chipNr < NumberConditionChips);
//
//    return true;
//}
//
//
///**
// * workXML parse the XML-File
// *
// * @param parser The parser to use for parsing the XML-File
// *
// * @return true if succeeded, false if an error occured.
// */
//
//
//bool L1GtTriggerMenuXmlParser::workXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser)
//{
//
//    XERCES_CPP_NAMESPACE_USE
//
//    DOMDocument* doc = parser->getDocument();
//    DOMNode* n1 = doc->getFirstChild();
//
//    if (n1 == 0) {
//
//        edm::LogError("L1GtTriggerMenuXmlParser") << "Error: Found no XML child"
//        << std::endl;
//
//        return false;
//    }
//
//    char* nodeName = XMLString::transcode(n1->getNodeName());
//    // TODO def as static std::string
//    if (XMLString::compareIString(nodeName, "def")) {
//
//        edm::LogError("L1GtTriggerMenuXmlParser")
//        << "Error: First XML child is not \"def\""
//        << std::endl;
//
//        return false;
//    }
//
//    LogTrace("L1GtTriggerMenuXmlParser")
//    << "\nFirst node name is: " << nodeName
//    << std::endl;
//    XMLString::release(&nodeName);
//
//    // clear possible old conditions
//    clearConditionsMap();
//
//    if ( ! checkVersion(parser) ) {
//        return false;
//    }
//
//    if ( ! parseConditions(parser) ) {
//        clearConditionsMap();
//        return false;
//    }
//
//    if ( ! parseAllAlgos(parser) ) {
//        clearConditionsMap();
//        return false;
//    }
//
//    return true;
//
//}
//
//
//
//
//
//
//





// static class members

const std::string L1GtTriggerMenuXmlParser::xml_def_tag("def");
const std::string L1GtTriggerMenuXmlParser::xml_chip_tag("condition_chip_");
const std::string L1GtTriggerMenuXmlParser::xml_conditions_tag("conditions");
const std::string L1GtTriggerMenuXmlParser::xml_prealgos_tag("prealgos");
const std::string L1GtTriggerMenuXmlParser::xml_algos_tag("algos");

const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_condition("condition");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle("particle");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_type("type");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_condition_muon("muon");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_condition_calo("calo");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_condition_esums("esums");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_condition_jetcnts("jet_cnts");

const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_muon("muon");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_eg("eg");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_ieg("ieg");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_jet("jet");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_fwdjet("fwdjet");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_tau("tau");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_etm("etm");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_ett("ett");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_particle_htt("htt");



const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_type_1("1_s");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_type_2("2_s");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_type_2wsc("2_wsc");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_type_3("3");
const std::string L1GtTriggerMenuXmlParser::xml_condition_attr_type_4("4");


const std::string L1GtTriggerMenuXmlParser::xml_attr_mode("mode");
const std::string L1GtTriggerMenuXmlParser::xml_attr_mode_bit("bit");
const std::string L1GtTriggerMenuXmlParser::xml_attr_max("max");

const std::string L1GtTriggerMenuXmlParser::xml_attr_nr("nr");
const std::string L1GtTriggerMenuXmlParser::xml_attr_pin("pin");
const std::string L1GtTriggerMenuXmlParser::xml_attr_pin_a("a");

const std::string L1GtTriggerMenuXmlParser::xml_etthreshold_tag("et_threshold");

const std::string L1GtTriggerMenuXmlParser::xml_pththreshold_tag("pt_h_threshold");
const std::string L1GtTriggerMenuXmlParser::xml_ptlthreshold_tag("pt_l_threshold");
const std::string L1GtTriggerMenuXmlParser::xml_quality_tag("quality");
const std::string L1GtTriggerMenuXmlParser::xml_eta_tag("eta");
const std::string L1GtTriggerMenuXmlParser::xml_phi_tag("phi");
const std::string L1GtTriggerMenuXmlParser::xml_phih_tag("phi_h");
const std::string L1GtTriggerMenuXmlParser::xml_phil_tag("phi_l");
const std::string L1GtTriggerMenuXmlParser::xml_chargecorrelation_tag("charge_correlation");
const std::string L1GtTriggerMenuXmlParser::xml_enmip_tag("en_mip");
const std::string L1GtTriggerMenuXmlParser::xml_eniso_tag("en_iso");
const std::string L1GtTriggerMenuXmlParser::xml_enoverflow_tag("en_overflow");
const std::string L1GtTriggerMenuXmlParser::xml_deltaeta_tag("delta_eta");
const std::string L1GtTriggerMenuXmlParser::xml_deltaphi_tag("delta_phi");

const std::string L1GtTriggerMenuXmlParser::xml_output_tag("output");
const std::string L1GtTriggerMenuXmlParser::xml_outputpin_tag("output_pin");

const std::string L1GtTriggerMenuXmlParser::xml_geeq_tag("ge_eq");
const std::string L1GtTriggerMenuXmlParser::xml_value_tag("value");

const std::string L1GtTriggerMenuXmlParser::xml_chipdef_tag("chip_def");
const std::string L1GtTriggerMenuXmlParser::xml_chip1_tag("chip_1");
const std::string L1GtTriggerMenuXmlParser::xml_ca_tag("ca");

//vmexml std::strings
const std::string L1GtTriggerMenuXmlParser::vmexml_vme_tag("vme");
const std::string L1GtTriggerMenuXmlParser::vmexml_condchip_tag("cond_chip_");
const std::string L1GtTriggerMenuXmlParser::vmexml_address_tag("address");
const std::string L1GtTriggerMenuXmlParser::vmexml_value_tag("value");

const std::string L1GtTriggerMenuXmlParser::vmexml_attr_particle("particle");
const std::string L1GtTriggerMenuXmlParser::vmexml_attr_particle_muon("muon");

const char L1GtTriggerMenuXmlParser::m_vmePreambleFileName[] = "testxxoo3.data";
const double L1GtTriggerMenuXmlParser::m_vmePreambleTimeTick = 12.5;
const char L1GtTriggerMenuXmlParser::m_vmePreambleLineRest[] =
    " 3 00000 00000 00000 00000 00000 00000 00000 00000 0000000 0000000 = XXXXXXXXXXXXXX XXXXXXXXXXXXXX";
