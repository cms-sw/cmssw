/**
 * \class L1GlobalTriggerConfig
 * 
 * 
 * 
 * Description: see header file 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author  M. Eder      - HEPHY Vienna - ORCA version
 * \author  Vasile Ghete - HEPHY Vienna - CMSSW version
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

// system include files
#include <iostream>
#include <sstream>
#include <cstdio>
#include <string>
#include <fstream>
#include <iomanip>

// user include files
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerLogicParser.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerMuonTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerCaloTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerEsumsTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerJetCountsTemplate.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"

XERCES_CPP_NAMESPACE_USE

// constructors

L1GlobalTriggerConfig::L1GlobalTriggerConfig(
    L1GlobalTrigger* gt,
    std::string& defXmlFile, 
    std::string& vmeXmlFile ) 
    : m_GT(gt)
    {

    LogDebug ("Trace") << "Entering " << __PRETTY_FUNCTION__ << std::endl; 
    
    m_xmlErrHandler = 0;
    
    parseTriggerMenu(defXmlFile, vmeXmlFile);
    
}

// destructor
L1GlobalTriggerConfig::~L1GlobalTriggerConfig() {

    clearConditionsMap();
 
}

/**
 * parseTriggerMenu Start reading the XML-file and print the read data.
 *
 * @param dir The name of the directory the xml-File is looked for.
 *
 *
 */ 
 
 
void L1GlobalTriggerConfig::parseTriggerMenu(std::string& defFile, std::string& vmeFile) {
    
    LogDebug ("Trace") << "Entering " << __PRETTY_FUNCTION__ << std::endl;
                
    XercesDOMParser* parser;
    LogTrace("L1GlobalTriggerConfig") << "\nOpening XML-File: \n  " << defFile << std::endl;
    if ((parser = initXML(defFile)) != 0) {
        workXML(parser);
    }
    cleanupXML(parser);

    std::ostringstream myCoutStream;
    printThresholds(myCoutStream);
    LogTrace("L1GlobalTriggerConfig") << myCoutStream.str() << std::endl;


    // part to generate vme-bus preamble

    if (vmeFile != "") {
        
        LogTrace("L1GlobalTriggerConfig") << "Writing vme-bus preamble" << std::endl;
        if ((parser = initXML(vmeFile)) != 0) {
             parseVmeXML(parser);
        }
        cleanupXML(parser);
    }
        
}

/**
 * initXML - Initialize XML-utilities and try to create a parser for the specified file.
 *
 * @param xmlFile Filename of the XML-File
 *
 * @return A reference to a XercesDOMParser object if suceeded. 0 if an error occured.
 *
 */

XercesDOMParser* L1GlobalTriggerConfig::initXML(const std::string &xmlFile) {
  
    // try to initialize
    try {
        XMLPlatformUtils::Initialize();
    } catch (const XMLException& toCatch) {
        char *message = XMLString::transcode(toCatch.getMessage());
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error during Xerces-c initialization! :"
            << message << std::endl;
        XMLString::release(&message);
        return 0;
    }

    XercesDOMParser *parser = new XercesDOMParser();
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
        char *message = XMLString::transcode(toCatch.getMessage());
        edm::LogError("L1GlobalTriggerConfig") 
            << "Exception while parsing XML: \n"
            << message << std::endl;
        XMLString::release(&message);
        delete parser;
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
        return 0;
    } catch (const DOMException &toCatch) {
        char *message = XMLString::transcode(toCatch.msg);
        edm::LogError("L1GlobalTriggerConfig") 
            << "DOM-Exception while parsing XML: \n"
            << message << std::endl;
        XMLString::release(&message);
        delete parser;
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
        return 0;
    } catch (...) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Unexpected Exception while parsing XML!" 
            << std::endl;
        delete parser;
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
        return 0;
    }

    return parser;
}
    
/**
 * cleanupXML - Delete parser and error handler. Shutdown XMLPlatformUtils.
 *
 * @param parser A reference to the parser to be deleted.
 *
 */

void L1GlobalTriggerConfig::cleanupXML(XercesDOMParser *parser) {

    // just delete parser and errorhandler
    if (parser != 0) {
        delete parser;
    }
    if (m_xmlErrHandler != 0) {
        delete m_xmlErrHandler;
        m_xmlErrHandler = 0;
    }

    XMLPlatformUtils::Terminate();

}

 /**
  * findXMLChild - searches for a child with the given name, optionally only 
  * the beginning of a tag is matched
  *
  * @param startchild The child to start with.
  * @param tagname The name looked for. If an empty string is used every child matches.
  * @param beginonly Optional parameter. If true the specified tagname only 
  *        needs to match the beginning of the childname.
  * @param rest The not matched rest is written here if beginonly is true (optional)
  *
  * @return Th    reference to the found child node.
  */

DOMNode* L1GlobalTriggerConfig::findXMLChild(
    DOMNode *startchild, const std::string &tagname,
    bool beginonly = false, std::string *rest = 0) {

    char* nodeName = 0;
    
    DOMNode *n1 = startchild;
    if (n1 == 0) {
        return 0;
    }

    if ( tagname != "") {
        nodeName = XMLString::transcode(n1->getNodeName());

        if (!beginonly) {
            while (XMLString::compareIString(nodeName, tagname.c_str())) { //match the whole tag
                XMLString::release(&nodeName);
                n1 = n1->getNextSibling();
                if (n1 == 0) break;
                nodeName = XMLString::transcode(n1->getNodeName());
            }
        } else {
            // match only the beginning
            while (XMLString::compareNIString(nodeName, tagname.c_str(), tagname.length())) {
                XMLString::release(&nodeName);
                n1 = n1->getNextSibling();
                if (n1 == 0) break;
                nodeName = XMLString::transcode(n1->getNodeName());
            }
            if (n1 != 0 && rest != 0) {
                *rest = std::string(nodeName).substr(tagname.length(), strlen(nodeName) - 
                    tagname.length());
            }
        }
    } else { // empty string given
        while (n1->getNodeType() != DOMNode::ELEMENT_NODE) {
            n1 = n1->getNextSibling();
            if (n1 == 0) break;
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
 * @return The value of the attribute or "" if an error occured.
 */

std::string L1GlobalTriggerConfig::getXMLAttribute(
    const DOMNode* node, const std::string& name) {

    XMLCh* attrname;
    char* retcstr;
    std::string ret = "";

    //get attributes list
    DOMNamedNodeMap* attributes = node->getAttributes();
    if (attributes == 0) return ret;
    
    //get attribute node
    attrname = XMLString::transcode(name.c_str());
    DOMNode *attribnode = attributes->getNamedItem(attrname);
    XMLString::release(&attrname);
    if (attribnode == 0) return ret;
    
    retcstr = XMLString::transcode(attribnode->getNodeValue());
    ret = retcstr;
    XMLString::release(&retcstr);

    return ret;
}

/**
 * getXMLTextValue - get the textvalue from a specified node
 *
 * @param node The reference to the node.
 * @return The textvalue of the node or an empty std::string if no textvalue is available.
 *
 */

std::string L1GlobalTriggerConfig::getXMLTextValue(DOMNode *node) {

    char *retcstr;
    const XMLCh* retxmlch;
    std::string ret = "";

    DOMNode *n1 = node;
    
    if (n1 == 0) return ret;
    if ((retxmlch = n1->getTextContent()) == 0) {
        return ret;
    }
    retcstr = XMLString::transcode(retxmlch);
    ret = retcstr;
    XMLString::release(&retcstr);

    return ret;
}

/** 
 * getNumFromType - get the number of particles from a specified type name 
 * (for calos and muons)
 *
 * @param type The name of the type
 *
 * @return The number of particles in this condition. -1 if type not found.
 */

int L1GlobalTriggerConfig::getNumFromType(const std::string &type) {

    if (type == xml_condition_attr_type_1) return 1;
    if (type == xml_condition_attr_type_2) return 2;
    if (type == xml_condition_attr_type_3) return 3;
    if (type == xml_condition_attr_type_4) return 4;
    if (type == xml_condition_attr_type_2wsc) return 2;

    return -1;
}


/** 
 * clearConditionsMap - delete all conditions in the map and clear the map.
 *
 */


void L1GlobalTriggerConfig::clearConditionsMap() {


    for (unsigned int i = 0; i < NumberConditionChips; i++) {
        // delete all conditions
        for (ConditionsMap::iterator it = conditionsmap[i].begin(); 
            it != conditionsmap[i].end(); it++) {
            
            if (it->second != 0) {
                delete it->second;
            }
            it->second = 0;
        
        }
        conditionsmap[i].clear();
    }

    // also clear prealgo and algo map
    for (ConditionsMap::iterator it = prealgosmap.begin();
        it != prealgosmap.end(); it++) {

        if (it->second != 0) {
            delete it->second;
        }
        it->second = 0;

    }
    prealgosmap.clear();

    for (ConditionsMap::iterator it = algosmap.begin(); 
        it != algosmap.end(); it++) {
            
        if (it->second != 0) {
            delete it->second;
        }
        it->second = 0;
        
    }
    algosmap.clear();
}


/**
 * insertIntoConditionsMap - Safe insert into conditions map. 
 * Do a check if a conditionname already exists to prevent a memory leak.
 *
 * @param cond Pointer to a L1GlobalTriggerConditionXML object.
 * @param chipNr Number of chip the condition is on.
 *
 * @return 0 if suceeded, -1 if the name already exists and nothing was inserted. 
 */

int L1GlobalTriggerConfig::insertIntoConditionsMap(
    L1GlobalTriggerConditions *cond, unsigned int chipNr = 0) {

    std::string name = cond->getName();
    LogTrace("L1GlobalTriggerConfig") 
        << "    Trying to insert condition (" << name << ")" ;
    
    //no condition name has to appear twice!
    if (conditionsmap[chipNr].count(name) != 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "      Condition already exists - not inserted!" 
            << std::endl;
        return -1;
    }

    conditionsmap[chipNr][name] = cond;
    LogTrace("L1GlobalTriggerConfig") 
        << "      OK - condition inserted!" 
        << std::endl;
    
    return 0;
}

/**
 * insertAlgoIntoMap Insert a algo into a map.
 *
 * @param node       The node of the algo.
 * @param algoname   The name of the algo.
 * @param insertmap  A reference to the map where the algo should be inserted.
 * @param operandmap 
 * @param nummap
 *
 * @return 0 if succeeded, -1 if an error occured (eg. duplicate algo-names).
 *
 */

int L1GlobalTriggerConfig::insertAlgoIntoMap(DOMNode* node, 
    const std::string& algoname, 
    ConditionsMap* insertmap, ConditionsMap* operandmap,
    unsigned int chipnr, 
    unsigned int nummap) {

    if (node == 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "    Node is 0 in " << __PRETTY_FUNCTION__ 
            << std::endl;
        return -1;
    }

    L1GlobalTriggerLogicParser *algoparser;    // the reference to insert

    // check if there is already such name
    if (insertmap->count(algoname) != 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "    Duplicate algorithm with name: " << algoname << " not inserted" 
            << std::endl;
        return -1;
    }

    // create a new Logic parser object
    algoparser = new L1GlobalTriggerLogicParser(*m_GT, algoname);

    // get the expression from the node
    std::string expression = getXMLTextValue(node);
    
    if ( algoparser->setExpression(expression, operandmap, nummap) != 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "    Error parsing expression: " << expression 
            << std::endl;
        delete algoparser;
        return -1;
    }

    // determine output pin
    DOMNode* pinnode;
    std::string pinstring;
    // output tags
    // for algos
    pinnode = findXMLChild(node->getFirstChild(), xml_output_tag);
    if (pinnode != 0) {
        pinstring = getXMLAttribute(pinnode, xml_attr_nr);
        algoparser->setOutputPin(strtol(pinstring.c_str(), (char **)NULL, 10));
    } else {
        // for prealgos
        pinnode = node->getFirstChild();
        while ( ( pinnode = findXMLChild(pinnode, xml_outputpin_tag) ) != 0 ) {
            pinstring = getXMLAttribute(pinnode, xml_attr_pin); // we look for the "a" pin
            if (pinstring == xml_attr_pin_a) {
                // found pin a
                pinstring = getXMLAttribute(pinnode, xml_attr_nr);
                algoparser->setOutputPin(strtol(pinstring.c_str(), (char **)NULL, 10));
                algoparser->setAlgoNumber(strtol(pinstring.c_str(), (char **)NULL, 10) + 
                    (OrderConditionChip[chipnr] -1)*PinsOnConditionChip);
                
                break;
            }
            pinnode = pinnode->getNextSibling();
        }
        if (pinnode == 0) {
            algoparser->setOutputPin(0);
            algoparser->setAlgoNumber(0);
            LogTrace("L1GlobalTriggerConfig") 
                << "    Warning: No pin number found for algorithm: " 
                << algoparser->getName() 
                << std::endl;
        }
    }

    if (algoparser->getOutputPin() > (int) PinsOnConditionChip) {
        LogTrace("L1GlobalTriggerConfig") 
            << "    Warning: Pin out of range for algorithm:    "
            << algoparser->getName() << std::endl;
        algoparser->setOutputPin(0);
        algoparser->setAlgoNumber(0);
    }

    // check the number of algorithms
    if (insertmap->size() > MaxNumberAlgorithms) {
        LogTrace("L1GlobalTriggerConfig") 
            << "    Warning: too many algorithms! Can not insert algorithm "
            << algoparser->getName() << std::endl;
        algoparser->setOutputPin(0);
        algoparser->setAlgoNumber(0);
    }
    
        
    // insert into map 
    (*insertmap)[algoname] = algoparser;

    return 0;
}
    
/**
 * getBitFromNode Get a bit from a specified bitvalue node.
 * 
 * @param node The xml node.
 *
 * @return The value of the bit or -1 if an error occured.
 */

int L1GlobalTriggerConfig::getBitFromNode(DOMNode* node) {

        if (getXMLAttribute(node, xml_attr_mode) != xml_attr_mode_bit) {
            edm::LogError("L1GlobalTriggerConfig") 
                << "Invalid mode for single bit" 
                << std::endl;
            return -1;
        }    // only bit mode TODO: what modes else?        

        std::string tmpstr = getXMLTextValue(node); // temporary string
        if (tmpstr == "0") {
            return 0;
        } else if (tmpstr == "1") {
            return 1;
        } else {
            edm::LogError("L1GlobalTriggerConfig") 
                << "Bad bit value (" << tmpstr << ")" 
                << std::endl;
            return -1;
        }
}
 
/**
 * getGeEqFlag - get the greater equal flag from a condition
 *
 * @param node The xml node of the condition.
 * @nodeName The name of the node from which the flag is a subchild.
 *
 * @return The value of the flag or -1 if no flag was found.
 */


int L1GlobalTriggerConfig::getGeEqFlag(DOMNode* node, const std::string& nodeName) {
        
    if (node == 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "node == 0 in " << __PRETTY_FUNCTION__ 
            << std::endl;
        return -1;
    }
    
    DOMNode *n1;
    
    // usually the geeq flag is a child of the first child (thefirst element node)
    n1 = node->getFirstChild();
    n1 = findXMLChild(n1, nodeName);
    if (n1 != 0) {
        n1 = findXMLChild( n1->getFirstChild(), xml_geeq_tag);
        if (n1 == 0) {
            LogTrace("L1GlobalTriggerConfig") 
                << "No greater equal tag found" 
                << std::endl;
            return -1;
        }
        return getBitFromNode(n1);
    } else {
        return -1;
    }
        
}

/**
 * getXMLHexTextValue Get the integer representation of a xml text child 
 *     representing a hex value
 *
 * @param node The xml node to get the value from.
 * @param dst The destination the value is written to.
 *
 * @return 0 if suceeded, -1 if an error occured
 *
 */ 
     

int L1GlobalTriggerConfig::getXMLHexTextValue(DOMNode *node, u_int64_t& dst) {

    u_int64_t dummyh; // dummy for eventual higher 64bit
    u_int64_t tempuint;	// temporary unsigned integer

    if (getXMLHexTextValue128(node, tempuint, dummyh) != 0) return -1;

    if (dummyh != 0) {
        edm::LogError("L1GlobalTriggerConfig") << "Too large hex-value!" << std::endl;
        return -1;
    }

    dst = tempuint;
    return 0;
}

/**
 * getXMLHexTextValue128 Get the integer representation of a xml-node 
 *     containing a hexadezimal value. The value may contain up to 128 bits.
 *
 * node - The reference to the node to get the value from.
 * dstl - The destination for the lower 64bit
 * dsth - The destination for the higher 64bit
 *
 */
    

int L1GlobalTriggerConfig::getXMLHexTextValue128(DOMNode *node, 
    u_int64_t &dstl, u_int64_t &dsth) {

    if (node == 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "node == 0 in " << __PRETTY_FUNCTION__ 
            << std::endl; 
        return -1;
    }

    u_int64_t tempuinth, tempuintl; // temporary unsigned integer

    std::string tempstr = getXMLTextValue(node);
    if (hexString2UInt128(tempstr, tempuintl, tempuinth) != 0) return -1;

    dstl = tempuintl;
    dsth = tempuinth;
	
    return 0;
}
 
/**
 * hexString2UInt128 converts a up to 128 bit hexadecimal string to 2 u_int64_t
 *
 * @param hex The string to be converted.
 * @param dstl The target for the lower 64 bit.
 * @param dsth The target for the upper 64 bit.
 *
 * @return 0 if conversion suceeded, -1 if an error occured.
 */


int L1GlobalTriggerConfig::hexString2UInt128(const string& hex, 
    u_int64_t &dstl, u_int64_t &dsth) {

    std::string tempstr, tempstrh, tempstrl; // temporary std::string
    u_int64_t tempuinth, tempuintl;          // temporary unsigned integer
    char *endptr;                            // endpointer for conversion

    unsigned int hexstart;   // start position of the hex value in the string
    unsigned int hexend;     // end position of the hex value in the string
    
    // string to determine start of hex value don't ignore leading zeros
    static const std::string valid_hex_start("0123456789ABCDEFabcdef"); 
    // string to determine end of hex value
    static const std::string valid_hex_end("0123456789ABCDEFabcdef"); 
    
    tempstr = hex;

    hexstart = tempstr.find_first_of(valid_hex_start);
    hexend = tempstr.find_first_not_of(valid_hex_end, hexstart);

    if (hexstart == hexend) {
        LogTrace("L1GlobalTriggerConfig") 
            << "No hex value found in: " << tempstr << std::endl;
        return -1;
    }

    tempstr = tempstr.substr(hexstart, hexend - hexstart);
    
    if (tempstr == "") {
        LogTrace("L1GlobalTriggerConfig") 
            << "Empty value in " << __PRETTY_FUNCTION__ << std::endl;
        return -1;
    }

    // split the string    
    if (tempstr.length() > 16) { // more than 64 bit
        tempstrl=tempstr.substr(tempstr.length()-16, 16);
        tempstrh=tempstr.substr(0,tempstr.length()-16);
    } else {
        tempstrl=tempstr;
        tempstrh="0";
    }
    
    // convert lower 64bit
    endptr = (char*) tempstrl.c_str();
    tempuintl = strtoull(tempstrl.c_str(), &endptr, 16);
    if (*endptr != 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "Unable to convert " << tempstr << " to hex." << std::endl;
        return -1;
    }

    // convert higher64 bit
    endptr = (char*) tempstrh.c_str();
    tempuinth = strtoull(tempstrh.c_str(), &endptr, 16);
    if (*endptr != 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "Unable to convert " << tempstr << " to hex." << std::endl;
        return -1;
    }

    dstl = tempuintl;
    dsth = tempuinth;
	
    return 0;
}

/**
 * countConditionChildMaxBits Count the set bits in the max attribute. 
 *     Needed for the wsc-values to determine 180 degree.
 *
 * @param node The xml node of the condition.
 * @param childname The name of the child 
 * @param dst The destination to write the number of bits.
 *
 * @return 0 if the Bits could be determined, otherwise -1.
 */

int L1GlobalTriggerConfig::countConditionChildMaxBits(DOMNode *node, 
    const std::string& childname, unsigned int& dst) {

    // should never happen, but lets check 
    if (node == 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "node == 0 in " << __PRETTY_FUNCTION__ << std::endl;
        return -1;
    }
    
    DOMNode *n2;
    DOMNode *n1;
    n1 = findXMLChild(node->getFirstChild(), childname);

    // if child not found
    if (n1 == 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "Child of condition not found ( " << childname << ")" 
            << std::endl;
        return -1;
    }

    n2 = findXMLChild(n1->getFirstChild(), xml_value_tag);

    if (n2 == 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "No value tag found for child " << childname << " in " << __PRETTY_FUNCTION__ 
            << std::endl;
        return -1;
    }
    
    // first try direct
    std::string maxstring = getXMLAttribute(n1, xml_attr_max); // string for the maxbits

    if (maxstring == "") {
        maxstring = getXMLAttribute(n2, xml_attr_max); // try next value tag
        // if no max was found neither in value nor in the childname tag
        if (maxstring == "") {
            LogTrace("L1GlobalTriggerConfig") 
                << "No Max value found for " << childname 
                << std::endl;
            return -1;
        }
    }

    // do the hex conversion

    u_int64_t maxbitsl, maxbitsh;   // unsigned integer values which bits are count.
    if (hexString2UInt128(maxstring, maxbitsl, maxbitsh) != 0 ) return -1;

    // count the bits
    LogTrace("L1GlobalTriggerConfig") 
        << "      Counting maxbits: high value = " << maxbitsh << " low value = " << maxbitsl 
        << std::endl;

    unsigned int counter = 0;

    while (maxbitsl != 0) {
        // check if bits set countinously
        if ( (maxbitsl & 1) == 0 ) {
            edm::LogError("L1GlobalTriggerConfig") 
                << "      Confused by not continous set bits for max value " 
                << maxstring << "(child=" << childname << ")" 
                << std::endl;
            return -1;
        }
        maxbitsl >>= 1;
        counter++;
    }
    
    if (maxbitsh != 0 && counter != 64) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "      Confused by not continous set bits for max value " 
            << maxstring << "(child=" << childname << ")" 
            << std::endl;
        return -1;
    }

        
    while (maxbitsh != 0) {
        //check if bits set countinously
        if ( (maxbitsh & 1) == 0 ) {
            edm::LogError("L1GlobalTriggerConfig") 
                << "      Confused by not continous set bits for max value " 
                << maxstring << "(child=" << childname << ")" 
                << std::endl;
            return -1;
        }
        maxbitsh >>= 1;
        counter++;
    }

    dst = counter;
    return 0;
    
}

/**
 * getConditionChildValues - Get values from a child of a condition.
 * 
 * @param node The xml node of the condition.
 * @param childname The name of the child the values should be extracted from.
 * @param num The number of values needed.
 * @param dst A pointer to a array of u_int64_t where the results are written.
 *
 * @return 0 if suceeded. -1 if an error occured or not enough values found. 
 */

int L1GlobalTriggerConfig::getConditionChildValues(DOMNode *node, 
    const std::string &childname, unsigned int num, u_int64_t *dst) {

    // should never happen, but lets check 
    if (node == 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "node == 0 in " << __PRETTY_FUNCTION__ 
            << std::endl;
        return -1;
    }

    DOMNode *n1 = findXMLChild(node->getFirstChild(), childname);

    // if child not found
    if (n1 == 0) {
        LogTrace("L1GlobalTriggerConfig") 
            << "Child of condition not found ( " << childname << ")" 
            << std::endl;    
        return -1;
    }

    // no values are sucessfull
    if (num == 0) return 0;

    n1 = findXMLChild(n1->getFirstChild(), xml_value_tag);
    for (unsigned int i = 0; i < num; i++) {
        if (n1 == 0) { 
            LogTrace("L1GlobalTriggerConfig") 
                << "Not enough values in condition child ( " << childname << ")" 
                << std::endl;
            return -1; // too less values
        }

        if (getXMLHexTextValue(n1, dst[i]) != 0) {
            edm::LogError("L1GlobalTriggerConfig") 
                << "Error converting condition child ( " << childname << ") value." 
                << std::endl;
            return -1;
        }
        
        n1 = findXMLChild(n1->getNextSibling(), xml_value_tag); // next child
    }

    // if we get here we are successfull
    return 0;
}
    

/**
 * getMuonMipIsoBits - Get Mip and Iso bits from a muon.
 *
 * @param node The node of the condition. 
 * @param num The number of bits required.
 * @param mipdst A pointer to an boolean array for the mip bits.
 * @param isodst A pointer to an boolean array for the iso pits.
 *
 * @return 0 if suceeded, -1 if an error occured.
 */

int L1GlobalTriggerConfig::getMuonMipIsoBits(DOMNode *node, 
    unsigned int num, bool *mipdst, bool *isodst) {

    if (node == 0) return -1;
    
    // find pt_l_threshold child
    DOMNode *n1;        
    n1 = findXMLChild(node->getFirstChild(), xml_ptlthreshold_tag);

    if (n1 == 0) return -1;

    // get first value tag
    n1 = findXMLChild(n1->getFirstChild(), xml_value_tag);

    for (unsigned int i = 0; i < num; i++) {
        if (n1 == 0) return -1; 	// too less values
        
        //mip bit
        DOMNode *bitnode = findXMLChild(n1->getFirstChild(), xml_enmip_tag); // working node
        if (bitnode == 0) return 0;

        int tmpint = getBitFromNode(bitnode);
        if (tmpint < 0) return -1;

        mipdst[i] = (tmpint != 0);

        LogTrace("L1GlobalTriggerConfig") 
            << "      MIP bit value for muon " << i << " = " << mipdst[i]
            << std::endl;


        //iso bit
        bitnode = findXMLChild(n1->getFirstChild(), xml_eniso_tag);
        if (bitnode == 0) return 0;

        tmpint = getBitFromNode(bitnode);
        if (tmpint < 0) return -1;

        isodst[i] = (tmpint != 0);

        LogTrace("L1GlobalTriggerConfig") 
            << "      Iso bit value for muon " << i << " = " << isodst[i]
            << std::endl;

        n1 = findXMLChild(n1->getNextSibling(), xml_value_tag); // next value
    }

    return 0; 
}


        
/**
 * parseMuon Parse a muon condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return 0 if suceeded. -1 if an error occured.
 *
 */

int L1GlobalTriggerConfig::parseMuon(DOMNode* node, 
    const std::string& name, unsigned int chipNr=0) {

    // temporary storage of the parameters
    L1GlobalTriggerMuonTemplate::ParticleParameter particleparameter[4];
    L1GlobalTriggerMuonTemplate::ConditionParameter conditionparameter;

    // get condition, particle name (must be muon) and type name
    std::string condition = getXMLAttribute(node, xml_condition_attr_condition);
    std::string particle = getXMLAttribute(node, xml_condition_attr_particle);
    std::string type = getXMLAttribute(node, xml_condition_attr_type);

    if (particle != xml_condition_attr_particle_muon) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Wrong particle for muon-condition (" << particle << ")" 
            << std::endl;
        return -1;
    }

    int numtype = getNumFromType(type); // the number of particles
    if (numtype < 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Unknown type for muon-condition (" << type << ")" 
            << std::endl;
        return -1;
    }

    if (numtype > 4) numtype = 4; // TODO: macro instead of 4

    // get greater equal flag
    
    // temporary integer for greater equal flag
    int ge_eqint = getGeEqFlag(node, xml_pththreshold_tag);
    if (ge_eqint < 0) {
        edm::LogError("L1GlobalTriggerConfig") 
        << "Error getting greater equal flag" 
        << std::endl;
        return -1;
    }
    // set the boolean value for the ge_eq mode
    bool ge_eq = ( ge_eqint != 0);

    // get values
    
    u_int64_t tmpvalues[4]; // temporary storage of values
    
    // get pt_h_thresholds    
    if (getConditionChildValues(node, xml_pththreshold_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        particleparameter[i].pt_h_threshold = tmpvalues[i];

        LogTrace("L1GlobalTriggerConfig") 
            << "      Muon high threshold (hex) for muon " << i << " = " 
            << std::hex << tmpvalues[i] << std::dec 
            << std::endl;
    }

    // get pt_l_thresholds

    if (getConditionChildValues(node, xml_ptlthreshold_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        LogTrace("L1GlobalTriggerConfig") 
            << "      Muon low threshold word (hex) for muon " << i << " = " 
            << std::hex << tmpvalues[i] << std::dec 
            << std::endl;

        // one takes mipbit at therefore one divide by 16 // TODO ?
        tmpvalues[i] = tmpvalues[i]/16;
        
        LogTrace("L1GlobalTriggerConfig") 
            << "        Muon low threshold (hex) for muon " << i << " = " 
            << std::hex << tmpvalues[i] << std::dec 
            << std::endl;
         particleparameter[i].pt_l_threshold = tmpvalues[i];
    }

    // get quality
    if (getConditionChildValues(node, xml_quality_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        particleparameter[i].quality = tmpvalues[i];
    }

    // get eta
    if (getConditionChildValues(node, xml_eta_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        particleparameter[i].eta = tmpvalues[i];
    }

    // get phi_h 
    if (getConditionChildValues(node, xml_phih_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        particleparameter[i].phi_h = tmpvalues[i];
    }
    
    // get phi_l
    if (getConditionChildValues(node, xml_phil_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        particleparameter[i].phi_l = tmpvalues[i];
    }

    // get charge correlation
    if (getXMLHexTextValue(findXMLChild(node->getFirstChild(), 
        xml_chargecorrelation_tag), tmpvalues[0]) != 0) {
        
        LogTrace("L1GlobalTriggerConfig") 
            << "    Error getting charge correlation from muon condition (" 
            << name << ")" << std::endl;
        return -1;
    }
    conditionparameter.charge_correlation = tmpvalues[0];

    // get mip and iso bits

    bool tmpiso[4];     // temporary storage for mip and iso bits    
    bool tmpmip[4];     // TODO: macro instead of 4

    if (getMuonMipIsoBits(node, numtype, tmpmip, tmpiso) != 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "    Could not get mip and iso bits from muon condition (" 
            << name << ")" << std::endl;
        return -1;
    }

    for (int i = 0; i < numtype; i++) {
        particleparameter[i].en_mip = tmpmip[i];
        particleparameter[i].en_iso = tmpiso[i];
    }

    // indicates if a correlation is used
    bool wsc = ( type == xml_condition_attr_type_2wsc );

    if (wsc) {
        // get deltaeta
        if (getConditionChildValues(node, xml_deltaeta_tag, 1, tmpvalues) != 0) {
            return -1;
        }
        
        conditionparameter.delta_eta = tmpvalues[0];
        
        // deltaphi is larger than 64bit
        if (getXMLHexTextValue128(findXMLChild(node->getFirstChild(), 
            xml_deltaphi_tag), tmpvalues[0], tmpvalues[1]) != 0) {
            edm::LogError("L1GlobalTriggerConfig") 
                << "    Could not get delta_phi for muon condition with wsc (" << name << ")" 
                << std::endl;
            return -1;
        }
        
        conditionparameter.delta_phil = tmpvalues[0];
        conditionparameter.delta_phih = tmpvalues[1];

        // get maxbits for deltaeta

        unsigned int maxbits;     // maximal bit counts for wsc values        
        if (countConditionChildMaxBits(node, xml_deltaeta_tag, maxbits) != 0) {
            return -1;
        }
        
        conditionparameter.delta_eta_maxbits = maxbits;

        // get maxbits for deltaphi
        if (countConditionChildMaxBits(node, xml_deltaphi_tag, maxbits) != 0) {
            return -1;
        }
        
        conditionparameter.delta_phi_maxbits = maxbits;
    }
        
    // now create a new MuonCondition

    L1GlobalTriggerMuonTemplate* muoncond;
    muoncond = new L1GlobalTriggerMuonTemplate(*m_GT, name);

    muoncond->setConditionParameter(numtype, particleparameter, &conditionparameter, wsc);
    muoncond->setGeEq(ge_eq);

    // enter it to the map
    if (insertIntoConditionsMap(muoncond, chipNr) != 0) {
        delete muoncond;
        muoncond = 0;
        edm::LogError("L1GlobalTriggerConfig") 
            << "    Error: duplicate condition (" << name << ")" 
            << std::endl;
        return -1;
    }
    
    return 0; 
}

/**
 * parseCalo Parse a calo-condition.
 *
 * @param node The corresponding node.
 * @param name The condition name.
 * @param chipNr The number of the chip this condition is located. 
 *
 * @return 0 if succeeded. -1 on Error.
 *
 */

int L1GlobalTriggerConfig::parseCalo(DOMNode* node, 
    const std::string& name, unsigned int chipNr = 0 ) {

    // temporary storage of the parameters
    L1GlobalTriggerCaloTemplate::ParticleParameter particleparameter[4];
    L1GlobalTriggerCaloTemplate::ConditionParameter conditionparameter;
    
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, xml_condition_attr_condition);
    std::string particle = getXMLAttribute(node, xml_condition_attr_particle);
    std::string type = getXMLAttribute(node, xml_condition_attr_type);

    // determine particle type
    L1GlobalTriggerCaloTemplate::ParticleType particletype;

    if (particle == xml_condition_attr_particle_eg) {
        particletype = L1GlobalTriggerCaloTemplate::EG;
    } else if (particle == xml_condition_attr_particle_ieg) {
        particletype = L1GlobalTriggerCaloTemplate::IEG;
    } else if (particle == xml_condition_attr_particle_jet) {
        particletype = L1GlobalTriggerCaloTemplate::CJET;
    } else if (particle == xml_condition_attr_particle_tau) {
        particletype = L1GlobalTriggerCaloTemplate::TJET;
    } else if (particle == xml_condition_attr_particle_fwdjet) {
        particletype = L1GlobalTriggerCaloTemplate::FJET;
    } else {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Wrong particle for calo-condition (" << particle << ")" 
            << std::endl;
        return -1;
    }

    int numtype = getNumFromType(type); // the number of particles
    if (numtype < 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Unknown type for calo-condition (" << type << ")" 
            << std::endl;
        return -1;
    }

    if (numtype > 4) numtype = 4; // TODO: macro instead of 4

    // get greater equal flag
    
    // temporary integer for greater equal flag
    int ge_eqint = getGeEqFlag(node, xml_etthreshold_tag);
    if (ge_eqint < 0) {
        edm::LogError("L1GlobalTriggerConfig") << "Error getting greater equal flag" << std::endl;
        return -1;
    }
    // set the boolean value for the ge_eq mode
    bool ge_eq = ( ge_eqint != 0);

    // get values

    u_int64_t tmpvalues[4]; // temporary storage of values
    
    // get et_thresholds
    if (getConditionChildValues(node, xml_etthreshold_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        particleparameter[i].et_threshold = tmpvalues[i];
    }

    // get eta_thresholds
    if (getConditionChildValues(node, xml_eta_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        particleparameter[i].eta = tmpvalues[i];
    }

    // get phi
    if (getConditionChildValues(node, xml_phi_tag, numtype, tmpvalues) != 0) {
        return -1;
    }
    // fill into structure
    for (int i = 0; i < numtype; i++) {
        particleparameter[i].phi = tmpvalues[i];
    }

    // indicates if a correlation is used
    bool wsc = ( type == xml_condition_attr_type_2wsc );

    if (wsc) {
        if (getConditionChildValues(node, xml_deltaeta_tag, 1, tmpvalues) != 0) {
            return -1;
        }
        
        conditionparameter.delta_eta = tmpvalues[0];        
        LogTrace("L1GlobalTriggerConfig") 
            << "      delta eta calo = " << tmpvalues[0] 
            << std::endl;
        
        if (getConditionChildValues(node, xml_deltaphi_tag, 1, tmpvalues) != 0) {
            return -1;
        }
        
        conditionparameter.delta_phi = tmpvalues[0];
        LogTrace("L1GlobalTriggerConfig") 
            << "      delta phi calo = " << tmpvalues[0] 
            << std::endl;

        // get maxbits for deltaeta

        LogTrace("L1GlobalTriggerConfig") 
            << "      Counting delta_eta_maxbits" 
            << std::endl; 
        unsigned int maxbits;     // maximal bit counts for wsc values        
        if (countConditionChildMaxBits(node, xml_deltaeta_tag, maxbits) != 0) {
            return -1;
        }
        
        conditionparameter.delta_eta_maxbits = maxbits;

        // get maxbits for deltaphi
        LogTrace("L1GlobalTriggerConfig") 
            << "      Counting delta_phi_maxbits" 
            << std::endl; 
        if (countConditionChildMaxBits(node, xml_deltaphi_tag, maxbits) != 0) {
            return -1;
        }
        
        conditionparameter.delta_phi_maxbits = maxbits;

    }

    // now create a new caloCondition

    L1GlobalTriggerCaloTemplate* calocond;
    calocond = new L1GlobalTriggerCaloTemplate(*m_GT, name);

    calocond->setConditionParameter(numtype, particleparameter, 
        &conditionparameter, particletype, wsc);
    calocond->setGeEq(ge_eq);

    // enter it to the map
    if (insertIntoConditionsMap(calocond, chipNr) != 0) {
        delete calocond;
        calocond = 0;
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error: duplicate condition (" << name << ")" 
            << std::endl;
        return -1;
    }
    
    return 0; 
}

/**
 * parseESums - Parse a Esums-condition .
 *
 * @param node The corresponding node.
 * @param name The condition name.
 * @param chipNr The number of the chip this condition is located on.
 *
 * @return 0 if succeeded. -1 on Error
 *
 */


int L1GlobalTriggerConfig::parseESums(DOMNode* node, 
    const std::string& name, unsigned int chipNr = 0) {

    // temporary storage of the parameters
    L1GlobalTriggerEsumsTemplate::ConditionParameter conditionparameter;
    
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, xml_condition_attr_condition);
    std::string particle = getXMLAttribute(node, xml_condition_attr_particle);
    std::string type = getXMLAttribute(node, xml_condition_attr_type);

    L1GlobalTriggerEsumsTemplate::SumType sumtype;

    if (particle == xml_condition_attr_particle_etm && 
        type == xml_condition_attr_particle_etm) {

        sumtype = L1GlobalTriggerEsumsTemplate::ETM;

    } else if (particle == xml_condition_attr_particle_ett &&
         type == xml_condition_attr_particle_ett) {
        
        sumtype = L1GlobalTriggerEsumsTemplate::ETT;
        
    } else if (particle == xml_condition_attr_particle_htt &&
         type == xml_condition_attr_particle_htt) {
        
        sumtype = L1GlobalTriggerEsumsTemplate::HTT;
        
    } else {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Wrong particle or type for Esums-condition (" 
            << particle << ", " << type << ")" 
            << std::endl;
        return -1;
    }

    // get greater equal flag
    int ge_eqint = getGeEqFlag(node,xml_etthreshold_tag);
    if (ge_eqint < 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error getting greater equal flag" 
            << std::endl;
        return -1;
    }
    // set the boolean value
    bool ge_eq = ( ge_eqint != 0);

    // get et_thresholds TODO &tmpvalue here?
    u_int64_t tmpvalue; 
    if (getConditionChildValues(node, xml_etthreshold_tag, 1, &tmpvalue) != 0) { 
        return -1;
    }
    // fill into structure
    conditionparameter.et_threshold = tmpvalue;
    
    // for etm read phi value
    if (sumtype == L1GlobalTriggerEsumsTemplate::ETM) {
         if (getConditionChildValues(node, xml_phi_tag, 1, &tmpvalue) != 0) {
            return -1;
         }
         
         // fill into structure
         conditionparameter.phi = tmpvalue;
    }


    // get the en_overflow flag
    DOMNode *n1;
    if ( (n1 = findXMLChild(node->getFirstChild(), xml_etthreshold_tag)) == 0) {
        return -1;
    }
    if ( (n1 = findXMLChild(n1->getFirstChild(), xml_enoverflow_tag)) == 0) {
        return -1;
    }

    int tmpint = getBitFromNode(n1);
    if (tmpint == 0) {
        conditionparameter.en_overflow = false;
    } else if (tmpint == 1) {
        conditionparameter.en_overflow = true;
    } else {
        return -1;
    }
        
    // now create a new EsumsCondition

    L1GlobalTriggerEsumsTemplate* esumscond;
    esumscond = new L1GlobalTriggerEsumsTemplate(*m_GT, name);

    esumscond->setConditionParameter(&conditionparameter, sumtype);
    esumscond->setGeEq(ge_eq);

    // enter it to the map
    if (insertIntoConditionsMap(esumscond, chipNr) != 0) {
        delete esumscond;
        esumscond = 0;
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error duplicate condition (" << name << ")" 
            << std::endl;
        return -1;
    }
    
    return 0; 

}

/**
 * parseJetCounts Parse a JetCounts-condition and insert an entry to the conditions map.
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located on.
 *
 * @return 0 if suceeded. -1 if an error occured.
 *
 */

int L1GlobalTriggerConfig::parseJetCounts(DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    // temporary storage of the parameters
    L1GlobalTriggerJetCountsTemplate::ConditionParameter conditionparameter;
    
    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, xml_condition_attr_condition);
    std::string particle = getXMLAttribute(node, xml_condition_attr_particle);
    std::string type = getXMLAttribute(node, xml_condition_attr_type);

    // pointer to detect an error while conversing type to integer
    // TODO check here
    char *endptr;       
    endptr = (char*) type.c_str();
    long int typeint = strtol(type.c_str(), &endptr, 10);

    if (*endptr != 0) { // conversion error
        return -1;
    }

    // maximum integer value of type TODO keep it here hardcoded?
    static const int maxtype = 11;  

    if (typeint < 0 || typeint > maxtype) { // out of range
        return -1;
    }

    conditionparameter.type = (unsigned int) typeint;

    // get greater equal flag
    int ge_eqint = getGeEqFlag(node,xml_etthreshold_tag);
    if (ge_eqint < 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error getting greater equal flag" 
            << std::endl;
        return -1;
    }
    // set the boolean value
    bool ge_eq = ( ge_eqint != 0);

    // get et_thresholds
    u_int64_t tmpvalue; 
    if (getConditionChildValues(node, xml_etthreshold_tag, 1, &tmpvalue) != 0) {
        return -1;
    }
    // fill into structure
    conditionparameter.et_threshold = tmpvalue;

    // now create a new JetCountsCondition

    L1GlobalTriggerJetCountsTemplate* jetcntscond;
    jetcntscond = new L1GlobalTriggerJetCountsTemplate(*m_GT, name);

    jetcntscond->setConditionParameter(&conditionparameter);
    jetcntscond->setGeEq(ge_eq);

    // enter it to the map
    if (insertIntoConditionsMap(jetcntscond, chipNr) != 0) {
        delete jetcntscond;
        jetcntscond = 0;
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error duplicate condition (" << name << ")" 
            << std::endl;
        return -1;
    }
    
    return 0; 

}
    
        
/**
 * workCondition - call the apropiate function to parse this condition.
 *
 * @param node The corresponding node to the condition.
 * @param name The name of the condition.
 * @param chipNr The number of the chip the condition is located on.
 *
 * @return 0 on success, -1 if an error occured.
 *
 */

int L1GlobalTriggerConfig::workCondition(
    DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    // get condition, particle name and type name
    std::string condition = getXMLAttribute(node, xml_condition_attr_condition);
    std::string particle = getXMLAttribute(node, xml_condition_attr_particle);
    std::string type = getXMLAttribute(node, xml_condition_attr_type);

    if (condition == "" || particle == "" || type == "") {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Missing attributes for condition " 
            << name << std::endl;
        return -1;
    }

    LogTrace("L1GlobalTriggerConfig") 
        << "    condition: " << condition << ", particle: " << particle 
        << ", type: " << type << std::endl;

    // call the appropiate function for this condition

    if (condition == xml_condition_attr_condition_muon) {
        return parseMuon(node, name, chipNr);
    } else if (condition == xml_condition_attr_condition_calo) {
        return parseCalo(node, name, chipNr);
    } else if (condition == xml_condition_attr_condition_esums) {
        return parseESums(node, name, chipNr);
    } else if (condition == xml_condition_attr_condition_jetcnts) {
        return parseJetCounts(node, name, chipNr);
    } else {
        edm::LogError("L1GlobalTriggerConfig") << "Unknown condition (" << condition << ")" << std::endl;
        return -1;
    }
     
    return 0;

}

/**
 * parseConditions - look for conditions and call the workCondition 
 *     function for each node
 *
 * @param parser The parser to parse the XML file with.
 *
 * @return 0 if succeeded. -1 if an error occured.
 *
 */

    
int L1GlobalTriggerConfig::parseConditions(XercesDOMParser *parser) {

 
    LogTrace("L1GlobalTriggerConfig") << "\nParsing conditions" << std::endl;

    DOMNode *doc = parser->getDocument();
    DOMNode *n1 = doc->getFirstChild();

    // we assume that the first child is "def" because it was checked in workXML

    // TODO def tag
    DOMNode *chipNode = n1->getFirstChild();
    if (chipNode == 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error: No child of <def> found" 
            << std::endl;
        return -1;
    }
    
    // find chip

    std::string chipName;        // name of the actual chip
    chipNode = findXMLChild(chipNode, xml_chip_tag, true, &chipName);
    if (chipNode == 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "  Error: Could not find <" << xml_chip_tag 
            << std::endl;
        return -1;
    }

    unsigned int chipNr = 0;    
    do {    

        // find conditions
        DOMNode *conditionsNode = chipNode->getFirstChild();
        conditionsNode = findXMLChild(conditionsNode, xml_conditions_tag);
        if (conditionsNode == 0) {
            edm::LogError("L1GlobalTriggerConfig") 
                << "Error: No <" << xml_conditions_tag << "> child found in Chip " 
                << chipName << std::endl;
            return -1;
        }             

        char* nodeName = XMLString::transcode(chipNode->getNodeName());
        LogTrace("L1GlobalTriggerConfig") 
            << "\n  Found Chip: " << nodeName << " Name: " << chipName 
            << std::endl;
        XMLString::release(&nodeName);
    
        // walk through conditions
        DOMNode *conditionNameNode = conditionsNode->getFirstChild();
        std::string conditionNameNodeName;
        conditionNameNode = findXMLChild(conditionNameNode, "", true, &conditionNameNodeName);
        while (conditionNameNode != 0) {
            LogTrace("L1GlobalTriggerConfig") 
                << "\n    Found a condition with name: "<< conditionNameNodeName 
                << std::endl;
            if (workCondition(conditionNameNode, conditionNameNodeName, chipNr) != 0) {
                return -1;
            }
            conditionNameNode = findXMLChild(conditionNameNode->getNextSibling(), "",
                true, &conditionNameNodeName);
    
        }
        // next chip
        chipNode = findXMLChild(chipNode->getNextSibling(), xml_chip_tag, true, &chipName);
        chipNr++;
        
    } while (chipNode != 0 && chipNr < NumberConditionChips);

    return 0;
}

/**
 * parsePreAlgos Parse all prealgos in all chips and insert them in 
 *     the prealgomap.
 *
 * @param node The def node.
 *
 * @return 0 if succeeded, -1 if an error occured.
 */

int L1GlobalTriggerConfig::parsePreAlgos(DOMNode *node) {

    DOMNode *chipNode = node->getFirstChild();
    if (chipNode == 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "  Error: No child of <def> found" 
            << std::endl;
        return -1;
    }
    
    // find first chip
    std::string chipName;   // the name of a chip

    chipNode = findXMLChild(chipNode, xml_chip_tag, true, &chipName);
    if (chipNode == 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "  Error: Could not find <" << xml_chip_tag 
            << std::endl;
        return -1;
    }

    unsigned int chipNr = 0;    
    do {    

        LogTrace("L1GlobalTriggerConfig") << std::endl;

        std::string nodeName = xml_chip_tag + chipName;
        LogTrace("L1GlobalTriggerConfig") 
            << "  Found Chip: " << nodeName << " Name: " << chipName 
            << std::endl;
 
        // find prealgos
        DOMNode *prealgosNode = chipNode->getFirstChild();
        prealgosNode = findXMLChild(prealgosNode, xml_prealgos_tag);
        if (prealgosNode == 0) {
            edm::LogError("L1GlobalTriggerConfig") 
                << "    Error: No <" << xml_prealgos_tag << "> child found in chip " << chipName 
                << std::endl;
            return -1;
        }
    
        // walk through conditions
        DOMNode *prealgoNameNode = prealgosNode->getFirstChild();
        std::string prealgoNameNodeName;
        prealgoNameNode = findXMLChild(prealgoNameNode, "", true, &prealgoNameNodeName);
        while (prealgoNameNode != 0) {
            LogTrace("L1GlobalTriggerConfig") 
                << "    Found a prealgo with name: " << prealgoNameNodeName 
                << std::endl;
            if (insertAlgoIntoMap(prealgoNameNode, prealgoNameNodeName, 
                &prealgosmap, &(conditionsmap[chipNr]), chipNr) != 0) {
                    return -1;
            }
            prealgoNameNode = findXMLChild(prealgoNameNode->getNextSibling(), "", true, 
                &prealgoNameNodeName);
    
        }
        // next chip
        chipNode = findXMLChild(chipNode->getNextSibling(), xml_chip_tag, true, &chipName);
        chipNr++;
        
    } while (chipNode != 0 && chipNr < NumberConditionChips);

    return 0;
}

/**
 * parseAlgos Parse all algos.
 *
 * @param node The def node.
 *
 * @return 0 if succeeded, -1 if an error occured.
 *
 */

int L1GlobalTriggerConfig::parseAlgos(DOMNode* node)
{

    LogTrace("L1GlobalTriggerConfig") << std::endl;
    
    DOMNode *algosNode = node->getFirstChild();
    if (algosNode == 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error: No child of <def> found" 
            << std::endl;
        return -1;
    }
    
    // find algos tag 
    algosNode=findXMLChild(algosNode, xml_algos_tag);
    if (algosNode == 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error: Could not find <" << xml_algos_tag << ">" 
            << std::endl;
        return -1;
    }

    DOMNode *algoNameNode = algosNode->getFirstChild();
    std::string algoNameNodeName;

    // walk through all nodes
    unsigned int algoNr = 0;

    algoNameNode = findXMLChild(algoNameNode, "", true, &algoNameNodeName);
    while (algoNameNode != 0) {
        algoNr++;
        std::ostringstream algoNrOss;
        algoNrOss << algoNr;          // convert algoNr to string
        LogTrace("L1GlobalTriggerConfig") 
            << "    Found an algo with name: "<< algoNameNodeName 
            << std::endl;
        
        // add a number to make the names unique
        algoNameNodeName = algoNameNodeName + " " + algoNrOss.str();
        unsigned int dummyChipNr = 999; 
        if (insertAlgoIntoMap(algoNameNode, algoNameNodeName, 
            &algosmap, &prealgosmap, dummyChipNr) != 0) {
                return -1;
        }
        algoNameNode = findXMLChild(algoNameNode->getNextSibling(), "", true, &algoNameNodeName);
    }
    
    return 0;
 
}

/*
 * parseAllAlgos Parse the prealgos and algos
 * 
 * @param parser A reference to the XercesDOMParser to use.
 * 
 * @return 0 if succeeded, -1 if an error occured.
 *
 */


int L1GlobalTriggerConfig::parseAllAlgos(XercesDOMParser* parser) {

    LogTrace("L1GlobalTriggerConfig") << "\nParsing prealgos and algos" << std::endl;

    DOMNode *doc = parser->getDocument();
    DOMNode *n1 = doc->getFirstChild();

    // we assume that the first child is "def" because it was checked in workXML
     
    // parse algos and prealgos
    if (parsePreAlgos(n1) != 0) return -1;  
    if (p_xmlfileversion == VERSION_PROTOTYPE) {    //algos only for prototype version
        if (parseAlgos(n1) != 0 ) return -1;
    }

    return 0;
}

/**
 * checkVersion Try to find out the version of the xml file by checking 
 *     the 1st chip connections
 *
 * @param parser A reference to the XercesDOMParser to use.
 *
 * @return 0 if succeeded, -1 if an error occured.
 *
 */

int L1GlobalTriggerConfig::checkVersion(XercesDOMParser* parser) {
    
    LogTrace("L1GlobalTriggerConfig") << "\nChecking XML-file version....." << std::endl;

    DOMNode *doc = parser->getDocument();
    DOMNode *n1 = doc->getFirstChild();

    // we assume that the first child is "def" because it was checked in workXML

    // find chip_def tag
    n1 = findXMLChild(n1->getFirstChild(), xml_chipdef_tag);
    if (n1 == 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error: Could not find <" << xml_chipdef_tag << "> tag." 
            << std::endl;
        return -1;
    }

    // find chip1 tag
    n1 = findXMLChild(n1->getFirstChild(), xml_chip1_tag);
    if (n1 == 0) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error: Could not find <" << xml_chip1_tag << "> tag." 
            << std::endl;
        return -1;
    }
    
    // we check if there are more than 4 ca tags

    // n1 is the first child in the chip (usually a ca-tag)
    n1 = n1->getFirstChild();
    // TODO remove hardwired ca tags
    for (int i=0; i < 4; i++) {
        n1 = findXMLChild(n1, xml_ca_tag, true);
        if (n1 == 0) {
            edm::LogError("L1GlobalTriggerConfig") 
                << "Error: Too less <ca >-tags" 
                << std::endl;
        }
        n1 = n1->getNextSibling();
    }

    // if there is a 5th ca tag the version is the final version
    n1 = findXMLChild(n1, xml_ca_tag, true);
    if (n1 == 0) {
        p_xmlfileversion = VERSION_PROTOTYPE;
        LogTrace("L1GlobalTriggerConfig") 
            << "  XML-File is prototype version (6U)" 
            << std::endl;
    } else {
        p_xmlfileversion = VERSION_FINAL;
        LogTrace("L1GlobalTriggerConfig") 
            << "  XML-File is final version (9U)" 
            << std::endl;
    }

    return 0;
}

/**
 * workXML Do all what need to be done on the XML-File
 *
 * @param parser The parser to use for parsing the XML-File
 *
 * @return 0 if succeeded, -1 if an error occured.
 */    
    

int L1GlobalTriggerConfig::workXML(XercesDOMParser *parser) {

    DOMDocument *doc = parser->getDocument();
    DOMNode *n1 = doc->getFirstChild();

    if (n1 == 0) {
        edm::LogError("L1GlobalTriggerConfig") << "Error: Found no XML child" 
            << std::endl;
        return -1;
    }

    char* nodeName = XMLString::transcode(n1->getNodeName());
    // TODO def as static std::string
    if (XMLString::compareIString(nodeName, "def")) {
        edm::LogError("L1GlobalTriggerConfig") 
            << "Error: First XML child is not \"def\"" 
            << std::endl;
        return -1;
    }

    LogTrace("L1GlobalTriggerConfig") 
        << "\nFirst node name is: " << nodeName 
        << std::endl;
    XMLString::release(&nodeName);

    // clear possible old conditions
    clearConditionsMap();

    if (checkVersion(parser) != 0) {
        return -1;
    }
    
    if (parseConditions(parser) != 0) {
        clearConditionsMap();
        return -1;
    }

    if (parseAllAlgos(parser) != 0) {
        clearConditionsMap();
        return -1;
    }

    return 0;

}
            
/**
 * printThresholds - print all conditions stored in the conditionsmap    
 *
 */

void L1GlobalTriggerConfig::printThresholds(std::ostream& myCout) {


    for (unsigned int i = 0; i < NumberConditionChips; i++) {
        myCout << "\n-------Chip " << i+1 << " --------" << std::endl;
        // print all
        for (ConditionsMap::iterator it = conditionsmap[i].begin(); 
            it != conditionsmap[i].end(); it++) {
            
            it->second->printThresholds(myCout);
        }
    }

    myCout << "\n-------- prealgos --------" << std::endl;
    for (ConditionsMap::iterator it = prealgosmap.begin(); 
        it != prealgosmap.end(); it++) {
            
            it->second->printThresholds(myCout);
    }

    myCout << "\n-------- algos --------" << std::endl;
    for (ConditionsMap::iterator it = algosmap.begin(); 
        it != algosmap.end(); it++) {
        
            it->second->printThresholds(myCout);
    }


}

/**
 * parseVmeXML parse a xml file for vme bus preamble specification, 
 *     write it to a file and store the time
 *
 * @param parser The parser to use for parsing the file.
 *
 * @return 0 if succeeded, -1 if an error occured.
 *
 */

int L1GlobalTriggerConfig::parseVmeXML(XercesDOMParser *parser) {        

    // simply search for adress tags within the chips and write them to the file

    DOMDocument *doc = parser->getDocument();
    DOMNode *n1 = doc->getFirstChild();

    if (n1 == 0) {
        edm::LogError("L1GlobalTriggerConfig") << "Error: Found no XML child" << std::endl;
        return -1;
    }

    // find "vme"-tag
    n1 = findXMLChild(n1, vmexml_vme_tag);
    if (n1 == 0) {
        edm::LogError("L1GlobalTriggerConfig") << "Error: No vme tag found." << std::endl;
        return -1;
    }
    n1 = n1->getFirstChild();

    // open the file
    std::ofstream ofs(p_vmePreambleFileName);
    // reset the time
    p_vmePreambleTime = 0.0;

    unsigned int chipCounter = 0; // count chips
         
    while (chipCounter < NumberConditionChips) {
        DOMNode *particlenode;	// node for a particle
        DOMNode *walknode;		// node for walking through a particle
        DOMNode *addressnode;	// an adress node
        
        n1 = findXMLChild(n1, vmexml_condchip_tag, true);
        if (n1 == 0) break;	// just break if no more chips found
        
        particlenode = n1->getFirstChild();
        while ((particlenode = findXMLChild(particlenode, "")) != 0) {            
            // check if muon
            if (getXMLAttribute(particlenode, vmexml_attr_particle) == vmexml_attr_particle_muon) {
                walknode = particlenode->getFirstChild();
                while ((walknode = findXMLChild(walknode, "")) != 0) {
                    addressnode=walknode->getFirstChild();
                    while ((addressnode = findXMLChild(addressnode, vmexml_address_tag)) != 0) {
                        // LogTrace("L1GlobalTriggerConfig") << getXMLTextValue(addressnode);
                        addVmeAddress(addressnode, ofs);
                        addressnode = addressnode->getNextSibling();
                    }
                    walknode = walknode->getNextSibling();
                }
            } else { // other particles than muon just contain adress nodes
                addressnode = particlenode->getFirstChild();
                while ((addressnode = findXMLChild(addressnode, vmexml_address_tag)) != 0) {
                    // LogTrace("L1GlobalTriggerConfig") << getXMLTextValue(addressnode);
                    addVmeAddress(addressnode, ofs);
                    addressnode = addressnode->getNextSibling();
                }
            }
            particlenode = particlenode->getNextSibling();
        } // end while particle

        n1 = n1->getNextSibling();
        chipCounter++;
    } // end while chipCounter

    return 0;

}

/**
 * addVmeAddress add 2 lines to the preamble and increase the preamble time
 *
 * @param node The node to add to the preamble.
 * @param ofs The filestream for writing out the lines.
 * @return 0 if succeeded, -1 if an error occured.
 *
 */

int L1GlobalTriggerConfig::addVmeAddress(DOMNode *node, std::ofstream& ofs) {
            
    std::string addrsrc = getXMLTextValue(node); // source string for the address
    std::string binarynumbers = "01";

    unsigned int startpos = addrsrc.find_first_of(binarynumbers); // start position of the
    unsigned int endpos = addrsrc.find_first_not_of(binarynumbers,startpos); // end position

    if (startpos == endpos) { // TODO write a better message
        edm::LogError("L1GlobalTriggerConfig") << "Error: No address value found." << std::endl;
        return -1; 
    }

    if (startpos < endpos-1) {
        endpos = endpos-1; // the last digit is ignored
    }

    addrsrc = addrsrc.substr(startpos, endpos - startpos);
    char* endptr = (char *) addrsrc.c_str(); // end pointer for conversion

    unsigned long int address = strtoul(addrsrc.c_str(), &endptr, 2); // integer value of address

    if (*endptr != 0) {
        edm::LogError("L1GlobalTriggerConfig") << "Error converting binary address." << std::endl;
        return -1;
    }

    // we got the address, lets go for the value
    DOMNode* valuenode;

    valuenode = findXMLChild(node->getFirstChild(), vmexml_value_tag);
    if (valuenode == 0) {
        edm::LogError("L1GlobalTriggerConfig") << "Found no value node for address." << hex << address 
            << dec << std::endl;
        return -1;
    }

    std::string valuesrc = getXMLTextValue(valuenode); // source string for the value 

    startpos = valuesrc.find_first_of(binarynumbers);
    endpos = valuesrc.find_first_not_of(binarynumbers, startpos);

    if (startpos == endpos) {
        edm::LogError("L1GlobalTriggerConfig") << "Error: No binary value found." << std::endl;
        return -1;
    }

    valuesrc = valuesrc.substr(startpos , endpos - startpos);
    endptr = (char*) valuesrc.c_str();
    unsigned long int value = strtoul(valuesrc.c_str(), &endptr, 2); // integer value of value
    if (*endptr != 0) {
        edm::LogError("L1GlobalTriggerConfig") << "Error converting binary value." << std::endl;
        return -1;
    }

    writeVmeLine(1, address, value, ofs);
    writeVmeLine(0, address, value, ofs);

    return 0;

}

/**
 * writeVmeLine Write a line of the vme bus preamble to the output file
 *
 * @param clkcond
 * @param address The address to be written.
 * @param value The value to be written.
 * @param ofs The output stream where the line is written to.
 *
 */


void L1GlobalTriggerConfig::writeVmeLine(unsigned int clkcond, 
    unsigned long int address, unsigned int value, std::ofstream& ofs) {

    ofs << " "; // begin with a space
    ofs << fixed << setprecision(1) << setw(5); // 1 digit after dot for Time
    ofs << dec << p_vmePreambleTime;
    ofs << "> ";
    ofs << setw(1); // width 1 for clkcond
    ofs << clkcond << " ";
    ofs << setw(6); // width 6 for address
    ofs << setfill('0'); // leading zeros
    ofs << hex << uppercase << address << " ";	// switch to hexadecimal uppercase and write address
    ofs << setw(2); // width 2 for value
    ofs << setfill(' '); // no leading zeros for value
    ofs << value << dec << nouppercase; 
    ofs << p_vmePreambleLineRest; // write the rest
    ofs << std::endl; // end of line

    p_vmePreambleTime += p_vmePreambleTimeTick;
}
    
        

// static class members

// correspondence "condition chip - GTL algorithm word" in the hardware
// chip 2: 0 - 95;  chip 1: 96 - 128 (191)
const int L1GlobalTriggerConfig::OrderConditionChip[L1GlobalTriggerConfig::NumberConditionChips] = {2, 1};      

const std::string L1GlobalTriggerConfig::xml_def_tag("def");
const std::string L1GlobalTriggerConfig::xml_chip_tag("condition_chip_");
const std::string L1GlobalTriggerConfig::xml_conditions_tag("conditions");
const std::string L1GlobalTriggerConfig::xml_prealgos_tag("prealgos");
const std::string L1GlobalTriggerConfig::xml_algos_tag("algos");

const std::string L1GlobalTriggerConfig::xml_condition_attr_condition("condition");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle("particle");
const std::string L1GlobalTriggerConfig::xml_condition_attr_type("type");
const std::string L1GlobalTriggerConfig::xml_condition_attr_condition_muon("muon");
const std::string L1GlobalTriggerConfig::xml_condition_attr_condition_calo("calo");
const std::string L1GlobalTriggerConfig::xml_condition_attr_condition_esums("esums");
const std::string L1GlobalTriggerConfig::xml_condition_attr_condition_jetcnts("jet_cnts");

const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_muon("muon");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_eg("eg");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_ieg("ieg");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_jet("jet");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_fwdjet("fwdjet");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_tau("tau");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_etm("etm");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_ett("ett");
const std::string L1GlobalTriggerConfig::xml_condition_attr_particle_htt("htt");



const std::string L1GlobalTriggerConfig::xml_condition_attr_type_1("1_s");
const std::string L1GlobalTriggerConfig::xml_condition_attr_type_2("2_s");
const std::string L1GlobalTriggerConfig::xml_condition_attr_type_2wsc("2_wsc");
const std::string L1GlobalTriggerConfig::xml_condition_attr_type_3("3");
const std::string L1GlobalTriggerConfig::xml_condition_attr_type_4("4");


const std::string L1GlobalTriggerConfig::xml_attr_mode("mode");
const std::string L1GlobalTriggerConfig::xml_attr_mode_bit("bit");
const std::string L1GlobalTriggerConfig::xml_attr_max("max");

const std::string L1GlobalTriggerConfig::xml_attr_nr("nr");
const std::string L1GlobalTriggerConfig::xml_attr_pin("pin");
const std::string L1GlobalTriggerConfig::xml_attr_pin_a("a");

const std::string L1GlobalTriggerConfig::xml_etthreshold_tag("et_threshold");

const std::string L1GlobalTriggerConfig::xml_pththreshold_tag("pt_h_threshold");
const std::string L1GlobalTriggerConfig::xml_ptlthreshold_tag("pt_l_threshold");
const std::string L1GlobalTriggerConfig::xml_quality_tag("quality");
const std::string L1GlobalTriggerConfig::xml_eta_tag("eta");
const std::string L1GlobalTriggerConfig::xml_phi_tag("phi");
const std::string L1GlobalTriggerConfig::xml_phih_tag("phi_h");
const std::string L1GlobalTriggerConfig::xml_phil_tag("phi_l");
const std::string L1GlobalTriggerConfig::xml_chargecorrelation_tag("charge_correlation");
const std::string L1GlobalTriggerConfig::xml_enmip_tag("en_mip");
const std::string L1GlobalTriggerConfig::xml_eniso_tag("en_iso");
const std::string L1GlobalTriggerConfig::xml_enoverflow_tag("en_overflow");
const std::string L1GlobalTriggerConfig::xml_deltaeta_tag("delta_eta");
const std::string L1GlobalTriggerConfig::xml_deltaphi_tag("delta_phi");

const std::string L1GlobalTriggerConfig::xml_output_tag("output");
const std::string L1GlobalTriggerConfig::xml_outputpin_tag("output_pin");

const std::string L1GlobalTriggerConfig::xml_geeq_tag("ge_eq");
const std::string L1GlobalTriggerConfig::xml_value_tag("value");

const std::string L1GlobalTriggerConfig::xml_chipdef_tag("chip_def");
const std::string L1GlobalTriggerConfig::xml_chip1_tag("chip_1");
const std::string L1GlobalTriggerConfig::xml_ca_tag("ca");

//vmexml std::strings
const std::string L1GlobalTriggerConfig::vmexml_vme_tag("vme");
const std::string L1GlobalTriggerConfig::vmexml_condchip_tag("cond_chip_");
const std::string L1GlobalTriggerConfig::vmexml_address_tag("address");
const std::string L1GlobalTriggerConfig::vmexml_value_tag("value");

const std::string L1GlobalTriggerConfig::vmexml_attr_particle("particle");
const std::string L1GlobalTriggerConfig::vmexml_attr_particle_muon("muon");


//

const char L1GlobalTriggerConfig::p_vmePreambleFileName[] = "testxxoo3.data";
const double L1GlobalTriggerConfig::p_vmePreambleTimeTick = 12.5;
const char L1GlobalTriggerConfig::p_vmePreambleLineRest[] =
" 3 00000 00000 00000 00000 00000 00000 00000 00000 0000000 0000000 = XXXXXXXXXXXXXX XXXXXXXXXXXXXX";
