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

#include "tmEventSetup/tmEventSetup.hh"
#include "tmEventSetup/esTriggerMenu.hh"
#include "tmEventSetup/esAlgorithm.hh"
#include "tmEventSetup/esCondition.hh"
#include "tmEventSetup/esObject.hh"
#include "tmEventSetup/esCut.hh"
#include "tmEventSetup/esScale.hh"
#include "tmGrammar/Algorithm.hh"



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
/*
void l1t::TriggerMenuXmlParser::setGtNumberTechTriggers(
        const unsigned int& numberTechTriggersValue) {

    m_numberTechTriggers = numberTechTriggersValue;

}
*/


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
        const std::vector<std::vector<EnergySumTemplate> >& vecEnergySumTempl) {

    m_vecEnergySumTemplate = vecEnergySumTempl;
}



void l1t::TriggerMenuXmlParser::setVecExternalTemplate(
        const std::vector<std::vector<ExternalTemplate> >& vecExternalTempl) {

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
        const std::vector<std::vector<EnergySumTemplate> >& corEnergySumTempl) {

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

/*
// set the technical trigger map
void l1t::TriggerMenuXmlParser::setGtTechnicalTriggerMap(const AlgorithmMap& ttMap) {
    m_technicalTriggerMap = ttMap;
}
*/

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


// parse def.xml file
void l1t::TriggerMenuXmlParser::parseXmlFileV2(const std::string& defXmlFile) {


    // resize the vector of condition maps
    // the number of condition chips should be correctly set before calling parseXmlFile
    m_conditionMap.resize(m_numberConditionChips);

    m_vecMuonTemplate.resize(m_numberConditionChips);
    m_vecCaloTemplate.resize(m_numberConditionChips);
    m_vecEnergySumTemplate.resize(m_numberConditionChips);
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

   // LogTrace("TriggerMenuXmlParser") << "\nOpening XML-File V2: \n  " << defXmlFile << std::endl;

  LogDebug("TriggerMenuXmlParser") << "\nOpening XML-File V2: \n  " << defXmlFile << std::endl;
  
  using namespace tmeventsetup;
  using namespace Algorithm;
  
  const esTriggerMenu* menu = tmeventsetup::getTriggerMenu(defXmlFile);


  //get the meta data
  m_triggerMenuDescription = menu->getComment();
  m_triggerMenuDate = menu->getDatetime();
  m_triggerMenuImplementation = menu->getFirmwareUuid(); //BLW: correct descriptor?
  m_triggerMenuName = menu->getName();
  m_triggerMenuInterface = menu->getVersion(); //BLW: correct descriptor?

  const std::map<std::string, esAlgorithm>& algoMap = menu->getAlgorithmMap();
  const std::map<std::string, esCondition>& condMap = menu->getConditionMap();
  const std::map<std::string, esScale>& scaleMap = menu->getScaleMap();

  // parse the scales
  m_gtScales.setScalesName( menu->getScaleSetName() );
  parseScales(scaleMap);


  //loop over the algorithms
  for (std::map<std::string, esAlgorithm>::const_iterator cit = algoMap.begin();
       cit != algoMap.end(); cit++)
  {
    //condition chip (artifact)  TO DO: Update
    int chipNr = 0;
  
    //get algorithm
    const esAlgorithm& algo = cit->second;

    //parse the algorithm
    parseAlgorithmV2(algo,chipNr); //blw

    //get conditions for this algorithm
    const std::vector<std::string>& rpn_vec = algo.getRpnVector();
    for (size_t ii = 0; ii < rpn_vec.size(); ii++)
    {
      const std::string& token = rpn_vec.at(ii);
      if (isGate(token)) continue;
//      long hash = getHash(token);
      const esCondition& condition = condMap.find(token)->second;
     
      //check to see if this condtion already exists
      if ((m_conditionMap[chipNr]).count(condition.getName()) == 0) {
     	  
	  // parse Calo Conditions (EG, Jets, Taus)      
	  if(condition.getType() == esConditionType::SingleEgamma || 
             condition.getType() == esConditionType::DoubleEgamma ||
	     condition.getType() == esConditionType::TripleEgamma ||
	     condition.getType() == esConditionType::QuadEgamma   ||
	     condition.getType() == esConditionType::SingleTau    ||
	     condition.getType() == esConditionType::DoubleTau    ||
	     condition.getType() == esConditionType::TripleTau    ||
	     condition.getType() == esConditionType::QuadTau      ||
	     condition.getType() == esConditionType::SingleJet    ||
	     condition.getType() == esConditionType::DoubleJet    ||
	     condition.getType() == esConditionType::TripleJet    ||
	     condition.getType() == esConditionType::QuadJet      ) 
	  {
             parseCaloV2(condition,chipNr,false); //blw 

	  // parse Energy Sums	 
	  } else if(condition.getType() == esConditionType::TotalEt ||
                    condition.getType() == esConditionType::TotalHt ||
		    condition.getType() == esConditionType::MissingEt ||
		    condition.getType() == esConditionType::MissingHt )
	  {
             parseEnergySumV2(condition,chipNr,false); 	

	  //parse Muons	 	
	  } else if(condition.getType() == esConditionType::SingleMuon    ||
	            condition.getType() == esConditionType::DoubleMuon    ||
	            condition.getType() == esConditionType::TripleMuon    ||
	            condition.getType() == esConditionType::QuadMuon      )       
	  {
             parseMuonV2(condition,chipNr,false);
             
	     
	  //parse Correlation Conditions	 	
	  } else if(condition.getType() == esConditionType::MuonMuonCorrelation    ||
	            condition.getType() == esConditionType::MuonEsumCorrelation    ||
	            condition.getType() == esConditionType::CaloMuonCorrelation    ||
	            condition.getType() == esConditionType::CaloCaloCorrelation    ||
		    condition.getType() == esConditionType::CaloEsumCorrelation    ||
		    condition.getType() == esConditionType::InvariantMass )       
	  {
             parseCorrelationV2(condition,chipNr);

	  //parse Muons	 	
	  } else if(condition.getType() == esConditionType::Externals      )       
	  {
             parseExternalV2(condition,chipNr);
	     	    
	  }      
      
      }//if condition is a new one
    }//loop over conditions
  }//loop over algorithms

  return;


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
    LogTrace("TriggerMenuXmlParser")
    << "    Trying to insert condition \"" << cName << "\" in the condition map." ;

    // no condition name has to appear twice!
    if ((m_conditionMap[chipNr]).count(cName) != 0) {
        LogTrace("TriggerMenuXmlParser") << "      Condition " << cName
            << " already exists - not inserted!" << std::endl;
        return false;
    }

    (m_conditionMap[chipNr])[cName] = &cond;
     LogTrace("TriggerMenuXmlParser")
     << "      OK - condition inserted!"
    << std::endl;


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
/*
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

*/

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
int l1t::TriggerMenuXmlParser::l1t2int( l1t::RelativeBx data ){  //l1t::RelativeBx
  std::stringstream ss;
  ss << data;
  int value;
  ss >> value;
  return value;
}
int l1t::TriggerMenuXmlParser::l1tstr2int( const std::string data ){ 
  std::stringstream ss;
  ss << data;
  int value;
  ss >> value;
  return value;
}


/**
 * parseScales Parse Et, Eta, and Phi Scales
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuXmlParser::parseScales(std::map<std::string, tmeventsetup::esScale> scaleMap) {
	
    using namespace tmeventsetup;
 
//  Setup ScaleParameter to hold information from parsing
    L1TGlobalScales::ScaleParameters muScales; 
    L1TGlobalScales::ScaleParameters egScales; 
    L1TGlobalScales::ScaleParameters tauScales;
    L1TGlobalScales::ScaleParameters jetScales;
    L1TGlobalScales::ScaleParameters ettScales;
    L1TGlobalScales::ScaleParameters etmScales;
    L1TGlobalScales::ScaleParameters httScales;
    L1TGlobalScales::ScaleParameters htmScales; 
 
// Start by parsing the Scale Map
    for (std::map<std::string, esScale>::const_iterator cit = scaleMap.begin();
       cit != scaleMap.end(); cit++)
  {
     const esScale& scale = cit->second;
 
    L1TGlobalScales::ScaleParameters *scaleParam;
    if      (scale.getObjectType() == esObjectType::Muon)   scaleParam = &muScales;
    else if (scale.getObjectType() == esObjectType::Egamma) scaleParam = &egScales;
    else if (scale.getObjectType() == esObjectType::Tau)    scaleParam = &tauScales;
    else if (scale.getObjectType() == esObjectType::Jet)    scaleParam = &jetScales;
    else if (scale.getObjectType() == esObjectType::ETT)    scaleParam = &ettScales;
    else if (scale.getObjectType() == esObjectType::ETM)    scaleParam = &etmScales;
    else if (scale.getObjectType() == esObjectType::HTT)    scaleParam = &httScales;
    else if (scale.getObjectType() == esObjectType::HTM)    scaleParam = &htmScales;
    else scaleParam = 0;
    
    if(scaleParam != 0) {	
        switch(scale.getScaleType()) {
	    case esScaleType::EtScale: {
	        scaleParam->etMin  = scale.getMinimum();
		scaleParam->etMax  = scale.getMaximum();
		scaleParam->etStep = scale.getStep();
		
		//Get bin edges
		const std::vector<esBin> binsV = scale.getBins();
		for(unsigned int i=0; i<binsV.size(); i++) {
		   const esBin& bin = binsV.at(i); 
		   std::pair<double, double> binLimits(bin.minimum, bin.maximum);
		   scaleParam->etBins.push_back(binLimits);
		}
		
		// If this is an energy sum fill dummy values for eta and phi
		// There are no scales for these in the XML so the other case statements will not be seen....do it here.
		if(scale.getObjectType() == esObjectType::ETT || scale.getObjectType() == esObjectType::HTT || 
		   scale.getObjectType() == esObjectType::ETM || scale.getObjectType() == esObjectType::HTM ) {
		   
	           scaleParam->etaMin  = -1.;
		   scaleParam->etaMax  = -1.;
		   scaleParam->etaStep = -1.;		   
		   if(scale.getObjectType() == esObjectType::ETT || scale.getObjectType() == esObjectType::HTT) {
	              scaleParam->phiMin  = -1.;
		      scaleParam->phiMax  = -1.;
		      scaleParam->phiStep = -1.;		   		   
		   }
		}   
	    }
		break;
	    case esScaleType::EtaScale: {
	        scaleParam->etaMin  = scale.getMinimum();
		scaleParam->etaMax  = scale.getMaximum();
		scaleParam->etaStep = scale.getStep();
		
		//Get bin edges
		const std::vector<esBin> binsV = scale.getBins();
		for(unsigned int i=0; i<binsV.size(); i++) {
		   const esBin& bin = binsV.at(i); 
		   std::pair<double, double> binLimits(bin.minimum, bin.maximum);
		   scaleParam->etaBins.push_back(binLimits);
		}
	    }
		break;
	    case esScaleType::PhiScale: {
	        scaleParam->phiMin  = scale.getMinimum();
		scaleParam->phiMax  = scale.getMaximum();
		scaleParam->phiStep = scale.getStep();
		
		//Get bin edges
		const std::vector<esBin> binsV = scale.getBins();
		for(unsigned int i=0; i<binsV.size(); i++) {
		   const esBin& bin = binsV.at(i); 
		   std::pair<double, double> binLimits(bin.minimum, bin.maximum);
		   scaleParam->phiBins.push_back(binLimits);
		}
	    }
		break;				
	    default:
	    
	        break;			
	} //end switch 
    } //end valid scale	
  } //end loop over scaleMap
  
  // put the ScaleParameters into the class
  m_gtScales.setMuonScales(muScales);
  m_gtScales.setEGScales(egScales);
  m_gtScales.setTauScales(tauScales);
  m_gtScales.setJetScales(jetScales);
  m_gtScales.setETTScales(ettScales);
  m_gtScales.setETMScales(etmScales);
  m_gtScales.setHTTScales(httScales);
  m_gtScales.setHTMScales(htmScales);
  
 
    
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

    LogDebug("TriggerMenuXmlParser")
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
        edm::LogError("TriggerMenuXmlParser") << "Unknown type for muon-condition (" << type
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
    for( l1t::MuonObjectRequirements::objectRequirement_const_iterator objPar = condMu.objectRequirements().objectRequirement().begin();
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

	LogDebug("TriggerMenuXmlParser")
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

	LogDebug("TriggerMenuXmlParser")
	  << "\n isolation flag " << cntIso << " = " << flag
	  << "\n isolationLUT = " << isolationLUT 
	  << std::endl;

	cntIso++;
      }

      objParameter[cnt].isolationLUT = isolationLUT;


      int cntEta=0;
      unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
      // Temporary before translation
      for( l1t::MuonObjectRequirement::etaWindow_const_iterator etaWindow =objPar->etaWindow().begin();
	   etaWindow != objPar->etaWindow().end(); ++etaWindow ){
	
	LogDebug("TriggerMenuXmlParser")
	  << "\n etaWindow lower = " << etaWindow->lower()
	  << "\n etaWindow upper = " << etaWindow->upper() 
	  << std::endl;
	if( cntEta==0 ){      etaWindow1Lower = etaWindow->lower(); etaWindow1Upper = etaWindow->upper(); }
	else if( cntEta==1 ){ etaWindow2Lower = etaWindow->lower(); etaWindow2Upper = etaWindow->upper(); }
	cntEta++;
      }

      int cntPhi=0;
      unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
      for( l1t::MuonObjectRequirement::phiWindow_const_iterator phiWindow =objPar->phiWindow().begin();
	   phiWindow != objPar->phiWindow().end(); ++phiWindow ){
 
	LogDebug("TriggerMenuXmlParser")
	  << "\n phiWindow begin = " << phiWindow->lower()
	  << "\n phiWindow end   = " << phiWindow->upper() 
	  << std::endl;

	if( cntPhi==0 ){      phiWindow1Lower = phiWindow->lower(); phiWindow1Upper = phiWindow->upper(); }
	else if( cntPhi==1 ){ phiWindow2Lower = phiWindow->lower(); phiWindow2Upper = phiWindow->upper(); }
	cntPhi++;
      }

      objParameter[cnt].etaWindow1Lower     = etaWindow1Lower;
      objParameter[cnt].etaWindow1Upper     = etaWindow1Upper;
      objParameter[cnt].etaWindow2Lower = etaWindow2Lower;
      objParameter[cnt].etaWindow2Upper = etaWindow2Upper;

      objParameter[cnt].phiWindow1Lower     = phiWindow1Lower;
      objParameter[cnt].phiWindow1Upper     = phiWindow1Upper;
      objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
      objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

      
      // Output for debugging
      LogDebug("TriggerMenuXmlParser") 
	<< "\n      Muon PT high threshold (hex) for muon object " << cnt << " = "
	<< std::hex << objParameter[cnt].ptHighThreshold 
	<< "\n      etaWindow Lower / Upper for muon object " << cnt << " = 0x"
	<< objParameter[cnt].etaWindow1Lower << " / 0x" << objParameter[cnt].etaWindow1Upper
	<< "\n      etaWindowVeto Lower / Upper for muon object " << cnt << " = 0x"
	<< objParameter[cnt].etaWindow2Lower << " / 0x" << objParameter[cnt].etaWindow2Upper
	<< "\n      phiWindow Lower / Upper for muon object " << cnt << " = 0x"
	<< objParameter[cnt].phiWindow1Lower << " / 0x" << objParameter[cnt].phiWindow1Upper
	<< "\n      phiWindowVeto Lower / Upper for muon object " << cnt << " = 0x"
	<< objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper << std::dec
	<< std::endl;

      cnt++;
    }



    // indicates if a correlation is used
    bool wscVal = (type == m_xmlConditionAttrType2wsc );

    if( wscVal ){

      xsd::cxx::tree::optional<l1t::DeltaRequirement> condRanges = condMu.deltaRequirement();
      LogDebug("TriggerMenuXmlParser") 
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


    LogDebug("TriggerMenuXmlParser") 
      << "\n intGEq  = " << intGEq
      << " nrObj   = " << nrObj 
      << "\n ****************************************** " 
      << std::endl;

    //
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

bool l1t::TriggerMenuXmlParser::parseMuonV2(tmeventsetup::esCondition condMu,
        unsigned int chipNr, const bool corrFlag) {


//    XERCES_CPP_NAMESPACE_USE
    using namespace tmeventsetup;

    // get condition, particle name (must be muon) and type name
    std::string condition = "muon";
    std::string particle = "muon";//l1t2string( condMu.objectType() );
    std::string type = l1t2string( condMu.getType() );
    std::string name = l1t2string( condMu.getName() );
    int nrObj = -1;

    if (condMu.getType() == esConditionType::SingleMuon) {
	type = "1_s";
	nrObj = 1;
    } else if (condMu.getType() == esConditionType::DoubleMuon) {
	type = "2_s";
	nrObj = 2;	
    } else if (condMu.getType() == esConditionType::TripleMuon) {
	type = "3";
	nrObj = 3;
    } else if (condMu.getType() == esConditionType::QuadMuon) {
	type = "4";
	nrObj = 4;
    } else {
        edm::LogError("TriggerMenuXmlParser") << "Wrong type for muon-condition ("
            << type << ")" << std::endl;
        return false;
    }


    if (nrObj < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Unknown type for muon-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    LogDebug("TriggerMenuXmlParser")
      << "\n ****************************************** "
      << "\n      parseMuon  "
      << "\n condition = " << condition
      << "\n particle  = " << particle
      << "\n type      = " << type
      << "\n name      = " << name
      << std::endl;



//     // get values

    // temporary storage of the parameters
    std::vector<MuonTemplate::ObjectParameter> objParameter(nrObj);
    
    // Do we need this?
    MuonTemplate::CorrelationParameter corrParameter;

    // need at least two values for deltaPhi
    std::vector<boost::uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);
    tmpValues.reserve( nrObj );

    if( int(condMu.getObjects().size())!=nrObj ){
      edm::LogError("TriggerMenuXmlParser") << " condMu objects: nrObj = " << nrObj
					    << "condMu.getObjects().size() = " 
					    << condMu.getObjects().size()
					    << std::endl;
      return false;
    }


//  Look for cuts on the objects in the condition
     unsigned int chargeCorrelation = 1;
     const std::vector<esCut>& cuts = condMu.getCuts();      
     for (size_t jj = 0; jj < cuts.size(); jj++)
      {
        const esCut cut = cuts.at(jj);
	if(cut.getCutType() == esCutType::ChargeCorrelation) { 
	   if( cut.getData()=="ls" )      chargeCorrelation = 2;
	   else if( cut.getData()=="os" ) chargeCorrelation = 4;
	   else chargeCorrelation = 1; //ignore correlation
        }
      }
   
    //set charge correlation parameter
    corrParameter.chargeCorrelation = chargeCorrelation;//tmpValues[0];


    int cnt = 0;


// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    
// Loop over objects and extract the cuts on the objects
    const std::vector<esObject>& objects = condMu.getObjects();
    for (size_t jj = 0; jj < objects.size(); jj++) {   

       const esObject object = objects.at(jj);
       gEq =  (object.getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
       relativeBx = object.getBxOffset();

//  Loop over the cuts for this object
        int upperThresholdInd = -1; 
	int lowerThresholdInd = 0;
        int cntEta = 0;
        unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
	int cntPhi = 0;
	unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
        int isolationLUT = 0xF; //default is to ignore unless specified.
	int charge = -1; //default value is to ignore unless specified
	int qualityLUT = 0xFFFF; //default is to ignore unless specified.		
	
        const std::vector<esCut>& cuts = object.getCuts();
        for (size_t kk = 0; kk < cuts.size(); kk++)
        {
          const esCut cut = cuts.at(kk); 
	 
	  switch(cut.getCutType()){
	     case esCutType::Threshold:
	       lowerThresholdInd = cut.getMinimum().index;
	       upperThresholdInd = cut.getMaximum().index;
	       break;
	       
	     case esCutType::Eta: {
	       
                 if(cntEta == 0) {
		    etaWindow1Lower = cut.getMinimum().index;
		    etaWindow1Upper = cut.getMaximum().index;
		 } else if(cntEta == 1) {
		    etaWindow2Lower = cut.getMinimum().index;
		    etaWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuXmlParser") << "Too Many Eta Cuts for muon-condition ("
        	       << particle << ")" << std::endl;
        	   return false;
		 }
		 cntEta++; 

	       } break;
	       
	     case esCutType::Phi: {

                if(cntPhi == 0) {
		    phiWindow1Lower = cut.getMinimum().index;
		    phiWindow1Upper = cut.getMaximum().index;
		 } else if(cntPhi == 1) {
		    phiWindow2Lower = cut.getMinimum().index;
		    phiWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuXmlParser") << "Too Many Phi Cuts for muon-condition ("
        	       << particle << ")" << std::endl;
        	   return false;
		 }
		 cntPhi++; 

	       }break;
	       
	     case esCutType::Charge:
               std::cout << "Found Charge Cut " << std::endl;
	       if( cut.getData()=="positive" ) charge = 0;
               else if( cut.getData()=="negative" ) charge = 1;
	       else charge = -1;
	       break;
	     case esCutType::Quality:
	     
                qualityLUT = l1tstr2int(cut.getData());
	     
	       break;
	     case esCutType::Isolation: {

                isolationLUT = l1tstr2int(cut.getData());
		       
	       } break;
	     default:
	       break; 	       	       	       	       
	  } //end switch 
	  
        } //end loop over cuts


// Set the parameter cuts
	objParameter[cnt].ptHighThreshold = upperThresholdInd;
	objParameter[cnt].ptLowThreshold  = lowerThresholdInd;

	objParameter[cnt].etaWindow1Lower     = etaWindow1Lower;
	objParameter[cnt].etaWindow1Upper     = etaWindow1Upper;
	objParameter[cnt].etaWindow2Lower = etaWindow2Lower;
	objParameter[cnt].etaWindow2Upper = etaWindow2Upper;

	objParameter[cnt].phiWindow1Lower     = phiWindow1Lower;
	objParameter[cnt].phiWindow1Upper     = phiWindow1Upper;
	objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
	objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

// BLW TO DO: Do we need these anymore?  Drop them?   
        objParameter[cnt].enableMip = false;//tmpMip[i];
        objParameter[cnt].enableIso = false;//tmpEnableIso[i];
        objParameter[cnt].requestIso = false;//tmpRequestIso[i];

        objParameter[cnt].charge = charge;
        objParameter[cnt].qualityLUT = qualityLUT;
        objParameter[cnt].isolationLUT = isolationLUT;


        cnt++;
    } //end loop over objects	


    // get the type of the condition, as defined in enum, from the condition type
    // as defined in the XML file
    // BLW TO DO: What the heck is this for?
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
        LogDebug("TriggerMenuXmlParser") << "Added Condition " << name << " to the ConditionMap" << std::endl;
        if (corrFlag) {
	    
            (m_corMuonTemplate[chipNr]).push_back(muonCond);
        }
        else {
	    LogDebug("TriggerMenuXmlParser") << "Added Condition " << name << " to the vecMuonTemplate vector" << std::endl;
            (m_vecMuonTemplate[chipNr]).push_back(muonCond);
        }

    }

    //
    return true;
}


bool l1t::TriggerMenuXmlParser::parseMuonCorr(const tmeventsetup::esObject* corrMu,
        unsigned int chipNr) {


//    XERCES_CPP_NAMESPACE_USE
    using namespace tmeventsetup;

    // get condition, particle name (must be muon) and type name
    std::string condition = "muon";
    std::string particle = "muon";//l1t2string( condMu.objectType() );
    std::string type = l1t2string( corrMu->getType() );
    std::string name = l1t2string( corrMu->getName() );
    int nrObj = 1;
    type = "1_s";



    if (nrObj < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Unknown type for muon-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    LogDebug("TriggerMenuXmlParser")
      << "\n ****************************************** "
      << "\n      parseMuon  "
      << "\n condition = " << condition
      << "\n particle  = " << particle
      << "\n type      = " << type
      << "\n name      = " << name
      << std::endl;



//     // get values

    // temporary storage of the parameters
    std::vector<MuonTemplate::ObjectParameter> objParameter(nrObj);
    
    // Do we need this?
    MuonTemplate::CorrelationParameter corrParameter;

    // need at least two values for deltaPhi
    std::vector<boost::uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);
    tmpValues.reserve( nrObj );


// BLW TO DO: How do we deal with these in the new format    
//    std::string str_chargeCorrelation = l1t2string( condMu.requestedChargeCorr() );
    std::string str_chargeCorrelation = "ig";
    unsigned int chargeCorrelation = 0;
    if( str_chargeCorrelation=="ig" )      chargeCorrelation = 1;
    else if( str_chargeCorrelation=="ls" ) chargeCorrelation = 2;
    else if( str_chargeCorrelation=="os" ) chargeCorrelation = 4;

    //getXMLHexTextValue("1", dst);
    corrParameter.chargeCorrelation = chargeCorrelation;//tmpValues[0];



 // BLW TO DO: These needs to the added to the object rather than the whole condition.
   int relativeBx = 0;
   bool gEq = false;


   //const esObject* object = condMu;
   gEq =  (corrMu->getComparisonOperator() == esComparisonOperator::GE);

 //  BLW TO DO: This needs to be added to the Object Parameters   
   relativeBx = corrMu->getBxOffset();

 //  Loop over the cuts for this object
    int upperThresholdInd = -1;
    int lowerThresholdInd = 0;
    int cntEta = 0;
    unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
    int cntPhi = 0;
    unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
    int isolationLUT = 0xF; //default is to ignore unless specified.
    int charge = -1;       //defaut is to ignore unless specified
    int qualityLUT = 0xFFFF; //default is to ignore unless specified.		

    const std::vector<esCut>& cuts = corrMu->getCuts();
    for (size_t kk = 0; kk < cuts.size(); kk++)
    {
      const esCut cut = cuts.at(kk); 

      switch(cut.getCutType()){
	 case esCutType::Threshold:
	   lowerThresholdInd = cut.getMinimum().index;
	   upperThresholdInd = cut.getMaximum().index;
	   break;

	 case esCutType::Eta: {

             if(cntEta == 0) {
		etaWindow1Lower = cut.getMinimum().index;
		etaWindow1Upper = cut.getMaximum().index;
	     } else if(cntEta == 1) {
		etaWindow2Lower = cut.getMinimum().index;
		etaWindow2Upper = cut.getMaximum().index;
             } else {
               edm::LogError("TriggerMenuXmlParser") << "Too Many Eta Cuts for muon-condition ("
        	   << particle << ")" << std::endl;
               return false;
	     }
	     cntEta++; 

	   } break;

	 case esCutType::Phi: {

            if(cntPhi == 0) {
		phiWindow1Lower = cut.getMinimum().index;
		phiWindow1Upper = cut.getMaximum().index;
	     } else if(cntPhi == 1) {
		phiWindow2Lower = cut.getMinimum().index;
		phiWindow2Upper = cut.getMaximum().index;
             } else {
               edm::LogError("TriggerMenuXmlParser") << "Too Many Phi Cuts for muon-condition ("
        	   << particle << ")" << std::endl;
               return false;
	     }
	     cntPhi++; 

	   }break;

	 case esCutType::Charge:
	   std::cout << "Found Charge Cut " << std::endl;  
	   if( cut.getData()=="positive" ) charge = 0;
           else if( cut.getData()=="negative" ) charge = 1;
	   else charge = -1; 
	   break;
	 case esCutType::Quality:

            qualityLUT = l1tstr2int(cut.getData());

	   break;
	 case esCutType::Isolation: {

            isolationLUT = l1tstr2int(cut.getData());

	   } break;
	 default:
	   break; 	       	       	       	       
      } //end switch 

    } //end loop over cuts


 // Set the parameter cuts
    objParameter[0].ptHighThreshold = upperThresholdInd;
    objParameter[0].ptLowThreshold  = lowerThresholdInd;

    objParameter[0].etaWindow1Lower     = etaWindow1Lower;
    objParameter[0].etaWindow1Upper     = etaWindow1Upper;
    objParameter[0].etaWindow2Lower = etaWindow2Lower;
    objParameter[0].etaWindow2Upper = etaWindow2Upper;

    objParameter[0].phiWindow1Lower     = phiWindow1Lower;
    objParameter[0].phiWindow1Upper     = phiWindow1Upper;
    objParameter[0].phiWindow2Lower = phiWindow2Lower;
    objParameter[0].phiWindow2Upper = phiWindow2Upper;

 // BLW TO DO: Do we need these anymore?  Drop them?   
    objParameter[0].enableMip = false;//tmpMip[i];
    objParameter[0].enableIso = false;//tmpEnableIso[i];
    objParameter[0].requestIso = false;//tmpRequestIso[i];

    objParameter[0].charge = charge;
    objParameter[0].qualityLUT = qualityLUT;
    objParameter[0].isolationLUT = isolationLUT;



    // get the type of the condition, as defined in enum, from the condition type
    // as defined in the XML file
    // BLW TO DO: What the heck is this for?
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
        LogDebug("TriggerMenuXmlParser") << "Added Condition " << name << " to the ConditionMap" << std::endl;
            (m_corMuonTemplate[chipNr]).push_back(muonCond);
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

bool l1t::TriggerMenuXmlParser::parseCalo(l1t::CalorimeterCondition condCalo,
        unsigned int chipNr, const bool corrFlag) {

    XERCES_CPP_NAMESPACE_USE

    // get condition, particle name and type name

    std::string condition = "calo";
    std::string particle = l1t2string( condCalo.objectType() );
    std::string type = l1t2string( condCalo.type() );
    std::string name = l1t2string( condCalo.name() );

    LogDebug("TriggerMenuXmlParser")
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
      if( cnt<nrObj ) {
         objParameter[cnt].etLowThreshold = objPar->etThreshold();
         objParameter[cnt].etHighThreshold = 999; //not implemented in old grammar
      }

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


      int cntIso=0;
      int isolationLUT = 0;
      for( l1t::CalorimeterIsolationLUT::isolation_const_iterator iIsoFlag = objPar->isolationLut().isolation().begin();
	   iIsoFlag != objPar->isolationLut().isolation().end(); ++iIsoFlag ){
	
	bool flag = (*iIsoFlag);

	isolationLUT |= (flag << cntIso);

	LogDebug("TriggerMenuXmlParser")
	  << "\n isolation flag " << cntIso << " = " << flag
	  << "\n isolationLUT = " << isolationLUT 
	  << std::endl;

	cntIso++;
      }

      objParameter[cnt].isolationLUT = isolationLUT;


      int cntEta=0;
      unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
      // Temporary before translation
      for( l1t::CalorimeterObjectRequirement::etaWindow_const_iterator etaWindow =objPar->etaWindow().begin();
	   etaWindow != objPar->etaWindow().end(); ++etaWindow ){
	
	LogDebug("TriggerMenuXmlParser")
	  << "\n etaWindow lower = " << etaWindow->lower()
	  << "\n etaWindow upper = " << etaWindow->upper() 
	  << std::endl;
	if( cntEta==0 ){      etaWindow1Lower = etaWindow->lower(); etaWindow1Upper = etaWindow->upper(); }
	else if( cntEta==1 ){ etaWindow2Lower = etaWindow->lower(); etaWindow2Upper = etaWindow->upper(); }
	cntEta++;
      }

      int cntPhi=0;
      unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
      for( l1t::CalorimeterObjectRequirement::phiWindow_const_iterator phiWindow =objPar->phiWindow().begin();
	   phiWindow != objPar->phiWindow().end(); ++phiWindow ){
 
	LogDebug("TriggerMenuXmlParser")
	  << "\n phiWindow begin = " << phiWindow->lower()
	  << "\n phiWindow end   = " << phiWindow->upper() 
	  << std::endl;

	if( cntPhi==0 ){      phiWindow1Lower = phiWindow->lower(); phiWindow1Upper = phiWindow->upper(); }
	else if( cntPhi==1 ){ phiWindow2Lower = phiWindow->lower(); phiWindow2Upper = phiWindow->upper(); }
	cntPhi++;
      }

      objParameter[cnt].etaWindow1Lower     = etaWindow1Lower;
      objParameter[cnt].etaWindow1Upper     = etaWindow1Upper;
      objParameter[cnt].etaWindow2Lower = etaWindow2Lower;
      objParameter[cnt].etaWindow2Upper = etaWindow2Upper;

      objParameter[cnt].phiWindow1Lower     = phiWindow1Lower;
      objParameter[cnt].phiWindow1Upper     = phiWindow1Upper;
      objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
      objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

      
      // Output for debugging
      LogDebug("TriggerMenuXmlParser") 
	<< "\n      Calo ET high threshold (hex) for calo object " << cnt << " = "
	<< std::hex << objParameter[cnt].etLowThreshold 
	<< "\n      etaWindow Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].etaWindow1Lower << " / 0x" << objParameter[cnt].etaWindow1Upper
	<< "\n      etaWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].etaWindow2Lower << " / 0x" << objParameter[cnt].etaWindow2Upper
	<< "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].phiWindow1Lower << " / 0x" << objParameter[cnt].phiWindow1Upper
	<< "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper << std::dec
	<< std::endl;

      cnt++;
    }



    // indicates if a correlation is used
    bool wscVal = (type == m_xmlConditionAttrType2wsc );

    if( wscVal ){

      xsd::cxx::tree::optional<l1t::DeltaRequirement> condRanges = condCalo.deltaRequirement();
      LogDebug("TriggerMenuXmlParser") 
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

    LogDebug("TriggerMenuXmlParser") 
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

bool l1t::TriggerMenuXmlParser::parseCaloV2(tmeventsetup::esCondition condCalo,
        unsigned int chipNr, const bool corrFlag) {


//    XERCES_CPP_NAMESPACE_USE
    using namespace tmeventsetup;
    
    // get condition, particle name and type name

    std::string condition = "calo";
    std::string particle = "test-fix" ;
    std::string type = l1t2string( condCalo.getType() );
    std::string name = l1t2string( condCalo.getName() );

    LogDebug("TriggerMenuXmlParser")
      << "\n ****************************************** " 
      << "\n      (in parseCaloV2) " 
      << "\n condition = " << condition 
      << "\n particle  = " << particle 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;


    // determine object type type
    // BLW TO DO:  Can this object type wait and be done later in the parsing. Or done differently completely..
    L1GtObject caloObjType;
    int nrObj = -1;

    if (condCalo.getType() == esConditionType::SingleEgamma) {
        caloObjType = NoIsoEG;
	type = "1_s";
	nrObj = 1;
    } else if (condCalo.getType() == esConditionType::DoubleEgamma) {
        caloObjType = NoIsoEG;
	type = "2_s";
	nrObj = 2;	
    } else if (condCalo.getType() == esConditionType::TripleEgamma) {
        caloObjType = NoIsoEG;
	type = "3";
	nrObj = 3;
    } else if (condCalo.getType() == esConditionType::QuadEgamma) {
        caloObjType = NoIsoEG;
	type = "4";
	nrObj = 4;
    } else if (condCalo.getType() == esConditionType::SingleJet) {
        caloObjType = CenJet;
	type = "1_s";
	nrObj = 1;
    } else if (condCalo.getType() == esConditionType::DoubleJet) {
        caloObjType = CenJet;
	type = "2_s";
	nrObj = 2;	
    } else if (condCalo.getType() == esConditionType::TripleJet) {
        caloObjType = CenJet;
	type = "3";
	nrObj = 3;
    } else if (condCalo.getType() == esConditionType::QuadJet) {
        caloObjType = CenJet;
	type = "4";
	nrObj = 4;			
    } else if (condCalo.getType() == esConditionType::SingleTau) {
        caloObjType = TauJet;
	type = "1_s";
	nrObj = 1;
    } else if (condCalo.getType() == esConditionType::DoubleTau) {
        caloObjType = TauJet;
	type = "2_s";
	nrObj = 2;	
    } else if (condCalo.getType() == esConditionType::TripleTau) {
        caloObjType = TauJet;
	type = "3";
	nrObj = 3;
    } else if (condCalo.getType() == esConditionType::QuadTau) {
        caloObjType = TauJet;
	type = "4";
	nrObj = 4;		
    } else {
        edm::LogError("TriggerMenuXmlParser") << "Wrong particle for calo-condition ("
            << particle << ")" << std::endl;
        return false;
    }

//    std::string str_etComparison = l1t2string( condCalo.comparison_operator() );

    if (nrObj < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Unknown type for calo-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    // get values

    // temporary storage of the parameters
    std::vector<CaloTemplate::ObjectParameter> objParameter(nrObj);

    //BLW TO DO:  Can this be dropped?
    CaloTemplate::CorrelationParameter corrParameter;

    // need at least one value for deltaPhiRange
    std::vector<boost::uint64_t> tmpValues((nrObj > 1) ? nrObj : 1);
    tmpValues.reserve( nrObj );


    if( int(condCalo.getObjects().size())!=nrObj ){
      edm::LogError("TriggerMenuXmlParser") << " condCalo objects: nrObj = " << nrObj
						    << "condCalo.getObjects().size() = " 
						    << condCalo.getObjects().size()
						    << std::endl;
      return false;
    }


//    std::string str_condCalo = "";
//    boost::uint64_t tempUIntH, tempUIntL;
//    boost::uint64_t dst;
    int cnt = 0;

// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    
// Loop over objects and extract the cuts on the objects
    const std::vector<esObject>& objects = condCalo.getObjects();
    for (size_t jj = 0; jj < objects.size(); jj++) {   

       const esObject object = objects.at(jj);
       gEq =  (object.getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
       relativeBx = object.getBxOffset();

//  Loop over the cuts for this object
        int upperThresholdInd = -1;
	int lowerThresholdInd = 0;
        int cntEta = 0;
        unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
	int cntPhi = 0;
	unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
        int isolationLUT = 0xF; //default is to ignore isolation unless specified.
	int qualityLUT   = 0xF; //default is to ignore quality unless specified.	
		
	
        const std::vector<esCut>& cuts = object.getCuts();
        for (size_t kk = 0; kk < cuts.size(); kk++)
        {
          const esCut cut = cuts.at(kk); 
	 
	  switch(cut.getCutType()){
	     case esCutType::Threshold:
	       lowerThresholdInd = cut.getMinimum().index;
	       upperThresholdInd = cut.getMaximum().index;
	       break;
	     case esCutType::Eta: {
	       
                 if(cntEta == 0) {
		    etaWindow1Lower = cut.getMinimum().index;
		    etaWindow1Upper = cut.getMaximum().index;
		 } else if(cntEta == 1) {
		    etaWindow2Lower = cut.getMinimum().index;
		    etaWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuXmlParser") << "Too Many Eta Cuts for calo-condition ("
        	       << particle << ")" << std::endl;
        	   return false;
		 }
		 cntEta++; 

	       } break;
	       
	     case esCutType::Phi: {

                if(cntPhi == 0) {
		    phiWindow1Lower = cut.getMinimum().index;
		    phiWindow1Upper = cut.getMaximum().index;
		 } else if(cntPhi == 1) {
		    phiWindow2Lower = cut.getMinimum().index;
		    phiWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuXmlParser") << "Too Many Phi Cuts for calo-condition ("
        	       << particle << ")" << std::endl;
        	   return false;
		 }
		 cntPhi++; 

	       }break;
	       
	     case esCutType::Charge: {

       	         edm::LogError("TriggerMenuXmlParser") << "No charge cut for calo-condition ("
        	       << particle << ")" << std::endl;
        	   return false;

	       }break;
	     case esCutType::Quality: {
             
	       qualityLUT = l1tstr2int(cut.getData());

	       }break;
	     case esCutType::Isolation: {

               isolationLUT = l1tstr2int(cut.getData());
		       
	       } break;
	     default:
	       break; 	       	       	       	       
	  } //end switch 
	  
        } //end loop over cuts

// Fill the object parameters
	objParameter[cnt].etHighThreshold = upperThresholdInd;
	objParameter[cnt].etLowThreshold  = lowerThresholdInd;
	objParameter[cnt].etaWindow1Lower     = etaWindow1Lower;
	objParameter[cnt].etaWindow1Upper     = etaWindow1Upper;
	objParameter[cnt].etaWindow2Lower = etaWindow2Lower;
	objParameter[cnt].etaWindow2Upper = etaWindow2Upper;
	objParameter[cnt].phiWindow1Lower     = phiWindow1Lower;
	objParameter[cnt].phiWindow1Upper     = phiWindow1Upper;
	objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
	objParameter[cnt].phiWindow2Upper = phiWindow2Upper;
        objParameter[cnt].isolationLUT       = isolationLUT;
        objParameter[cnt].qualityLUT         = qualityLUT; //TO DO: Must add 

      // Output for debugging
      LogDebug("TriggerMenuXmlParser") 
	<< "\n      Calo ET high thresholds (hex) for calo object " << caloObjType << " " << cnt << " = "
	<< std::hex << objParameter[cnt].etLowThreshold << " - " << objParameter[cnt].etHighThreshold 
	<< "\n      etaWindow Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].etaWindow1Lower << " / 0x" << objParameter[cnt].etaWindow1Upper
	<< "\n      etaWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].etaWindow2Lower << " / 0x" << objParameter[cnt].etaWindow2Upper
	<< "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].phiWindow1Lower << " / 0x" << objParameter[cnt].phiWindow1Upper
	<< "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
	<< objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper
	<< "\n      Isolation LUT for calo object " << cnt << " = 0x"
	<< objParameter[cnt].isolationLUT
	<< "\n      Quality LUT for calo object " << cnt << " = 0x"
	<< objParameter[cnt].qualityLUT << std::dec
	<< std::endl;

      cnt++;
    } //end loop over objects



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


    

    // now create a new calo condition
    CaloTemplate caloCond(name);

    caloCond.setCondType(cType);
    caloCond.setObjectType(objType);
    
    //BLW TO DO: This needs to be added to the object rather than the whole condition
    caloCond.setCondGEq(gEq);
    caloCond.setCondChipNr(chipNr);
    
    //BLW TO DO: This needs to be added to the object rather than the whole condition
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

bool l1t::TriggerMenuXmlParser::parseCaloCorr(const tmeventsetup::esObject* corrCalo,
        unsigned int chipNr) {


//    XERCES_CPP_NAMESPACE_USE
    using namespace tmeventsetup;
    
    // get condition, particle name and type name

    std::string condition = "calo";
    std::string particle = "test-fix" ;
    std::string type = l1t2string( corrCalo->getType() );
    std::string name = l1t2string( corrCalo->getName() );

    LogDebug("TriggerMenuXmlParser")
      << "\n ****************************************** " 
      << "\n      (in parseCaloV2) " 
      << "\n condition = " << condition 
      << "\n particle  = " << particle 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;


    // determine object type type
    // BLW TO DO:  Can this object type wait and be done later in the parsing. Or done differently completely..
    L1GtObject caloObjType;
    int nrObj = 1;
    type = "1_s";


    if (corrCalo->getType() == esObjectType::Egamma) {
        caloObjType = NoIsoEG;
    } else if (corrCalo->getType() == esObjectType::Jet) {
        caloObjType = CenJet;
    } else if (corrCalo->getType() == esObjectType::Tau) {
        caloObjType = TauJet;
    } else {
        edm::LogError("TriggerMenuXmlParser") << "Wrong particle for calo-condition ("
            << particle << ")" << std::endl;
        return false;
    }


//    std::string str_etComparison = l1t2string( condCalo.comparison_operator() );

    if (nrObj < 0) {
        edm::LogError("TriggerMenuXmlParser") << "Unknown type for calo-condition (" << type
            << ")" << "\nCan not determine number of trigger objects. " << std::endl;
        return false;
    }

    // get values

    // temporary storage of the parameters
    std::vector<CaloTemplate::ObjectParameter> objParameter(nrObj);

    //BLW TO DO:  Can this be dropped?
    CaloTemplate::CorrelationParameter corrParameter;

    // need at least one value for deltaPhiRange
    std::vector<boost::uint64_t> tmpValues((nrObj > 1) ? nrObj : 1);
    tmpValues.reserve( nrObj );



// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    

    gEq =  (corrCalo->getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
    relativeBx = corrCalo->getBxOffset();

//  Loop over the cuts for this object
     int upperThresholdInd = -1;
     int lowerThresholdInd = 0;
     int cntEta = 0;
     unsigned int etaWindow1Lower=-1, etaWindow1Upper=-1, etaWindow2Lower=-1, etaWindow2Upper=-1;
     int cntPhi = 0;
     unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
     int isolationLUT = 0xF; //default is to ignore isolation unless specified.
     int qualityLUT   = 0xF; //default is to ignore quality unless specified.	


     const std::vector<esCut>& cuts = corrCalo->getCuts();
     for (size_t kk = 0; kk < cuts.size(); kk++)
     {
       const esCut cut = cuts.at(kk); 

       switch(cut.getCutType()){
	  case esCutType::Threshold:
	    lowerThresholdInd = cut.getMinimum().index;
	    upperThresholdInd = cut.getMaximum().index;
	    break;
	  case esCutType::Eta: {

              if(cntEta == 0) {
		 etaWindow1Lower = cut.getMinimum().index;
		 etaWindow1Upper = cut.getMaximum().index;
	      } else if(cntEta == 1) {
		 etaWindow2Lower = cut.getMinimum().index;
		 etaWindow2Upper = cut.getMaximum().index;
              } else {
        	edm::LogError("TriggerMenuXmlParser") << "Too Many Eta Cuts for calo-condition ("
        	    << particle << ")" << std::endl;
        	return false;
	      }
	      cntEta++; 

	    } break;

	  case esCutType::Phi: {

             if(cntPhi == 0) {
		 phiWindow1Lower = cut.getMinimum().index;
		 phiWindow1Upper = cut.getMaximum().index;
	      } else if(cntPhi == 1) {
		 phiWindow2Lower = cut.getMinimum().index;
		 phiWindow2Upper = cut.getMaximum().index;
              } else {
        	edm::LogError("TriggerMenuXmlParser") << "Too Many Phi Cuts for calo-condition ("
        	    << particle << ")" << std::endl;
        	return false;
	      }
	      cntPhi++; 

	    }break;

	  case esCutType::Charge: {

       	      edm::LogError("TriggerMenuXmlParser") << "No charge cut for calo-condition ("
        	    << particle << ")" << std::endl;
        	return false;

	    }break;
	  case esCutType::Quality: {

	    qualityLUT = l1tstr2int(cut.getData());

	    }break;
	  case esCutType::Isolation: {

            isolationLUT = l1tstr2int(cut.getData());

	    } break;
	  default:
	    break; 	       	       	       	       
       } //end switch 

     } //end loop over cuts

// Fill the object parameters
     objParameter[0].etLowThreshold  = lowerThresholdInd;
     objParameter[0].etHighThreshold = upperThresholdInd;
     objParameter[0].etaWindow1Lower = etaWindow1Lower;
     objParameter[0].etaWindow1Upper = etaWindow1Upper;
     objParameter[0].etaWindow2Lower = etaWindow2Lower;
     objParameter[0].etaWindow2Upper = etaWindow2Upper;
     objParameter[0].phiWindow1Lower = phiWindow1Lower;
     objParameter[0].phiWindow1Upper = phiWindow1Upper;
     objParameter[0].phiWindow2Lower = phiWindow2Lower;
     objParameter[0].phiWindow2Upper = phiWindow2Upper;
     objParameter[0].isolationLUT    = isolationLUT;
     objParameter[0].qualityLUT      = qualityLUT; //TO DO: Must add 

   // Output for debugging
   LogDebug("TriggerMenuXmlParser") 
     << "\n      Calo ET high threshold (hex) for calo object " << caloObjType << " "  << " = "
     << std::hex << objParameter[0].etLowThreshold << " - " << objParameter[0].etHighThreshold 
     << "\n      etaWindow Lower / Upper for calo object "  << " = 0x"
     << objParameter[0].etaWindow1Lower << " / 0x" << objParameter[0].etaWindow1Upper
     << "\n      etaWindowVeto Lower / Upper for calo object "  << " = 0x"
     << objParameter[0].etaWindow2Lower << " / 0x" << objParameter[0].etaWindow2Upper
     << "\n      phiWindow Lower / Upper for calo object "  << " = 0x"
     << objParameter[0].phiWindow1Lower << " / 0x" << objParameter[0].phiWindow1Upper
     << "\n      phiWindowVeto Lower / Upper for calo object "  << " = 0x"
     << objParameter[0].phiWindow2Lower << " / 0x" << objParameter[0].phiWindow2Upper
     << "\n      Isolation LUT for calo object "  << " = 0x"
     << objParameter[0].isolationLUT
     << "\n      Quality LUT for calo object "  << " = 0x"
     << objParameter[0].qualityLUT << std::dec
     << std::endl;



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


    

    // now create a new calo condition
    CaloTemplate caloCond(name);

    caloCond.setCondType(cType);
    caloCond.setObjectType(objType);
    
    //BLW TO DO: This needs to be added to the object rather than the whole condition
    caloCond.setCondGEq(gEq);
    caloCond.setCondChipNr(chipNr);
    
    //BLW TO DO: This needs to be added to the object rather than the whole condition
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
            (m_corCaloTemplate[chipNr]).push_back(caloCond);
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

bool l1t::TriggerMenuXmlParser::parseEnergySum(l1t::EnergySumsCondition condEnergySum,
        unsigned int chipNr, const bool corrFlag) {

    XERCES_CPP_NAMESPACE_USE

    // get condition, particle name and type name

    std::string condition = "calo";
    std::string type = l1t2string( condEnergySum.objectType() );
    std::string name = l1t2string( condEnergySum.name() );

    LogDebug("TriggerMenuXmlParser")
      << "\n ****************************************** " 
      << "\n      DARRENS TEST OUTPUT (in parseEnergySum) " 
      << "\n condition = " << condition 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;

    // determine object type type
    L1GtObject energySumObjType;
    GtConditionType cType;

    if( type == m_xmlConditionAttrObjectETM ){
      energySumObjType = ETM;
      cType = TypeETM;
    }
    else if( type == m_xmlConditionAttrObjectETT ){
      energySumObjType = ETT;
      cType = TypeETT;
    }
    else if( type == m_xmlConditionAttrObjectHTT ){
      energySumObjType = HTT;
      cType = TypeHTT;
    }
    else if( type == m_xmlConditionAttrObjectHTM ){
      energySumObjType = HTM;
      cType = TypeHTM;
    }
    else {
      edm::LogError("TriggerMenuXmlParser")
	<< "Wrong type for energy-sum condition (" << type
	<< ")" << std::endl;
      return false;
    }



    // global object
    int nrObj = 1;

    std::string str_etComparison = l1t2string( condEnergySum.comparison_operator() );

    // get greater equal flag
    int intGEq = ( str_etComparison=="ge" ) ? 1 : 0;
    if( intGEq < 0 ){
      edm::LogError("TriggerMenuXmlParser") 
	<< "Error getting \"greater or equal\" flag"
	<< std::endl;
      return false;
    }
    // set the boolean value for the ge_eq mode
    bool gEq = (intGEq != 0);


    // get values

    // temporary storage of the parameters
    std::vector<EnergySumTemplate::ObjectParameter> objParameter(nrObj);


    int cnt = 0;

    l1t::EnergySumsObjectRequirement objPar = condEnergySum.objectRequirement();

    // ET Threshold
    objParameter[cnt].etLowThreshold = objPar.etThreshold();
    objParameter[cnt].etLowThreshold = 999; //not implemented in old grammar

    int cntPhi=0;
    unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
    for( l1t::EnergySumsObjectRequirement::phiWindow_const_iterator phiWindow =objPar.phiWindow().begin();
	 phiWindow != objPar.phiWindow().end(); ++phiWindow ){
      
      LogDebug("TriggerMenuXmlParser")
	<< "\n phiWindow begin = " << phiWindow->lower()
	<< "\n phiWindow end   = " << phiWindow->upper() 
	<< std::endl;

      if( cntPhi==0 ){      phiWindow1Lower = phiWindow->lower(); phiWindow1Upper = phiWindow->upper(); }
      else if( cntPhi==1 ){ phiWindow2Lower = phiWindow->lower(); phiWindow2Upper = phiWindow->upper(); }
      cntPhi++;
    }

    objParameter[cnt].phiWindow1Lower     = phiWindow1Lower;
    objParameter[cnt].phiWindow1Upper     = phiWindow1Upper;
    objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
    objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

      
    // Output for debugging
    LogDebug("TriggerMenuXmlParser") 
      << "\n      EnergySum ET high threshold (hex) for energy sum object " << cnt << " = "
      << std::hex << objParameter[cnt].etLowThreshold << " - " << objParameter[cnt].etHighThreshold << std::hex 
      << "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[cnt].phiWindow1Lower << " / 0x" << objParameter[cnt].phiWindow1Upper
      << "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper <<std::dec
      << std::endl;




    int relativeBx = l1t2int( condEnergySum.relativeBx() );

    // object types - all same energySumObjType
    std::vector<L1GtObject> objType(nrObj, energySumObjType);

    // now create a new energySum condition

    EnergySumTemplate energySumCond(name);

    energySumCond.setCondType(cType);
    energySumCond.setObjectType(objType);
    energySumCond.setCondGEq(gEq);
    energySumCond.setCondChipNr(chipNr);
    energySumCond.setCondRelativeBx(relativeBx);

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



    /*



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

      */

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

bool l1t::TriggerMenuXmlParser::parseEnergySumV2(tmeventsetup::esCondition condEnergySum,
        unsigned int chipNr, const bool corrFlag) {


//    XERCES_CPP_NAMESPACE_USE
     using namespace tmeventsetup;
     
    // get condition, particle name and type name

    std::string condition = "calo";
    std::string type = l1t2string( condEnergySum.getType() );
    std::string name = l1t2string( condEnergySum.getName() );

    LogDebug("TriggerMenuXmlParser")
      << "\n ****************************************** " 
      << "\n      (in parseEnergySumV2) " 
      << "\n condition = " << condition 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;



    // determine object type type
    L1GtObject energySumObjType;
    GtConditionType cType;

    if( condEnergySum.getType() == esConditionType::MissingEt ){
      energySumObjType = L1GtObject::ETM;
      cType = TypeETM;
    }
    else if( condEnergySum.getType() == esConditionType::TotalEt ){
      energySumObjType = L1GtObject::ETT;
      cType = TypeETT;
    }
    else if( condEnergySum.getType() == esConditionType::TotalHt ){
      energySumObjType = L1GtObject::HTT;
      cType = TypeHTT;
    }
    else if( condEnergySum.getType() == esConditionType::MissingHt ){
      energySumObjType = L1GtObject::HTM;
      cType = TypeHTM;
    }
    else {
      edm::LogError("TriggerMenuXmlParser")
	<< "Wrong type for energy-sum condition (" << type
	<< ")" << std::endl;
      return false;
    }



    // global object
    int nrObj = 1;

//    std::string str_etComparison = l1t2string( condEnergySum.comparison_operator() );

    // get values

    // temporary storage of the parameters
    std::vector<EnergySumTemplate::ObjectParameter> objParameter(nrObj);


    int cnt = 0;

// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    
//    l1t::EnergySumsObjectRequirement objPar = condEnergySum.objectRequirement();

// Loop over objects and extract the cuts on the objects
    const std::vector<esObject>& objects = condEnergySum.getObjects();
    for (size_t jj = 0; jj < objects.size(); jj++) {   

       const esObject object = objects.at(jj);
       gEq =  (object.getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
       relativeBx = object.getBxOffset();

//  Loop over the cuts for this object
        int lowerThresholdInd = 0;
	int upperThresholdInd = -1;
	int cntPhi = 0;
	unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;
		
	
        const std::vector<esCut>& cuts = object.getCuts();
        for (size_t kk = 0; kk < cuts.size(); kk++)
        {
          const esCut cut = cuts.at(kk); 
	 
	  switch(cut.getCutType()){
	     case esCutType::Threshold:
	       lowerThresholdInd = cut.getMinimum().index;
	       upperThresholdInd = cut.getMaximum().index;
	       break;

	     case esCutType::Eta: 
	       break;
	       
	     case esCutType::Phi: {

                if(cntPhi == 0) {
		    phiWindow1Lower = cut.getMinimum().index;
		    phiWindow1Upper = cut.getMaximum().index;
		 } else if(cntPhi == 1) {
		    phiWindow2Lower = cut.getMinimum().index;
		    phiWindow2Upper = cut.getMaximum().index;
                 } else {
        	   edm::LogError("TriggerMenuXmlParser") << "Too Many Phi Cuts for esum-condition ("
        	       << type << ")" << std::endl;
        	   return false;
		 }
		 cntPhi++; 

	       }
	       break;
	       
	     default:
	       break; 	       	       	       	       
	  } //end switch 
	  
        } //end loop over cuts



    // Fill the object parameters
    objParameter[cnt].etLowThreshold   = lowerThresholdInd;
    objParameter[cnt].etHighThreshold = upperThresholdInd;
    objParameter[cnt].phiWindow1Lower = phiWindow1Lower;
    objParameter[cnt].phiWindow1Upper = phiWindow1Upper;
    objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
    objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

      
    // Output for debugging
    LogDebug("TriggerMenuXmlParser") 
      << "\n      EnergySum ET high threshold (hex) for energy sum object " << cnt << " = "
      << std::hex << objParameter[cnt].etLowThreshold << " - " << objParameter[cnt].etHighThreshold 
      << "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[cnt].phiWindow1Lower << " / 0x" << objParameter[cnt].phiWindow1Upper
      << "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper << std::dec
      << std::endl;

      cnt++;
    } //end loop over objects
    
    // object types - all same energySumObjType
    std::vector<L1GtObject> objType(nrObj, energySumObjType);

    // now create a new energySum condition

    EnergySumTemplate energySumCond(name);

    energySumCond.setCondType(cType);
    energySumCond.setObjectType(objType);
    energySumCond.setCondGEq(gEq);
    energySumCond.setCondChipNr(chipNr);
    energySumCond.setCondRelativeBx(relativeBx);

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

bool l1t::TriggerMenuXmlParser::parseEnergySumCorr(const tmeventsetup::esObject* corrESum,
        unsigned int chipNr) {


//    XERCES_CPP_NAMESPACE_USE
     using namespace tmeventsetup;
     
    // get condition, particle name and type name

    std::string condition = "calo";
    std::string type = l1t2string( corrESum->getType() );
    std::string name = l1t2string( corrESum->getName() );

    LogDebug("TriggerMenuXmlParser")
      << "\n ****************************************** " 
      << "\n      (in parseEnergySumV2) " 
      << "\n condition = " << condition 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;



    // determine object type type
    L1GtObject energySumObjType;
    GtConditionType cType;

    if( corrESum->getType()== esObjectType::ETM ){
      energySumObjType = L1GtObject::ETM;
      cType = TypeETM;
    }
    else if( corrESum->getType()== esObjectType::HTM ){
      energySumObjType = L1GtObject::HTM;
      cType = TypeHTM;
    }
    else {
      edm::LogError("TriggerMenuXmlParser")
	<< "Wrong type for energy-sum correclation condition (" << type
	<< ")" << std::endl;
      return false;
    }



    // global object
    int nrObj = 1;

//    std::string str_etComparison = l1t2string( condEnergySum.comparison_operator() );

    // get values

    // temporary storage of the parameters
    std::vector<EnergySumTemplate::ObjectParameter> objParameter(nrObj);


    int cnt = 0;

// BLW TO DO: These needs to the added to the object rather than the whole condition.
    int relativeBx = 0;
    bool gEq = false;
    
//    l1t::EnergySumsObjectRequirement objPar = condEnergySum.objectRequirement();


   gEq =  (corrESum->getComparisonOperator() == esComparisonOperator::GE);

//  BLW TO DO: This needs to be added to the Object Parameters   
   relativeBx = corrESum->getBxOffset();

//  Loop over the cuts for this object
    int lowerThresholdInd = 0;
    int upperThresholdInd = -1;
    int cntPhi = 0;
    unsigned int phiWindow1Lower=-1, phiWindow1Upper=-1, phiWindow2Lower=-1, phiWindow2Upper=-1;


    const std::vector<esCut>& cuts = corrESum->getCuts();
    for (size_t kk = 0; kk < cuts.size(); kk++)
    {
      const esCut cut = cuts.at(kk); 

      switch(cut.getCutType()){
	 case esCutType::Threshold:
	   lowerThresholdInd = cut.getMinimum().index;
	   upperThresholdInd = cut.getMaximum().index;
	   break;

	 case esCutType::Eta: 
	   break;

	 case esCutType::Phi: {

            if(cntPhi == 0) {
		phiWindow1Lower = cut.getMinimum().index;
		phiWindow1Upper = cut.getMaximum().index;
	     } else if(cntPhi == 1) {
		phiWindow2Lower = cut.getMinimum().index;
		phiWindow2Upper = cut.getMaximum().index;
             } else {
               edm::LogError("TriggerMenuXmlParser") << "Too Many Phi Cuts for esum-condition ("
        	   << type << ")" << std::endl;
               return false;
	     }
	     cntPhi++; 

	   }
	   break;

	 default:
	   break; 	       	       	       	       
      } //end switch 

    } //end loop over cuts



    // Fill the object parameters
    objParameter[0].etLowThreshold  = lowerThresholdInd;
    objParameter[0].etHighThreshold = upperThresholdInd;
    objParameter[0].phiWindow1Lower = phiWindow1Lower;
    objParameter[0].phiWindow1Upper = phiWindow1Upper;
    objParameter[0].phiWindow2Lower = phiWindow2Lower;
    objParameter[0].phiWindow2Upper = phiWindow2Upper;

      
    // Output for debugging
    LogDebug("TriggerMenuXmlParser") 
      << "\n      EnergySum ET high threshold (hex) for energy sum object " << cnt << " = "
      << std::hex << objParameter[0].etLowThreshold << " - " << objParameter[0].etLowThreshold 
      << "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[0].phiWindow1Lower << " / 0x" << objParameter[0].phiWindow1Upper
      << "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
      << objParameter[0].phiWindow2Lower << " / 0x" << objParameter[0].phiWindow2Upper << std::dec
      << std::endl;

    
    // object types - all same energySumObjType
    std::vector<L1GtObject> objType(nrObj, energySumObjType);

    // now create a new energySum condition

    EnergySumTemplate energySumCond(name);

    energySumCond.setCondType(cType);
    energySumCond.setObjectType(objType);
    energySumCond.setCondGEq(gEq);
    energySumCond.setCondChipNr(chipNr);
    energySumCond.setCondRelativeBx(relativeBx);

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

       (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);

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

bool l1t::TriggerMenuXmlParser::parseExternalV2(tmeventsetup::esCondition condExt,
        unsigned int chipNr) {


    using namespace tmeventsetup;

    
    // get condition, particle name and type name
    std::string condition = "ext";     
    std::string particle = "test-fix";
    std::string type = l1t2string( condExt.getType() );
    std::string name = l1t2string( condExt.getName() );


    LogDebug("TriggerMenuXmlParser")
      << "\n ****************************************** " 
      << "\n      (in parseExternalV2) " 
      << "\n condition = " << condition 
      << "\n particle  = " << particle 
      << "\n type      = " << type 
      << "\n name      = " << name 
      << std::endl;


    // object type and condition type
    // object type - irrelevant for External conditions
    GtConditionType cType = TypeExternal;

    int relativeBx = 0;    
    unsigned int channelID = 0;

    // Get object for External conditions
    const std::vector<esObject>& objects = condExt.getObjects();
    for (size_t jj = 0; jj < objects.size(); jj++) {   

       const esObject object = objects.at(jj);
       if(object.getType() == esObjectType::EXT) {
          relativeBx = object.getBxOffset();
          channelID = object.getExternalChannelId();
       }
    }   


    // set the boolean value for the ge_eq mode - irrelevant for External conditions
    bool gEq = false;

    // now create a new External condition
    ExternalTemplate externalCond(name);

    externalCond.setCondType(cType);
    externalCond.setCondGEq(gEq);
    externalCond.setCondChipNr(chipNr);
    externalCond.setCondRelativeBx(relativeBx);
    externalCond.setExternalChannel(channelID);

    LogTrace("TriggerMenuXmlParser") 
             << externalCond << "\n" << std::endl;

    // insert condition into the map
    if ( !insertConditionIntoMap(externalCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser")
            << "    Error: duplicate condition (" << name
            << ")" << std::endl;

        return false;
    } else {

        (m_vecExternalTemplate[chipNr]).push_back(externalCond);

    }
         
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
/*
bool l1t::TriggerMenuXmlParser::parseExternal(XERCES_CPP_NAMESPACE::DOMNode* node,
    const std::string& name, unsigned int chipNr) {

    XERCES_CPP_NAMESPACE_USE
*/
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
/*    
    return true;
}
*/

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

bool l1t::TriggerMenuXmlParser::parseCorrelationV2(
        tmeventsetup::esCondition corrCond,
        unsigned int chipNr) {

    using namespace tmeventsetup;

    std::string condition = "corr";
    std::string particle = "test-fix" ;
    std::string type = l1t2string( corrCond.getType() );
    std::string name = l1t2string( corrCond.getName() );

    LogDebug("TriggerMenuXmlParser") << " ****************************************** " << std::endl
     << "     (in parseCorrelation) " << std::endl
     << " condition = " << condition << std::endl
     << " particle  = " << particle << std::endl
     << " type      = " << type << std::endl
     << " name      = " << name << std::endl;


   

    // create a new correlation condition
    CorrelationTemplate correlationCond(name);

    // check that the condition does not exist already in the map
    if ( !insertConditionIntoMap(correlationCond, chipNr)) {

        edm::LogError("TriggerMenuXmlParser")
                << "    Error: duplicate correlation condition (" << name << ")"
                << std::endl;

        return false;
    }


// Define some of the quantities to store the parased information

    // condition type BLW  (Do we change this to the type of correlation condition?)
    GtConditionType cType = l1t::Type2cor;

    // two objects (for sure)
    const int nrObj = 2;

    // object types and greater equal flag - filled in the loop
    int intGEq[nrObj] = { -1, -1 };
    std::vector<L1GtObject> objType(nrObj);   //BLW do we want to define these as a different type?
    std::vector<GtConditionCategory> condCateg(nrObj);   //BLW do we want to change these categories

    // correlation flag and index in the cor*vector
    const bool corrFlag = true;
    int corrIndexVal[nrObj] = { -1, -1 };


    // Storage of the correlation selection
    CorrelationTemplate::CorrelationParameter corrParameter;
    corrParameter.chargeCorrelation = 1; //ignore charge correlation

// Get the correlation Cuts on the legs
        
      const std::vector<esCut>& cuts = corrCond.getCuts();      
      for (size_t jj = 0; jj < cuts.size(); jj++)
      {
        const esCut cut = cuts.at(jj);
        std::cout << "    cut name = " << cut.getName() << "\n";
        std::cout << "    cut target = " << cut.getObjectType() << "\n";
        std::cout << "    cut type = " << cut.getCutType() << "\n";
        std::cout << "    cut min. value  index = " << cut.getMinimum().value << " " << cut.getMinimum().index << "\n";
        std::cout << "    cut max. value  index = " << cut.getMaximum().value << " " << cut.getMaximum().index << "\n";
        std::cout << "    cut data = " << cut.getData() << "\n";


	if(cut.getCutType() == esCutType::ChargeCorrelation) { 
	   if( cut.getData()=="ls" )      corrParameter.chargeCorrelation = 2;
	   else if( cut.getData()=="os" ) corrParameter.chargeCorrelation = 4;
	   else corrParameter.chargeCorrelation = 1; //ignore charge correlation
        } else {
	  //keep the type from what the correlation is.
          corrParameter.corrCutType = cut.getCutType();
	  corrParameter.minCutValue = cut.getMinimum().value;
	  corrParameter.maxCutValue = cut.getMaximum().value;
	}  

      }


// Get the two objects that form the legs
      const std::vector<esObject>& objects = corrCond.getObjects();
      if(objects.size() != 2) {
            edm::LogError("TriggerMenuXmlParser")
                    << "incorrect number of objects for the correlation condition " << name << " corrFlag " << corrFlag << std::endl;
            return false;      
      }
      
// loop over legs      
      for (size_t jj = 0; jj < objects.size(); jj++)
      {
        const esObject object = objects.at(jj);
/*        std::cout << "      obj name = " << object->getName() << "\n";
        std::cout << "      obj type = " << object->getType() << "\n";
        std::cout << "      obj op = " << object->getComparisonOperator() << "\n";
        std::cout << "      obj bx = " << object->getBxOffset() << "\n";
*/

// check the leg type
        if(object.getType() == esObjectType::Muon) {
	  // we have a muon  

          //BLW Is there a problem here with not entering second instanance into the m_corMuonTemplate[]?
          if ((m_conditionMap[chipNr]).count(object.getName()) == 0) {
	   	                  
              parseMuonCorr(&object,chipNr);	     
	    
          } else {
	     LogDebug("TriggerMenuXmlParser")  << "Not Adding Correlation Muon Condition." << std::endl;
	  }
	  
          //Now set some flags for this subCondition
	  intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
          objType[jj] = Mu;
          condCateg[jj] = CondMuon;
          corrIndexVal[jj] = (m_corMuonTemplate[chipNr]).size() - 1;


	  
        } else if(object.getType() == esObjectType::Egamma ||
	          object.getType() == esObjectType::Jet    ||
		  object.getType() == esObjectType::Tau ) {
	  // we have an Calo object

          //BLW Is there a problem here with not entering second instanance into the m_corMuonTemplate[]?
          if ((m_conditionMap[chipNr]).count(object.getName()) == 0) {
	   	                  
              parseCaloCorr(&object,chipNr);	     
	    
          } else {
	     LogDebug("TriggerMenuXmlParser")  << "Not Adding Correlation Calo Condition." << std::endl;
	  }
	  

          //Now set some flags for this subCondition
	  intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
          switch(object.getType()) {
	     case esObjectType::Egamma: { 
	      objType[jj] = NoIsoEG;
	     }
	        break;
	     case esObjectType::Jet: { 
	      objType[jj] = CenJet;
	     }
	        break;
	     case esObjectType::Tau: { 
	      objType[jj] = TauJet;
	     }
	        break;
	      default: {
	      }
	        break;	
          }		 
          condCateg[jj] = CondCalo;
          corrIndexVal[jj] = (m_corCaloTemplate[chipNr]).size() - 1;
	  
	  
        } else if(object.getType() == esObjectType::ETM  ||
	          object.getType() == esObjectType::HTM ) {
	  // we have Energy Sum

          //BLW Is there a problem here with not entering second instanance into the m_corMuonTemplate[]?
          if ((m_conditionMap[chipNr]).count(object.getName()) == 0) {
	   	                  
              parseEnergySumCorr(&object,chipNr);	     
	    
          } else {
	     LogDebug("TriggerMenuXmlParser")  << "Not Adding Correlation EtSum Condition." << std::endl;
	  }
	  

          //Now set some flags for this subCondition
	  intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
          switch(object.getType()) {
	     case esObjectType::ETM: { 
	      objType[jj] = L1GtObject::ETM;
	     }
	        break;
	     case esObjectType::HTM: { 
	      objType[jj] = L1GtObject::HTM;
	     }
	        break;
	      default: {
	      }
	        break;			
          }		 
          condCateg[jj] = CondEnergySum;
          corrIndexVal[jj] = (m_corEnergySumTemplate[chipNr]).size() - 1;

	} else {
	
          edm::LogError("TriggerMenuXmlParser")
                  << "Illegal Object Type "
                  << " for the correlation condition " << name << std::endl;
          return false;	     

	}  //if block on leg types

      }  //loop over legs
    

    // get greater equal flag for the correlation condition
    bool gEq = true;
    if (intGEq[0] != intGEq[1]) {
        edm::LogError("TriggerMenuXmlParser")
                << "Inconsistent GEq flags for sub-conditions "
                << " for the correlation condition " << name << std::endl;
        return false;

    }
    else {
        gEq = (intGEq[0] != 0);

    }
    

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

      LogDebug("TriggerMenuXmlParser")
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
    LogDebug("TriggerMenuXmlParser")
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

    LogDebug("TriggerMenuXmlParser")
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
    LogDebug("TriggerMenuXmlParser") << " ====> condCalorimeter" << std::endl;
    for (l1t::ConditionList::condCalorimeter_const_iterator condCalo = conditions.condCalorimeter().begin();
	 condCalo != conditions.condCalorimeter().end(); ++condCalo ){

      LogDebug("TriggerMenuXmlParser")
	<< condCalo->name()  << " {"                    
	<< "  comment: " << condCalo->comment()
	<< "  locked: "      << condCalo->locked()     
	<< "}" 
	<< std::endl;

      l1t::CalorimeterCondition condition = (*condCalo);

      parseCalo( condition, chipNr );
    }

    LogDebug("TriggerMenuXmlParser")  << " ====> condMuon " << std::endl;
    for (l1t::ConditionList::condMuon_const_iterator condMu = conditions.condMuon().begin();
	 condMu != conditions.condMuon().end(); ++condMu ){

      LogDebug("TriggerMenuXmlParser")
	<< condMu->name()  << " {"                    
	<< "  comment: " << condMu->comment()
	<< "  locked: "      << condMu->locked()     
	<< "}" 
	<< std::endl;

      l1t::MuonCondition condition = (*condMu);

      parseMuon( condition, chipNr );
    }

    LogDebug("TriggerMenuXmlParser")  << " ====> condEnergySums " << std::endl;
    for (l1t::ConditionList::condEnergySums_const_iterator condEnergySums = conditions.condEnergySums().begin();
	 condEnergySums != conditions.condEnergySums().end(); ++condEnergySums ){

      LogDebug("TriggerMenuXmlParser")
	<< condEnergySums->name()  << " {"                    
	<< "  comment: " << condEnergySums->comment()
	<< "  locked: "      << condEnergySums->locked()     
	<< "}" 
	<< std::endl;

      l1t::EnergySumsCondition condition = (*condEnergySums);

      parseEnergySum( condition, chipNr );
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

bool l1t::TriggerMenuXmlParser::parseAlgorithmV2( tmeventsetup::esAlgorithm algorithm,
    unsigned int chipNr) {

  
  using namespace tmeventsetup;
  using namespace Algorithm;
  

    // get alias
    std::string algAlias = algorithm.getName();
    std::string algName  = algorithm.getName();

    if (algAlias == "") {
        algAlias = algName;
        LogDebug("TriggerMenuXmlParser")
                << "\n    No alias defined for algorithm. Alias set to algorithm name."
                << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
                << std::endl;
    } else {
      //LogDebug("TriggerMenuXmlParser") 
      LogDebug("TriggerMenuXmlParser")  << "\n    Alias defined for algorithm."
			      << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
			      << std::endl;
    }

    // get the logical expression
    std::string logExpression = algorithm.getExpressionInCondition();

    LogDebug("TriggerMenuXmlParser")
      << "      Logical expression: " << logExpression
      << "      Chip number:        " << chipNr
      << std::endl;

    // determine output pin
    int outputPin = algorithm.getIndex();


    //LogTrace("TriggerMenuXmlParser")
    LogDebug("TriggerMenuXmlParser")  << "      Output pin:         " << outputPin
			    << std::endl;


    // compute the bit number from chip number, output pin and order of the chips
    // pin numbering start with 1, bit numbers with 0
    int bitNumber = outputPin;// + (m_orderConditionChip[chipNr] -1)*m_pinsOnConditionChip -1;

    //LogTrace("TriggerMenuXmlParser")
    LogDebug("TriggerMenuXmlParser")  << "      Bit number:         " << bitNumber
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
      LogDebug("TriggerMenuXmlParser")  << "\n    Alias defined for algorithm."
			      << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
			      << std::endl;
    }

    // get the logical expression
    std::string logExpression = l1t2string( algorithm.logical_expression() );

    LogDebug("TriggerMenuXmlParser")
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
    LogDebug("TriggerMenuXmlParser")  << "      Output pin:         " << outputPin
			    << std::endl;


    // compute the bit number from chip number, output pin and order of the chips
    // pin numbering start with 1, bit numbers with 0
    int bitNumber = outputPin;// + (m_orderConditionChip[chipNr] -1)*m_pinsOnConditionChip -1;

    //LogTrace("TriggerMenuXmlParser")
    LogDebug("TriggerMenuXmlParser")  << "      Bit number:         " << bitNumber
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
    LogDebug("TriggerMenuXmlParser")  << " ====> algorithms " << std::endl;
    for( l1t::AlgorithmList::algorithm_const_iterator i = algorithms.algorithm().begin();
         i != algorithms.algorithm().end(); ++i ){

      l1t::Algorithm algorithm = (*i);
      LogDebug("TriggerMenuXmlParser") 
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
/*
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
*/
/*
 * parseTechTriggers Parse the technical triggers
 *
 * @param parser A reference to the XercesDOMParser to use.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */
/*
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
*/

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

