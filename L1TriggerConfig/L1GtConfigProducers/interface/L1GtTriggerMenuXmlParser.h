#ifndef L1GtConfigProducers_L1GtTriggerMenuXmlParser_h
#define L1GtConfigProducers_L1GtTriggerMenuXmlParser_h

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
 * \author  M. Eder            - HEPHY Vienna - ORCA version, reduced functionality
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

// user include files
// base class
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

// forward declarations
class L1GtCondition;
class L1GtAlgorithm;

// class declaration
class L1GtTriggerMenuXmlParser : public L1GtXmlParserTags
{

public:

    /// constructor
    ///    empty
    L1GtTriggerMenuXmlParser();

    /// destructor
    virtual ~L1GtTriggerMenuXmlParser();

public:

    ///   get / set the number of condition chips in GTL
    inline const unsigned int gtNumberConditionChips() const
    {
        return m_numberConditionChips;
    }

    void setGtNumberConditionChips(const unsigned int&);

    ///   get / set the number of pins on the GTL condition chips
    inline const unsigned int gtPinsOnConditionChip() const
    {
        return m_pinsOnConditionChip;
    }

    void setGtPinsOnConditionChip(const unsigned int&);

    ///   get / set the correspondence "condition chip - GTL algorithm word"
    ///   in the hardware
    inline std::vector<int> gtOrderConditionChip() const
    {
        return m_orderConditionChip;
    }

    void setGtOrderConditionChip(const std::vector<int>&);

    /// get / set the number of physics trigger algorithms
    inline const unsigned int gtNumberPhysTriggers() const
    {
        return m_numberPhysTriggers;
    }

    void setGtNumberPhysTriggers(const unsigned int&);

    ///  get / set the number of L1 jet counts received by GT
    inline const unsigned int gtNumberL1JetCounts() const
    {
        return m_numberL1JetCounts;
    }

    void setGtNumberL1JetCounts(const unsigned int&);


public:

    /// get / set the trigger menu name
    inline const std::string& gtTriggerMenuName() const {
        return m_triggerMenuName;
    }

    void setGtTriggerMenuName(const std::string&);

    /// get / set the condition maps
    inline const std::vector<ConditionMap>& gtConditionMap() const
    {
        return m_conditionMap;
    }

    void setGtConditionMap(const std::vector<ConditionMap>&);

    /// get / set the algorithm map
    inline const AlgorithmMap& gtAlgorithmMap() const
    {
        return m_algorithmMap;
    }

    void setGtAlgorithmMap(const AlgorithmMap&);


public:

    /// parse def.xml and vme.xml files
    void parseXmlFile(const std::string& defXmlFile,
                      const std::string& vmeXmlFile);


private:

    // XML methods

    /// init xml system
    XERCES_CPP_NAMESPACE::XercesDOMParser* initXML(const std::string& xmlFile);

    /// find a named child of a xml node
    XERCES_CPP_NAMESPACE::DOMNode* findXMLChild(
        XERCES_CPP_NAMESPACE::DOMNode* startChild, const std::string& tagName,
        bool beginOnly, std::string* rest);

    /// get a named attribute for an xml node as string
    std::string getXMLAttribute(
        const XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name);

    /// get the text value of a xml node as string
    std::string getXMLTextValue(XERCES_CPP_NAMESPACE::DOMNode* node);

    /// convert a hexadecimal string with up to 128 to 2 boost::uint64_t
    bool hexString2UInt128(const std::string& hexString,
                           boost::uint64_t& dstL, boost::uint64_t& dstH);

    /// get a hexadecimal value of a xml node containing text with up to 128 bit
    bool getXMLHexTextValue128(XERCES_CPP_NAMESPACE::DOMNode* node,
                               boost::uint64_t& dstL, boost::uint64_t& dstH);

    /// get a hexadecimal value of a xml node containing text
    bool getXMLHexTextValue(XERCES_CPP_NAMESPACE::DOMNode* node, boost::uint64_t& dst);

    /// get the number of bits in the max attribute of a condition child
    bool countConditionChildMaxBits(XERCES_CPP_NAMESPACE::DOMNode* node,
                                    const std::string& childName, unsigned int& dst);

    /// get values from a child of a condition
    bool getConditionChildValues(XERCES_CPP_NAMESPACE::DOMNode* node,
                                 const std::string& childName,
                                 unsigned int num,
                                 std::vector<boost::uint64_t>& dst);

    /// shutdown the xml utils and deallocate parser and error handler
    void cleanupXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
    
    /// FIXME remove it after new L1 Trigger Menu Editor available
    /// mirrors the LUT table from GTgui format to correct bit format
    boost::uint64_t mirror(const boost::uint64_t oldLUT, int maxBitsLUT, int maxBitsReal);

private:

    // methods for the VME file

    /// parse the vme xml file
    bool parseVmeXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

private:

    // methods for conditions and algorithms

    /// clearMaps - delete all conditions and algorithms in
    /// the maps and clear the maps.
    void clearMaps();

    /// insertConditionIntoMap - safe insert of condition into condition map.
    /// if the condition name already exists, do not insert it and return false
    bool insertConditionIntoMap(L1GtCondition* cond, const int chipNr);

    /// insert an algorithm into algorithm map
    bool insertAlgorithmIntoMap(L1GtAlgorithm* alg);

    /// get the type of the condition, as defined in enum, from the condition type
    /// as defined in the XML file
    L1GtConditionType getTypeFromType(const std::string& type);

    /// get number of particles from condition type
    int getNumFromType(const std::string& type);

    /// get bit from a bit node
    int getBitFromNode(XERCES_CPP_NAMESPACE::DOMNode* node);

    /// getGEqFlag - get the "greater or equal flag" from a condition
    int getGEqFlag(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& nodeName);

    /// get MIP and Isolation bits from a muon
    bool getMuonMipIsoBits(XERCES_CPP_NAMESPACE::DOMNode* node,
                           unsigned int num, std::vector<bool>& mipDst,
                           std::vector<bool>& isoEnDst, std::vector<bool>& isoReqDst);

    /// parse a muon condition
    bool parseMuon(XERCES_CPP_NAMESPACE::DOMNode* node,
                   const std::string& name, unsigned int chipNr);

    /// parse a calorimeter condition
    bool parseCalo(XERCES_CPP_NAMESPACE::DOMNode* node,
                   const std::string& name, unsigned int chipNr);

    /// parse an "energy sum" condition
    bool parseEnergySum(XERCES_CPP_NAMESPACE::DOMNode* node,
                        const std::string& name, unsigned int chipNr);

    /// parse a "jet counts" condition
    bool parseJetCounts(XERCES_CPP_NAMESPACE::DOMNode* node,
                        const std::string& name, unsigned int chipNr);

    /// choose the parser for a particular condition
    bool workCondition(XERCES_CPP_NAMESPACE::DOMNode* node,
                       const std::string& name, unsigned int chipNr);

    /// parse all conditions
    bool parseConditions(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

    /// parse an algorithm and insert it into algorithm map.
    bool workAlgorithm(XERCES_CPP_NAMESPACE::DOMNode* node,
                       const std::string& name, unsigned int chipNr);

    /// parse all algorithms
    bool parseAlgorithms(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

    /// do all the steps for filling a trigger menu
    bool workXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);



private:

    /// error handler for xml-parser
    XERCES_CPP_NAMESPACE::ErrorHandler* m_xmlErrHandler;

    /// hardware limits

    /// number of condition chips
    unsigned int m_numberConditionChips;

    /// number of pins on the GTL condition chips
    unsigned int m_pinsOnConditionChip;
    
    /// correspondence "condition chip - GTL algorithm word" in the hardware
    /// chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    std::vector<int> m_orderConditionChip;    

    /// number of physics trigger algorithms
    unsigned int m_numberPhysTriggers;

    /// jet counts
    unsigned int m_numberL1JetCounts;

private:

    // the trigger menu

    /// menu name 
    std::string m_triggerMenuName;

    /// map containing the conditions (per condition chip)
    std::vector<ConditionMap> m_conditionMap;

    /// map containing the algorithms (global map)
    AlgorithmMap m_algorithmMap;


};

#endif /*L1GtConfigProducers_L1GtTriggerMenuXmlParser_h*/
