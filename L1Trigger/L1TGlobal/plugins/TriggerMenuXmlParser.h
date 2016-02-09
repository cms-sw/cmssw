#ifndef L1TGlobal_TriggerMenuXmlParser_h
#define L1TGlobal_TriggerMenuXmlParser_h

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
#include "L1Trigger/L1TGlobal/interface/TriggerMenuFwd.h"

#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CaloTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ExternalTemplate.h"

#include "L1Trigger/L1TGlobal/interface/L1TGlobalScales.h"

#include "L1Trigger/L1TGlobal/src/L1TMenuEditor/L1TriggerMenu.hxx"

#include "tmEventSetup/tmEventSetup.hh"
#include "tmEventSetup/esTriggerMenu.hh"
#include "tmEventSetup/esAlgorithm.hh"
#include "tmEventSetup/esCondition.hh"
#include "tmEventSetup/esObject.hh"
#include "tmEventSetup/esCut.hh"
#include "tmEventSetup/esScale.hh"
#include "tmGrammar/Algorithm.hh"

// forward declarations
class GtCondition;
class L1GtAlgorithm;

namespace l1t {

// class declaration
class TriggerMenuXmlParser : public L1GtXmlParserTags
{

public:

    /// constructor
    ///    empty
    TriggerMenuXmlParser();

    /// destructor
    virtual ~TriggerMenuXmlParser();

public:

    ///   get / set the number of condition chips in GTL
    inline const unsigned int gtNumberConditionChips() const {
        return m_numberConditionChips;
    }

    void setGtNumberConditionChips(const unsigned int&);

    ///   get / set the number of pins on the GTL condition chips
    inline const unsigned int gtPinsOnConditionChip() const {
        return m_pinsOnConditionChip;
    }

    void setGtPinsOnConditionChip(const unsigned int&);

    ///   get / set the correspondence "condition chip - GTL algorithm word"
    ///   in the hardware
    inline std::vector<int> gtOrderConditionChip() const {
        return m_orderConditionChip;
    }

    void setGtOrderConditionChip(const std::vector<int>&);

    /// get / set the number of physics trigger algorithms
    inline const unsigned int gtNumberPhysTriggers() const {
        return m_numberPhysTriggers;
    }

    void setGtNumberPhysTriggers(const unsigned int&);

/*
    /// get / set the number of technical triggers
    inline const unsigned int gtNumberTechTriggers() const {
        return m_numberTechTriggers;
    }

    void setGtNumberTechTriggers(const unsigned int&);
*/



public:

    /// get / set / build the condition maps
    inline const std::vector<ConditionMap>& gtConditionMap() const {
        return m_conditionMap;
    }

    void setGtConditionMap(const std::vector<ConditionMap>&);

    /// get / set the trigger menu names
    inline const std::string& gtTriggerMenuInterface() const {
        return m_triggerMenuInterface;
    }

    void setGtTriggerMenuInterface(const std::string&);

    //
    inline const std::string& gtTriggerMenuName() const {
        return m_triggerMenuName;
    }

    void setGtTriggerMenuName(const std::string&);

    //
    inline const std::string& gtTriggerMenuImplementation() const {
        return m_triggerMenuImplementation;
    }

    void setGtTriggerMenuImplementation(const std::string&);

    /// menu associated scale key
    inline const std::string& gtScaleDbKey() const {
        return m_scaleDbKey;
    }

    /// menu associated scales
    inline const L1TGlobalScales& gtScales() const {
        return m_gtScales;
    }

    void setGtScaleDbKey(const std::string&);

    /// get / set the vectors containing the conditions
    inline const std::vector<std::vector<MuonTemplate> >& vecMuonTemplate() const {
        return m_vecMuonTemplate;
    }
    void setVecMuonTemplate(const std::vector<std::vector<MuonTemplate> >&);

    //
    inline const std::vector<std::vector<CaloTemplate> >& vecCaloTemplate() const {
        return m_vecCaloTemplate;
    }

    void setVecCaloTemplate(const std::vector<std::vector<CaloTemplate> >&);

    //
    inline const std::vector<std::vector<EnergySumTemplate> >& vecEnergySumTemplate() const {

        return m_vecEnergySumTemplate;
    }

    void setVecEnergySumTemplate(
            const std::vector<std::vector<EnergySumTemplate> >&);

    //

    inline const std::vector<std::vector<ExternalTemplate> >&
        vecExternalTemplate() const {

        return m_vecExternalTemplate;
    }

    void setVecExternalTemplate(
            const std::vector<std::vector<ExternalTemplate> >&);

    //
    inline const std::vector<std::vector<CorrelationTemplate> >& vecCorrelationTemplate() const {

        return m_vecCorrelationTemplate;
    }

    void setVecCorrelationTemplate(
            const std::vector<std::vector<CorrelationTemplate> >&);

    // get / set the vectors containing the conditions for correlation templates
    //
    inline const std::vector<std::vector<MuonTemplate> >& corMuonTemplate() const {
        return m_corMuonTemplate;
    }

    void setCorMuonTemplate(const std::vector<std::vector<MuonTemplate> >&);

    //
    inline const std::vector<std::vector<CaloTemplate> >& corCaloTemplate() const {
        return m_corCaloTemplate;
    }

    void setCorCaloTemplate(const std::vector<std::vector<CaloTemplate> >&);

    //
    inline const std::vector<std::vector<EnergySumTemplate> >& corEnergySumTemplate() const {

        return m_corEnergySumTemplate;
    }

    void setCorEnergySumTemplate(
            const std::vector<std::vector<EnergySumTemplate> >&);

    /// get / set the algorithm map (by name)
    inline const AlgorithmMap& gtAlgorithmMap() const {
        return m_algorithmMap;
    }

    void setGtAlgorithmMap(const AlgorithmMap&);

    /// get / set the algorithm map (by alias)
    inline const AlgorithmMap& gtAlgorithmAliasMap() const {
        return m_algorithmAliasMap;
    }

    void setGtAlgorithmAliasMap(const AlgorithmMap&);
/*
    /// get / set the technical trigger map
    inline const AlgorithmMap& gtTechnicalTriggerMap() const {
        return m_technicalTriggerMap;
    }

//    void setGtTechnicalTriggerMap(const AlgorithmMap&);
*/

public:

    /// parse def.xml and vme.xml files
    void parseXmlFile(const std::string& defXmlFile,
            const std::string& vmeXmlFile);
	    
	    
    void parseXmlFileV2(const std::string& defXmlFile);	    

public:

    /// get / set the XML parser creation date, author, description for menu interface, menu
    inline const std::string& gtTriggerMenuInterfaceDate() const {
        return m_triggerMenuInterfaceDate;
    }

    void setGtTriggerMenuInterfaceDate(const std::string&);

    inline const std::string& gtTriggerMenuInterfaceAuthor() const {
        return m_triggerMenuInterfaceAuthor;
    }

    void setGtTriggerMenuInterfaceAuthor(const std::string&);

    inline const std::string& gtTriggerMenuInterfaceDescription() const {
        return m_triggerMenuInterfaceDescription;
    }

    void setGtTriggerMenuInterfaceDescription(const std::string&);

    //

    inline const std::string& gtTriggerMenuDate() const {
        return m_triggerMenuDate;
    }

    void setGtTriggerMenuDate(const std::string&);

    inline const std::string& gtTriggerMenuAuthor() const {
        return m_triggerMenuAuthor;
    }

    void setGtTriggerMenuAuthor(const std::string&);

    inline const std::string& gtTriggerMenuDescription() const {
        return m_triggerMenuDescription;
    }

    void setGtTriggerMenuDescription(const std::string&);


    inline const std::string& gtAlgorithmImplementation() const {
        return m_algorithmImplementation;
    }

    void setGtAlgorithmImplementation(const std::string&);


private:

    // XML methods

    /// init xml system
    XERCES_CPP_NAMESPACE::XercesDOMParser* initXML(const std::string& xmlFile);

    /// find a named child of a xml node
    XERCES_CPP_NAMESPACE::DOMNode* findXMLChild(
            XERCES_CPP_NAMESPACE::DOMNode* startChild,
            const std::string& tagName, bool beginOnly = false,
            std::string* rest = 0);

    /// get a named attribute for an xml node as string
    std::string getXMLAttribute(const XERCES_CPP_NAMESPACE::DOMNode* node,
            const std::string& name);

    /// get the text value of a xml node as string
    std::string getXMLTextValue(XERCES_CPP_NAMESPACE::DOMNode* node);

    /// convert a hexadecimal string with up to 128 to 2 boost::uint64_t
    bool hexString2UInt128(const std::string& hexString, boost::uint64_t& dstL,
            boost::uint64_t& dstH);

    /// get a hexadecimal value of a xml node containing text with up to 128 bit
    bool getXMLHexTextValue128Old(XERCES_CPP_NAMESPACE::DOMNode* node,
            boost::uint64_t& dstL, boost::uint64_t& dstH);
    bool getXMLHexTextValue128(const std::string& childName,
            boost::uint64_t& dstL, boost::uint64_t& dstH);

    /// get a hexadecimal value of a xml node containing text
    bool getXMLHexTextValueOld(XERCES_CPP_NAMESPACE::DOMNode* node,
            boost::uint64_t& dst);
    bool getXMLHexTextValue(const std::string& childName,
            boost::uint64_t& dst);

    /// get the number of bits in the max attribute of a condition child
    bool countConditionChildMaxBits(const std::string& childName, 
	    unsigned int& dst);

    /// get values from a child of a condition
    bool getConditionChildValuesOld(XERCES_CPP_NAMESPACE::DOMNode* node,
            const std::string& childName, unsigned int num,
            std::vector<boost::uint64_t>& dst);

    /// shutdown the xml utils and deallocate parser and error handler
    void cleanupXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);


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
    bool insertConditionIntoMap(GtCondition& cond, const int chipNr);

    /// insert an algorithm into algorithm map
    bool insertAlgorithmIntoMap(const L1GtAlgorithm& alg);

    /// insert a technical trigger into technical trigger map
    //bool insertTechTriggerIntoMap(const L1GtAlgorithm& alg);

    /// get the type of the condition, as defined in enum, from the condition type
    /// as defined in the XML file
    l1t::GtConditionType getTypeFromType(const std::string& type);

    /// get number of particles from condition type
    int getNumFromType(const std::string& type);

    /// get bit from a bit node
    int getBitFromNode(XERCES_CPP_NAMESPACE::DOMNode* node);

    /// getGEqFlag - get the "greater or equal flag" from a condition
    int getGEqFlag(XERCES_CPP_NAMESPACE::DOMNode* node,
            const std::string& nodeName);

    /// get MIP and Isolation bits from a muon
    bool getMuonMipIsoBits(XERCES_CPP_NAMESPACE::DOMNode* node,
            unsigned int num, std::vector<bool>& mipDst,
            std::vector<bool>& isoEnDst, std::vector<bool>& isoReqDst);

    template <typename T> std::string l1t2string( T );
    std::string l1tDateTime2string( l1t::DateTime );
    int l1t2int( l1t::RelativeBx );
    int l1tstr2int( const std::string data );


    /// parse scales
/*     bool parseScale(tmeventsetup::esScale scale); */
//    bool parseScales( tmeventsetup::esScale scale);
	bool parseScales(std::map<std::string, tmeventsetup::esScale> scaleMap);
 
    /// parse a muon condition
/*     bool parseMuon(XERCES_CPP_NAMESPACE::DOMNode* node, */
/*             const std::string& name, unsigned int chipNr = 0, */
/*             const bool corrFlag = false); */
    bool parseMuon( l1t::MuonCondition condMu,
            unsigned int chipNr = 0, const bool corrFlag = false);

    /// parse a muon condition
/*     bool parseMuon(XERCES_CPP_NAMESPACE::DOMNode* node, */
/*             const std::string& name, unsigned int chipNr = 0, */
/*             const bool corrFlag = false); */
    bool parseMuonV2( tmeventsetup::esCondition condMu,
            unsigned int chipNr = 0, const bool corrFlag = false);

    bool parseMuonCorr(const tmeventsetup::esObject* condMu,
             unsigned int chipNr = 0);


    /// parse a calorimeter condition
/*     bool parseCalo(XERCES_CPP_NAMESPACE::DOMNode* node, */
/*             const std::string& name, unsigned int chipNr = 0, */
/*             const bool corrFlag = false); */
    bool parseCalo( l1t::CalorimeterCondition condCalo,
            unsigned int chipNr = 0, const bool corrFlag = false);

    /// parse a calorimeter condition
/*     bool parseCalo(XERCES_CPP_NAMESPACE::DOMNode* node, */
/*             const std::string& name, unsigned int chipNr = 0, */
/*             const bool corrFlag = false); */
    bool parseCaloV2( tmeventsetup::esCondition condCalo,
            unsigned int chipNr = 0, const bool corrFlag = false);

    bool parseCaloCorr(const tmeventsetup::esObject* corrCalo,
            unsigned int chipNr = 0);

    /// parse an "energy sum" condition
    /* bool parseEnergySum(XERCES_CPP_NAMESPACE::DOMNode* node, */
    /*         const std::string& name, unsigned int chipNr = 0, */
    /*         const bool corrFlag = false); */

    bool parseEnergySum( l1t::EnergySumsCondition condEnergySums,
            unsigned int chipNr = 0, const bool corrFlag = false);

    /// parse an "energy sum" condition
    /* bool parseEnergySum(XERCES_CPP_NAMESPACE::DOMNode* node, */
    /*         const std::string& name, unsigned int chipNr = 0, */
    /*         const bool corrFlag = false); */

    bool parseEnergySumV2( tmeventsetup::esCondition condEnergySums,
            unsigned int chipNr = 0, const bool corrFlag = false);


    bool parseEnergySumCorr(const tmeventsetup::esObject* corrESum,
            unsigned int chipNr = 0);


    /// parse an External condition
/*
    bool parseExternal(XERCES_CPP_NAMESPACE::DOMNode* node,
            const std::string& name, unsigned int chipNr = 0);
*/
      bool parseExternalV2(tmeventsetup::esCondition condExt,
        unsigned int chipNr = 0);


    /// parse a correlation condition
    bool parseCorrelation(XERCES_CPP_NAMESPACE::DOMNode* node,
            const std::string& name, unsigned int chipNr = 0);

    /// parse a correlation condition
    bool parseCorrelationV2(tmeventsetup::esCondition corrCond, unsigned int chipNr = 0);

    /// parse all parse all identification attributes (trigger menu names, scale DB key, etc)
    //bool parseId(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
    bool parseId( std::auto_ptr<l1t::L1TriggerMenu> tm );
    bool parseId( l1t::Meta meta );

    /// choose the parser for a particular condition
    bool workCondition(XERCES_CPP_NAMESPACE::DOMNode* node,
            const std::string& name, unsigned int chipNr);

    /// parse all conditions
    //bool parseConditions(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
    //bool parseConditions( std::auto_ptr<l1t::L1TriggerMenu> tm );
    bool parseConditions( l1t::ConditionList conditions );

    /// parse an algorithm and insert it into algorithm map.
/*     bool workAlgorithm(XERCES_CPP_NAMESPACE::DOMNode* node, */
/*             const std::string& name, unsigned int chipNr); */
    bool workAlgorithm( l1t::Algorithm algorithm, 
            unsigned int chipNr);

    /// parse all algorithms
    //bool parseAlgorithms(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
    bool parseAlgorithms( l1t::AlgorithmList algorithms );

    /// parse all algorithms
    //bool parseAlgorithms(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
    bool parseAlgorithmV2( tmeventsetup::esAlgorithm algorithm,
            unsigned int chipNr = 0 );


    /// parse an algorithm and insert it into algorithm map.
//    bool workTechTrigger(XERCES_CPP_NAMESPACE::DOMNode* node,
//            const std::string& name);

    /// parse all algorithms
//   bool parseTechTriggers(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

    /// do all the steps for filling a trigger menu
    //bool workXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
    bool workXML( std::auto_ptr<l1t::L1TriggerMenu> tm );

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

    /// number of technical triggers
    //unsigned int m_numberTechTriggers;


private:

    ///  members  for XML parser only (do not appear in CondFormats)

    std::string m_triggerMenuInterfaceDate;
    std::string m_triggerMenuInterfaceAuthor;
    std::string m_triggerMenuInterfaceDescription;

    std::string m_triggerMenuDate;
    std::string m_triggerMenuAuthor;
    std::string m_triggerMenuDescription;

    std::string m_algorithmImplementation;


private:

    /// map containing the conditions (per condition chip) - transient
    std::vector<ConditionMap> m_conditionMap;

private:

    /// menu names
    std::string m_triggerMenuInterface;
    std::string m_triggerMenuName;
    std::string m_triggerMenuImplementation;

    /// menu associated scale key
    std::string m_scaleDbKey;

    /// vectors containing the conditions
    /// explicit, due to persistency...
    std::vector<std::vector<MuonTemplate> > m_vecMuonTemplate;
    std::vector<std::vector<CaloTemplate> > m_vecCaloTemplate;
    std::vector<std::vector<EnergySumTemplate> > m_vecEnergySumTemplate;
    std::vector<std::vector<ExternalTemplate> > m_vecExternalTemplate;

    std::vector<std::vector<CorrelationTemplate> > m_vecCorrelationTemplate;
    std::vector<std::vector<MuonTemplate> > m_corMuonTemplate;
    std::vector<std::vector<CaloTemplate> > m_corCaloTemplate;
    std::vector<std::vector<EnergySumTemplate> > m_corEnergySumTemplate;

    /// map containing the physics algorithms (by name)
    AlgorithmMap m_algorithmMap;

    /// map containing the physics algorithms (by alias)
    AlgorithmMap m_algorithmAliasMap;

    /// map containing the technical triggers
//    AlgorithmMap m_technicalTriggerMap;

    // class containing the scales from the L1 Menu XML
    L1TGlobalScales m_gtScales;

};

}
#endif /*L1TGlobal_TriggerMenuXmlParser_h*/
