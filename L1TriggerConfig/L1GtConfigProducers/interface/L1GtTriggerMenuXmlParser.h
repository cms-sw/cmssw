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
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>

#include <boost/cstdint.hpp>

#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

// user include files

// forward declarations

// class declaration
class L1GtTriggerMenuXmlParser
{

public:

    /// constructor
    L1GtTriggerMenuXmlParser();

    /// destructor
    virtual ~L1GtTriggerMenuXmlParser();

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
                                 unsigned int num, boost::uint64_t* dst);

    /// shutdown the xml utils and deallocate parser and error handler
    void cleanupXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

private:

    // methods for the VME file

    /// write a vme line
    void writeVmeLine(unsigned int clkcond, unsigned long int address,
         unsigned int value, std::ofstream& ofs);

    /// add two lines to the preamble
    bool addVmeAddress(XERCES_CPP_NAMESPACE::DOMNode* node, std::ofstream& ofs);

    /// parse the vme bus preamble xml file, write the preamble file and store the time
    bool parseVmeXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);



private:

    // methods for conditions and algorithms

    /// get number of particles from condition type
    int getNumFromType(const std::string& type);

    /// get bit from a bit node        
    int getBitFromNode(XERCES_CPP_NAMESPACE::DOMNode* node);    

    /// get greater equal flag from a condition
    int getGeEqFlag(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& nodeName);

//    // parse a muon condition
//    int parseMuon(XERCES_CPP_NAMESPACE::DOMNode* node, 
//    const std::string& name, unsigned int chipNr);
//
//    /// choose the parser for a particular condition
//    bool workCondition(XERCES_CPP_NAMESPACE::DOMNode* node, 
//    const std::string& name, unsigned int chipNr);
//
//    /// parse all conditions
//    bool parseConditions(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
//
//private:
//
//    /// do all the steps filling the configuration
//    bool workXML(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
//
private:

    /// error handler for xml-parser
    XERCES_CPP_NAMESPACE::ErrorHandler* m_xmlErrHandler;

    /// time for the vme bus preamble
    double m_vmePreambleTime;

private:

    // strings for the def.xml-syntax
    static const std::string xml_condition_attr_condition;
    static const std::string xml_condition_attr_particle;
    static const std::string xml_condition_attr_type;
    static const std::string xml_condition_attr_condition_muon;
    static const std::string xml_condition_attr_condition_calo;
    static const std::string xml_condition_attr_condition_esums;
    static const std::string xml_condition_attr_condition_jetcnts;

    static const std::string xml_condition_attr_particle_muon;
    static const std::string xml_condition_attr_particle_eg;
    static const std::string xml_condition_attr_particle_ieg;
    static const std::string xml_condition_attr_particle_jet;
    static const std::string xml_condition_attr_particle_fwdjet;
    static const std::string xml_condition_attr_particle_tau;
    static const std::string xml_condition_attr_particle_etm;
    static const std::string xml_condition_attr_particle_ett;
    static const std::string xml_condition_attr_particle_htt;

    static const std::string xml_condition_attr_type_1;
    static const std::string xml_condition_attr_type_2;
    static const std::string xml_condition_attr_type_2wsc;
    static const std::string xml_condition_attr_type_3;
    static const std::string xml_condition_attr_type_4;

    static const std::string xml_attr_mode;
    static const std::string xml_attr_mode_bit;
    static const std::string xml_attr_max;

    static const std::string xml_attr_pin;
    static const std::string xml_attr_pin_a;
    static const std::string xml_attr_nr;


    static const std::string xml_conditions_tag;
    static const std::string xml_prealgos_tag;
    static const std::string xml_algos_tag;
    static const std::string xml_chip_tag;
    static const std::string xml_value_tag;
    static const std::string xml_def_tag;

    static const std::string xml_etthreshold_tag;

    static const std::string xml_pththreshold_tag;
    static const std::string xml_ptlthreshold_tag;
    static const std::string xml_quality_tag;
    static const std::string xml_eta_tag;
    static const std::string xml_phi_tag;
    static const std::string xml_phih_tag;
    static const std::string xml_phil_tag;
    static const std::string xml_geeq_tag;
    static const std::string xml_chargecorrelation_tag;
    static const std::string xml_eniso_tag;
    static const std::string xml_enmip_tag;
    static const std::string xml_enoverflow_tag;
    static const std::string xml_deltaeta_tag;
    static const std::string xml_deltaphi_tag;

    static const std::string xml_output_tag;
    static const std::string xml_outputpin_tag;


    static const std::string xml_chipdef_tag;
    static const std::string xml_chip1_tag;
    static const std::string xml_ca_tag;

    // strings for the vme bus xml file syntax
    static const std::string vmexml_vme_tag;
    static const std::string vmexml_condchip_tag;
    static const std::string vmexml_address_tag;
    static const std::string vmexml_value_tag;
    static const std::string vmexml_attr_particle;
    static const std::string vmexml_attr_particle_muon;

    // name of the file to write the vme bus preamble
    static const char m_vmePreambleFileName[];

    // time for one line
    static const double m_vmePreambleTimeTick;

    // the rest of a line in the preamble
    static const char m_vmePreambleLineRest[];

};

#endif /*L1GtConfigProducers_L1GtTriggerMenuXmlParser_h*/
