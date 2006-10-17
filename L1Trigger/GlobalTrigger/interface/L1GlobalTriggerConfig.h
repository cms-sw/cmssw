#ifndef GlobalTrigger_L1GlobalTriggerConfig_h
#define GlobalTrigger_L1GlobalTriggerConfig_h

/**
 * \class L1GlobalTriggerConfig
 * 
 * 
 * 
 * Description: Configuration parameters for L1GlobalTrigger 
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

// system include files
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <bitset>

// user include files
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GlobalTrigger;


XERCES_CPP_NAMESPACE_USE

// class declaration
class L1GlobalTriggerConfig {

public:

    // constructors
    L1GlobalTriggerConfig(L1GlobalTrigger*, std::string&, std::string&);
     
    // destructor 
    virtual ~L1GlobalTriggerConfig();
    
public:

    // number of maximum chips defined in the xml file
    static const unsigned int NumberConditionChips = 2;

    // number of pins on the GTL condition chips
    static const unsigned int PinsOnConditionChip = 96;

    // correspondence "condition chip - GTL algorithm word" in the hardware
    // chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    static const int OrderConditionChip[NumberConditionChips];      

    // maximum number of algorithms accepted by FDL board
    static const unsigned int MaxNumberAlgorithms = L1GlobalTriggerReadoutRecord::NumberPhysTriggers;       
    
    // number of input modules: 2 (GMT, GCT)
    static const unsigned int NumberInputModules = 2;  

    // map containing the conditions from the xml file
    typedef std::map<std::string, L1GlobalTriggerConditions*> ConditionsMap;
    ConditionsMap conditionsmap[NumberConditionChips];

    // map for prealgos
    ConditionsMap prealgosmap;
    // map for algos
    ConditionsMap algosmap;

    enum XMLFileVersionType {	// version of the xml-file
        VERSION_PROTOTYPE,		// prototype version (6U)
        VERSION_FINAL		    // final version
    };	

    inline XMLFileVersionType getVersion() const {return p_xmlfileversion;} 

    virtual void parseTriggerMenu(std::string&, std::string&);
    
    // return / set mask to block output pins
    inline const std::bitset<MaxNumberAlgorithms>& getTriggerMask() const { return p_triggermask; }
    void setTriggerMask(std::bitset<MaxNumberAlgorithms> trigMask) { p_triggermask = trigMask; }

    // return / set mask to block input: bit 0 GCT, bit 1 GMT 
    inline const std::bitset<NumberInputModules>& getInputMask() const { return p_inputmask; }
    void setInputMask(std::bitset<NumberInputModules> inMask) { p_inputmask = inMask; }

    // return time of the vme bus preamble
    inline double getVmePreambleTime() const { return p_vmePreambleTime; }
    
private:

    std::bitset<MaxNumberAlgorithms> p_triggermask;
    std::bitset<NumberInputModules> p_inputmask;
 
    // strings for the xml-syntax
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
    static const char p_vmePreambleFileName[];

    // time for the vme bus preamble
    double p_vmePreambleTime;
    // time for one line
    static const double p_vmePreambleTimeTick;

    // the rest of a line in the preamble
    static const char p_vmePreambleLineRest[];


    // the version of the xml-file is stored here
    XMLFileVersionType p_xmlfileversion;

    // error handler for xml-parser
    ErrorHandler* m_xmlErrHandler;

    // clear the conditions map
    void clearConditionsMap();

    // insert condition into conditionsmap
    int insertIntoConditionsMap(L1GlobalTriggerConditions *cond, unsigned int chipnr);

    // insert algo into a map
    int insertAlgoIntoMap(DOMNode *node, const std::string& algoname,
        ConditionsMap *insertmap, ConditionsMap* operandmap, unsigned int chipnr,
        unsigned int nummap=1);

    // get number of particles from condition type    
    int getNumFromType(const std::string& type);
    
    // parse all conditions
    int parseConditions(XercesDOMParser* parser);

    // parse all prealgos
    int parsePreAlgos(DOMNode *node);

    // parse all algos
    int parseAlgos(DOMNode *node);

    // parse prealgos and algos
    int parseAllAlgos(XercesDOMParser* parser);

    // get the version of the xml file
    int checkVersion(XercesDOMParser* parser);

    // get bit from a bit node        
    int getBitFromNode(DOMNode* node);    

    // get mip and iso bits from a muon
    int getMuonMipIsoBits(DOMNode* node, unsigned int num, bool *mipdst, bool *isodst);
    
    // parse a muon condition
    int parseMuon(DOMNode* node, const std::string& name, unsigned int chipnr);

    // parse a calo condition
    int parseCalo(DOMNode* node, const std::string& name, unsigned int chipnr);

    // parse a energy sums condition
    int parseESums(DOMNode* node, const std::string& name, unsigned int chipnr);

    // parse a jet counts condtition
    int parseJetCounts(DOMNode* node, const std::string& name, unsigned int chipnr);

    // choose the parser for a particular condition
    int workCondition(DOMNode* node, const std::string& name, unsigned int chipnr);

    // get greater equal flag from a condition
    int getGeEqFlag(DOMNode* node, const std::string &nodename);

    // get values from a child of a condition 
    int getConditionChildValues(DOMNode *node, const std::string &childname, 
        unsigned int num, u_int64_t* dst); 

    // init xml system
    XercesDOMParser* initXML(const std::string& xmlFile);

    // find a named child of a xml node
    DOMNode* findXMLChild(DOMNode* startchild, const std::string& tagname, 
        bool beginonly, std::string* rest);

    // get a named attribute fo a xml node as string
    std::string getXMLAttribute(const DOMNode* node, const std::string& name);

    // get the text value of a xml node as string
    std::string getXMLTextValue(DOMNode* node);

    // get a hexadecimal value of a xml node containing text
    int getXMLHexTextValue(DOMNode* node, u_int64_t& dst); 

    // get a hexadecimal value of a xml node containing text with up to 128 bit
    int getXMLHexTextValue128(DOMNode *node, u_int64_t& dstl, u_int64_t& dsth);

    // convert a hexadecimal string with up to 128 to 2 u_int64_t
    int hexString2UInt128(const std::string& hex, u_int64_t& dstl, u_int64_t& dsth);

    // get the number of bits in the max attribute of a condition child
    int countConditionChildMaxBits(DOMNode *node, const std::string& childname,
        unsigned int& dst);
    
    // do all the steps filling the configuration
    int workXML(XercesDOMParser* parser);

    // shutdown the xml utils and deallocate parser and errorhandler
    void cleanupXML(XercesDOMParser* parser); 

    // parse the vme bus preamble xml file, write the preamble file and store the time
    int parseVmeXML(XercesDOMParser* parser);

    // add 2 lines to the preamble
    int addVmeAddress(DOMNode *node, std::ofstream& ofs);

    // write a vme line
    void writeVmeLine(unsigned int clkcond, unsigned long int address,
         unsigned int value, std::ofstream& ofs);
        
    // print all thresholds from conditions
    void printThresholds();
    
    L1GlobalTrigger* m_GT;
    

};

#endif
