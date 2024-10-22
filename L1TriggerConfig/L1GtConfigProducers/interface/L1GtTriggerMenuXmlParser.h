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
 *
 */

// system include files
#include <string>
#include <vector>

#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

// user include files
// base class
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCastorTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtHfBitCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtHfRingEtSumsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCorrelationTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtBptxTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtExternalTemplate.h"
#include <cstdint>

// forward declarations
class L1GtCondition;
class L1GtAlgorithm;

// class declaration
class L1GtTriggerMenuXmlParser : public L1GtXmlParserTags {
public:
  /// constructor
  ///    empty
  L1GtTriggerMenuXmlParser();

  /// destructor
  ~L1GtTriggerMenuXmlParser() override;

public:
  ///   get / set the number of condition chips in GTL
  inline const unsigned int gtNumberConditionChips() const { return m_numberConditionChips; }

  void setGtNumberConditionChips(const unsigned int&);

  ///   get / set the number of pins on the GTL condition chips
  inline const unsigned int gtPinsOnConditionChip() const { return m_pinsOnConditionChip; }

  void setGtPinsOnConditionChip(const unsigned int&);

  ///   get / set the correspondence "condition chip - GTL algorithm word"
  ///   in the hardware
  inline std::vector<int> gtOrderConditionChip() const { return m_orderConditionChip; }

  void setGtOrderConditionChip(const std::vector<int>&);

  /// get / set the number of physics trigger algorithms
  inline const unsigned int gtNumberPhysTriggers() const { return m_numberPhysTriggers; }

  void setGtNumberPhysTriggers(const unsigned int&);

  /// get / set the number of technical triggers
  inline const unsigned int gtNumberTechTriggers() const { return m_numberTechTriggers; }

  void setGtNumberTechTriggers(const unsigned int&);

  ///  get / set the number of L1 jet counts received by GT
  inline const unsigned int gtNumberL1JetCounts() const { return m_numberL1JetCounts; }

  void setGtNumberL1JetCounts(const unsigned int&);

public:
  /// get / set / build the condition maps
  inline const std::vector<ConditionMap>& gtConditionMap() const { return m_conditionMap; }

  void setGtConditionMap(const std::vector<ConditionMap>&);

  /// get / set the trigger menu names
  inline const std::string& gtTriggerMenuInterface() const { return m_triggerMenuInterface; }

  void setGtTriggerMenuInterface(const std::string&);

  //
  inline const std::string& gtTriggerMenuName() const { return m_triggerMenuName; }

  void setGtTriggerMenuName(const std::string&);

  //
  inline const std::string& gtTriggerMenuImplementation() const { return m_triggerMenuImplementation; }

  void setGtTriggerMenuImplementation(const std::string&);

  /// menu associated scale key
  inline const std::string& gtScaleDbKey() const { return m_scaleDbKey; }

  void setGtScaleDbKey(const std::string&);

  /// get / set the vectors containing the conditions
  inline const std::vector<std::vector<L1GtMuonTemplate> >& vecMuonTemplate() const { return m_vecMuonTemplate; }
  void setVecMuonTemplate(const std::vector<std::vector<L1GtMuonTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtCaloTemplate> >& vecCaloTemplate() const { return m_vecCaloTemplate; }

  void setVecCaloTemplate(const std::vector<std::vector<L1GtCaloTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtEnergySumTemplate> >& vecEnergySumTemplate() const {
    return m_vecEnergySumTemplate;
  }

  void setVecEnergySumTemplate(const std::vector<std::vector<L1GtEnergySumTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtJetCountsTemplate> >& vecJetCountsTemplate() const {
    return m_vecJetCountsTemplate;
  }

  void setVecJetCountsTemplate(const std::vector<std::vector<L1GtJetCountsTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtCastorTemplate> >& vecCastorTemplate() const { return m_vecCastorTemplate; }

  void setVecCastorTemplate(const std::vector<std::vector<L1GtCastorTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtHfBitCountsTemplate> >& vecHfBitCountsTemplate() const {
    return m_vecHfBitCountsTemplate;
  }

  void setVecHfBitCountsTemplate(const std::vector<std::vector<L1GtHfBitCountsTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtHfRingEtSumsTemplate> >& vecHfRingEtSumsTemplate() const {
    return m_vecHfRingEtSumsTemplate;
  }

  void setVecHfRingEtSumsTemplate(const std::vector<std::vector<L1GtHfRingEtSumsTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtBptxTemplate> >& vecBptxTemplate() const { return m_vecBptxTemplate; }

  void setVecBptxTemplate(const std::vector<std::vector<L1GtBptxTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtExternalTemplate> >& vecExternalTemplate() const {
    return m_vecExternalTemplate;
  }

  void setVecExternalTemplate(const std::vector<std::vector<L1GtExternalTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtCorrelationTemplate> >& vecCorrelationTemplate() const {
    return m_vecCorrelationTemplate;
  }

  void setVecCorrelationTemplate(const std::vector<std::vector<L1GtCorrelationTemplate> >&);

  // get / set the vectors containing the conditions for correlation templates
  //
  inline const std::vector<std::vector<L1GtMuonTemplate> >& corMuonTemplate() const { return m_corMuonTemplate; }

  void setCorMuonTemplate(const std::vector<std::vector<L1GtMuonTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtCaloTemplate> >& corCaloTemplate() const { return m_corCaloTemplate; }

  void setCorCaloTemplate(const std::vector<std::vector<L1GtCaloTemplate> >&);

  //
  inline const std::vector<std::vector<L1GtEnergySumTemplate> >& corEnergySumTemplate() const {
    return m_corEnergySumTemplate;
  }

  void setCorEnergySumTemplate(const std::vector<std::vector<L1GtEnergySumTemplate> >&);

  /// get / set the algorithm map (by name)
  inline const AlgorithmMap& gtAlgorithmMap() const { return m_algorithmMap; }

  void setGtAlgorithmMap(const AlgorithmMap&);

  /// get / set the algorithm map (by alias)
  inline const AlgorithmMap& gtAlgorithmAliasMap() const { return m_algorithmAliasMap; }

  void setGtAlgorithmAliasMap(const AlgorithmMap&);

  /// get / set the technical trigger map
  inline const AlgorithmMap& gtTechnicalTriggerMap() const { return m_technicalTriggerMap; }

  void setGtTechnicalTriggerMap(const AlgorithmMap&);

public:
  /// parse def.xml and vme.xml files
  void parseXmlFile(const std::string& defXmlFile, const std::string& vmeXmlFile);

public:
  /// get / set the XML parser creation date, author, description for menu interface, menu
  inline const std::string& gtTriggerMenuInterfaceDate() const { return m_triggerMenuInterfaceDate; }

  void setGtTriggerMenuInterfaceDate(const std::string&);

  inline const std::string& gtTriggerMenuInterfaceAuthor() const { return m_triggerMenuInterfaceAuthor; }

  void setGtTriggerMenuInterfaceAuthor(const std::string&);

  inline const std::string& gtTriggerMenuInterfaceDescription() const { return m_triggerMenuInterfaceDescription; }

  void setGtTriggerMenuInterfaceDescription(const std::string&);

  //

  inline const std::string& gtTriggerMenuDate() const { return m_triggerMenuDate; }

  void setGtTriggerMenuDate(const std::string&);

  inline const std::string& gtTriggerMenuAuthor() const { return m_triggerMenuAuthor; }

  void setGtTriggerMenuAuthor(const std::string&);

  inline const std::string& gtTriggerMenuDescription() const { return m_triggerMenuDescription; }

  void setGtTriggerMenuDescription(const std::string&);

  inline const std::string& gtAlgorithmImplementation() const { return m_algorithmImplementation; }

  void setGtAlgorithmImplementation(const std::string&);

private:
  // XML methods

  /// init xml system
  XERCES_CPP_NAMESPACE::XercesDOMParser* initXML(const std::string& xmlFile);

  /// find a named child of a xml node
  XERCES_CPP_NAMESPACE::DOMNode* findXMLChild(XERCES_CPP_NAMESPACE::DOMNode* startChild,
                                              const std::string& tagName,
                                              bool beginOnly = false,
                                              std::string* rest = nullptr);

  /// get a named attribute for an xml node as string
  std::string getXMLAttribute(const XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name);

  /// get the text value of a xml node as string
  std::string getXMLTextValue(XERCES_CPP_NAMESPACE::DOMNode* node);

  /// convert a hexadecimal string with up to 128 to 2 uint64_t
  bool hexString2UInt128(const std::string& hexString, uint64_t& dstL, uint64_t& dstH);

  /// get a hexadecimal value of a xml node containing text with up to 128 bit
  bool getXMLHexTextValue128(XERCES_CPP_NAMESPACE::DOMNode* node, uint64_t& dstL, uint64_t& dstH);

  /// get a hexadecimal value of a xml node containing text
  bool getXMLHexTextValue(XERCES_CPP_NAMESPACE::DOMNode* node, uint64_t& dst);

  /// get the number of bits in the max attribute of a condition child
  bool countConditionChildMaxBits(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& childName, unsigned int& dst);

  /// get values from a child of a condition
  bool getConditionChildValues(XERCES_CPP_NAMESPACE::DOMNode* node,
                               const std::string& childName,
                               unsigned int num,
                               std::vector<uint64_t>& dst);

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
  bool insertConditionIntoMap(L1GtCondition& cond, const int chipNr);

  /// insert an algorithm into algorithm map
  bool insertAlgorithmIntoMap(const L1GtAlgorithm& alg);

  /// insert a technical trigger into technical trigger map
  bool insertTechTriggerIntoMap(const L1GtAlgorithm& alg);

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
                         unsigned int num,
                         std::vector<bool>& mipDst,
                         std::vector<bool>& isoEnDst,
                         std::vector<bool>& isoReqDst);

  /// parse a muon condition
  bool parseMuon(XERCES_CPP_NAMESPACE::DOMNode* node,
                 const std::string& name,
                 unsigned int chipNr = 0,
                 const bool corrFlag = false);

  /// parse a calorimeter condition
  bool parseCalo(XERCES_CPP_NAMESPACE::DOMNode* node,
                 const std::string& name,
                 unsigned int chipNr = 0,
                 const bool corrFlag = false);

  /// parse an "energy sum" condition
  bool parseEnergySum(XERCES_CPP_NAMESPACE::DOMNode* node,
                      const std::string& name,
                      unsigned int chipNr = 0,
                      const bool corrFlag = false);

  /// parse a "jet counts" condition
  bool parseJetCounts(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr = 0);

  /// parse a CASTOR condition
  bool parseCastor(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr = 0);

  /// parse a HfBitCounts condition
  bool parseHfBitCounts(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr = 0);

  /// parse a HfRingEtSums condition
  bool parseHfRingEtSums(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr = 0);

  /// parse a Bptx condition
  bool parseBptx(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr = 0);

  /// parse an External condition
  bool parseExternal(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr = 0);

  /// parse a correlation condition
  bool parseCorrelation(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr = 0);

  /// parse all parse all identification attributes (trigger menu names, scale DB key, etc)
  bool parseId(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

  /// choose the parser for a particular condition
  bool workCondition(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr);

  /// parse all conditions
  bool parseConditions(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

  /// parse an algorithm and insert it into algorithm map.
  bool workAlgorithm(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name, unsigned int chipNr);

  /// parse all algorithms
  bool parseAlgorithms(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

  /// parse an algorithm and insert it into algorithm map.
  bool workTechTrigger(XERCES_CPP_NAMESPACE::DOMNode* node, const std::string& name);

  /// parse all algorithms
  bool parseTechTriggers(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);

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

  /// number of technical triggers
  unsigned int m_numberTechTriggers;

  /// jet counts
  unsigned int m_numberL1JetCounts;

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
  std::vector<std::vector<L1GtMuonTemplate> > m_vecMuonTemplate;
  std::vector<std::vector<L1GtCaloTemplate> > m_vecCaloTemplate;
  std::vector<std::vector<L1GtEnergySumTemplate> > m_vecEnergySumTemplate;
  std::vector<std::vector<L1GtJetCountsTemplate> > m_vecJetCountsTemplate;
  std::vector<std::vector<L1GtCastorTemplate> > m_vecCastorTemplate;
  std::vector<std::vector<L1GtHfBitCountsTemplate> > m_vecHfBitCountsTemplate;
  std::vector<std::vector<L1GtHfRingEtSumsTemplate> > m_vecHfRingEtSumsTemplate;
  std::vector<std::vector<L1GtBptxTemplate> > m_vecBptxTemplate;
  std::vector<std::vector<L1GtExternalTemplate> > m_vecExternalTemplate;

  std::vector<std::vector<L1GtCorrelationTemplate> > m_vecCorrelationTemplate;
  std::vector<std::vector<L1GtMuonTemplate> > m_corMuonTemplate;
  std::vector<std::vector<L1GtCaloTemplate> > m_corCaloTemplate;
  std::vector<std::vector<L1GtEnergySumTemplate> > m_corEnergySumTemplate;

  /// map containing the physics algorithms (by name)
  AlgorithmMap m_algorithmMap;

  /// map containing the physics algorithms (by alias)
  AlgorithmMap m_algorithmAliasMap;

  /// map containing the technical triggers
  AlgorithmMap m_technicalTriggerMap;
};

#endif /*L1GtConfigProducers_L1GtTriggerMenuXmlParser_h*/
