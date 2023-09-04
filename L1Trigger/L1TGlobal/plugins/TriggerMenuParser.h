#ifndef L1TGlobal_TriggerMenuParser_h
#define L1TGlobal_TriggerMenuParser_h

/**
 * \class TriggerMenuParser
 *
 *
 * Description: Xerces-C XML parser for the L1 Trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * \author  M. Eder            - HEPHY Vienna - ORCA version, reduced functionality
 * \author  Vladimir Rekovic
 *                - indexing
 *                - correlations with overlap object removal
 * \author R. Cavanaugh
 *                - displaced muons
 * \author Elisa Fontanesi                                                                               
 *                - extended for three-body correlation conditions                                                               
 *                                                                  
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <vector>

#include "L1Trigger/L1TGlobal/interface/TriggerMenuFwd.h"

#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/MuonShowerTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CaloTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumZdcTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationThreeBodyTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationWithOverlapRemovalTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ExternalTemplate.h"

#include "L1Trigger/L1TGlobal/interface/GlobalScales.h"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"

#include <cmath>
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1TUtmCondition.h"
#include "CondFormats/L1TObjects/interface/L1TUtmObject.h"
#include "CondFormats/L1TObjects/interface/L1TUtmCut.h"
#include "CondFormats/L1TObjects/interface/L1TUtmScale.h"

// forward declarations
class GlobalCondition;
class GlobalAlgorithm;

namespace l1t {

  typedef enum { COS, SIN } TrigFunc_t;

  // class declaration
  class TriggerMenuParser {
  public:
    /// constructor
    ///    empty
    TriggerMenuParser();

    /// destructor
    virtual ~TriggerMenuParser();

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
    inline const unsigned long gtTriggerMenuImplementation() const { return m_triggerMenuImplementation; }

    void setGtTriggerMenuImplementation(const unsigned long&);

    /// menu associated scale key
    inline const std::string& gtScaleDbKey() const { return m_scaleDbKey; }

    /// menu associated scales
    inline const GlobalScales& gtScales() const { return m_gtScales; }

    void setGtScaleDbKey(const std::string&);

    /// get / set the vectors containing the conditions
    inline const std::vector<std::vector<MuonTemplate> >& vecMuonTemplate() const { return m_vecMuonTemplate; }
    void setVecMuonTemplate(const std::vector<std::vector<MuonTemplate> >&);

    //
    inline const std::vector<std::vector<MuonShowerTemplate> >& vecMuonShowerTemplate() const {
      return m_vecMuonShowerTemplate;
    }
    void setVecMuonShowerTemplate(const std::vector<std::vector<MuonShowerTemplate> >&);

    //
    inline const std::vector<std::vector<CaloTemplate> >& vecCaloTemplate() const { return m_vecCaloTemplate; }

    void setVecCaloTemplate(const std::vector<std::vector<CaloTemplate> >&);

    //
    inline const std::vector<std::vector<EnergySumTemplate> >& vecEnergySumTemplate() const {
      return m_vecEnergySumTemplate;
    }

    void setVecEnergySumTemplate(const std::vector<std::vector<EnergySumTemplate> >&);

    //
    inline const std::vector<std::vector<EnergySumZdcTemplate> >& vecEnergySumZdcTemplate() const {
      return m_vecEnergySumZdcTemplate;
    }

    void setVecEnergySumZdcTemplate(const std::vector<std::vector<EnergySumZdcTemplate> >&);

    //
    inline const std::vector<std::vector<ExternalTemplate> >& vecExternalTemplate() const {
      return m_vecExternalTemplate;
    }

    void setVecExternalTemplate(const std::vector<std::vector<ExternalTemplate> >&);

    //
    inline const std::vector<std::vector<CorrelationTemplate> >& vecCorrelationTemplate() const {
      return m_vecCorrelationTemplate;
    }

    void setVecCorrelationTemplate(const std::vector<std::vector<CorrelationTemplate> >&);

    //
    inline const std::vector<std::vector<CorrelationThreeBodyTemplate> >& vecCorrelationThreeBodyTemplate() const {
      return m_vecCorrelationThreeBodyTemplate;
    }

    void setVecCorrelationThreeBodyTemplate(const std::vector<std::vector<CorrelationThreeBodyTemplate> >&);

    //
    inline const std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> >&
    vecCorrelationWithOverlapRemovalTemplate() const {
      return m_vecCorrelationWithOverlapRemovalTemplate;
    }

    void setVecCorrelationWithOverlapRemovalTemplate(
        const std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> >&);

    // get / set the vectors containing the conditions for correlation templates
    //
    inline const std::vector<std::vector<MuonTemplate> >& corMuonTemplate() const { return m_corMuonTemplate; }

    void setCorMuonTemplate(const std::vector<std::vector<MuonTemplate> >&);

    //
    inline const std::vector<std::vector<CaloTemplate> >& corCaloTemplate() const { return m_corCaloTemplate; }

    void setCorCaloTemplate(const std::vector<std::vector<CaloTemplate> >&);

    //
    inline const std::vector<std::vector<EnergySumTemplate> >& corEnergySumTemplate() const {
      return m_corEnergySumTemplate;
    }

    void setCorEnergySumTemplate(const std::vector<std::vector<EnergySumTemplate> >&);

    /// get / set the algorithm map (by name)
    inline const AlgorithmMap& gtAlgorithmMap() const { return m_algorithmMap; }

    void setGtAlgorithmMap(const AlgorithmMap&);

    /// get / set the algorithm map (by alias)
    inline const AlgorithmMap& gtAlgorithmAliasMap() const { return m_algorithmAliasMap; }

    void setGtAlgorithmAliasMap(const AlgorithmMap&);

  public:
    void parseCondFormats(const L1TUtmTriggerMenu* utmMenu);

    std::map<std::string, unsigned int> getExternalSignals(const L1TUtmTriggerMenu* utmMenu);

  public:
    /// get / set the XML parser creation date, author, description for menu interface, menu
    inline const std::string& gtTriggerMenuInterfaceDate() const { return m_triggerMenuInterfaceDate; }

    void setGtTriggerMenuInterfaceDate(const std::string&);

    inline const std::string& gtTriggerMenuInterfaceAuthor() const { return m_triggerMenuInterfaceAuthor; }

    void setGtTriggerMenuInterfaceAuthor(const std::string&);

    inline const std::string& gtTriggerMenuInterfaceDescription() const { return m_triggerMenuInterfaceDescription; }

    void setGtTriggerMenuInterfaceDescription(const std::string&);

    //

    inline const int gtTriggerMenuUUID() const { return m_triggerMenuUUID; }

    void setGtTriggerMenuUUID(const int);

    inline const std::string& gtTriggerMenuDate() const { return m_triggerMenuDate; }

    void setGtTriggerMenuDate(const std::string&);

    inline const std::string& gtTriggerMenuAuthor() const { return m_triggerMenuAuthor; }

    void setGtTriggerMenuAuthor(const std::string&);

    inline const std::string& gtTriggerMenuDescription() const { return m_triggerMenuDescription; }

    void setGtTriggerMenuDescription(const std::string&);

    inline const std::string& gtAlgorithmImplementation() const { return m_algorithmImplementation; }

    void setGtAlgorithmImplementation(const std::string&);

  private:
    // methods for conditions and algorithms

    /// clearMaps - delete all conditions and algorithms in
    /// the maps and clear the maps.
    void clearMaps();

    /// insertConditionIntoMap - safe insert of condition into condition map.
    /// if the condition name already exists, do not insert it and return false
    bool insertConditionIntoMap(GlobalCondition& cond, const int chipNr);

    /// insert an algorithm into algorithm map
    bool insertAlgorithmIntoMap(const GlobalAlgorithm& alg);

    template <typename T>
    std::string l1t2string(T);
    int l1tstr2int(const std::string data);

    /// parse scales
    /*     bool parseScale(L1TUtmScale scale); */
    //    bool parseScales( L1TUtmScale scale);
    bool parseScales(std::map<std::string, tmeventsetup::esScale> scaleMap);

    /// parse a muon condition
    /*     bool parseMuon(XERCES_CPP_NAMESPACE::DOMNode* node, */
    /*             const std::string& name, unsigned int chipNr = 0, */
    /*             const bool corrFlag = false); */
    bool parseMuon(L1TUtmCondition condMu, unsigned int chipNr = 0, const bool corrFlag = false);

    bool parseMuonCorr(const L1TUtmObject* condMu, unsigned int chipNr = 0);

    /// parse a muon shower condition
    bool parseMuonShower(L1TUtmCondition condMu, unsigned int chipNr = 0, const bool corrFlag = false);

    /// parse a calorimeter condition
    /*     bool parseCalo(XERCES_CPP_NAMESPACE::DOMNode* node, */
    /*             const std::string& name, unsigned int chipNr = 0, */
    /*             const bool corrFlag = false); */
    bool parseCalo(L1TUtmCondition condCalo, unsigned int chipNr = 0, const bool corrFlag = false);

    bool parseCaloCorr(const L1TUtmObject* corrCalo, unsigned int chipNr = 0);

    /// parse an "energy sum" condition
    /* bool parseEnergySum(XERCES_CPP_NAMESPACE::DOMNode* node, */
    /*         const std::string& name, unsigned int chipNr = 0, */
    /*         const bool corrFlag = false); */

    bool parseEnergySum(L1TUtmCondition condEnergySums, unsigned int chipNr = 0, const bool corrFlag = false);

    bool parseEnergySumZdc(L1TUtmCondition condEnergySumZdcs, unsigned int chipNr = 0, const bool corrFlag = false);

    bool parseEnergySumCorr(const L1TUtmObject* corrESum, unsigned int chipNr = 0);

    bool parseExternal(L1TUtmCondition condExt, unsigned int chipNr = 0);

    /// parse a correlation condition
    bool parseCorrelation(L1TUtmCondition corrCond, unsigned int chipNr = 0);

    /// parse a three-body correlation condition
    bool parseCorrelationThreeBody(L1TUtmCondition corrCond, unsigned int chipNr = 0);

    /// parse a correlation condition with overlap removal
    bool parseCorrelationWithOverlapRemoval(const L1TUtmCondition& corrCond, unsigned int chipNr = 0);

    /// parse all algorithms
    //bool parseAlgorithms(XERCES_CPP_NAMESPACE::XercesDOMParser* parser);
    bool parseAlgorithm(L1TUtmAlgorithm algorithm, unsigned int chipNr = 0);

    // Parse LUT for Cal Mu Eta
    void parseCalMuEta_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap, std::string obj1, std::string obj2);

    // Parse LUT for Cal Mu Phi
    void parseCalMuPhi_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap, std::string obj1, std::string obj2);

    // Parse LUT for Pt LUT in Mass calculation
    void parsePt_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap,
                      std::string lutpfx,
                      std::string obj1,
                      unsigned int prec);

    // Parse LUT for Upt LUT in Mass calculation for displaced muons
    void parseUpt_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap,
                       std::string lutpfx,
                       std::string obj1,
                       unsigned int prec);

    // Parse LUT for Delta Eta and Cosh
    void parseDeltaEta_Cosh_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap,
                                 std::string obj1,
                                 std::string obj2,
                                 unsigned int prec1,
                                 unsigned int prec2);

    // Parse LUT for Delta Eta and Cosh
    void parseDeltaPhi_Cos_LUTS(const std::map<std::string, tmeventsetup::esScale>& scaleMap,
                                const std::string& obj1,
                                const std::string& obj2,
                                unsigned int prec1,
                                unsigned int prec2);

    // Parse LUT for Sin(Phi),Cos(Phi) in TwoBodyPt algorithm calculation
    void parsePhi_Trig_LUTS(const std::map<std::string, tmeventsetup::esScale>& scaleMap,
                            const std::string& obj,
                            TrigFunc_t func,
                            unsigned int prec);

  private:
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
    unsigned long m_triggerMenuImplementation;
    unsigned long m_triggerMenuUUID;

    /// menu associated scale key
    std::string m_scaleDbKey;

    /// vectors containing the conditions
    /// explicit, due to persistency...
    std::vector<std::vector<MuonTemplate> > m_vecMuonTemplate;
    std::vector<std::vector<MuonShowerTemplate> > m_vecMuonShowerTemplate;
    std::vector<std::vector<CaloTemplate> > m_vecCaloTemplate;
    std::vector<std::vector<EnergySumTemplate> > m_vecEnergySumTemplate;
    std::vector<std::vector<EnergySumZdcTemplate> > m_vecEnergySumZdcTemplate;
    std::vector<std::vector<ExternalTemplate> > m_vecExternalTemplate;

    std::vector<std::vector<CorrelationTemplate> > m_vecCorrelationTemplate;
    std::vector<std::vector<CorrelationThreeBodyTemplate> > m_vecCorrelationThreeBodyTemplate;
    std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> > m_vecCorrelationWithOverlapRemovalTemplate;
    std::vector<std::vector<MuonTemplate> > m_corMuonTemplate;
    std::vector<std::vector<CaloTemplate> > m_corCaloTemplate;
    std::vector<std::vector<EnergySumTemplate> > m_corEnergySumTemplate;

    /// map containing the physics algorithms (by name)
    AlgorithmMap m_algorithmMap;

    /// map containing the physics algorithms (by alias)
    AlgorithmMap m_algorithmAliasMap;

    // class containing the scales from the L1 Menu XML
    GlobalScales m_gtScales;
  };

}  // namespace l1t
#endif /*L1TGlobal_TriggerMenuParser_h*/
