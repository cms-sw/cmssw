#ifndef L1Trigger_L1TGlobal_TriggerMenu_h
#define L1Trigger_L1TGlobal_TriggerMenu_h

/**
 * \class TriggerMenu
 *
 *
 * Description: L1 trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *          Vladimir Rekovic - extend for overlap removal
 *          Elisa Fontanesi - extended for three-body correlation conditions
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <vector>
#include <map>

#include <iosfwd>

// user include files
#include "L1Trigger/L1TGlobal/interface/TriggerMenuFwd.h"

#include "L1Trigger/L1TGlobal/interface/GlobalScales.h"

#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CaloTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ExternalTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationThreeBodyTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationWithOverlapRemovalTemplate.h"

// forward declarations
class GlobalCondition;
class L1GtAlgorithm;
class GlobalScales;

// class declaration
class TriggerMenu {
public:
  // constructor
  TriggerMenu();

  TriggerMenu(const std::string&,
              const unsigned int numberConditionChips,
              const std::vector<std::vector<MuonTemplate> >&,
              const std::vector<std::vector<CaloTemplate> >&,
              const std::vector<std::vector<EnergySumTemplate> >&,
              const std::vector<std::vector<ExternalTemplate> >&,
              const std::vector<std::vector<CorrelationTemplate> >&,
              const std::vector<std::vector<CorrelationThreeBodyTemplate> >&,
              const std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> >&,
              const std::vector<std::vector<MuonTemplate> >&,
              const std::vector<std::vector<CaloTemplate> >&,
              const std::vector<std::vector<EnergySumTemplate> >&);

  // copy constructor
  TriggerMenu(const TriggerMenu&);

  // destructor
  virtual ~TriggerMenu();

  // assignment operator
  TriggerMenu& operator=(const TriggerMenu&);

public:
  /// get / set / build the condition maps
  inline const std::vector<l1t::ConditionMap>& gtConditionMap() const { return m_conditionMap; }

  void setGtConditionMap(const std::vector<l1t::ConditionMap>&);
  void buildGtConditionMap();

  /// get / set the trigger menu names
  inline const std::string& gtTriggerMenuInterface() const { return m_triggerMenuInterface; }

  void setGtTriggerMenuInterface(const std::string&);

  //
  inline const std::string& gtTriggerMenuName() const { return m_triggerMenuName; }

  void setGtTriggerMenuName(const std::string&);

  //
  inline const unsigned long gtTriggerMenuUUID() const { return m_triggerMenuUUID; }

  void setGtTriggerMenuUUID(const unsigned long uuid);

  //
  inline const unsigned long gtTriggerMenuImplementation() const { return m_triggerMenuImplementation; }

  void setGtTriggerMenuImplementation(const unsigned long);

  /// menu associated scale key
  inline const std::string& gtScaleDbKey() const { return m_scaleDbKey; }

  void setGtScaleDbKey(const std::string&);

  /// get / set the vectors containing the conditions
  inline const std::vector<std::vector<MuonTemplate> >& vecMuonTemplate() const { return m_vecMuonTemplate; }

  void setVecMuonTemplate(const std::vector<std::vector<MuonTemplate> >&);

  //
  inline const std::vector<std::vector<CaloTemplate> >& vecCaloTemplate() const { return m_vecCaloTemplate; }

  void setVecCaloTemplate(const std::vector<std::vector<CaloTemplate> >&);

  //
  inline const std::vector<std::vector<EnergySumTemplate> >& vecEnergySumTemplate() const {
    return m_vecEnergySumTemplate;
  }

  void setVecEnergySumTemplate(const std::vector<std::vector<EnergySumTemplate> >&);

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

  //
  inline const std::vector<std::vector<MuonTemplate> >& corMuonTemplate() const { return m_corMuonTemplate; }

  void setCorMuonTemplate(const std::vector<std::vector<MuonTemplate> >&);

  //
  inline const std::vector<std::vector<CaloTemplate> >& corCaloTemplate() const { return m_corCaloTemplate; }

  void setCorCaloTemplate(const std::vector<std::vector<CaloTemplate> >&);

  // get / set the vectors containing the conditions for correlation templates
  //
  inline const std::vector<std::vector<EnergySumTemplate> >& corEnergySumTemplate() const {
    return m_corEnergySumTemplate;
  }

  void setCorEnergySumTemplate(const std::vector<std::vector<EnergySumTemplate> >&);

  /// get / set the algorithm map (by name)
  inline const l1t::AlgorithmMap& gtAlgorithmMap() const { return m_algorithmMap; }

  void setGtAlgorithmMap(const l1t::AlgorithmMap&);

  /// get / set the algorithm map (by alias)
  inline const l1t::AlgorithmMap& gtAlgorithmAliasMap() const { return m_algorithmAliasMap; }

  void setGtAlgorithmAliasMap(const l1t::AlgorithmMap&);

  /// get the scales
  inline const l1t::GlobalScales& gtScales() const { return m_gtScales; }

  void setGtScales(const l1t::GlobalScales&);

  /*
    /// get / set the technical trigger map
    inline const l1t::AlgorithmMap& gtTechnicalTriggerMap() const {
        return m_technicalTriggerMap;
    }

//    void setGtTechnicalTriggerMap(const l1t::AlgorithmMap&);
*/
  /// print the trigger menu
  /// allow various verbosity levels
  void print(std::ostream&, int&) const;

public:
  /// get the result for algorithm with name algName
  /// use directly the format of decisionWord (no typedef)
  const bool gtAlgorithmResult(const std::string& algName, const std::vector<bool>& decWord) const;

private:
  /// map containing the conditions (per condition chip) - transient
  std::vector<l1t::ConditionMap> m_conditionMap;

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
  std::vector<std::vector<CaloTemplate> > m_vecCaloTemplate;
  std::vector<std::vector<EnergySumTemplate> > m_vecEnergySumTemplate;

  std::vector<std::vector<ExternalTemplate> > m_vecExternalTemplate;

  std::vector<std::vector<CorrelationTemplate> > m_vecCorrelationTemplate;
  std::vector<std::vector<CorrelationThreeBodyTemplate> > m_vecCorrelationThreeBodyTemplate;
  std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> > m_vecCorrelationWithOverlapRemovalTemplate;
  std::vector<std::vector<MuonTemplate> > m_corMuonTemplate;
  std::vector<std::vector<CaloTemplate> > m_corCaloTemplate;
  std::vector<std::vector<EnergySumTemplate> > m_corEnergySumTemplate;

  /// map containing the physics algorithms (by name)
  l1t::AlgorithmMap m_algorithmMap;

  /// map containing the physics algorithms (by alias)
  l1t::AlgorithmMap m_algorithmAliasMap;

  /// map containing the technical triggers
  //    l1t::AlgorithmMap m_technicalTriggerMap;

  // class containing the scales from the L1 Menu XML
  l1t::GlobalScales m_gtScales;
};

#endif /*L1Trigger_L1TGlobal_TriggerMenu_h*/
