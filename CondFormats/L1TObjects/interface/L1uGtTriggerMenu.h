#ifndef CondFormats_L1TObjects_L1uGtTriggerMenu_h
#define CondFormats_L1TObjects_L1uGtTriggerMenu_h

/**
 * \class L1uGtTriggerMenu
 *
 *
 * Description: L1 trigger menu.
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
#include <vector>
#include <map>

#include <iosfwd>

// user include files
#include "CondFormats/L1TObjects/interface/L1uGtTriggerMenuFwd.h"

#include "CondFormats/L1TObjects/interface/L1uGtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1uGtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCastorTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtHfBitCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtHfRingEtSumsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1uGtCorrelationTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtBptxTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtExternalTemplate.h"

// forward declarations
class L1uGtCondition;
class L1GtAlgorithm;

// class declaration
class L1uGtTriggerMenu
{

public:

    // constructor
    L1uGtTriggerMenu();

    L1uGtTriggerMenu(const std::string&, const unsigned int numberConditionChips,
            const std::vector<std::vector<L1uGtMuonTemplate> >&,
            const std::vector<std::vector<L1uGtCaloTemplate> >&,
            const std::vector<std::vector<L1GtEnergySumTemplate> >&,
            const std::vector<std::vector<L1GtJetCountsTemplate> >&,
            const std::vector<std::vector<L1GtCastorTemplate> >&,
            const std::vector<std::vector<L1GtHfBitCountsTemplate> >&,
            const std::vector<std::vector<L1GtHfRingEtSumsTemplate> >&,
            const std::vector<std::vector<L1GtBptxTemplate> >&,
            const std::vector<std::vector<L1GtExternalTemplate> >&,
            const std::vector<std::vector<L1uGtCorrelationTemplate> >&,
            const std::vector<std::vector<L1uGtMuonTemplate> >&,
            const std::vector<std::vector<L1uGtCaloTemplate> >&,
            const std::vector<std::vector<L1GtEnergySumTemplate> >&
    );

    // copy constructor
    L1uGtTriggerMenu(const L1uGtTriggerMenu&);

    // destructor
    virtual ~L1uGtTriggerMenu();

    // assignment operator
    L1uGtTriggerMenu& operator=(const L1uGtTriggerMenu&);

public:

    /// get / set / build the condition maps
    inline const std::vector<l1t::ConditionMap>& gtConditionMap() const {
        return m_conditionMap;
    }

    void setGtConditionMap(const std::vector<l1t::ConditionMap>&);
    void buildGtConditionMap();

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

    void setGtScaleDbKey(const std::string&);

    /// get / set the vectors containing the conditions
    inline const std::vector<std::vector<L1uGtMuonTemplate> >& vecMuonTemplate() const {
        return m_vecMuonTemplate;
    }

    void setVecMuonTemplate(const std::vector<std::vector<L1uGtMuonTemplate> >&);

    //
    inline const std::vector<std::vector<L1uGtCaloTemplate> >& vecCaloTemplate() const {
        return m_vecCaloTemplate;
    }

    void setVecCaloTemplate(const std::vector<std::vector<L1uGtCaloTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtEnergySumTemplate> >&
        vecEnergySumTemplate() const {

        return m_vecEnergySumTemplate;
    }

    void setVecEnergySumTemplate(
            const std::vector<std::vector<L1GtEnergySumTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtJetCountsTemplate> >&
        vecJetCountsTemplate() const {

        return m_vecJetCountsTemplate;
    }

    void setVecJetCountsTemplate(
            const std::vector<std::vector<L1GtJetCountsTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtCastorTemplate> >&
        vecCastorTemplate() const {

        return m_vecCastorTemplate;
    }

    void setVecCastorTemplate(
            const std::vector<std::vector<L1GtCastorTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtHfBitCountsTemplate> >&
        vecHfBitCountsTemplate() const {

        return m_vecHfBitCountsTemplate;
    }

    void setVecHfBitCountsTemplate(
            const std::vector<std::vector<L1GtHfBitCountsTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtHfRingEtSumsTemplate> >&
        vecHfRingEtSumsTemplate() const {

        return m_vecHfRingEtSumsTemplate;
    }

    void setVecHfRingEtSumsTemplate(
            const std::vector<std::vector<L1GtHfRingEtSumsTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtBptxTemplate> >&
        vecBptxTemplate() const {

        return m_vecBptxTemplate;
    }

    void setVecBptxTemplate(
            const std::vector<std::vector<L1GtBptxTemplate> >&);

    //

    inline const std::vector<std::vector<L1GtExternalTemplate> >&
        vecExternalTemplate() const {

        return m_vecExternalTemplate;
    }

    void setVecExternalTemplate(
            const std::vector<std::vector<L1GtExternalTemplate> >&);

    //
    inline const std::vector<std::vector<L1uGtCorrelationTemplate> >&
        vecCorrelationTemplate() const {

        return m_vecCorrelationTemplate;
    }

    void setVecCorrelationTemplate(
            const std::vector<std::vector<L1uGtCorrelationTemplate> >&);

    //
    inline const std::vector<std::vector<L1uGtMuonTemplate> >& corMuonTemplate() const {
        return m_corMuonTemplate;
    }

    void setCorMuonTemplate(const std::vector<std::vector<L1uGtMuonTemplate> >&);

    //
    inline const std::vector<std::vector<L1uGtCaloTemplate> >& corCaloTemplate() const {
        return m_corCaloTemplate;
    }

    void setCorCaloTemplate(const std::vector<std::vector<L1uGtCaloTemplate> >&);

    // get / set the vectors containing the conditions for correlation templates
    //
    inline const std::vector<std::vector<L1GtEnergySumTemplate> >&
        corEnergySumTemplate() const {

        return m_corEnergySumTemplate;
    }

    void setCorEnergySumTemplate(
            const std::vector<std::vector<L1GtEnergySumTemplate> >&);


    /// get / set the algorithm map (by name)
    inline const l1t::AlgorithmMap& gtAlgorithmMap() const {
        return m_algorithmMap;
    }

    void setGtAlgorithmMap(const l1t::AlgorithmMap&);

    /// get / set the algorithm map (by alias)
    inline const l1t::AlgorithmMap& gtAlgorithmAliasMap() const {
        return m_algorithmAliasMap;
    }

    void setGtAlgorithmAliasMap(const l1t::AlgorithmMap&);

    /// get / set the technical trigger map
    inline const l1t::AlgorithmMap& gtTechnicalTriggerMap() const {
        return m_technicalTriggerMap;
    }

    void setGtTechnicalTriggerMap(const l1t::AlgorithmMap&);

    /// print the trigger menu
    /// allow various verbosity levels
    void print(std::ostream&, int&) const;

public:

    /// get the result for algorithm with name algName
    /// use directly the format of decisionWord (no typedef)
    const bool gtAlgorithmResult(const std::string& algName,
            const std::vector<bool>& decWord) const;

private:

    /// map containing the conditions (per condition chip) - transient
    mutable std::vector<l1t::ConditionMap> m_conditionMap;

private:

    /// menu names
    std::string m_triggerMenuInterface;
    std::string m_triggerMenuName;
    std::string m_triggerMenuImplementation;

    /// menu associated scale key
    std::string m_scaleDbKey;

    /// vectors containing the conditions
    /// explicit, due to persistency...
    std::vector<std::vector<L1uGtMuonTemplate> > m_vecMuonTemplate;
    std::vector<std::vector<L1uGtCaloTemplate> > m_vecCaloTemplate;
    std::vector<std::vector<L1GtEnergySumTemplate> > m_vecEnergySumTemplate;
    std::vector<std::vector<L1GtJetCountsTemplate> > m_vecJetCountsTemplate;
    std::vector<std::vector<L1GtCastorTemplate> > m_vecCastorTemplate;
    std::vector<std::vector<L1GtHfBitCountsTemplate> > m_vecHfBitCountsTemplate;
    std::vector<std::vector<L1GtHfRingEtSumsTemplate> > m_vecHfRingEtSumsTemplate;
    std::vector<std::vector<L1GtBptxTemplate> > m_vecBptxTemplate;
    std::vector<std::vector<L1GtExternalTemplate> > m_vecExternalTemplate;

    std::vector<std::vector<L1uGtCorrelationTemplate> > m_vecCorrelationTemplate;
    std::vector<std::vector<L1uGtMuonTemplate> > m_corMuonTemplate;
    std::vector<std::vector<L1uGtCaloTemplate> > m_corCaloTemplate;
    std::vector<std::vector<L1GtEnergySumTemplate> > m_corEnergySumTemplate;

    /// map containing the physics algorithms (by name)
    l1t::AlgorithmMap m_algorithmMap;

    /// map containing the physics algorithms (by alias)
    l1t::AlgorithmMap m_algorithmAliasMap;

    /// map containing the technical triggers
    l1t::AlgorithmMap m_technicalTriggerMap;


};

#endif /*CondFormats_L1TObjects_L1uGtTriggerMenu_h*/
