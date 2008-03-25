#ifndef CondFormats_L1TObjects_L1GtTriggerMenu_h
#define CondFormats_L1TObjects_L1GtTriggerMenu_h

/**
 * \class L1GtTriggerMenu
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

#include <ostream>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtCorrelationTemplate.h"

// forward declarations
class L1GtCondition;
class L1GtAlgorithm;

// class declaration
class L1GtTriggerMenu
{

public:

    // constructor
    L1GtTriggerMenu();

    L1GtTriggerMenu(const std::string&, const unsigned int numberConditionChips,
            const std::vector<std::vector<L1GtMuonTemplate> >&,
            const std::vector<std::vector<L1GtCaloTemplate> >&,
            const std::vector<std::vector<L1GtEnergySumTemplate> >&,
            const std::vector<std::vector<L1GtJetCountsTemplate> >&,
            const std::vector<std::vector<L1GtCorrelationTemplate> >&,
            const std::vector<std::vector<L1GtMuonTemplate> >&,
            const std::vector<std::vector<L1GtCaloTemplate> >&,
            const std::vector<std::vector<L1GtEnergySumTemplate> >&
    );

    // destructor
    virtual ~L1GtTriggerMenu();

public:

    /// get / set / build the condition maps
    inline const std::vector<ConditionMap>& gtConditionMap() const {
        return m_conditionMap;
    }

    void setGtConditionMap(const std::vector<ConditionMap>&);
    void buildGtConditionMap();

    /// get / set the trigger menu name
    inline const std::string& gtTriggerMenuName() const {
        return m_triggerMenuName;
    }

    void setGtTriggerMenuName(const std::string&);

    /// get / set the vectors containing the conditions
    inline const std::vector<std::vector<L1GtMuonTemplate> >& vecMuonTemplate() const {
        return m_vecMuonTemplate;
    }

    void setVecMuonTemplate(const std::vector<std::vector<L1GtMuonTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtCaloTemplate> >& vecCaloTemplate() const {
        return m_vecCaloTemplate;
    }

    void setVecCaloTemplate(const std::vector<std::vector<L1GtCaloTemplate> >&);

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
    inline const std::vector<std::vector<L1GtCorrelationTemplate> >& 
        vecCorrelationTemplate() const {
        
        return m_vecCorrelationTemplate;
    }

    void setVecCorrelationTemplate(
            const std::vector<std::vector<L1GtCorrelationTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtMuonTemplate> >& corMuonTemplate() const {
        return m_corMuonTemplate;
    }

    void setCorMuonTemplate(const std::vector<std::vector<L1GtMuonTemplate> >&);

    //
    inline const std::vector<std::vector<L1GtCaloTemplate> >& corCaloTemplate() const {
        return m_corCaloTemplate;
    }

    void setCorCaloTemplate(const std::vector<std::vector<L1GtCaloTemplate> >&);

    // get / set the vectors containing the conditions for correlation templates
    //
    inline const std::vector<std::vector<L1GtEnergySumTemplate> >& 
        corEnergySumTemplate() const {
        
        return m_corEnergySumTemplate;
    }

    void setCorEnergySumTemplate(
            const std::vector<std::vector<L1GtEnergySumTemplate> >&);


    /// get / set the algorithm map
    inline const AlgorithmMap& gtAlgorithmMap() const {
        return m_algorithmMap;
    }

    void setGtAlgorithmMap(const AlgorithmMap&);

    /// print the trigger menu
    /// allow various verbosity levels
    void print(std::ostream&, int&) const;

public:
    
    /// get the result for algorithm with name algName
    /// use directly the format of decisionWord (no typedef) 
    bool gtAlgorithmResult(const std::string& algName, const std::vector<bool>& decWord);
    
private:

    /// map containing the conditions (per condition chip) - transient 
    std::vector<ConditionMap> m_conditionMap;

private:

    /// menu name 
    std::string m_triggerMenuName;
    
    /// vectors containing the conditions
    /// explicit, due to persistency...    
    std::vector<std::vector<L1GtMuonTemplate> > m_vecMuonTemplate;
    std::vector<std::vector<L1GtCaloTemplate> > m_vecCaloTemplate;
    std::vector<std::vector<L1GtEnergySumTemplate> > m_vecEnergySumTemplate;
    std::vector<std::vector<L1GtJetCountsTemplate> > m_vecJetCountsTemplate;
    
    std::vector<std::vector<L1GtCorrelationTemplate> > m_vecCorrelationTemplate;
    std::vector<std::vector<L1GtMuonTemplate> > m_corMuonTemplate;
    std::vector<std::vector<L1GtCaloTemplate> > m_corCaloTemplate;
    std::vector<std::vector<L1GtEnergySumTemplate> > m_corEnergySumTemplate;

    /// map containing the algorithms (global map)
    AlgorithmMap m_algorithmMap;

};

#endif /*CondFormats_L1TObjects_L1GtTriggerMenu_h*/
