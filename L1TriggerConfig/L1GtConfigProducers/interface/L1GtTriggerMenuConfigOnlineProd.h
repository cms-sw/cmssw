#ifndef L1GtConfigProducers_L1GtTriggerMenuConfigOnlineProd_h
#define L1GtConfigProducers_L1GtTriggerMenuConfigOnlineProd_h

/**
 * \class L1GtTriggerMenuConfigOnlineProd
 *
 *
 * Description: online producer for L1GtTriggerMenu.
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
#include "boost/shared_ptr.hpp"
#include <string>
#include <vector>

// user include files
//   base class
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// forward declarations

// class declaration
class L1GtTriggerMenuConfigOnlineProd :
        public L1ConfigOnlineProdBase<L1GtTriggerMenuRcd, L1GtTriggerMenu>
{

public:

    /// constructor
    L1GtTriggerMenuConfigOnlineProd(const edm::ParameterSet&);

    /// destructor
    ~L1GtTriggerMenuConfigOnlineProd();

    /// public methods
    virtual boost::shared_ptr<L1GtTriggerMenu> newObject(const std::string& objectKey);

private:

    /// define simple structures to get the tables from DB

    struct TableMenuGeneral {
        std::string menuInterface;
        std::string menuImplementation;
        std::string algoImplTag;
        std::string scalesKey;
    };

    struct TableMenuAlgo {
        short bitNumberSh;
        int algChipNr;
        std::string algName;
        std::string algAlias;
        std::string logExpression;
    };

    struct TableMenuAlgoCond {
        short bitNumberSh;
        float condIndexF;
        std::string condFK;
    };

    struct TableMenuCond {
        std::string cond;
        std::string condCategory;
        std::string condType;
        std::string gtObject1;
        std::string gtObject2;
        bool condGeq;
        float countIndex;
        float countThreshold;
        float chargeCorrelation;
        std::string objectParameter1FK;
        std::string objectParameter2FK;
        std::string objectParameter3FK;
        std::string objectParameter4FK;
        std::string deltaEtaRange;
        std::string deltaPhiRange;
    };

    struct TableMenuObjectParameters {
        std::string opId;
        float ptHighThreshold;
        float ptLowThreshold;
        float enableMip;
        float enableIso;
        float requestIso;
        float energyOverflow;
        float etThreshold;
        std::string etaRange;
        std::string phiRange;
        float phiHigh;
        float phiLow;
        std::string qualityRange;
        float charge;

    };

    /// methods to retrieve the tables from DB

    /// retrieve table with general menu parameters from DB
    bool tableMenuGeneralFromDB(const std::string& gtSchema, const std::string& objKey);

    /// retrieve table with physics algorithms from DB
    bool tableMenuAlgoFromDB(const std::string& gtSchema, const std::string& objKey);

    /// retrieve table with conditions associated to physics algorithms from DB
    bool tableMenuAlgoCondFromDB(const std::string& gtSchema, const std::string& objKey);

    /// retrieve table with list of conditions in the menu
    bool tableMenuCondFromDB(const std::string& gtSchema, const std::string& objKey);

    /// retrieve table with object parameters from DB
    bool tableMenuObjectParametersFromDB(const std::string& gtSchema, const std::string& objKey);


    /// member to keep various tables retrieved from DB

    TableMenuGeneral m_tableMenuGeneral;
    std::vector<TableMenuAlgo> m_tableMenuAlgo;
    std::vector<TableMenuAlgoCond> m_tableMenuAlgoCond;
    std::vector<TableMenuCond> m_tableMenuCond;
    std::vector<TableMenuObjectParameters> m_tableMenuObjectParameters;



    /// menu representation

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

#endif
