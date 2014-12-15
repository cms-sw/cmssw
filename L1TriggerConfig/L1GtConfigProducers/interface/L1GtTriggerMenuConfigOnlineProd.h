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
 *
 */

// system include files
#include "boost/shared_ptr.hpp"
#include "boost/lexical_cast.hpp"

#include <string>
#include <vector>
#include <iomanip>
#include <iostream>

// user include files
//   base class
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

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

    /// initialize the class (mainly reserve/resize)
    void init(const int numberConditionChips);


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
        bool condGEq;
        short countIndex;
        short countThreshold;

        // Oracle / Coral pretends that chargeCorrelation is bool in OMDS
        //   can not be - it has three values...
        //   but it reads/writes correctly the numerical value from OMDS (1, 2...)
        bool chargeCorrelation;
        std::string objectParameter1FK;
        std::string objectParameter2FK;
        std::string objectParameter3FK;
        std::string objectParameter4FK;
        std::string deltaEtaRange;
        std::string deltaPhiRange;
    };

    struct TableMenuObjectParameters {
        std::string opId;
        short ptHighThreshold;
        short ptLowThreshold;
        bool enableMip;
        bool enableIso;
        bool requestIso;
        bool energyOverflow;
        float etThreshold;
        std::string etaRange;
        std::string phiRange;
        short phiHigh;
        short phiLow;
        std::string qualityRange;
        bool charge;

    };

    struct TableMenuTechTrig {
        short bitNumberSh;
        std::string techName;
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

    /// retrieve table with technical triggers from DB
    bool tableMenuTechTrigFromDB(const std::string& gtSchema, const std::string& objKey);

private:

    /// return for an algorithm with bitNr the mapping between the integer index in logical expression
    /// and the condition name (FK)
    const std::map<int, std::string> condIndexNameMap(const short bitNr) const;

    /// convert a logical expression with indices to a logical expression with names
    std::string convertLogicalExpression(const std::string&, const std::map<int, std::string>&) const;

    /// return the chip number for an algorithm with index bitNumberSh
    int chipNumber(short) const;

    /// build the algorithm map in the menu
    void buildAlgorithmMap();

    /// build the technical trigger map in the menu
    void buildTechnicalTriggerMap();

    /// string to enum L1GtConditionCategory conversion
    L1GtConditionCategory strToEnumCondCategory(const std::string& strCategory);

    /// string to enum L1GtConditionType conversion
    L1GtConditionType strToEnumCondType(const std::string& strType);

    /// string to enum L1GtObject conversion
    L1GtObject strToEnumL1GtObject(const std::string& strObject);

    /// split a hex string in two 64-bit words returned as hex strings
    void splitHexStringInTwo64bitWords(
            const std::string& hexStr, std::string& hex0WordStr, std::string& hex1WordStr);

    /// get a list of chip numbers from the m_tableMenuAlgoCond table for a condition
    std::list<int> listChipNumber(const std::string&);

    void fillMuonObjectParameter(const std::string& opFK, L1GtMuonTemplate::ObjectParameter&);
    void addMuonCondition(const TableMenuCond&);

    void fillCaloObjectParameter(const std::string& opFK, L1GtCaloTemplate::ObjectParameter&);
    void addCaloCondition(const TableMenuCond&);

    void fillEnergySumObjectParameter(
            const std::string& opFK, L1GtEnergySumTemplate::ObjectParameter&, const L1GtObject&);
    void addEnergySumCondition(const TableMenuCond&);

    void addJetCountsCondition(const TableMenuCond&);
    void addHfBitCountsCondition(const TableMenuCond&);
    void addHfRingEtSumsCondition(const TableMenuCond&);
    void addCastorCondition(const TableMenuCond&);
    void addBptxCondition(const TableMenuCond&);
    void addExternalCondition(const TableMenuCond&);
    void addCorrelationCondition(const TableMenuCond&);

    /// add the conditions from a menu to the corresponding list
    void addConditions();


private:
    template<typename Result, typename Source>
    Result lexical_cast_from_hex(Source & value) const;


private:

    /// member to keep various tables retrieved from DB

    TableMenuGeneral m_tableMenuGeneral;
    std::vector<TableMenuAlgo> m_tableMenuAlgo;
    std::vector<TableMenuAlgoCond> m_tableMenuAlgoCond;
    std::vector<TableMenuCond> m_tableMenuCond;
    std::vector<TableMenuObjectParameters> m_tableMenuObjectParameters;
    std::vector<TableMenuTechTrig> m_tableMenuTechTrig;

private:

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

private:

    bool m_isDebugEnabled;


};



template<typename Result, typename Source>
Result L1GtTriggerMenuConfigOnlineProd::lexical_cast_from_hex(Source & value) const {

    std::stringstream convertor;
    convertor << value;

    Result result;
    if (! ( convertor >> std::hex >> result ) || !convertor.eof()) {
        throw boost::bad_lexical_cast();
    }

    return result;
}


#endif
