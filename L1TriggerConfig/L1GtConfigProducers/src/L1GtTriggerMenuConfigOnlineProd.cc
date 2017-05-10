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

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuConfigOnlineProd.h"

// system include files
#include <string>
#include <vector>
#include <map>
#include <list>
#include <utility>
#include <algorithm>

#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"


// constructor
L1GtTriggerMenuConfigOnlineProd::L1GtTriggerMenuConfigOnlineProd(const edm::ParameterSet& parSet) :
    L1ConfigOnlineProdBase<L1GtTriggerMenuRcd, L1GtTriggerMenu> (parSet), m_isDebugEnabled(
            edm::isDebugEnabled()) {

    // empty

}

// destructor
L1GtTriggerMenuConfigOnlineProd::~L1GtTriggerMenuConfigOnlineProd() {

    // empty

}

// public methods

// initialize the class (mainly reserve/resize)
void L1GtTriggerMenuConfigOnlineProd::init(const int numberConditionChips) {

    // resize the vectors of condition maps
    // the number of condition chips should be correctly set

    m_vecMuonTemplate.resize(numberConditionChips);
    m_vecCaloTemplate.resize(numberConditionChips);
    m_vecEnergySumTemplate.resize(numberConditionChips);
    m_vecJetCountsTemplate.resize(numberConditionChips);
    m_vecCastorTemplate.resize(numberConditionChips);
    m_vecHfBitCountsTemplate.resize(numberConditionChips);
    m_vecHfRingEtSumsTemplate.resize(numberConditionChips);
    m_vecBptxTemplate.resize(numberConditionChips);
    m_vecExternalTemplate.resize(numberConditionChips);

    m_vecCorrelationTemplate.resize(numberConditionChips);
    m_corMuonTemplate.resize(numberConditionChips);
    m_corCaloTemplate.resize(numberConditionChips);
    m_corEnergySumTemplate.resize(numberConditionChips);


}

boost::shared_ptr<L1GtTriggerMenu> L1GtTriggerMenuConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // FIXME seems to not work anymore in constructor...
    m_isDebugEnabled = edm::isDebugEnabled();

    // shared pointer for L1GtTriggerMenu - empty menu
    boost::shared_ptr<L1GtTriggerMenu> pL1GtTriggerMenuEmpty =
            boost::shared_ptr<L1GtTriggerMenu>(new L1GtTriggerMenu());

    // FIXME get it from L1GtStableParameters?
    //       initialize once, from outside
    const unsigned int numberConditionChips = 2;
    init(numberConditionChips);

    const std::string gtSchema = "CMS_GT";

    // l1GtTriggerMenu: parameters in table GTFE_SETUP_FK
    // the objectKey for the menu obtained from GT_SETUP is the L1T_MENU_IMPL key

    // SQL queries:

    // retrieve table with general menu parameters from DB, view L1T_MENU_GENERAL_VIEW
    if (!tableMenuGeneralFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenuEmpty;

    }

    // build the menu name
    std::string menuName = m_tableMenuGeneral.menuInterface + "/" + m_tableMenuGeneral.scalesKey
            + "/" + m_tableMenuGeneral.algoImplTag;


    // retrieve table with physics algorithms from DB, view L1T_MENU_ALGO_VIEW
    if (!tableMenuAlgoFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenuEmpty;

    }

    // retrieve table with conditions associated to physics algorithms from DB
    if (!tableMenuAlgoCondFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenuEmpty;

    }

    // retrieve table with list of conditions in the menu
    if (!tableMenuCondFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenuEmpty;

    }


    // retrieve table with object parameters from DB, view CMS_GT.L1T_MENU_OP_VIEW
    if (!tableMenuObjectParametersFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenuEmpty;

    }

    // retrieve table with technical triggers from DB, view L1T_MENU_TECHTRIG_VIEW
    if (!tableMenuTechTrigFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenuEmpty;

    }

    // build the algorithm map in the menu
    buildAlgorithmMap();

    // build the technical trigger map in the menu
    buildTechnicalTriggerMap();

    // add the conditions from a menu to the corresponding list
    addConditions();

    // fill the record
    boost::shared_ptr<L1GtTriggerMenu> pL1GtTriggerMenu = boost::shared_ptr<L1GtTriggerMenu>(
                new L1GtTriggerMenu(menuName, numberConditionChips,
                        m_vecMuonTemplate,
                        m_vecCaloTemplate,
                        m_vecEnergySumTemplate,
                        m_vecJetCountsTemplate,
                        m_vecCastorTemplate,
                        m_vecHfBitCountsTemplate,
                        m_vecHfRingEtSumsTemplate,
                        m_vecBptxTemplate,
                        m_vecExternalTemplate,
                        m_vecCorrelationTemplate,
                        m_corMuonTemplate,
                        m_corCaloTemplate,
                        m_corEnergySumTemplate) );

    pL1GtTriggerMenu->setGtTriggerMenuInterface(m_tableMenuGeneral.menuInterface);
    pL1GtTriggerMenu->setGtTriggerMenuImplementation(m_tableMenuGeneral.menuImplementation);
    pL1GtTriggerMenu->setGtScaleDbKey(m_tableMenuGeneral.scalesKey);

    pL1GtTriggerMenu->setGtAlgorithmMap(m_algorithmMap);
    pL1GtTriggerMenu->setGtAlgorithmAliasMap(m_algorithmAliasMap);
    pL1GtTriggerMenu->setGtTechnicalTriggerMap(m_technicalTriggerMap);


    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
            << "\nThe following L1GtTriggerMenu record was read from OMDS: \n"
            << std::endl;

        std::ostringstream myCoutStream;
        int verbosity = 0;
        pL1GtTriggerMenu->print(myCoutStream, verbosity);
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n" << std::endl;

        verbosity = 2;
        pL1GtTriggerMenu->print(myCoutStream, verbosity);
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << myCoutStream.str() << "\n" << std::endl;

    }

    return pL1GtTriggerMenu;
}

// retrieve table with general menu parameters from DB
bool L1GtTriggerMenuConfigOnlineProd::tableMenuGeneralFromDB(
        const std::string& gtSchema, const std::string& objectKey) {

    // select * from CMS_GT.L1T_MENU_GENERAL_VIEW
    //     where L1T_MENU_GENERAL_VIEW.MENU_IMPLEMENTATION = objectKey

    const std::vector<std::string>& columns = m_omdsReader.columnNamesView(
            gtSchema, "L1T_MENU_GENERAL_VIEW");

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n List of columns in L1T_MENU_GENERAL_VIEW:\n" << std::endl;
        for (std::vector<std::string>::const_iterator iter = columns.begin(); iter != columns.end(); iter++) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << ( *iter ) << std::endl;

        }
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n\n" << std::endl;
    }

    l1t::OMDSReader::QueryResults results = m_omdsReader.basicQueryView(
            columns, gtSchema, "L1T_MENU_GENERAL_VIEW",
            "L1T_MENU_GENERAL_VIEW.MENU_IMPLEMENTATION", m_omdsReader.singleAttribute(objectKey));

    // check if query was successful
    if (results.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem to get content of L1T_MENU_GENERAL_VIEW for L1GtTriggerMenu implementation key: "
                << objectKey;
        return false;
    }

    // retrieve menu interface name, scales key, algorithm implementation tag

    for (std::vector<std::string>::const_iterator constIt = columns.begin(); constIt
            != columns.end(); ++constIt) {

        if ( ( *constIt ) == "MENU_IMPLEMENTATION") {
            results.fillVariable(*constIt, m_tableMenuGeneral.menuImplementation);

        } else if ( ( *constIt ) == "INTERFACE") {
            results.fillVariable(*constIt, m_tableMenuGeneral.menuInterface);

        } else if ( ( *constIt ) == "SCALES_KEY") {
            results.fillVariable(*constIt, m_tableMenuGeneral.scalesKey);

        } else if ( ( *constIt ) == "ALGO_IMPL_TAG") {
            results.fillVariable(*constIt, m_tableMenuGeneral.algoImplTag);

        } else {
            // do nothing

        }

    }

    // cross checks
    if (m_tableMenuGeneral.menuImplementation != objectKey)   {

        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n L1 trigger menu implementation read from querying view not identical"
                << "\n with menu key:"
                << "\n   from view: " << m_tableMenuGeneral.menuImplementation
                << "\n   from key:  " <<  objectKey
                << "\n Menu implementation name set from key."
                << std::endl;
        m_tableMenuGeneral.menuImplementation = objectKey;
    }

    //
    return true;

}

// retrieve table with physics algorithms from DB
bool L1GtTriggerMenuConfigOnlineProd::tableMenuAlgoFromDB(
        const std::string& gtSchema, const std::string& objectKey) {

    // select * from CMS_GT.L1T_MENU_ALGO_VIEW
    //     where L1T_MENU_ALGO_VIEW.MENU_IMPLEMENTATION = objectKey

    const std::vector<std::string>& columnsMenuAlgo = m_omdsReader.columnNamesView(
            gtSchema, "L1T_MENU_ALGO_VIEW");

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n List of columns in L1T_MENU_ALGO_VIEW:\n"
            << std::endl;
        for (std::vector<std::string>::const_iterator iter = columnsMenuAlgo.begin(); iter != columnsMenuAlgo.end(); iter++) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << ( *iter ) << std::endl;

        }
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n\n" << std::endl;
    }

    l1t::OMDSReader::QueryResults resultsMenuAlgo = m_omdsReader.basicQueryView(
            columnsMenuAlgo, gtSchema, "L1T_MENU_ALGO_VIEW", "L1T_MENU_ALGO_VIEW.MENU_IMPLEMENTATION",
            m_omdsReader.singleAttribute(objectKey));


    // check if query was successful is based on size of returned list
    // BUT one can have menus w/o physics algorithms - no error for empty list, but a warning!
    if (resultsMenuAlgo.queryFailed()) {
        edm::LogWarning("L1-O2O")
                << "Warning: Content of L1T_MENU_ALGO_VIEW for L1GtTriggerMenu implementation key: "
                << "\n  " << objectKey << "\nis empty!"
                << "\nNo physics algorithms are found for this menu.";

    }

    TableMenuAlgo menuAlgo;
    int resultsMenuAlgoRows = resultsMenuAlgo.numberRows();

    for (int iRow = 0; iRow < resultsMenuAlgoRows; ++iRow) {

        for (std::vector<std::string>::const_iterator constIt = columnsMenuAlgo.begin(); constIt
                != columnsMenuAlgo.end(); ++constIt) {

            if ( ( *constIt ) == "ALGO_INDEX") {
                resultsMenuAlgo.fillVariableFromRow(*constIt, iRow, menuAlgo.bitNumberSh);

            } else if ( ( *constIt ) == "NAME") {
                resultsMenuAlgo.fillVariableFromRow(*constIt, iRow, menuAlgo.algName);

            } else if ( ( *constIt ) == "ALIAS") {
                resultsMenuAlgo.fillVariableFromRow(*constIt, iRow, menuAlgo.algAlias);

            } else if ( ( *constIt ) == "LOGICEXPR") {
                resultsMenuAlgo.fillVariableFromRow(*constIt, iRow, menuAlgo.logExpression);

            } else {
                // do nothing

            }

        }

        LogTrace("L1GtTriggerMenuConfigOnlineProd")
            << "Row  " << iRow << ": index = " << menuAlgo.bitNumberSh << " algName = " << menuAlgo.algName
            << " algAlias = " << menuAlgo.algAlias << " logExpression = '" << menuAlgo.logExpression << "'"
            << std::endl;

        m_tableMenuAlgo.push_back(menuAlgo);


    }

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
            << "\n Number of rows read from L1T_MENU_ALGO_VIEW: " << resultsMenuAlgoRows
            << std::endl;

    }

    return true;

}

// retrieve table with conditions associated to physics algorithms from DB
bool L1GtTriggerMenuConfigOnlineProd::tableMenuAlgoCondFromDB(
        const std::string& gtSchema, const std::string& objectKey) {

    // get list of conditions associated with the algorithms

    // select * from CMS_GT.L1T_MENU_ALGO_COND_VIEW
    //     where L1T_MENU_ALGO_COND_VIEW.MENU_IMPLEMENTATION = objectKey

    const std::vector<std::string>& columnsMenuAlgoCond = m_omdsReader.columnNamesView(
            gtSchema, "L1T_MENU_ALGO_COND_VIEW");

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n List of columns in L1T_MENU_ALGO_COND_VIEW:\n" << std::endl;
        for (std::vector<std::string>::const_iterator iter = columnsMenuAlgoCond.begin(); iter
                != columnsMenuAlgoCond.end(); iter++) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << ( *iter ) << std::endl;

        }
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n\n" << std::endl;
    }

    l1t::OMDSReader::QueryResults resultsMenuAlgoCond = m_omdsReader.basicQueryView(
            columnsMenuAlgoCond, gtSchema, "L1T_MENU_ALGO_COND_VIEW",
            "L1T_MENU_ALGO_COND_VIEW.MENU_IMPLEMENTATION", m_omdsReader.singleAttribute(objectKey));

    // check if query was successful is based on size of returned list
    // BUT one can have menus w/o algorithms and conditions - no error for empty list, but a warning!
    if (resultsMenuAlgoCond.queryFailed()) {
        edm::LogWarning("L1-O2O")
                << "Warning: Content of L1T_MENU_ALGO_COND_VIEW for L1GtTriggerMenu implementation key: "
                << "\n  " << objectKey << "\nis empty!"
                << "\nNo list of condition associated to each algorithm are found for this menu.";

    }

    //
    TableMenuAlgoCond menuAlgoCond;
    int resultsMenuAlgoCondRows = resultsMenuAlgoCond.numberRows();

    for (int iRow = 0; iRow < resultsMenuAlgoCondRows; ++iRow) {

        for (std::vector<std::string>::const_iterator constIt = columnsMenuAlgoCond.begin(); constIt
                != columnsMenuAlgoCond.end(); ++constIt) {

            if ( ( *constIt ) == "ALGO_INDEX") {
                resultsMenuAlgoCond.fillVariableFromRow(
                        *constIt, iRow, menuAlgoCond.bitNumberSh);

            } else if ( ( *constIt ) == "COND_INDEX") {
                resultsMenuAlgoCond.fillVariableFromRow(
                        *constIt, iRow, menuAlgoCond.condIndexF);

            } else if ( ( *constIt ) == "COND_FK") {
                resultsMenuAlgoCond.fillVariableFromRow(*constIt, iRow, menuAlgoCond.condFK);

            } else {
                // do nothing

            }

        }

        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "Row  " << iRow << ": ALGO_INDEX = "
                << menuAlgoCond.bitNumberSh << " COND_INDEX = " << menuAlgoCond.condIndexF
                << " COND_FK = " << menuAlgoCond.condFK << std::endl;

        m_tableMenuAlgoCond.push_back(menuAlgoCond);

    }

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n Number of rows read from L1T_MENU_ALGO_COND_VIEW: "
                << resultsMenuAlgoCondRows << std::endl;
    }


    return true;

}

// retrieve table with list of conditions in the menu
bool L1GtTriggerMenuConfigOnlineProd::tableMenuCondFromDB(
        const std::string& gtSchema, const std::string& objectKey) {

    // select * from CMS_GT.L1T_MENU_COND_VIEW
    //     where L1T_MENU_COND_VIEW.MENU_IMPLEMENTATION = objectKey

    const std::vector<std::string>& columnsMenuCond = m_omdsReader.columnNamesView(
            gtSchema, "L1T_MENU_COND_VIEW");

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n List of columns in L1T_MENU_COND_VIEW:\n" << std::endl;
        for (std::vector<std::string>::const_iterator iter = columnsMenuCond.begin(); iter
                != columnsMenuCond.end(); iter++) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << ( *iter ) << std::endl;

        }
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n\n" << std::endl;
    }

    l1t::OMDSReader::QueryResults resultsMenuCond = m_omdsReader.basicQueryView(
            columnsMenuCond, gtSchema, "L1T_MENU_COND_VIEW",
            "L1T_MENU_COND_VIEW.MENU_IMPLEMENTATION", m_omdsReader.singleAttribute(objectKey));

    // check if query was successful is based on size of returned list
    // BUT one can have menus w/o conditions - no error for empty list, but a warning!
    if (resultsMenuCond.queryFailed()) {
        edm::LogWarning("L1-O2O")
                << "Warning: Content of L1T_MENU_COND_VIEW for L1GtTriggerMenu implementation key: "
                << "\n  " << objectKey << "\nis empty!"
                << "\nNo conditions associated to menu are found for this menu.";

    }

    //
    TableMenuCond menuCond;
    int resultsMenuCondRows = resultsMenuCond.numberRows();

    for (int iRow = 0; iRow < resultsMenuCondRows; ++iRow) {

        for (std::vector<std::string>::const_iterator constIt = columnsMenuCond.begin(); constIt
                != columnsMenuCond.end(); ++constIt) {

            if ( ( *constIt ) == "COND") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.cond);

            } else if ( ( *constIt ) == "COND_CATEGORY") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.condCategory);

            } else if ( ( *constIt ) == "COND_TYPE") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.condType);

            } else if ( ( *constIt ) == "GT_OBJECT_1") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.gtObject1);

            } else if ( ( *constIt ) == "GT_OBJECT_2") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.gtObject2);

            } else if ( ( *constIt ) == "COND_GEQ") {
                //float condGEqFloat = -1;
                //resultsMenuCond.fillVariableFromRow(*constIt, iRow, condGEqFloat);
                //menuCond.condGEq = (condGEqFloat > 0.5) ? true : false;
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.condGEq);

            } else if ( ( *constIt ) == "COUNT_INDEX") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.countIndex);

            } else if ( ( *constIt ) == "COUNT_THRESHOLD") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.countThreshold);

            } else if ( ( *constIt ) == "CHARGE_CORRELATION") {
                resultsMenuCond.fillVariableFromRow(
                        *constIt, iRow, menuCond.chargeCorrelation);

            } else if ( ( *constIt ) == "OBJECT_PARAMETER_1_FK") {
                resultsMenuCond.fillVariableFromRow(
                        *constIt, iRow, menuCond.objectParameter1FK);

            } else if ( ( *constIt ) == "OBJECT_PARAMETER_2_FK") {
                resultsMenuCond.fillVariableFromRow(
                        *constIt, iRow, menuCond.objectParameter2FK);

            } else if ( ( *constIt ) == "OBJECT_PARAMETER_3_FK") {
                resultsMenuCond.fillVariableFromRow(
                        *constIt, iRow, menuCond.objectParameter3FK);

            } else if ( ( *constIt ) == "OBJECT_PARAMETER_4_FK") {
                resultsMenuCond.fillVariableFromRow(
                        *constIt, iRow, menuCond.objectParameter4FK);

            } else if ( ( *constIt ) == "DELTA_ETA_RANGE") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.deltaEtaRange);

            } else if ( ( *constIt ) == "DELTA_PHI_RANGE") {
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.deltaPhiRange);

            } else {
                // do nothing

            }

        }

        LogTrace("L1GtTriggerMenuConfigOnlineProd")
            << " COND " << menuCond.cond
            << " COND_CATEGORY " << menuCond.condCategory
            << " COND_TYPE " << menuCond.condType
            << " GT_OBJECT_1 " << menuCond.gtObject1
            << " GT_OBJECT_2 " << menuCond.gtObject2
            << " COND_GEQ " << menuCond.condGEq << "\n"
            << " COUNT_INDEX " << menuCond.countIndex
            << " COUNT_THRESHOLD " << menuCond.countThreshold << "\n"
            << " CHARGE_CORRELATION " << menuCond.chargeCorrelation << "\n"
            << " OBJECT_PARAMETER_1_FK " << menuCond.objectParameter1FK
            << " OBJECT_PARAMETER_2_FK " << menuCond.objectParameter2FK
            << " OBJECT_PARAMETER_3_FK " << menuCond.objectParameter3FK
            << " OBJECT_PARAMETER_4_FK " << menuCond.objectParameter4FK << "\n"
            << " DELTA_ETA_RANGE " << menuCond.deltaEtaRange
            << " DELTA_PHI_RANGE " << menuCond.deltaPhiRange;

        m_tableMenuCond.push_back(menuCond);


    }

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
            << "\n Number of rows read from L1T_MENU_COND_VIEW: " << resultsMenuCondRows
            << std::endl;

    }

    return true;

}

// retrieve table with object parameters from DB
bool L1GtTriggerMenuConfigOnlineProd::tableMenuObjectParametersFromDB(
        const std::string& gtSchema, const std::string& objectKey) {

    // get the list of object parameters in the menu

    // select * from CMS_GT.L1T_MENU_OP_VIEW
    //     where L1T_MENU_OP_VIEW.MENU_IMPLEMENTATION = objectKey

    const std::vector<std::string>& columnsMenuOp = m_omdsReader.columnNamesView(
            gtSchema, "L1T_MENU_OP_VIEW");

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n List of columns in L1T_MENU_OP_VIEW:\n"
                << std::endl;
        for (std::vector<std::string>::const_iterator iter = columnsMenuOp.begin(); iter
                != columnsMenuOp.end(); iter++) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << ( *iter ) << std::endl;

        }
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n\n" << std::endl;
    }

    l1t::OMDSReader::QueryResults resultsMenuOp = m_omdsReader.basicQueryView(
            columnsMenuOp, gtSchema, "L1T_MENU_OP_VIEW", "L1T_MENU_OP_VIEW.MENU_IMPLEMENTATION",
            m_omdsReader.singleAttribute(objectKey));

    // check if query was successful is based on size of returned list
    // BUT one can have menus w/o object parameters - no error for empty list, but a warning!
    if (resultsMenuOp.queryFailed()) {
        edm::LogWarning("L1-O2O")
                << "Warning: Content of L1T_MENU_OP_VIEW for L1GtTriggerMenu implementation key: "
                << "\n  " << objectKey << "\nis empty!"
                << "\nNo object parameters associated to menu are found for this menu.";

    }

    TableMenuObjectParameters menuObjectParameters;
    int resultsMenuOpRows = resultsMenuOp.numberRows();

    for (int iRow = 0; iRow < resultsMenuOpRows; ++iRow) {

        for (std::vector<std::string>::const_iterator constIt = columnsMenuOp.begin(); constIt
                != columnsMenuOp.end(); ++constIt) {

            if ( ( *constIt ) == "ID") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.opId);

            } else if ( ( *constIt ) == "PT_HIGH_THRESHOLD") {
                resultsMenuOp.fillVariableFromRow(
                        *constIt, iRow, menuObjectParameters.ptHighThreshold);

            } else if ( ( *constIt ) == "PT_LOW_THRESHOLD") {
                resultsMenuOp.fillVariableFromRow(
                        *constIt, iRow, menuObjectParameters.ptLowThreshold);

            } else if ( ( *constIt ) == "ENABLE_MIP") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.enableMip);

            } else if ( ( *constIt ) == "ENABLE_ISO") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.enableIso);

            } else if ( ( *constIt ) == "REQUEST_ISO") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.requestIso);

            } else if ( ( *constIt ) == "ENERGY_OVERFLOW") {
                resultsMenuOp.fillVariableFromRow(
                        *constIt, iRow, menuObjectParameters.energyOverflow);

            } else if ( ( *constIt ) == "ET_THRESHOLD") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.etThreshold);

            } else if ( ( *constIt ) == "ETA_RANGE") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.etaRange);

            } else if ( ( *constIt ) == "PHI_RANGE") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.phiRange);

            } else if ( ( *constIt ) == "PHI_LOW") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.phiLow);

            } else if ( ( *constIt ) == "PHI_HIGH") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.phiHigh);

            } else if ( ( *constIt ) == "QUALITY_RANGE") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.qualityRange);

            } else if ( ( *constIt ) == "CHARGE") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.charge);

            } else {
                // do nothing

            }

        }

        LogTrace("L1GtTriggerMenuConfigOnlineProd")
            << " ID " << menuObjectParameters.opId
            << " PT_HIGH_THRESHOLD " << menuObjectParameters.ptHighThreshold
            << " PT_LOW_THRESHOLD " << menuObjectParameters.ptLowThreshold
            << " ENABLE_MIP " << menuObjectParameters.enableMip
            << " ENABLE_ISO " << menuObjectParameters.enableIso
            << " REQUEST_ISO " << menuObjectParameters.requestIso
            << " ENERGY_OVERFLOW " << menuObjectParameters.energyOverflow
            << " ET_THRESHOLD " << menuObjectParameters.etThreshold
            << " ETA_RANGE " << menuObjectParameters.etaRange
            << " PHI_RANGE " << menuObjectParameters.phiRange
            << " PHI_LOW " << menuObjectParameters.phiLow
            << " PHI_HIGH " << menuObjectParameters.phiHigh
            << " QUALITY_RANGE " << menuObjectParameters.qualityRange
            << " CHARGE " << menuObjectParameters.charge
            << std::endl;

        m_tableMenuObjectParameters.push_back(menuObjectParameters);
    }

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n Number of rows read from L1T_MENU_OP_VIEW: " << resultsMenuOpRows
                << std::endl;

    }

    return true;

}

// retrieve table with technical triggers from DB
bool L1GtTriggerMenuConfigOnlineProd::tableMenuTechTrigFromDB(
        const std::string& gtSchema, const std::string& objectKey) {

    // select * from CMS_GT.L1T_MENU_TECHTRIG_VIEW
    //     where L1T_MENU_TECHTRIG_VIEW.MENU_IMPLEMENTATION = objectKey

    const std::vector<std::string>& columnsMenuTechTrig =
            m_omdsReader.columnNamesView(gtSchema, "L1T_MENU_TECHTRIG_VIEW");

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n List of columns in L1T_MENU_TECHTRIG_VIEW:\n"
                << std::endl;
        for (std::vector<std::string>::const_iterator iter =
                columnsMenuTechTrig.begin(); iter != columnsMenuTechTrig.end(); iter++) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << (*iter) << std::endl;

        }
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n\n" << std::endl;
    }

    l1t::OMDSReader::QueryResults resultsMenuTechTrig =
            m_omdsReader.basicQueryView(columnsMenuTechTrig, gtSchema,
                    "L1T_MENU_TECHTRIG_VIEW",
                    "L1T_MENU_TECHTRIG_VIEW.MENU_IMPLEMENTATION",
                    m_omdsReader.singleAttribute(objectKey));

    // check if query was successful is based on size of returned list
    // BUT one can have menus w/o technical triggers - no error for empty list, but a warning!
    if (resultsMenuTechTrig.queryFailed()) {
        edm::LogWarning("L1-O2O")
                << "Warning: Content of L1T_MENU_TECHTRIG_VIEW for L1GtTriggerMenu implementation key: "
                << "\n  " << objectKey << "\nis empty!"
                << "\nNo technical triggers are found for this menu.";

    }

    TableMenuTechTrig menuTechTrig;
    int resultsMenuTechTrigRows = resultsMenuTechTrig.numberRows();

    for (int iRow = 0; iRow < resultsMenuTechTrigRows; ++iRow) {

        for (std::vector<std::string>::const_iterator constIt =
                columnsMenuTechTrig.begin(); constIt
                != columnsMenuTechTrig.end(); ++constIt) {

            if ((*constIt) == "TECHTRIG_INDEX") {
                resultsMenuTechTrig.fillVariableFromRow(*constIt, iRow,
                        menuTechTrig.bitNumberSh);

            } else if ((*constIt) == "NAME") {
                resultsMenuTechTrig.fillVariableFromRow(*constIt, iRow,
                        menuTechTrig.techName);

            } else {
                // do nothing

            }

        }

        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "Row  " << iRow
                << ": index = " << menuTechTrig.bitNumberSh << " techName = "
                << menuTechTrig.techName << std::endl;

        m_tableMenuTechTrig.push_back(menuTechTrig);

    }

    if (m_isDebugEnabled) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n Number of rows read from L1T_MENU_TECHTRIG_VIEW: "
                << resultsMenuTechTrigRows << std::endl;

    }

    return true;

}

// return for an algorithm with bitNr the mapping between the integer index in logical expression
// and the condition name (FK)
const std::map<int, std::string> L1GtTriggerMenuConfigOnlineProd::condIndexNameMap(
        const short bitNr) const {

    std::map<int, std::string> mapIndexName;

    for (std::vector<TableMenuAlgoCond>::const_iterator constIt = m_tableMenuAlgoCond.begin(); constIt
            != m_tableMenuAlgoCond.end(); ++constIt) {

        if (bitNr == (*constIt).bitNumberSh) {
            mapIndexName[static_cast<int>((*constIt).condIndexF)] = (*constIt).condFK;
        }

    }

    if (m_isDebugEnabled) {

        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Bit number : " << bitNr << std::endl;

        for (std::map<int, std::string>::const_iterator constIt = mapIndexName.begin(); constIt
                != mapIndexName.end(); ++constIt) {

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "  Condition index -> name: "
                    << ( ( *constIt ).first ) << " " << ( ( *constIt ).second ) << std::endl;

        }

    }


    return mapIndexName;

}

// convert a logical expression with indices to a logical expression with names
std::string L1GtTriggerMenuConfigOnlineProd::convertLogicalExpression(
        const std::string& expressionIndices, const std::map<int, std::string>& mapCondIndexName) const {

    std::string expressionNames;

    L1GtLogicParser parserIndices = L1GtLogicParser(expressionIndices);
    parserIndices.convertIntToNameLogicalExpression(mapCondIndexName);
    expressionNames = parserIndices.logicalExpression();

    return expressionNames;

}

// return the chip number for an algorithm with index bitNumberSh
int L1GtTriggerMenuConfigOnlineProd::chipNumber(short bitNumberSh) const {

    // FIXME get it from Event Setup
    const unsigned numberConditionChips = 2;
    const unsigned pinsOnConditionChip = 96;
    std::vector<int> orderConditionChip;
    orderConditionChip.push_back(2);
    orderConditionChip.push_back(1);

    int posChip = ( static_cast<unsigned> (bitNumberSh) / pinsOnConditionChip ) + 1;
    for (unsigned int iChip = 0; iChip < numberConditionChips; ++iChip) {
        if (posChip == orderConditionChip[iChip]) {
            return static_cast<int>(iChip);
        }
    }

    // chip number not found
    return -1;

}

// build the algorithm map in the menu
void L1GtTriggerMenuConfigOnlineProd::buildAlgorithmMap() {


     // temporary value
    int bitNumber = -1;
    std::string logicalExpression;

    // loop over m_tableMenuAlgo
    for (std::vector<TableMenuAlgo>::const_iterator constIt = m_tableMenuAlgo.begin(); constIt
            != m_tableMenuAlgo.end(); constIt++) {

        bitNumber = static_cast<int> ((*constIt).bitNumberSh);

        const std::map<int, std::string>& condIndexName = condIndexNameMap((*constIt).bitNumberSh);
        logicalExpression = convertLogicalExpression((*constIt).logExpression, condIndexName);

        // create a new algorithm and insert it into algorithm map
        L1GtAlgorithm alg((*constIt).algName, logicalExpression, bitNumber);
        alg.setAlgoAlias((*constIt).algAlias);

        // set algorithm chip number:
        int algChipNr = chipNumber((*constIt).bitNumberSh);
        alg.setAlgoChipNumber(algChipNr);

        // insert algorithm
        m_algorithmMap[(*constIt).algName] = alg;
        m_algorithmAliasMap[(*constIt).algAlias] = alg;

    }

}

// build the technical trigger map in the menu
void L1GtTriggerMenuConfigOnlineProd::buildTechnicalTriggerMap() {

    // temporary value
    int bitNumber = -1;
    std::string logicalExpression;

    // loop over m_tableMenuTechTrig
    for (std::vector<TableMenuTechTrig>::const_iterator constIt =
            m_tableMenuTechTrig.begin(); constIt != m_tableMenuTechTrig.end(); constIt++) {

        bitNumber = static_cast<int> ((*constIt).bitNumberSh);

        // create a new technical trigger and insert it into technical trigger map
        // technical triggers have L1GtAlgorithm class
        L1GtAlgorithm techTrig((*constIt).techName, logicalExpression, bitNumber);

        // chip number set in constructor to -1 - no meaning for technical triggers

        // insert technical trigger
        m_technicalTriggerMap[(*constIt).techName] = techTrig;

        // no alias is defined for technical triggers
    }

}


L1GtConditionCategory L1GtTriggerMenuConfigOnlineProd::strToEnumCondCategory(
        const std::string& strCategory) {

    if (strCategory == "CondMuon") {
        return CondMuon;
    } else if (strCategory == "CondCalo") {
        return CondCalo;
    } else if (strCategory == "CondEnergySum") {
        return CondEnergySum;
    } else if (strCategory == "CondJetCounts") {
        return CondJetCounts;
    } else if (strCategory == "CondCorrelation") {
        return CondCorrelation;
    } else if (strCategory == "CondCastor") {
        return CondCastor;
    } else if (strCategory == "CondHfBitCounts") {
        return CondHfBitCounts;
    } else if (strCategory == "CondHfRingEtSums") {
        return CondHfRingEtSums;
    } else if (strCategory == "CondBptx") {
        return CondBptx;
    } else if (strCategory == "CondExternal") {
        return CondExternal;
    } else if (strCategory == "CondNull") {
        return CondNull;
    } else {
        edm::LogWarning("L1GtTriggerMenuConfigOnlineProd")
                << "\n Warning: string " << strCategory
                << " not defined. Returning CondNull.\n" << std::endl;
        return CondNull;
    }

    return CondNull;

}

// string to enum L1GtConditionType conversion
L1GtConditionType L1GtTriggerMenuConfigOnlineProd::strToEnumCondType(const std::string& strType) {

    if (strType == "1s") {
        return Type1s;
    } else if (strType == "2s") {
        return Type2s;
    } else if (strType == "2wsc") {
        return Type2wsc;
    } else if (strType == "2cor") {
        return Type2cor;
    } else if (strType == "3s") {
        return Type3s;
    } else if (strType == "4s") {
        return Type4s;
    } else if (strType == "ETM") {
        return TypeETM;
    } else if (strType == "ETT") {
        return TypeETT;
    } else if (strType == "HTT") {
        return TypeHTT;
    } else if (strType == "HTM") {
        return TypeHTM;
    } else if (strType == "JetCounts") {
        return TypeJetCounts;
    } else if (strType == "Castor") {
        return TypeCastor;
    } else if (strType == "HfBitCounts") {
        return TypeHfBitCounts;
    } else if (strType == "HfRingEtSums") {
        return TypeHfRingEtSums;
    } else if (strType == "Bptx") {
        return TypeBptx;
    } else if (strType == "TypeExternal") {
        return TypeExternal;
    } else {
        edm::LogWarning("L1GtTriggerMenuConfigOnlineProd")
                << "\n Warning: string " << strType
                << " not associated to any L1GtConditionType. Returning TypeNull.\n"
                << std::endl;
        return TypeNull;
    }

    return TypeNull;

}

// string to enum L1GtObject conversion
L1GtObject L1GtTriggerMenuConfigOnlineProd::strToEnumL1GtObject(const std::string& strObject) {

    if (strObject == "Mu") {
        return Mu;
    } else if (strObject == "NoIsoEG") {
        return NoIsoEG;
    } else if (strObject == "IsoEG") {
        return IsoEG;
    } else if (strObject == "CenJet") {
        return CenJet;
    } else if (strObject == "ForJet") {
        return ForJet;
    } else if (strObject == "TauJet") {
        return TauJet;
    } else if (strObject == "ETM") {
        return ETM;
    } else if (strObject == "ETT") {
        return ETT;
    } else if (strObject == "HTT") {
        return HTT;
    } else if (strObject == "HTM") {
        return HTM;
    } else if (strObject == "JetCounts") {
        return JetCounts;
    } else if (strObject == "HfBitCounts") {
        return HfBitCounts;
    } else if (strObject == "HfRingEtSums") {
        return HfRingEtSums;
    } else if (strObject == "TechTrig") {
        return TechTrig;
    } else if (strObject == "Castor") {
        return Castor;
    } else if (strObject == "BPTX") {
        return BPTX;
    } else if (strObject == "GtExternal") {
        return GtExternal;
    } else {
        edm::LogWarning("L1GtTriggerMenuConfigOnlineProd")
                << "\n Warning: string " << strObject
                << " not associated to any L1GtObject. Returning Mu (no Null type).\n"
                << std::endl;
        return Mu;
    }

    // no null type, so returning Mu - should never arrive here
    return Mu;

}

void L1GtTriggerMenuConfigOnlineProd::splitHexStringInTwo64bitWords(
        const std::string& hexStr, std::string& hex0WordStr, std::string& hex1WordStr) {

    unsigned int lenHexStr = hexStr.length();

    if (lenHexStr < 3) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Warning: string " << hexStr
                << " has length less than 3." << "\n Not possible to split it in two 64-bit words."
                << "\n Return two zero strings." << std::endl;
        hex0WordStr = "0x0";
        hex1WordStr = "0x0";

        return;
    }

    unsigned int lenHex = lenHexStr - 2;
    unsigned int len0Word = lenHex > 16 ? 16 : lenHex;
    unsigned int len1Word = lenHex - len0Word;

    unsigned int pos0Word = lenHexStr - len0Word;
    hex0WordStr = "0x" + hexStr.substr(pos0Word, len0Word);

    if (len1Word > 0) {
        unsigned int pos1Word = pos0Word - len1Word;
        hex1WordStr = "0x" + hexStr.substr(pos1Word, len1Word);
    } else {
        hex1WordStr = "0x0";
    }
}

// get a list of chip numbers from the m_tableMenuAlgoCond table for a condition
std::list<int> L1GtTriggerMenuConfigOnlineProd::listChipNumber(const std::string& condFK) {

    std::list<int> chipList;

    // loop over m_tableMenuAlgoCond
    for (std::vector<TableMenuAlgoCond>::const_iterator constIt = m_tableMenuAlgoCond.begin(); constIt
            != m_tableMenuAlgoCond.end(); constIt++) {

        if (condFK == ( *constIt ).condFK) {
            int chipNr = chipNumber( ( *constIt ).bitNumberSh);
            chipList.push_back(chipNr);
        }
    }

    return chipList;

}

void L1GtTriggerMenuConfigOnlineProd::fillMuonObjectParameter(
        const std::string& opFK, L1GtMuonTemplate::ObjectParameter& objPar) {

    // loop over m_tableMenuCond
    for (std::vector<TableMenuObjectParameters>::const_iterator constIt =
            m_tableMenuObjectParameters.begin(); constIt != m_tableMenuObjectParameters.end(); constIt++) {

        if (opFK == ( *constIt ).opId) {
            objPar.ptHighThreshold = static_cast<unsigned int> ( ( *constIt ).ptHighThreshold);
            objPar.ptLowThreshold = static_cast<unsigned int> ( ( *constIt ).ptLowThreshold);
            objPar.enableMip = static_cast<bool> ( ( *constIt ).enableMip);
            objPar.enableIso = static_cast<bool> ( ( *constIt ).enableIso);
            objPar.requestIso = static_cast<bool> ( ( *constIt ).requestIso);
            objPar.etaRange = lexical_cast_from_hex<unsigned long long>( ( *constIt ).etaRange);
            objPar.phiHigh = static_cast<unsigned int> ( ( *constIt ).phiHigh);
            objPar.phiLow = static_cast<unsigned int> ( ( *constIt ).phiLow);
            objPar.qualityRange = lexical_cast_from_hex<unsigned int>( ( *constIt ).qualityRange);
            //objPar.charge = static_cast<unsigned int> ( ( *constIt ).charge);

            // can break after it is found - DB consistency
            break;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addMuonCondition(const TableMenuCond& condDB) {


    L1GtMuonTemplate muonCond(condDB.cond);
    muonCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - all muons
    int nrObj = muonCond.nrObjects();
    std::vector<L1GtObject> objType(nrObj, Mu);
    muonCond.setObjectType(objType);

    muonCond.setCondGEq(condDB.condGEq);

    // temporary storage of the parameters
    std::vector<L1GtMuonTemplate::ObjectParameter> objParameter(nrObj);

    for (int iObj = 0; iObj < nrObj; ++iObj) {
        if (iObj == 0) {
            fillMuonObjectParameter(condDB.objectParameter1FK, objParameter[iObj]);
        } else if (iObj == 1) {
            fillMuonObjectParameter(condDB.objectParameter2FK, objParameter[iObj]);
        } else if (iObj == 2) {
            fillMuonObjectParameter(condDB.objectParameter3FK, objParameter[iObj]);
        } else if (iObj == 3) {
            fillMuonObjectParameter(condDB.objectParameter4FK, objParameter[iObj]);
        } else {
            LogTrace("L1GtTriggerMenuConfigOnlineProd")
                    << "\n Warning: number of objects requested " << nrObj
                    << " not available in DB." << "\n Maximum 4 object parameters implemented. \n"
                    << std::endl;
        }
    }

    L1GtMuonTemplate::CorrelationParameter corrParameter;
    corrParameter.chargeCorrelation = static_cast<unsigned int> (condDB.chargeCorrelation);
    if (muonCond.wsc()) {
        corrParameter.deltaEtaRange = lexical_cast_from_hex<unsigned long long> (
                condDB.deltaEtaRange);

        std::string word0;
        std::string word1;
        splitHexStringInTwo64bitWords(condDB.deltaPhiRange, word0, word1);

        corrParameter.deltaPhiRange0Word = lexical_cast_from_hex<unsigned long long> (word0);
        corrParameter.deltaPhiRange1Word = lexical_cast_from_hex<unsigned long long> (word1);

        corrParameter.deltaPhiMaxbits = 0; // not used anymore
    }

    muonCond.setConditionParameter(objParameter, corrParameter);

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);

    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        muonCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecMuonTemplate[*itChip] ).push_back(muonCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << muonCond << "\n" << std::endl;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::fillCaloObjectParameter(
        const std::string& opFK, L1GtCaloTemplate::ObjectParameter& objPar) {

    // loop over m_tableMenuCond
    for (std::vector<TableMenuObjectParameters>::const_iterator constIt =
            m_tableMenuObjectParameters.begin(); constIt != m_tableMenuObjectParameters.end(); constIt++) {

        if (opFK == ( *constIt ).opId) {
            objPar.etThreshold = static_cast<unsigned int> ( ( *constIt ).etThreshold);
            objPar.etaRange = lexical_cast_from_hex<unsigned int> ( ( *constIt ).etaRange);
            objPar.phiRange = lexical_cast_from_hex<unsigned int> ( ( *constIt ).phiRange);

            // can break after it is found - DB consistency
            break;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addCaloCondition(const TableMenuCond& condDB) {

    L1GtCaloTemplate caloCond(condDB.cond);
    caloCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - all have the same type, so reading it for first object is OK
    int nrObj = caloCond.nrObjects();

    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    std::vector<L1GtObject> objType(nrObj, obj);
    caloCond.setObjectType(objType);

    caloCond.setCondGEq(condDB.condGEq);

    // temporary storage of the parameters
    std::vector<L1GtCaloTemplate::ObjectParameter> objParameter(nrObj);

    for (int iObj = 0; iObj < nrObj; ++iObj) {
        if (iObj == 0) {
            fillCaloObjectParameter(condDB.objectParameter1FK, objParameter[iObj]);
        } else if (iObj == 1) {
            fillCaloObjectParameter(condDB.objectParameter2FK, objParameter[iObj]);
        } else if (iObj == 2) {
            fillCaloObjectParameter(condDB.objectParameter3FK, objParameter[iObj]);
        } else if (iObj == 3) {
            fillCaloObjectParameter(condDB.objectParameter4FK, objParameter[iObj]);
        } else {
            LogTrace("L1GtTriggerMenuConfigOnlineProd")
                    << "\n Warning: number of objects requested " << nrObj
                    << " not available in DB." << "\n Maximum 4 object parameters implemented. \n"
                    << std::endl;
        }
    }

    L1GtCaloTemplate::CorrelationParameter corrParameter;
    if (caloCond.wsc()) {
        corrParameter.deltaEtaRange = lexical_cast_from_hex<unsigned long long> (
                condDB.deltaEtaRange);

        corrParameter.deltaPhiRange = lexical_cast_from_hex<unsigned long long> (
                condDB.deltaPhiRange);
        corrParameter.deltaPhiMaxbits = 0; // FIXME check correlations
    }

    caloCond.setConditionParameter(objParameter, corrParameter);

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);

    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        caloCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecCaloTemplate[*itChip] ).push_back(caloCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << caloCond << "\n" << std::endl;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::fillEnergySumObjectParameter(
        const std::string& opFK, L1GtEnergySumTemplate::ObjectParameter& objPar,
        const L1GtObject& obj) {

    // loop over m_tableMenuCond
    for (std::vector<TableMenuObjectParameters>::const_iterator constIt =
            m_tableMenuObjectParameters.begin(); constIt != m_tableMenuObjectParameters.end(); constIt++) {

        if (opFK == ( *constIt ).opId) {
            objPar.etThreshold = static_cast<unsigned int> ( ( *constIt ).etThreshold);
            objPar.energyOverflow = static_cast<bool> ( ( *constIt ).energyOverflow); // not used

            std::string word0;
            std::string word1;
            splitHexStringInTwo64bitWords( ( *constIt ).phiRange, word0, word1);

            if (obj == ETM) {
                objPar.phiRange0Word = lexical_cast_from_hex<unsigned long long> (word0);
                objPar.phiRange1Word = lexical_cast_from_hex<unsigned long long> (word1);

            } else if (obj == HTM) {
                objPar.phiRange0Word = lexical_cast_from_hex<unsigned long long> (word0);
            }

            // can break after it is found - DB consistency
            break;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addEnergySumCondition(const TableMenuCond& condDB) {

    L1GtEnergySumTemplate esumCond(condDB.cond);
    esumCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - all energy sums are global - so 1 object
    int nrObj = 1;

    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    std::vector<L1GtObject> objType(nrObj, obj);
    esumCond.setObjectType(objType);

    esumCond.setCondGEq(condDB.condGEq);

    // temporary storage of the parameters - no CorrelationParameter
    std::vector<L1GtEnergySumTemplate::ObjectParameter> objParameter(nrObj);
    fillEnergySumObjectParameter(condDB.objectParameter1FK, objParameter[0], objType[0]);

    esumCond.setConditionParameter(objParameter);

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);

    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        esumCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecEnergySumTemplate[*itChip] ).push_back(esumCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << esumCond << "\n" << std::endl;
        }
    }

}


void L1GtTriggerMenuConfigOnlineProd::addJetCountsCondition(const TableMenuCond& condDB) {

    L1GtJetCountsTemplate jcCond(condDB.cond);
    jcCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - jet counts are "global"
    int nrObj = 1;

    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    std::vector<L1GtObject> objType(nrObj, obj);
    jcCond.setObjectType(objType);

    jcCond.setCondGEq(condDB.condGEq);

    // temporary storage of the parameters - no CorrelationParameter
    // for counts, the DB implementation is without OP, directly in TableMenuCond
    std::vector<L1GtJetCountsTemplate::ObjectParameter> objParameter(nrObj);
    objParameter.at(0).countIndex = static_cast<unsigned int>(condDB.countIndex);
    objParameter.at(0).countThreshold = static_cast<unsigned int>(condDB.countThreshold);
    objParameter.at(0).countOverflow = false ; // not used

    jcCond.setConditionParameter(objParameter);

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);

    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        jcCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecJetCountsTemplate[*itChip] ).push_back(jcCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << jcCond << "\n" << std::endl;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addHfBitCountsCondition(const TableMenuCond& condDB) {

    L1GtHfBitCountsTemplate countsCond(condDB.cond);
    countsCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - HF bit counts are "global"
    int nrObj = 1;

    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    std::vector<L1GtObject> objType(nrObj, obj);
    countsCond.setObjectType(objType);

    countsCond.setCondGEq(condDB.condGEq);

    // temporary storage of the parameters - no CorrelationParameter
    // for counts, the DB implementation is without OP, directly in TableMenuCond
    std::vector<L1GtHfBitCountsTemplate::ObjectParameter> objParameter(nrObj);
    objParameter.at(0).countIndex = static_cast<unsigned int>(condDB.countIndex);
    objParameter.at(0).countThreshold = static_cast<unsigned int>(condDB.countThreshold);

    countsCond.setConditionParameter(objParameter);

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);

    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        countsCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecHfBitCountsTemplate[*itChip] ).push_back(countsCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << countsCond << "\n" << std::endl;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addHfRingEtSumsCondition(const TableMenuCond& condDB) {

    L1GtHfRingEtSumsTemplate esumCond(condDB.cond);
    esumCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - HF energy sums are "global"
    int nrObj = 1;

    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    std::vector<L1GtObject> objType(nrObj, obj);
    esumCond.setObjectType(objType);

    esumCond.setCondGEq(condDB.condGEq);

    // temporary storage of the parameters - no CorrelationParameter
    // for HF energy sums, the DB implementation is without OP, directly in TableMenuCond
    std::vector<L1GtHfRingEtSumsTemplate::ObjectParameter> objParameter(nrObj);
    objParameter.at(0).etSumIndex = static_cast<unsigned int>(condDB.countIndex);
    objParameter.at(0).etSumThreshold = static_cast<unsigned int>(condDB.countThreshold);

    esumCond.setConditionParameter(objParameter);

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);

    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        esumCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecHfRingEtSumsTemplate[*itChip] ).push_back(esumCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << esumCond << "\n" << std::endl;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addCastorCondition(const TableMenuCond& condDB) {

    L1GtCastorTemplate castorCond(condDB.cond);
    castorCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - logical conditions have no objects associated in GT
    // one put however a "Castor" object type
    int nrObj = 1;

    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    std::vector<L1GtObject> objType(nrObj, obj);
    castorCond.setObjectType(objType);

    // irrelevant, set to false for completeness
    castorCond.setCondGEq(false);

    // logical conditions have no ObjectParameter, no CorrelationParameter

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);

    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        castorCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecCastorTemplate[*itChip] ).push_back(castorCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << castorCond << "\n" << std::endl;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addBptxCondition(const TableMenuCond& condDB) {

    L1GtBptxTemplate bptxCond(condDB.cond);
    bptxCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - logical conditions have no objects associated in GT
    // one put however a "Bptx" object type
    int nrObj = 1;

    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    std::vector<L1GtObject> objType(nrObj, obj);
    bptxCond.setObjectType(objType);

    // irrelevant, set to false for completeness
    bptxCond.setCondGEq(false);

    // logical conditions have no ObjectParameter, no CorrelationParameter

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);



    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        bptxCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecBptxTemplate[*itChip] ).push_back(bptxCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << bptxCond << "\n" << std::endl;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addExternalCondition(const TableMenuCond& condDB) {

    L1GtExternalTemplate externalCond(condDB.cond);
    externalCond.setCondType(strToEnumCondType(condDB.condType));

    // object types - logical conditions have no objects associated in GT
    // one put however a "External" object type
    int nrObj = 1;

    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    std::vector<L1GtObject> objType(nrObj, obj);
    externalCond.setObjectType(objType);

    // irrelevant, set to false for completeness
    externalCond.setCondGEq(false);

    // logical conditions have no ObjectParameter, no CorrelationParameter

    // get chip number list
    std::list<int> chipList = listChipNumber(condDB.cond);



    // eliminate duplicates
    chipList.sort();
    chipList.unique();

    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        externalCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecExternalTemplate[*itChip] ).push_back(externalCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << externalCond << "\n" << std::endl;
        }
    }

}

void L1GtTriggerMenuConfigOnlineProd::addCorrelationCondition(const TableMenuCond& condDB) {

    // create a new correlation condition
    L1GtCorrelationTemplate correlationCond(condDB.cond);
    correlationCond.setCondType(strToEnumCondType(condDB.condType));

    // two objects (for sure) - type taken from DB
    const int nrObj = 2;

    std::vector<L1GtObject> objType(nrObj);
    L1GtObject obj = strToEnumL1GtObject(condDB.gtObject1);
    objType[0] = obj;

    obj = strToEnumL1GtObject(condDB.gtObject2);
    objType[1] = obj;

    correlationCond.setObjectType(objType);

    // irrelevant, it is set for each subcondition
    correlationCond.setCondGEq(condDB.condGEq);

    // get chip number list, eliminate duplicates
    std::list<int> chipList = listChipNumber(condDB.cond);
    chipList.sort();
    chipList.unique();

    // temporary vectors for sub-conditions
    std::vector<L1GtConditionCategory> subcondCategory(nrObj);
    std::vector<int> subcondIndex(nrObj);

    bool wrongSubcondition = false;

    for (int iObj = 0; iObj < nrObj; ++iObj) {

        L1GtObject gtObj = objType[iObj];

        // sub-conditions (all have the same condGEq as the correlation condition).
        switch (gtObj) {
            case Mu: {
                subcondCategory[iObj] = CondMuon;

                // temporary storage of the parameters
                std::vector<L1GtMuonTemplate::ObjectParameter> objParameter(1);

                std::string subcondName;
                if (iObj == 0) {
                    fillMuonObjectParameter(condDB.objectParameter1FK, objParameter[0]);
                    subcondName = condDB.objectParameter1FK;
                } else if (iObj == 1) {
                    subcondName = condDB.objectParameter2FK;
                    fillMuonObjectParameter(condDB.objectParameter2FK, objParameter[0]);
                }

                //  chargeCorrelation must be set for muons
                //  put it to ignore
                L1GtMuonTemplate::CorrelationParameter corrPar;
                corrPar.chargeCorrelation = 1;

                L1GtMuonTemplate subcond(subcondName, Type1s);
                subcond.setCondGEq(condDB.condGEq);
                subcond.setObjectType(std::vector<L1GtObject> (1, gtObj));
                subcond.setConditionParameter(objParameter, corrPar);

                // add the same sub-condition once to every chip where required
                for (std::list<int>::const_iterator itChip = chipList.begin(); itChip
                        != chipList.end(); ++itChip) {

                    subcond.setCondChipNr(*itChip);

                    // no check for uniqueness - done by DB
                    ( m_corMuonTemplate[*itChip] ).push_back(subcond);

                    // index
                    subcondIndex[iObj] = ( m_corMuonTemplate[*itChip] ).size() - 1;

                    if (m_isDebugEnabled) {
                        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition "
                                << ( condDB.cond ) << " on chip " << ( *itChip ) << "\n "
                                << std::endl;

                        LogTrace("L1GtTriggerMenuConfigOnlineProd") << subcond << "\n" << std::endl;
                    }
                }

            }
                break;

            case IsoEG:
            case NoIsoEG:
            case CenJet:
            case ForJet:
            case TauJet: {

                subcondCategory[iObj] = CondCalo;

                // temporary storage of the parameters
                std::vector<L1GtCaloTemplate::ObjectParameter> objParameter(1);

                std::string subcondName;
                if (iObj == 0) {
                    fillCaloObjectParameter(condDB.objectParameter1FK, objParameter[0]);
                    subcondName = condDB.objectParameter1FK;
                } else if (iObj == 1) {
                    subcondName = condDB.objectParameter2FK;
                    fillCaloObjectParameter(condDB.objectParameter2FK, objParameter[0]);
                }

                L1GtCaloTemplate::CorrelationParameter corrPar; //  dummy

                L1GtCaloTemplate subcond(subcondName, Type1s);
                subcond.setCondGEq(condDB.condGEq);
                subcond.setObjectType(std::vector<L1GtObject> (1, gtObj));
                subcond.setConditionParameter(objParameter, corrPar);

                // add the same sub-condition once to every chip where required
                for (std::list<int>::const_iterator itChip = chipList.begin(); itChip
                        != chipList.end(); ++itChip) {

                    subcond.setCondChipNr(*itChip);

                    // no check for uniqueness - done by DB
                    ( m_corCaloTemplate[*itChip] ).push_back(subcond);

                    // index
                    subcondIndex[iObj] = ( m_corCaloTemplate[*itChip] ).size() - 1;

                    if (m_isDebugEnabled) {
                        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition "
                                << ( condDB.cond ) << " on chip " << ( *itChip ) << "\n "
                                << std::endl;

                        LogTrace("L1GtTriggerMenuConfigOnlineProd") << subcond << "\n" << std::endl;
                    }
                }
            }
                break;

            case ETM:
            case HTM: {

                subcondCategory[iObj] = CondEnergySum;

                // temporary storage of the parameters
                std::vector<L1GtEnergySumTemplate::ObjectParameter> objParameter(1);

                std::string subcondName;
                if (iObj == 0) {
                    fillEnergySumObjectParameter(condDB.objectParameter1FK, objParameter[0], gtObj);
                    subcondName = condDB.objectParameter1FK;
                } else if (iObj == 1) {
                    subcondName = condDB.objectParameter2FK;
                    fillEnergySumObjectParameter(condDB.objectParameter2FK, objParameter[0], gtObj);
                }

                L1GtConditionType condType;

                switch (gtObj) {
                    case ETM: {
                        condType = TypeETM;
                    }
                        break;
                    case HTM: {
                        condType = TypeHTM;
                    }
                        break;
                    default: {
                        edm::LogWarning("L1GtTriggerMenuConfigOnlineProd")
                                << "\n Warning: wrong L1GtConditionType "
                                << gtObj << std::endl;

                    }
                        break;
                }

                L1GtEnergySumTemplate subcond(subcondName, condType);
                subcond.setCondGEq(condDB.condGEq);
                subcond.setObjectType(std::vector<L1GtObject> (1, gtObj));
                subcond.setConditionParameter(objParameter);

                // add the same sub-condition once to every chip where required
                for (std::list<int>::const_iterator itChip = chipList.begin(); itChip
                        != chipList.end(); ++itChip) {

                    subcond.setCondChipNr(*itChip);

                    // no check for uniqueness - done by DB
                    ( m_corEnergySumTemplate[*itChip] ).push_back(subcond);

                    // index
                    subcondIndex[iObj] = ( m_corEnergySumTemplate[*itChip] ).size() - 1;

                    if (m_isDebugEnabled) {
                        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition "
                                << ( condDB.cond ) << " on chip " << ( *itChip ) << "\n "
                                << std::endl;

                        LogTrace("L1GtTriggerMenuConfigOnlineProd") << subcond << "\n" << std::endl;
                    }
                }
            }
                break;
            case ETT:
            case HTT:
            case JetCounts:
            case HfBitCounts:
            case HfRingEtSums:
            case Castor:
            case BPTX:
            case GtExternal:
            case TechTrig: {
                wrongSubcondition = true;
                edm::LogWarning("L1GtTriggerMenuConfigOnlineProd")
                        << "\n Warning: correlation condition "
                        << (condDB.cond)
                        << " with invalid sub-condition object type " << gtObj
                        << "\n Condition ignored!" << std::endl;
            }
            default: {
                wrongSubcondition = true;
                edm::LogWarning("L1GtTriggerMenuConfigOnlineProd")
                        << "\n Warning: correlation condition "
                        << (condDB.cond)
                        << " with invalid sub-condition object type " << gtObj
                        << "\n Condition ignored!" << std::endl;

                //

            }
                break;
        }

    }

    if (wrongSubcondition) {
        edm::LogWarning("L1GtTriggerMenuConfigOnlineProd")
                << "\n Warning: wrong sub-condition for correlation condition "
                << (condDB.cond)
                << "\n Condition not inserted in menu. \n A sub-condition may be left in the menu"
                << std::endl;
        return;

    }

    // get the correlation parameters for the correlation condition (std::string)
    L1GtCorrelationTemplate::CorrelationParameter corrParameter;
    corrParameter.deltaEtaRange = condDB.deltaEtaRange;
    corrParameter.deltaPhiRange = condDB.deltaPhiRange;

    // set condition categories
    correlationCond.setCond0Category(subcondCategory[0]);
    correlationCond.setCond1Category(subcondCategory[1]);

    // set condition indices in correlation vector
    correlationCond.setCond0Index(subcondIndex[0]);
    correlationCond.setCond1Index(subcondIndex[1]);

    // set correlation parameter
    corrParameter.deltaPhiMaxbits = 0; //  TODO
    correlationCond.setCorrelationParameter(corrParameter);


    // add the same condition once to every chip where required
    for (std::list<int>::const_iterator itChip = chipList.begin(); itChip != chipList.end(); ++itChip) {

        correlationCond.setCondChipNr(*itChip);

        // no check for uniqueness - done by DB
        ( m_vecCorrelationTemplate[*itChip] ).push_back(correlationCond);

        if (m_isDebugEnabled) {
            LogTrace("L1GtTriggerMenuConfigOnlineProd") << "\n Adding condition " << ( condDB.cond )
                    << " on chip " << ( *itChip ) << "\n " << std::endl;

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << correlationCond << "\n" << std::endl;
        }
    }

}



// add the conditions from a menu to the corresponding list
void L1GtTriggerMenuConfigOnlineProd::addConditions() {

    // loop over m_tableMenuCond
    for (std::vector<TableMenuCond>::const_iterator constIt = m_tableMenuCond.begin(); constIt
            != m_tableMenuCond.end(); constIt++) {

        L1GtConditionCategory conCategory = strToEnumCondCategory((*constIt).condCategory);

        switch (conCategory) {
            case CondMuon: {

                addMuonCondition(*constIt);

            }
                break;
            case CondCalo: {

                addCaloCondition(*constIt);

            }
                break;
            case CondEnergySum: {

                addEnergySumCondition(*constIt);

            }
                break;
            case CondJetCounts: {
                addJetCountsCondition(*constIt);

            }
                break;
            case CondHfBitCounts: {
                addHfBitCountsCondition(*constIt);

            }
                break;
            case CondHfRingEtSums: {
                addHfRingEtSumsCondition(*constIt);

            }
                break;
            case CondCastor: {

                addCastorCondition(*constIt);

            }
                break;
            case CondBptx: {

                addBptxCondition(*constIt);

            }
                break;
            case CondExternal: {

                addExternalCondition(*constIt);

            }
                break;
            case CondCorrelation: {

                addCorrelationCondition(*constIt);

            }
                break;
            case CondNull: {

                // do nothing

            }
                break;
            default: {

                // do nothing

            }
                break;
        }

    }

}


DEFINE_FWK_EVENTSETUP_MODULE( L1GtTriggerMenuConfigOnlineProd);
