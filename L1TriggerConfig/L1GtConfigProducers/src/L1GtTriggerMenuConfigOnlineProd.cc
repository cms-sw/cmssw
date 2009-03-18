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

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuConfigOnlineProd.h"

// system include files
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"


// constructor
L1GtTriggerMenuConfigOnlineProd::L1GtTriggerMenuConfigOnlineProd(const edm::ParameterSet& parSet) :
    L1ConfigOnlineProdBase<L1GtTriggerMenuRcd, L1GtTriggerMenu> (parSet) {

    // empty

}

// destructor
L1GtTriggerMenuConfigOnlineProd::~L1GtTriggerMenuConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtTriggerMenu> L1GtTriggerMenuConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // shared pointer for L1GtTriggerMenu
    boost::shared_ptr<L1GtTriggerMenu> pL1GtTriggerMenu = boost::shared_ptr<L1GtTriggerMenu>(
            new L1GtTriggerMenu());

    const std::string gtSchema = "CMS_GT";

    // l1GtTriggerMenu: parameters in table GTFE_SETUP_FK
    // the objectKey for the menu obtained from GT_SETUP is the L1T_MENU_IMPL key

    // SQL queries:

    // retrieve table with general menu parameters from DB, view L1T_MENU_GENERAL_VIEW
    if (!tableMenuGeneralFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenu;

    }

    // build the menu name
    std::string menuName = m_tableMenuGeneral.menuInterface + "/" + m_tableMenuGeneral.scalesKey
            + "/" + m_tableMenuGeneral.algoImplTag;


    // retrieve table with physics algorithms from DB, view L1T_MENU_ALGO_VIEW
    if (!tableMenuAlgoFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenu;

    }

    // retrieve table with conditions associated to physics algorithms from DB
    if (!tableMenuAlgoCondFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenu;

    }

    // retrieve table with list of conditions in the menu
    if (!tableMenuCondFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenu;

    }


    // retrieve table with object parameters from DB, view CMS_GT.L1T_MENU_OP_VIEW
    if (!tableMenuObjectParametersFromDB(gtSchema, objectKey)) {
        return pL1GtTriggerMenu;

    }







    // fill the record

    pL1GtTriggerMenu->setGtTriggerMenuName(menuName);
    pL1GtTriggerMenu->setGtTriggerMenuInterface(m_tableMenuGeneral.menuInterface);
    pL1GtTriggerMenu->setGtTriggerMenuImplementation(m_tableMenuGeneral.menuImplementation);
    pL1GtTriggerMenu->setGtScaleDbKey(m_tableMenuGeneral.scalesKey);

    pL1GtTriggerMenu->setGtAlgorithmMap(m_algorithmMap);
    pL1GtTriggerMenu->setGtAlgorithmAliasMap(m_algorithmAliasMap);
    pL1GtTriggerMenu->setGtTechnicalTriggerMap(m_technicalTriggerMap);

    if (edm::isDebugEnabled()) {
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

    if (edm::isDebugEnabled()) {
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

    if (edm::isDebugEnabled()) {
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

    // check if query was successful
    if (resultsMenuAlgo.queryFailed()) {
        edm::LogError("L1-O2O")
            << "Problem to get content of L1T_MENU_ALGO_VIEW for L1GtTriggerMenu implementation key: "
            << objectKey;
        return false;
    }

    // temporary values
    int bitNumber = -1;

    // FIXME get it from Event Setup
     const unsigned numberConditionChips= 2;
     const unsigned pinsOnConditionChip= 96;
     std::vector<int> orderConditionChip;
     orderConditionChip.push_back(2);
     orderConditionChip.push_back(1);

    TableMenuAlgo menuAlgo;
    int resultsMenuAlgoRows = resultsMenuAlgo.numberRows();

    for (int iRow = 0; iRow < resultsMenuAlgoRows; ++iRow) {

        for (std::vector<std::string>::const_iterator constIt = columnsMenuAlgo.begin(); constIt
                != columnsMenuAlgo.end(); ++constIt) {

            if ( ( *constIt ) == "ALGO_INDEX") {
                resultsMenuAlgo.fillVariableFromRow(*constIt, iRow, menuAlgo.bitNumberSh);
                bitNumber = static_cast<int> (menuAlgo.bitNumberSh);

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
            << "Row  " << iRow << ": index = " << bitNumber << " algName = " << menuAlgo.algName
            << " algAlias = " << menuAlgo.algAlias << " logExpression = '" << menuAlgo.logExpression << "'"
            << std::endl;

        m_tableMenuAlgo.push_back(menuAlgo);

        // FIXME - move from here

        // create a new algorithm and insert it into algorithm map
        L1GtAlgorithm alg(menuAlgo.algName, menuAlgo.logExpression, bitNumber);
        alg.setAlgoAlias(menuAlgo.algAlias);

        // set algorithm chip number:
        int algChipNr = alg.algoChipNumber(
                numberConditionChips, pinsOnConditionChip, orderConditionChip);
        alg.setAlgoChipNumber(algChipNr);

        // insert algorithm
        m_algorithmMap[menuAlgo.algName] = alg;
        m_algorithmAliasMap[menuAlgo.algAlias] = alg;



    }

    if (edm::isDebugEnabled()) {
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

    if (edm::isDebugEnabled()) {
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

    // check if query was successful
    if (resultsMenuAlgoCond.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem to get content of L1T_MENU_ALGO_COND_VIEW for L1GtTriggerMenu"
                << " implementation key: " << objectKey;
        return false;
    }

    // temporary values
    int bitNumber = -1;
    int condIndex = -1;

    // define multimap (ALGO_INDEX, pair (COND_INDEX, COND_FK))
    std::pair<int, std::string> condIndexName;
    std::multimap<int, std::pair<int, std::string> > algoCondMap;

    //
    TableMenuAlgoCond menuAlgoCond;
    int resultsMenuAlgoCondRows = resultsMenuAlgoCond.numberRows();

    for (int iRow = 0; iRow < resultsMenuAlgoCondRows; ++iRow) {

        for (std::vector<std::string>::const_iterator constIt = columnsMenuAlgoCond.begin(); constIt
                != columnsMenuAlgoCond.end(); ++constIt) {

            if ( ( *constIt ) == "ALGO_INDEX") {
                resultsMenuAlgoCond.fillVariableFromRow(
                        *constIt, iRow, menuAlgoCond.bitNumberSh);
                bitNumber = static_cast<int> (menuAlgoCond.bitNumberSh);

            } else if ( ( *constIt ) == "COND_INDEX") {
                resultsMenuAlgoCond.fillVariableFromRow(
                        *constIt, iRow, menuAlgoCond.condIndexF);
                condIndex = static_cast<int> (menuAlgoCond.condIndexF);

            } else if ( ( *constIt ) == "COND_FK") {
                resultsMenuAlgoCond.fillVariableFromRow(*constIt, iRow, menuAlgoCond.condFK);

            } else {
                // do nothing

            }

        }

        LogTrace("L1GtTriggerMenuConfigOnlineProd") << "Row  " << iRow << ": ALGO_INDEX = "
                << bitNumber << " COND_INDEX = " << condIndex << " COND_FK = "
                << menuAlgoCond.condFK << std::endl;

        m_tableMenuAlgoCond.push_back(menuAlgoCond);

        // FIXME move from here

        // fill multimap (ALGO_INDEX, pair (COND_INDEX, COND_FK)
        condIndexName = make_pair(condIndex, menuAlgoCond.condFK);
        algoCondMap.insert(std::pair<int, std::pair<int, std::string> >(bitNumber, condIndexName));

    }

    if (edm::isDebugEnabled()) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n Number of rows read from L1T_MENU_ALGO_COND_VIEW: "
                << resultsMenuAlgoCondRows << std::endl;
    }

    if (edm::isDebugEnabled()) {
        for (std::multimap<int, std::pair<int, std::string> >::iterator it = algoCondMap.begin(); it
                != algoCondMap.end(); ++it) {

            LogTrace("L1GtTriggerMenuConfigOnlineProd") << " ALGO_INDEX: " << it->first
                    << " COND_INDEX " << ( it->second ).first << " COND_FK "
                    << ( it->second ).second << std::endl;
        }
    }

    // for each algorithm, replace in the logical expression the condition index with
    // the condition name

    // FIXME replace index with key

    return true;

}

// retrieve table with list of conditions in the menu
bool L1GtTriggerMenuConfigOnlineProd::tableMenuCondFromDB(
        const std::string& gtSchema, const std::string& objectKey) {

    // select * from CMS_GT.L1T_MENU_COND_VIEW
    //     where L1T_MENU_COND_VIEW.MENU_IMPLEMENTATION = objectKey

    const std::vector<std::string>& columnsMenuCond = m_omdsReader.columnNamesView(
            gtSchema, "L1T_MENU_COND_VIEW");

    if (edm::isDebugEnabled()) {
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

    // check if query was successful
    if (resultsMenuCond.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem to get content of L1T_MENU_COND_VIEW for L1GtTriggerMenu"
                << " implementation key: " << objectKey;
        return false;

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
                resultsMenuCond.fillVariableFromRow(*constIt, iRow, menuCond.condGeq);

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
            << " COND_GEQ " << menuCond.condGeq << "\n"
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

    if (edm::isDebugEnabled()) {
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

    if (edm::isDebugEnabled()) {
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

    // check if query was successful
    if (resultsMenuOp.queryFailed()) {
        edm::LogError("L1-O2O") << "Problem to get content of L1T_MENU_OP_VIEW for L1GtTriggerMenu"
                << " implementation key: " << objectKey;
        return false;
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
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.phiHigh);

            } else if ( ( *constIt ) == "PHI_HIGH") {
                resultsMenuOp.fillVariableFromRow(*constIt, iRow, menuObjectParameters.phiLow);

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
            << " PHI_LOW " << menuObjectParameters.phiHigh
            << " PHI_HIGH " << menuObjectParameters.phiLow
            << " QUALITY_RANGE " << menuObjectParameters.qualityRange
            << " CHARGE " << menuObjectParameters.charge
            << std::endl;

        m_tableMenuObjectParameters.push_back(menuObjectParameters);
    }

    if (edm::isDebugEnabled()) {
        LogTrace("L1GtTriggerMenuConfigOnlineProd")
                << "\n Number of rows read from L1T_MENU_OP_VIEW: " << resultsMenuOpRows
                << std::endl;

    }

    return true;

}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtTriggerMenuConfigOnlineProd);
