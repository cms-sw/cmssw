/**
 * \class L1GtPsbSetupConfigOnlineProd
 *
 *
 * Description: online producer for L1GtPsbSetup.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * \author: Thomas Themel      - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPsbSetupConfigOnlineProd.h"

// system include files
#include "boost/lexical_cast.hpp"
#include <algorithm>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtPsbSetupConfigOnlineProd::L1GtPsbSetupConfigOnlineProd(const edm::ParameterSet& parSet) :
    L1ConfigOnlineProdBase<L1GtPsbSetupRcd, L1GtPsbSetup> (parSet) {

    // empty

}

// destructor
L1GtPsbSetupConfigOnlineProd::~L1GtPsbSetupConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtPsbSetup> L1GtPsbSetupConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // shared pointer for L1GtPsbSetup
    boost::shared_ptr<L1GtPsbSetup> pL1GtPsbSetup = boost::shared_ptr<L1GtPsbSetup>(
            new L1GtPsbSetup());

    const std::string gtSchema = "CMS_GT";

    // the setup's contents, to be filled from database
    std::vector<L1GtPsbConfig> psbConfigurations;

    // SELECT PSB_SLOT_*_SETUP_FK FROM CMS_GT.GT_SETUP WHERE GT_SETUP.ID = MyKey
    std::vector<std::string> psbColumns = m_omdsReader.columnNames(gtSchema, "GT_SETUP");

    std::vector<std::string>::iterator newEnd = std::remove_if(
            psbColumns.begin(), psbColumns.end(), &notPsbColumnName);
    psbColumns.erase(newEnd, psbColumns.end());

    // select * from CMS_GT.GT_SETUP where GT_SETUP.ID = objectKey
    l1t::OMDSReader::QueryResults psbKeys = m_omdsReader.basicQuery(
            psbColumns, gtSchema, "GT_SETUP", "GT_SETUP.ID",
            m_omdsReader.singleAttribute(objectKey));

    // check if query was successful and produced one line only
    if (!checkOneLineResult(psbKeys, "GT_SETUP query for PSB keys with ID = " + objectKey)) {
        edm::LogError("L1-O2O")
            << "Problem to get content of CMS_GT.GT_SETUP for GT_SETUP.ID key: "
            << objectKey;
        return pL1GtPsbSetup;
    }

    // fill the psbConfigurations vector
    for (std::vector<std::string>::const_iterator it = psbColumns.begin(); it != psbColumns.end(); ++it) {

        std::string psbKey;
        psbKeys.fillVariable(*it, psbKey);

        if (psbKey.empty()) {
            addDefaultPsb(*it, psbConfigurations);
        } else {
            addPsbFromDb(psbKey, psbConfigurations);
        }
    }

    // assign to the result object
    pL1GtPsbSetup->setGtPsbSetup(psbConfigurations);

    if (edm::isDebugEnabled()) {
        LogTrace("L1-O2O") << (*pL1GtPsbSetup) << std::endl;
    }

    return pL1GtPsbSetup;
}

// Return true if columnName does NOT match the pattern
// PSB_SLOT_.*_SETUP_FK
bool L1GtPsbSetupConfigOnlineProd::notPsbColumnName(const std::string& columnName) {

    static std::string startMatch("PSB_SLOT_");
    static std::string endMatch("_SETUP_FK");

    unsigned len = columnName.size();

    // it's not a PSB column name if it's too short
    return len <= ( startMatch.size() + endMatch.size() ) ||
    // or the start doesn't match
            columnName.substr(0, startMatch.size()) != startMatch ||
    // or the end doesn't match
            columnName.substr(len - endMatch.size(), endMatch.size()) != endMatch;
}

void L1GtPsbSetupConfigOnlineProd::addPsbFromDb(const std::string& psbKey, std::vector<
        L1GtPsbConfig>& psbSetup) {

    // SQL> describe gt_psb_setup;
    // (heavily pruned to just the stuff we need to set up a GtPsbConfig)
    //  Name                                      Null?    Type
    //  ----------------------------------------- -------- ----------------------------
    //  ID                                        NOT NULL VARCHAR2(256)
    //  BOARD_SLOT                                         NUMBER(3)
    //  CH0_SEND_LVDS_NOT_DS92LV16                         NUMBER(1)
    //  CH1_SEND_LVDS_NOT_DS92LV16                         NUMBER(1)
    //  ENABLE_TT_LVDS_0                                   NUMBER(1)
    //  ENABLE_TT_LVDS_1                                   NUMBER(1)
    //  ENABLE_TT_LVDS_2                                   NUMBER(1)
    //  ENABLE_TT_LVDS_3                                   NUMBER(1)
    //  ENABLE_TT_LVDS_4                                   NUMBER(1)
    //  ENABLE_TT_LVDS_5                                   NUMBER(1)
    //  ENABLE_TT_LVDS_6                                   NUMBER(1)
    //  ENABLE_TT_LVDS_7                                   NUMBER(1)
    //  ENABLE_TT_LVDS_8                                   NUMBER(1)
    //  ENABLE_TT_LVDS_9                                   NUMBER(1)
    //  ENABLE_TT_LVDS_10                                  NUMBER(1)
    //  ENABLE_TT_LVDS_11                                  NUMBER(1)
    //  ENABLE_TT_LVDS_12                                  NUMBER(1)
    //  ENABLE_TT_LVDS_13                                  NUMBER(1)
    //  ENABLE_TT_LVDS_14                                  NUMBER(1)
    //  ENABLE_TT_LVDS_15                                  NUMBER(1)
    //  SERLINK_CH0_REC_ENABLE                             NUMBER(1)
    //  SERLINK_CH1_REC_ENABLE                             NUMBER(1)
    //  SERLINK_CH2_REC_ENABLE                             NUMBER(1)
    //  SERLINK_CH3_REC_ENABLE                             NUMBER(1)
    //  SERLINK_CH4_REC_ENABLE                             NUMBER(1)
    //  SERLINK_CH5_REC_ENABLE                             NUMBER(1)
    //  SERLINK_CH6_REC_ENABLE                             NUMBER(1)
    //  SERLINK_CH7_REC_ENABLE                             NUMBER(1)

    const std::string gtSchema = "CMS_GT";

    // setup up columns to query
    std::vector<std::string> columnNames;

    static const std::string lvdPrefix = "ENABLE_TT_LVDS_";
    static const std::string lvdSuffix = "";
    static const std::string serPrefix = "SERLINK_CH";
    static const std::string serSuffix = "_REC_ENABLE";
    static const std::string boardSlotColumn = "BOARD_SLOT";
    static const std::string ch0FormatColumn = "CH0_SEND_LVDS_NOT_DS92LV16";
    static const std::string ch1FormatColumn = "CH1_SEND_LVDS_NOT_DS92LV16";

    columnNames.push_back(boardSlotColumn);
    columnNames.push_back(ch0FormatColumn);
    columnNames.push_back(ch1FormatColumn);

    for (unsigned i = 0; i < (unsigned) L1GtPsbConfig::PsbNumberLvdsGroups; ++i) {
        columnNames.push_back(numberedColumnName(lvdPrefix, i, lvdSuffix));
    }

    for (unsigned i = 0; i < (unsigned) L1GtPsbConfig::PsbSerLinkNumberChannels; ++i) {
        columnNames.push_back(numberedColumnName(serPrefix, i, serSuffix));
    }

    // execute database query
    l1t::OMDSReader::QueryResults psbQuery = m_omdsReader.basicQuery(
            columnNames, gtSchema, "GT_PSB_SETUP", "GT_PSB_SETUP.ID", m_omdsReader.singleAttribute(
                    psbKey));

    // check if query was successful and we get only one line
    if (!checkOneLineResult(psbQuery, "GT_PSB_SETUP query for PSB keys with ID = " + psbKey)) {
        edm::LogError("L1-O2O")
            << "Problem to get setup for PSB keys with ID = " << psbKey;
        return; // FIXME: change method to bool?
    }

    // extract values
    int16_t boardSlot;

    getRequiredValue(psbQuery, boardSlotColumn, boardSlot);

    bool sendLvds0, sendLvds1;

    getRequiredValue(psbQuery, ch0FormatColumn, sendLvds0);
    getRequiredValue(psbQuery, ch1FormatColumn, sendLvds1);

    const std::vector<bool>& enableRecLvds = extractBoolVector(
            psbQuery, lvdPrefix, lvdSuffix, L1GtPsbConfig::PsbNumberLvdsGroups);
    const std::vector<bool>& serLinkRecEnable = extractBoolVector(
            psbQuery, serPrefix, serSuffix, L1GtPsbConfig::PsbSerLinkNumberChannels);

    // create new PSB object with db values
    L1GtPsbConfig toAdd;

    toAdd.setGtBoardSlot(boardSlot);
    toAdd.setGtPsbCh0SendLvds(sendLvds0);
    toAdd.setGtPsbCh1SendLvds(sendLvds1);
    toAdd.setGtPsbEnableRecLvds(enableRecLvds);
    toAdd.setGtPsbEnableRecSerLink(serLinkRecEnable);

    // add to vector
    psbSetup.push_back(toAdd);
}

void L1GtPsbSetupConfigOnlineProd::addDefaultPsb(const std::string& psbColumn, std::vector<
        L1GtPsbConfig>& psbSetup) const {

    // deduce the assigned board from the column name
    unsigned boardSlot = numberFromString(psbColumn);

    // create a default PsbConfig with the appropriate board slot and all links disabled
    L1GtPsbConfig toAdd;
    static std::vector<bool> allFalseLvds(L1GtPsbConfig::PsbNumberLvdsGroups, false);
    static std::vector<bool> allFalseSerLink(L1GtPsbConfig::PsbSerLinkNumberChannels, false);

    toAdd.setGtBoardSlot(boardSlot);
    toAdd.setGtPsbCh0SendLvds(false);
    toAdd.setGtPsbCh1SendLvds(false);
    toAdd.setGtPsbEnableRecLvds(allFalseLvds);
    toAdd.setGtPsbEnableRecSerLink(allFalseSerLink);

    psbSetup.push_back(toAdd);
}

std::vector<bool> L1GtPsbSetupConfigOnlineProd::extractBoolVector(
        const l1t::OMDSReader::QueryResults& query, const std::string& prefix,
        const std::string& suffix, unsigned nColumns) const {

    std::vector<bool> result(nColumns);

    for (unsigned i = 0; i < nColumns; ++i) {
        bool dbValue;
        getRequiredValue(query, numberedColumnName(prefix, i, suffix), dbValue);
        result[i] = dbValue;
    }

    return result;
}

bool L1GtPsbSetupConfigOnlineProd::checkOneLineResult(
        const l1t::OMDSReader::QueryResults& result, const std::string& queryDescription) const {

    // check if query was successful
    if (result.queryFailed()) {
        edm::LogError("L1-O2O") << "\n " << queryDescription + " failed: no match found!";
        return false;

    } else if (result.numberRows() != 1) {
        edm::LogError("L1-O2O") << "\nProblem with " << queryDescription << ": "
                << ( result.numberRows() ) << " rows were returned, expected 1.";
        return false;
    }

    return true;
}


std::string L1GtPsbSetupConfigOnlineProd::numberedColumnName(
        const std::string& prefix, unsigned number) const {

    return numberedColumnName(prefix, number, "");

}

std::string L1GtPsbSetupConfigOnlineProd::numberedColumnName(
        const std::string& prefix, unsigned number, const std::string& suffix) const {

    std::ostringstream colName;
    colName << prefix << number << suffix;

    return colName.str();
}

unsigned L1GtPsbSetupConfigOnlineProd::numberFromString(const std::string& aString) const {

    std::istringstream stream(aString);
    unsigned result;

    for (unsigned i = 0; i < aString.size(); ++i) {
        if (stream >> result) {
            // we got a value
            return result;
        } else {
            // clear error flags from failed >>
            stream.clear();
            // skip another character
            stream.seekg(i);
        }
    }

    // throw here
    throw cms::Exception("NumberNotFound") << "Failed to extract numeric value from " << aString;
}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtPsbSetupConfigOnlineProd);
