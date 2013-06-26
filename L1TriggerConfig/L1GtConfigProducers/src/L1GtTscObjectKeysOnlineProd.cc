/**
 * \class L1GtTscObjectKeysOnlineProd
 *
 *
 * Description: online producer for L1 GT record keys starting from TSC key.
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTscObjectKeysOnlineProd.h"

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtTscObjectKeysOnlineProd::L1GtTscObjectKeysOnlineProd(const edm::ParameterSet& parSet) :
    L1ObjectKeysOnlineProdBase(parSet),
    m_enableL1GtParameters(
            parSet.getParameter<bool> ("EnableL1GtParameters")),
    m_enableL1GtTriggerMenu(
            parSet.getParameter<bool> ("EnableL1GtTriggerMenu")),
    m_enableL1GtPsbSetup(
            parSet.getParameter<bool> ("EnableL1GtPsbSetup")) {

    // empty

}

// destructor
L1GtTscObjectKeysOnlineProd::~L1GtTscObjectKeysOnlineProd() {

    // empty

}

// private methods

std::string L1GtTscObjectKeysOnlineProd::keyL1GtParameters(
        const std::string& subsystemKey, const std::string& gtSchema) {

    std::string objectKey;

    if (!subsystemKey.empty()) {

        // Execute SQL queries to get keys from OMDS.

        // SELECT GTFE_SETUP_FK FROM CMS_GT.GT_SETUP WHERE GT_SETUP.ID = MyKey
        l1t::OMDSReader::QueryResults objectKeyResults = m_omdsReader.basicQuery(
                "GTFE_SETUP_FK", gtSchema, "GT_SETUP", "GT_SETUP.ID", m_omdsReader.singleAttribute(
                        subsystemKey));

        // check if query was successful
        if (objectKeyResults.queryFailed()) {
            edm::LogError("L1-O2O")
                    << "Problem with key for record L1GtParametersRcd: query failed ";
            return objectKey;
        } else if ( ( objectKeyResults.numberRows() != 1 )) {
            edm::LogError("L1-O2O") << "Problem with key for record L1GtParametersRcd: "
                    << ( objectKeyResults.numberRows() ) << " rows were returned";
            return objectKey;
        }

        objectKeyResults.fillVariable(objectKey);
    }

    return objectKey;
}

std::string L1GtTscObjectKeysOnlineProd::keyL1GtTriggerMenu(
        const std::string& subsystemKey, const std::string& gtSchema) {

    std::string objectKey;

    if (!subsystemKey.empty()) {

        // Execute SQL queries to get keys from OMDS.

        // SELECT L1T_MENU_FK FROM CMS_GT.GT_SETUP WHERE GT_SETUP.ID = MyKey
        l1t::OMDSReader::QueryResults objectKeyResults = m_omdsReader.basicQuery(
                "L1T_MENU_FK", gtSchema, "GT_SETUP", "GT_SETUP.ID", m_omdsReader.singleAttribute(
                        subsystemKey));

        // check if query was successful
        if (objectKeyResults.queryFailed()) {
            edm::LogError("L1-O2O")
                    << "Problem with key for record L1GtTriggerMenuRcd: query failed ";
            return objectKey;
        } else if ( ( objectKeyResults.numberRows() != 1 )) {
            edm::LogError("L1-O2O") << "Problem with key for record L1GtTriggerMenuRcd: "
                    << ( objectKeyResults.numberRows() ) << " rows were returned";
            return objectKey;
        }

        objectKeyResults.fillVariable(objectKey);
    }

    return objectKey;

}

std::string L1GtTscObjectKeysOnlineProd::keyL1GtPsbSetup(
        const std::string& subsystemKey, const std::string& gtSchema) {

    std::string objectKey;

    if (!subsystemKey.empty()) {

        // no need to query OMDS, one uses the GT_SETUP key to get the individual PSB keys.
        //     the L1GtPsbSetup key is GT_SETUP key
        objectKey = subsystemKey;

    }

    return objectKey;

}

// public methods
void L1GtTscObjectKeysOnlineProd::fillObjectKeys(ReturnType pL1TriggerKey) {

    // kMYSUBSYSTEM = kCSCTF, kDTTF, kRPC, kGMT, kRCT, mkGCT, kGT, or kTSP0
    // subsystemKey = TRIGGERSUP_CONF.{CSCTF_KEY, DTTF_KEY, RPC_KEY, GMT_KEY, RCT_KEY, GCT_KEY, GT_KEY}

    std::string subsystemKey = pL1TriggerKey->subsystemKey(L1TriggerKey::kGT);
    const std::string gtSchema = "CMS_GT";

    //
    if (m_enableL1GtParameters) {
        const std::string& objectKey = keyL1GtParameters(subsystemKey, gtSchema);
        if (!objectKey.empty()) {
            pL1TriggerKey->add("L1GtParametersRcd", "L1GtParameters", objectKey);
        }
    }

    //
    if (m_enableL1GtTriggerMenu) {
        const std::string& objectKey = keyL1GtTriggerMenu(subsystemKey, gtSchema);
        if (!objectKey.empty()) {
            pL1TriggerKey->add("L1GtTriggerMenuRcd", "L1GtTriggerMenu", objectKey);
        }
    }

    //
    if (m_enableL1GtPsbSetup) {
        const std::string& objectKey = keyL1GtPsbSetup(subsystemKey, gtSchema);
        if (!objectKey.empty()) {
            pL1TriggerKey->add("L1GtPsbSetupRcd", "L1GtPsbSetup", objectKey);
        }
    }

}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtTscObjectKeysOnlineProd);
