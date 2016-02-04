/**
 * \class L1GtRsObjectKeysOnlineProd
 *
 *
 * Description: online producer for L1 GT record keys from RUN SETTINGS.
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtRsObjectKeysOnlineProd.h"

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtRsObjectKeysOnlineProd::L1GtRsObjectKeysOnlineProd(const edm::ParameterSet& parSet) :
    L1ObjectKeysOnlineProdBase(parSet),
    m_partitionNumber(
            parSet.getParameter<int> ("PartitionNumber")),
    m_enableL1GtPrescaleFactorsAlgoTrig(
            parSet.getParameter<bool> ("EnableL1GtPrescaleFactorsAlgoTrig")),
    m_enableL1GtPrescaleFactorsTechTrig(
            parSet.getParameter<bool> ("EnableL1GtPrescaleFactorsTechTrig")),
    m_enableL1GtTriggerMaskAlgoTrig(
            parSet.getParameter<bool> ("EnableL1GtTriggerMaskAlgoTrig")),
    m_enableL1GtTriggerMaskTechTrig(
            parSet.getParameter<bool> ("EnableL1GtTriggerMaskTechTrig")),
    m_enableL1GtTriggerMaskVetoTechTrig(
            parSet.getParameter<bool> ("EnableL1GtTriggerMaskVetoTechTrig")) {

    // empty
}

// destructor
L1GtRsObjectKeysOnlineProd::~L1GtRsObjectKeysOnlineProd() {

    // empty

}

// private methods
std::string L1GtRsObjectKeysOnlineProd::keyL1GtPrescaleFactorsAlgoTrig(const std::string& gtSchema) {

    std::string objectKey;

    // SELECT GT_RUN_SETTINGS_FK FROM CMS_GT.GT_RUN_SETTINGS_KEY_CURRENT
    //        WHERE GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER = m_partitionNumber
    l1t::OMDSReader::QueryResults objectKeyResults =
            m_omdsReader.basicQueryGenericKey<int> ("GT_RUN_SETTINGS_FK",
                    gtSchema, "GT_RUN_SETTINGS_KEY_CURRENT",
                    "GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER",
                    m_omdsReader.singleAttribute(m_partitionNumber), "");

    // check if query was successful
    if (objectKeyResults.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem with key for record L1GtPrescaleFactorsAlgoTrigRcd: query failed ";
        return objectKey;
    } else if ( ( objectKeyResults.numberRows() != 1 )) {
        edm::LogError("L1-O2O") << "Problem with key for record L1GtPrescaleFactorsAlgoTrigRcd: "
                << ( objectKeyResults.numberRows() ) << " rows were returned";
        return objectKey;
    }

    objectKeyResults.fillVariable(objectKey);

    //
    if (edm::isDebugEnabled()) {
        LogTrace("L1GtRsObjectKeysOnlineProd")
                << "\nThe following GT_RUN_SETTINGS_FK key "
                << "was found for L1GtPrescaleFactorsAlgoTrigRcd: \n  "
                << objectKey << "\nfor partition " << m_partitionNumber << "\n"
                << std::endl;
    }

    return objectKey;
}


std::string L1GtRsObjectKeysOnlineProd::keyL1GtPrescaleFactorsTechTrig(const std::string& gtSchema) {

    std::string objectKey;

    // SELECT GT_RUN_SETTINGS_FK FROM CMS_GT.GT_RUN_SETTINGS_KEY_CURRENT
    //        WHERE GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER = m_partitionNumber
    l1t::OMDSReader::QueryResults objectKeyResults =
            m_omdsReader.basicQueryGenericKey<int> ("GT_RUN_SETTINGS_FK",
                    gtSchema, "GT_RUN_SETTINGS_KEY_CURRENT",
                    "GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER",
                    m_omdsReader.singleAttribute(m_partitionNumber), "");

    // check if query was successful
    if (objectKeyResults.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem with key for record L1GtPrescaleFactorsTechTrigRcd: query failed ";
        return objectKey;
    } else if ( ( objectKeyResults.numberRows() != 1 )) {
        edm::LogError("L1-O2O") << "Problem with key for record L1GtPrescaleFactorsTechTrigRcd: "
                << ( objectKeyResults.numberRows() ) << " rows were returned";
        return objectKey;
    }

    objectKeyResults.fillVariable(objectKey);

    //
    if (edm::isDebugEnabled()) {
        LogTrace("L1GtRsObjectKeysOnlineProd")
                << "\nThe following GT_RUN_SETTINGS_FK key "
                << "was found for L1GtPrescaleFactorsTechTrigRcd: \n  "
                << objectKey << "\nfor partition " << m_partitionNumber << "\n"
                << std::endl;
    }

    return objectKey;
}


std::string L1GtRsObjectKeysOnlineProd::keyL1GtTriggerMaskAlgoTrig(const std::string& gtSchema) {

    std::string objectKey;

    // SELECT FINOR_ALGO_FK FROM CMS_GT.GT_RUN_SETTINGS
    //        WHERE GT_RUN_SETTINGS.ID = (
    // SELECT GT_RUN_SETTINGS_FK FROM CMS_GT.GT_RUN_SETTINGS_KEY_CURRENT
    //        WHERE GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER = m_partitionNumber)
    l1t::OMDSReader::QueryResults objectKeyResults = m_omdsReader.basicQuery(
            "FINOR_ALGO_FK", gtSchema, "GT_RUN_SETTINGS", "GT_RUN_SETTINGS.ID",
            m_omdsReader.basicQueryGenericKey<int> (
                    "GT_RUN_SETTINGS_FK", gtSchema, "GT_RUN_SETTINGS_KEY_CURRENT",
                    "GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER",
                    m_omdsReader.singleAttribute(m_partitionNumber), ""));

    // check if query was successful
    if (objectKeyResults.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem with key for record L1GtTriggerMaskAlgoTrigRcd: query failed ";
        return objectKey;
    } else if ( ( objectKeyResults.numberRows() != 1 )) {
        edm::LogError("L1-O2O") << "Problem with key for record L1GtTriggerMaskAlgoTrigRcd: "
                << ( objectKeyResults.numberRows() ) << " rows were returned";
        return objectKey;
    }

    objectKeyResults.fillVariable(objectKey);

    //
    if (edm::isDebugEnabled()) {
        LogTrace("L1GtRsObjectKeysOnlineProd")
                << "\nThe following key was found for L1GtTriggerMaskAlgoTrigRcd: \n  " << objectKey
                << "\nfor partition " << m_partitionNumber << "\n" << std::endl;
    }

    return objectKey;
}

std::string L1GtRsObjectKeysOnlineProd::keyL1GtTriggerMaskTechTrig(const std::string& gtSchema) {

    std::string objectKey;

    // SELECT FINOR_TT_FK FROM CMS_GT.GT_RUN_SETTINGS
    //        WHERE GT_RUN_SETTINGS.ID = (
    // SELECT GT_RUN_SETTINGS_FK FROM CMS_GT.GT_RUN_SETTINGS_KEY_CURRENT
    //        WHERE GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER = m_partitionNumber)
    l1t::OMDSReader::QueryResults objectKeyResults = m_omdsReader.basicQuery(
            "FINOR_TT_FK", gtSchema, "GT_RUN_SETTINGS", "GT_RUN_SETTINGS.ID",
            m_omdsReader.basicQueryGenericKey<int> (
                    "GT_RUN_SETTINGS_FK", gtSchema, "GT_RUN_SETTINGS_KEY_CURRENT",
                    "GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER",
                    m_omdsReader.singleAttribute(m_partitionNumber), ""));

    // check if query was successful
    if (objectKeyResults.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem with key for record L1GtTriggerMaskTechTrigRcd: query failed ";
        return objectKey;
    } else if ( ( objectKeyResults.numberRows() != 1 )) {
        edm::LogError("L1-O2O") << "Problem with key for record L1GtTriggerMaskTechTrigRcd: "
                << ( objectKeyResults.numberRows() ) << " rows were returned";
        return objectKey;
    }

    objectKeyResults.fillVariable(objectKey);

    //
    if (edm::isDebugEnabled()) {
        LogTrace("L1GtRsObjectKeysOnlineProd")
                << "\nThe following key was found for L1GtTriggerMaskTechTrigRcd: \n  " << objectKey
                << "\nfor partition " << m_partitionNumber << "\n" << std::endl;
    }

    return objectKey;
}

std::string L1GtRsObjectKeysOnlineProd::keyL1GtTriggerMaskVetoTechTrig(const std::string& gtSchema) {

    std::string objectKey;

    // SELECT VETO_TT_FK FROM CMS_GT.GT_RUN_SETTINGS
    //        WHERE GT_RUN_SETTINGS.ID = (
    // SELECT GT_RUN_SETTINGS_FK FROM CMS_GT.GT_RUN_SETTINGS_KEY_CURRENT
    //        WHERE GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER = m_partitionNumber)
    l1t::OMDSReader::QueryResults objectKeyResults = m_omdsReader.basicQuery(
            "VETO_TT_FK", gtSchema, "GT_RUN_SETTINGS", "GT_RUN_SETTINGS.ID",
            m_omdsReader.basicQueryGenericKey<int> (
                    "GT_RUN_SETTINGS_FK", gtSchema, "GT_RUN_SETTINGS_KEY_CURRENT",
                    "GT_RUN_SETTINGS_KEY_CURRENT.GT_PARTITION_NUMBER",
                    m_omdsReader.singleAttribute(m_partitionNumber), ""));

    // check if query was successful
    if (objectKeyResults.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem with key for record L1GtTriggerMaskVetoTechTrigRcd: query failed ";
        return objectKey;
    } else if ( ( objectKeyResults.numberRows() != 1 )) {
        edm::LogError("L1-O2O") << "Problem with key for record L1GtTriggerMaskVetoTechTrigRcd: "
                << ( objectKeyResults.numberRows() ) << " rows were returned";
        return objectKey;
    }

    objectKeyResults.fillVariable(objectKey);

    //
    if (edm::isDebugEnabled()) {
        LogTrace("L1GtRsObjectKeysOnlineProd")
                << "\nThe following key was found for L1GtTriggerMaskVetoTechTrigRcd: \n  "
                << objectKey << "\nfor partition " << m_partitionNumber << "\n" << std::endl;
    }

    return objectKey;
}

// public methods
void L1GtRsObjectKeysOnlineProd::fillObjectKeys(ReturnType pL1TriggerKey) {

    const std::string gtSchema = "CMS_GT";

    if (m_enableL1GtPrescaleFactorsAlgoTrig) {
        const std::string& objectKey = keyL1GtPrescaleFactorsAlgoTrig(gtSchema);
        if (!objectKey.empty()) {
            pL1TriggerKey->add(
                    "L1GtPrescaleFactorsAlgoTrigRcd", "L1GtPrescaleFactors", objectKey);
        }
    }

    if (m_enableL1GtPrescaleFactorsTechTrig) {
        const std::string& objectKey = keyL1GtPrescaleFactorsTechTrig(gtSchema);
        if (!objectKey.empty()) {
            pL1TriggerKey->add(
                    "L1GtPrescaleFactorsTechTrigRcd", "L1GtPrescaleFactors", objectKey);
        }
    }

    if (m_enableL1GtTriggerMaskAlgoTrig) {
        const std::string& objectKey = keyL1GtTriggerMaskAlgoTrig(gtSchema);
        if (!objectKey.empty()) {
            pL1TriggerKey->add("L1GtTriggerMaskAlgoTrigRcd", "L1GtTriggerMask", objectKey);
        }
    }

    if (m_enableL1GtTriggerMaskTechTrig) {
        const std::string& objectKey = keyL1GtTriggerMaskTechTrig(gtSchema);
        if (!objectKey.empty()) {
            pL1TriggerKey->add("L1GtTriggerMaskTechTrigRcd", "L1GtTriggerMask", objectKey);
        }
    }

    if (m_enableL1GtTriggerMaskVetoTechTrig) {
        const std::string& objectKey = keyL1GtTriggerMaskVetoTechTrig(gtSchema);
        if (!objectKey.empty()) {
            pL1TriggerKey->add(
                    "L1GtTriggerMaskVetoTechTrigRcd", "L1GtTriggerMask", objectKey);
        }
    }

}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtRsObjectKeysOnlineProd);
