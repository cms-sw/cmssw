/**
 * \class L1GtPrescaleFactorsAlgoTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtPrescaleFactorsAlgoTrigRcd.
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPrescaleFactorsAlgoTrigConfigOnlineProd.h"

// system include files
#include <vector>

#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtPrescaleFactorsAlgoTrigConfigOnlineProd::L1GtPrescaleFactorsAlgoTrigConfigOnlineProd(
        const edm::ParameterSet& parSet) :
            L1ConfigOnlineProdBase<L1GtPrescaleFactorsAlgoTrigRcd, L1GtPrescaleFactors> (parSet),
            m_isDebugEnabled(edm::isDebugEnabled()) {

    // empty

}

// destructor
L1GtPrescaleFactorsAlgoTrigConfigOnlineProd::~L1GtPrescaleFactorsAlgoTrigConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtPrescaleFactors> L1GtPrescaleFactorsAlgoTrigConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // FIXME seems to not work anymore in constructor...
    m_isDebugEnabled = edm::isDebugEnabled();

    // shared pointer for L1GtPrescaleFactors
    boost::shared_ptr<L1GtPrescaleFactors> pL1GtPrescaleFactors =
            boost::shared_ptr<L1GtPrescaleFactors>(new L1GtPrescaleFactors());

    // Procedure:
    // objectKey received as input is GT_RUN_SETTINGS_FK
    //
    // get from GT_RUN_SETTINGS_PRESC_VIEW the PRESCALE_FACTORS_ALGO_FK and the index PRESCALE_INDEX
    // for the GT_RUN_SETTINGS_FK
    //
    // get the prescale factors for key PRESCALE_FACTORS_ALGO_FK
    // from GT_FDL_PRESCALE_FACTORS_ALGO table

    const std::string gtSchema = "CMS_GT";

    // select * from CMS_GT.GT_RUN_SETTINGS_PRESC_VIEW
    //     where GT_RUN_SETTINGS_PRESC_VIEW.ID = objectKey

    std::vector<std::string> columnsView;
    columnsView.push_back("PRESCALE_INDEX");
    columnsView.push_back("PRESCALE_FACTORS_ALGO_FK");

    l1t::OMDSReader::QueryResults resultsView = m_omdsReader.basicQueryView(
            columnsView, gtSchema, "GT_RUN_SETTINGS_PRESC_VIEW",
            "GT_RUN_SETTINGS_PRESC_VIEW.ID", m_omdsReader.singleAttribute(
                    objectKey));

    // check if query was successful
    if (resultsView.queryFailed()) {
        edm::LogError("L1-O2O")
                << "Problem to get content of GT_RUN_SETTINGS_PRESC_VIEW "
                << "for GT_RUN_SETTINGS_PRESC_VIEW.ID: " << objectKey;
        return pL1GtPrescaleFactors;
    }

    // retrieve PRESCALE_INDEX and PRESCALE_FACTORS_ALGO_FK for GT_RUN_SETTINGS_FK

    std::string objectKeyPrescaleFactorsSet = "WRONG_KEY_INITIAL";
    short prescaleFactorsSetIndex = 9999;

    int resultsViewRows = resultsView.numberRows();
    if (m_isDebugEnabled) {
        LogTrace("L1GtPrescaleFactorsAlgoTrigConfigOnlineProd") << "\nFound "
                << resultsViewRows << " prescale factors sets for \n  "
                << "GT_RUN_SETTINGS_PRESC_VIEW.ID = " << objectKey << "\n"
                << std::endl;
    }

    // vector to be filled (resultsViewRows prescale factors sets)
    std::vector<std::vector<int> > pFactors;
    pFactors.reserve(resultsViewRows);

    int countSet = -1;

    for (int iRow = 0; iRow < resultsViewRows; ++iRow) {

        for (std::vector<std::string>::const_iterator constIt =
                columnsView.begin(); constIt != columnsView.end(); ++constIt) {

            if ((*constIt) == "PRESCALE_INDEX") {
                resultsView.fillVariableFromRow(*constIt, iRow,
                        prescaleFactorsSetIndex);
            } else if ((*constIt) == "PRESCALE_FACTORS_ALGO_FK") {
                resultsView.fillVariableFromRow(*constIt, iRow,
                        objectKeyPrescaleFactorsSet);
            } else {

                LogTrace("L1GtPrescaleFactorsAlgoTrigConfigOnlineProd")
                        << "\nUnknown field " << (*constIt)
                        << " requested for columns in GT_RUN_SETTINGS_PRESC_VIEW"
                        << std::endl;

            }
        }

        if (m_isDebugEnabled) {
            LogTrace("L1GtPrescaleFactorsAlgoTrigConfigOnlineProd")
                    << "\nFound prescale factors set: \n  index = "
                    << prescaleFactorsSetIndex
                    << "\n  PRESCALE_FACTORS_ALGO_FK = "
                    << objectKeyPrescaleFactorsSet << "\n" << std::endl;
        }

        // retrive now the prescale factors for PRESCALE_FACTORS_ALGO_FK


        // SQL query:
        //
        // select * from CMS_GT.GT_FDL_PRESCALE_FACTORS_ALGO
        //        WHERE GT_FDL_PRESCALE_FACTORS_ALGO.ID = objectKeyPrescaleFactorsSet
        const std::vector<std::string>& columns = m_omdsReader.columnNames(
                gtSchema, "GT_FDL_PRESCALE_FACTORS_ALGO");

        if (m_isDebugEnabled) {
            LogTrace("L1GtPrescaleFactorsAlgoTrigConfigOnlineProd")
                    << "\nColumn names for GT_FDL_PRESCALE_FACTORS_ALGO"
                    << std::endl;

            for (std::vector<std::string>::const_iterator iter =
                    columns.begin(); iter != columns.end(); iter++) {
                LogTrace("L1GtPrescaleFactorsAlgoTrigConfigOnlineProd")
                        << (*iter) << std::endl;

            }
        }

        l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
                columns, gtSchema, "GT_FDL_PRESCALE_FACTORS_ALGO",
                "GT_FDL_PRESCALE_FACTORS_ALGO.ID",
                m_omdsReader.singleAttribute(objectKeyPrescaleFactorsSet));

        // check if query was successful
        if (results.queryFailed()) {
            edm::LogError("L1-O2O")
                    << "Problem with L1GtPrescaleFactorsAlgoTrigRcd key:"
                    << objectKeyPrescaleFactorsSet;
            return pL1GtPrescaleFactors;
        }




        // check if set indices are ordered, starting from 0 (initial value for countSet is -1)
        countSet++;
        if (prescaleFactorsSetIndex != countSet) {
            edm::LogError("L1-O2O")
                    << "L1GtPrescaleFactorsAlgoTrig has unordered sets PRESCALE_INDEX in DB for\n"
                    << " GT_RUN_SETTINGS_PRESC_VIEW.ID = " << objectKey << "\n"
                    << std::endl;
            return pL1GtPrescaleFactors;

        }


        // fill one set of prescale factors
        int pfSetSize = columns.size() - 1; // table ID is also in columns
        std::vector<int> pfSet(pfSetSize, 0);

        for (int i = 0; i < pfSetSize; i++) {
            results.fillVariable(columns[i + 1], pfSet[i]);
        }

        pFactors.push_back(pfSet);

    }

    // fill the record

    pL1GtPrescaleFactors->setGtPrescaleFactors(pFactors);

    if (m_isDebugEnabled) {
        std::ostringstream myCoutStream;
        pL1GtPrescaleFactors->print(myCoutStream);
        LogTrace("L1GtPrescaleFactorsAlgoTrigConfigOnlineProd")
                << "\nThe following L1GtPrescaleFactorsAlgoTrigRcd record was read from OMDS: \n"
                << myCoutStream.str() << "\n" << std::endl;
    }

    return pL1GtPrescaleFactors;
}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtPrescaleFactorsAlgoTrigConfigOnlineProd);
