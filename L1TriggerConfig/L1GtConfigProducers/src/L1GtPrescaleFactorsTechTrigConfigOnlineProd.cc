/**
 * \class L1GtPrescaleFactorsTechTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtPrescaleFactorsTechTrigRcd.
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPrescaleFactorsTechTrigConfigOnlineProd.h"

// system include files
#include <vector>

#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtPrescaleFactorsTechTrigConfigOnlineProd::L1GtPrescaleFactorsTechTrigConfigOnlineProd(
        const edm::ParameterSet& parSet) :
    L1ConfigOnlineProdBase<L1GtPrescaleFactorsTechTrigRcd, L1GtPrescaleFactors> (parSet) {

    // empty

}

// destructor
L1GtPrescaleFactorsTechTrigConfigOnlineProd::~L1GtPrescaleFactorsTechTrigConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtPrescaleFactors> L1GtPrescaleFactorsTechTrigConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // shared pointer for L1GtPrescaleFactors
    boost::shared_ptr<L1GtPrescaleFactors> pL1GtPrescaleFactors = boost::shared_ptr<
            L1GtPrescaleFactors>(new L1GtPrescaleFactors());

    // l1GtPrescaleFactorsTechTrig: key PRESCALE_FACTORS_TT_FK in GT_FDL_PRESCALE_FACTORS_TT table

    const std::string gtSchema = "CMS_GT";

    // SQL query:
    //
    // select * from CMS_GT.GT_FDL_PRESCALE_FACTORS_TT
    //        WHERE GT_FDL_PRESCALE_FACTORS_TT.ID = objectKey
    const std::vector<std::string>& columns = m_omdsReader.columnNames(
            gtSchema, "GT_FDL_PRESCALE_FACTORS_TT");

    if (edm::isDebugEnabled()) {
        for (std::vector<std::string>::const_iterator iter = columns.begin(); iter != columns.end(); iter++) {
            LogTrace("L1GtPrescaleFactorsTechTrigConfigOnlineProd") << ( *iter ) << std::endl;

        }
    }

    l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
            columns, gtSchema, "GT_FDL_PRESCALE_FACTORS_TT", "GT_FDL_PRESCALE_FACTORS_TT.ID",
            m_omdsReader.singleAttribute(objectKey));

    // check if query was successful
    if (results.queryFailed()) {
        edm::LogError("L1-O2O") << "Problem with L1GtPrescaleFactorsTechTrigRcd key:" << objectKey;
        return pL1GtPrescaleFactors;
    }

    // FIXME get all sets of PF, when defined!
    // TODO check column ordering - if needed, parse the column name for index

    // fill one set of prescale factors
    int pfSetSize = columns.size() - 1; // table ID is also in columns
    std::vector<int> pfSet(pfSetSize, 0);

    for (int i = 0; i < pfSetSize; i++) {
        results.fillVariable(columns[i+1], pfSet[i]);
    }

    std::vector<std::vector<int> > pFactors;
    pFactors.push_back(pfSet);

    // fill the record

    pL1GtPrescaleFactors->setGtPrescaleFactors(pFactors);

    if (edm::isDebugEnabled()) {
        std::ostringstream myCoutStream;
        pL1GtPrescaleFactors->print(myCoutStream);
        LogTrace("L1GtPrescaleFactorsTechTrigConfigOnlineProd")
                << "\nThe following L1GtPrescaleFactorsTechTrigRcd record was read from OMDS: \n"
                << myCoutStream.str() << "\n" << std::endl;
    }

    return pL1GtPrescaleFactors;
}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtPrescaleFactorsTechTrigConfigOnlineProd);
