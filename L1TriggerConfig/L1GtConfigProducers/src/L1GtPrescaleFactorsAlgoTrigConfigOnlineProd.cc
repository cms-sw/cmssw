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
    L1ConfigOnlineProdBase<L1GtPrescaleFactorsAlgoTrigRcd, L1GtPrescaleFactors> (parSet) {

    // empty

}

// destructor
L1GtPrescaleFactorsAlgoTrigConfigOnlineProd::~L1GtPrescaleFactorsAlgoTrigConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtPrescaleFactors> L1GtPrescaleFactorsAlgoTrigConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // shared pointer for L1GtPrescaleFactors
    boost::shared_ptr<L1GtPrescaleFactors> pL1GtPrescaleFactors = boost::shared_ptr<
            L1GtPrescaleFactors>(new L1GtPrescaleFactors());

    // l1GtPrescaleFactorsAlgoTrig: key PRESCALE_FACTORS_ALGO_FK in GT_FDL_PRESCALE_FACTORS_ALGO table

    const std::string gtSchema = "CMS_GT";

    // FIXME QUERY


    // SQL query:
    //
    // select * from CMS_GT.GT_FDL_PRESCALE_FACTORS_ALGO
    //        WHERE GT_FDL_PRESCALE_FACTORS_ALGO.ID = objectKey

    std::vector<std::string> columns;
    columns.push_back("PRESCALE_FACTOR_ALGO_000");
    columns.push_back("PRESCALE_FACTOR_ALGO_001");

    l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
            columns, gtSchema, "GT_FDL_PRESCALE_FACTORS_ALGO", "GT_FDL_PRESCALE_FACTORS_ALGO.ID",
            m_omdsReader.singleAttribute(objectKey));

    // check if query was successful
    if (results.queryFailed()) {
        edm::LogError("L1-O2O") << "Problem with L1GtPrescaleFactorsAlgoTrigRcd key:" << objectKey;
        return pL1GtPrescaleFactors;
    }

    int factor000 = 0;
    results.fillVariable("PRESCALE_FACTOR_ALGO_000", factor000);

    int factor001 = 0;
    results.fillVariable("PRESCALE_FACTOR_ALGO_001", factor001);

    std::vector<int> pfSet;
    pfSet.push_back(factor000);
    pfSet.push_back(factor001);

    std::vector<std::vector<int> > pFactors;

    pFactors.push_back(pfSet);

    // fill the record

    pL1GtPrescaleFactors->setGtPrescaleFactors(pFactors);

    if (edm::isDebugEnabled()) {
        std::ostringstream myCoutStream;
        pL1GtPrescaleFactors->print(myCoutStream);
        LogTrace("L1GtPrescaleFactorsAlgoTrigConfigOnlineProd")
                << "\nThe following L1GtPrescaleFactorsAlgoTrigRcd record was read from OMDS: \n"
                << myCoutStream.str() << "\n" << std::endl;
    }

    return pL1GtPrescaleFactors;
}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtPrescaleFactorsAlgoTrigConfigOnlineProd);
