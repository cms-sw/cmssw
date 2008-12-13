/**
 * \class L1GtTriggerMaskTechTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtTriggerMaskTechTrigRcd.
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMaskTechTrigConfigOnlineProd.h"

// system include files
#include <vector>
#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtTriggerMaskTechTrigConfigOnlineProd::L1GtTriggerMaskTechTrigConfigOnlineProd(
        const edm::ParameterSet& parSet) :
    L1ConfigOnlineProdBase<L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask> (parSet),
            m_partitionNumber(parSet.getParameter<int> ("PartitionNumber")) {

    // empty

}

// destructor
L1GtTriggerMaskTechTrigConfigOnlineProd::~L1GtTriggerMaskTechTrigConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtTriggerMask> L1GtTriggerMaskTechTrigConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // shared pointer for L1GtTriggerMask
    boost::shared_ptr<L1GtTriggerMask> pL1GtTriggerMask = boost::shared_ptr<L1GtTriggerMask>(
            new L1GtTriggerMask());

    // l1GtTriggerMaskTechTrig: FINOR_TT_FK key in GT_PARTITION_FINOR_TT

    const std::string gtSchema = "CMS_GT";

    // FIXME QUERY

    // SQL query:
    //
    // select * from CMS_GT.GT_PARTITION_FINOR_TT WHERE GT_PARTITION_FINOR_TT.ID = objectKey

    std::vector<std::string> columns;
    columns.push_back("FINOR_TT_000");
    columns.push_back("FINOR_TT_001");

    l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
            columns, gtSchema, "GT_PARTITION_FINOR_TT", "GT_PARTITION_FINOR_TT.ID",
            m_omdsReader.singleAttribute(objectKey));

    // check if query was successful
    if (results.queryFailed()) {
        edm::LogError("L1-O2O") << "Problem with L1GtTriggerMaskTechTrigRcd key:" << objectKey;
        return pL1GtTriggerMask;
    }

    bool mask_000 = 0;
    results.fillVariable("FINOR_TT_000", mask_000);

    bool mask_001 = 0;
    results.fillVariable("FINOR_TT_001", mask_001);

    // fill the record
    //
    //    // total Bx's in the event
    //    int totalBxInEventVal = boost::lexical_cast<int>(totalBxInEventStr);
    //    pL1GtTriggerMask->setGtTotalBxInEvent(totalBxInEventVal);
    //

    if (edm::isDebugEnabled()) {
        std::ostringstream myCoutStream;
        pL1GtTriggerMask->print(myCoutStream);
        LogTrace("L1GtTriggerMaskTechTrigConfigOnlineProd")
                << "\nThe following L1GtTriggerMaskTechTrigRcd record was read from OMDS: \n"
                << myCoutStream.str() << "\n" << std::endl;
    }

    return pL1GtTriggerMask;
}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtTriggerMaskTechTrigConfigOnlineProd);
