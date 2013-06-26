/**
 * \class L1GtTriggerMaskAlgoTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtTriggerMaskAlgoTrigRcd.
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMaskAlgoTrigConfigOnlineProd.h"

// system include files
#include <vector>

#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtTriggerMaskAlgoTrigConfigOnlineProd::L1GtTriggerMaskAlgoTrigConfigOnlineProd(
        const edm::ParameterSet& parSet) :
    L1ConfigOnlineProdBase<L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask> (parSet),
            m_partitionNumber(parSet.getParameter<int> ("PartitionNumber")) {

    // empty

}

// destructor
L1GtTriggerMaskAlgoTrigConfigOnlineProd::~L1GtTriggerMaskAlgoTrigConfigOnlineProd() {

    // empty

}

// public methods

boost::shared_ptr<L1GtTriggerMask> L1GtTriggerMaskAlgoTrigConfigOnlineProd::newObject(
        const std::string& objectKey) {

    // shared pointer for L1GtTriggerMask
    boost::shared_ptr<L1GtTriggerMask> pL1GtTriggerMask = boost::shared_ptr<L1GtTriggerMask>(
            new L1GtTriggerMask());

    // l1GtTriggerMaskAlgoTrig: FINOR_ALGO_FK key in GT_PARTITION_FINOR_ALGO

    const std::string gtSchema = "CMS_GT";

    // SQL query:
    // select * from CMS_GT.GT_PARTITION_FINOR_ALGO WHERE GT_PARTITION_FINOR_ALGO.ID = objectKey

    const std::vector<std::string>& columns = m_omdsReader.columnNames(
            gtSchema, "GT_PARTITION_FINOR_ALGO");

    if (edm::isDebugEnabled()) {
        for (std::vector<std::string>::const_iterator iter = columns.begin(); iter != columns.end(); iter++) {
            LogTrace("L1GtTriggerMaskAlgoTrigConfigOnlineProd") << ( *iter ) << std::endl;

        }
    }

    l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
            columns, gtSchema, "GT_PARTITION_FINOR_ALGO", "GT_PARTITION_FINOR_ALGO.ID",
            m_omdsReader.singleAttribute(objectKey));

    // check if query was successful
    if (results.queryFailed()) {
        edm::LogError("L1-O2O") << "Problem with L1GtTriggerMaskAlgoTrigRcd key:" << objectKey;
        return pL1GtTriggerMask;
    }

    // mask for other partitions than m_partitionNumber set to 1 (algorithm masked)
    int maskSize = columns.size() - 1; // table ID is also in columns
    std::vector<bool> trigMaskBool(maskSize, false);
    std::vector<unsigned int> trigMask(maskSize, 0);

    for (int i = 0; i < maskSize; i++) {
        bool tMask = trigMaskBool[i];
        results.fillVariable(columns[i + 1], tMask);
        trigMask[i] = 0xFF & (~( static_cast<unsigned int> (tMask) << m_partitionNumber ));
    }

    // fill the record
    pL1GtTriggerMask->setGtTriggerMask(trigMask);

    if (edm::isDebugEnabled()) {
        std::ostringstream myCoutStream;
        pL1GtTriggerMask->print(myCoutStream);
        LogTrace("L1GtTriggerMaskAlgoTrigConfigOnlineProd")
                << "\nThe following L1GtTriggerMaskAlgoTrigRcd record was read from OMDS: \n"
                << myCoutStream.str() << "\n" << std::endl;
    }

    return pL1GtTriggerMask;
}

DEFINE_FWK_EVENTSETUP_MODULE( L1GtTriggerMaskAlgoTrigConfigOnlineProd);
