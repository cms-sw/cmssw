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
L1GtTriggerMaskTechTrigConfigOnlineProd::L1GtTriggerMaskTechTrigConfigOnlineProd(const edm::ParameterSet& parSet)
    : L1ConfigOnlineProdBase<L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask>(parSet),
      m_partitionNumber(parSet.getParameter<int>("PartitionNumber")) {
  // empty
}

// destructor
L1GtTriggerMaskTechTrigConfigOnlineProd::~L1GtTriggerMaskTechTrigConfigOnlineProd() {
  // empty
}

// public methods

std::unique_ptr<L1GtTriggerMask> L1GtTriggerMaskTechTrigConfigOnlineProd::newObject(const std::string& objectKey) {
  auto pL1GtTriggerMask = std::make_unique<L1GtTriggerMask>();

  // l1GtTriggerMaskTechTrig: FINOR_TT_FK key in GT_PARTITION_FINOR_TT

  const std::string gtSchema = "CMS_GT";

  // SQL query:
  //
  // select * from CMS_GT.GT_PARTITION_FINOR_TT WHERE GT_PARTITION_FINOR_TT.ID = objectKey

  const std::vector<std::string>& columns = m_omdsReader.columnNames(gtSchema, "GT_PARTITION_FINOR_TT");

  if (edm::isDebugEnabled()) {
    for (std::vector<std::string>::const_iterator iter = columns.begin(); iter != columns.end(); iter++) {
      LogTrace("L1GtTriggerMaskTechTrigConfigOnlineProd") << (*iter) << std::endl;
    }
  }

  l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
      columns, gtSchema, "GT_PARTITION_FINOR_TT", "GT_PARTITION_FINOR_TT.ID", m_omdsReader.singleAttribute(objectKey));

  // check if query was successful
  if (results.queryFailed()) {
    edm::LogError("L1-O2O") << "Problem with L1GtTriggerMaskTechTrigRcd key:" << objectKey;
    return pL1GtTriggerMask;
  }

  // mask for other partitions than m_partitionNumber set to 1 (algorithm masked)
  int maskSize = columns.size() - 1;  // table ID is also in columns
  std::vector<bool> trigMaskBool(maskSize, false);
  std::vector<unsigned int> trigMask(maskSize, 0);

  for (int i = 0; i < maskSize; i++) {
    bool tMask = trigMaskBool[i];
    results.fillVariable(columns[i + 1], tMask);
    trigMask[i] = 0xFF & (~(static_cast<unsigned int>(tMask) << m_partitionNumber));
  }

  // fill the record
  pL1GtTriggerMask->setGtTriggerMask(trigMask);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    pL1GtTriggerMask->print(myCoutStream);
    LogTrace("L1GtTriggerMaskTechTrigConfigOnlineProd")
        << "\nThe following L1GtTriggerMaskTechTrigRcd record was read from OMDS: \n"
        << myCoutStream.str() << "\n"
        << std::endl;
  }

  return pL1GtTriggerMask;
}

DEFINE_FWK_EVENTSETUP_MODULE(L1GtTriggerMaskTechTrigConfigOnlineProd);
