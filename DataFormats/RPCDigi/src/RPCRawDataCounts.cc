#include "DataFormats/RPCDigi/interface/RPCRawDataCounts.h"
#include "DataFormats/RPCDigi/interface/ReadoutError.h"
#include "DataFormats/RPCDigi/interface/DataRecord.h"
#include "DataFormats/RPCDigi/interface/RecordSLD.h"
#include "DataFormats/RPCDigi/interface/ErrorRDDM.h"
#include "DataFormats/RPCDigi/interface/ErrorRDM.h"
#include "DataFormats/RPCDigi/interface/ErrorRCDM.h"
#include "DataFormats/RPCDigi/interface/ReadoutError.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <sstream>

using namespace rpcrawtodigi;
using namespace std;

typedef std::map<std::pair<int, int>, int>::const_iterator IT;

int RPCRawDataCounts::fedBxRecords(int fedId) const {
  int type = DataRecord::StartOfBXData;
  IT im = theRecordTypes.find(make_pair(fedId, type));
  return (im == theRecordTypes.end()) ? 0 : im->second;
}

int RPCRawDataCounts::fedFormatErrors(int fedId) const {
  for (IT im = theReadoutErrors.begin(); im != theReadoutErrors.end(); ++im) {
    if (im->first.first != fedId)
      continue;
    if (im->first.second > ReadoutError::NoProblem && im->first.second <= ReadoutError::InconsistentDataSize)
      return 1;
  }
  return 0;
}

int RPCRawDataCounts::fedErrorRecords(int fedId) const {
  for (IT im = theRecordTypes.begin(); im != theRecordTypes.end(); ++im) {
    if (im->first.first != fedId)
      continue;
    if (im->first.second > DataRecord::Empty)
      return 1;
  }
  return 0;
}

void RPCRawDataCounts::addDccRecord(int fed, const rpcrawtodigi::DataRecord& record, int weight) {
  DataRecord::DataRecordType type = record.type();
  switch (type) {
    case (DataRecord::StartOfTbLinkInputNumberData): {
      theGoodEvents[make_pair(fed, RecordSLD(record).rmb())] += weight;
      break;
    }
    case (DataRecord::RDDM): {
      theBadEvents[make_pair(fed, ErrorRDDM(record).rmb())] += weight;
      break;
    }
    case (DataRecord::RDM): {
      theBadEvents[make_pair(fed, ErrorRDM(record).rmb())] += weight;
      break;
    }
    case (DataRecord::RCDM): {
      theBadEvents[make_pair(fed, ErrorRCDM(record).rmb())] += weight;
      break;
    }
    default: {
    }
  }

  theRecordTypes[make_pair(fed, type)] += weight;
}

void RPCRawDataCounts::addReadoutError(int fed, const rpcrawtodigi::ReadoutError& e, int weight) {
  theReadoutErrors[make_pair(fed, e.rawCode())] += weight;
}

void RPCRawDataCounts::operator+=(const RPCRawDataCounts& o) {
  for (IT irt = o.theRecordTypes.begin(); irt != o.theRecordTypes.end(); ++irt) {
    theRecordTypes[make_pair(irt->first.first, irt->first.second)] += irt->second;
  }

  for (IT ire = o.theReadoutErrors.begin(); ire != o.theReadoutErrors.end(); ++ire) {
    theReadoutErrors[make_pair(ire->first.first, ire->first.second)] += ire->second;
  }

  for (IT ire = o.theGoodEvents.begin(); ire != o.theGoodEvents.end(); ++ire) {
    theGoodEvents[make_pair(ire->first.first, ire->first.second)] += ire->second;
  }

  for (IT ire = o.theBadEvents.begin(); ire != o.theBadEvents.end(); ++ire) {
    theBadEvents[make_pair(ire->first.first, ire->first.second)] += ire->second;
  }
}

std::string RPCRawDataCounts::print() const {
  std::ostringstream str;
  for (IT irt = theRecordTypes.begin(); irt != theRecordTypes.end(); ++irt) {
    str << "RECORD (" << irt->first.first << "," << irt->first.second << ")" << irt->second;
  }
  for (IT ire = theReadoutErrors.begin(); ire != theReadoutErrors.end(); ++ire) {
    str << "ERROR(" << ire->first.first << "," << ire->first.second << ")=" << ire->second << endl;
  }
  return str.str();
}
