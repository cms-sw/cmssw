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
  auto im = theRecordTypes.find(make_pair(fedId, type));
  return (im == theRecordTypes.end()) ? 0 : im->second;
}

int RPCRawDataCounts::fedFormatErrors(int fedId) const {
  for (const auto& theReadoutError : theReadoutErrors) {
    if (theReadoutError.first.first != fedId)
      continue;
    if (theReadoutError.first.second > ReadoutError::NoProblem &&
        theReadoutError.first.second <= ReadoutError::InconsistentDataSize)
      return 1;
  }
  return 0;
}

int RPCRawDataCounts::fedErrorRecords(int fedId) const {
  for (const auto& theRecordType : theRecordTypes) {
    if (theRecordType.first.first != fedId)
      continue;
    if (theRecordType.first.second > DataRecord::Empty)
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
  for (const auto& theRecordType : o.theRecordTypes) {
    theRecordTypes[make_pair(theRecordType.first.first, theRecordType.first.second)] += theRecordType.second;
  }

  for (const auto& theReadoutError : o.theReadoutErrors) {
    theReadoutErrors[make_pair(theReadoutError.first.first, theReadoutError.first.second)] += theReadoutError.second;
  }

  for (const auto& theGoodEvent : o.theGoodEvents) {
    theGoodEvents[make_pair(theGoodEvent.first.first, theGoodEvent.first.second)] += theGoodEvent.second;
  }

  for (const auto& theBadEvent : o.theBadEvents) {
    theBadEvents[make_pair(theBadEvent.first.first, theBadEvent.first.second)] += theBadEvent.second;
  }
}

std::string RPCRawDataCounts::print() const {
  std::ostringstream str;
  for (const auto& theRecordType : theRecordTypes) {
    str << "RECORD (" << theRecordType.first.first << "," << theRecordType.first.second << ")" << theRecordType.second;
  }
  for (const auto& theReadoutError : theReadoutErrors) {
    str << "ERROR(" << theReadoutError.first.first << "," << theReadoutError.first.second
        << ")=" << theReadoutError.second << endl;
  }
  return str.str();
}
