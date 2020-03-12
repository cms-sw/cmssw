//
//   \class ScaleRecordHelper
//
//   Description:  A helper class to extract L1 trigger scales from the database
//
//
//   Author :
//   Thomas Themel
//
//--------------------------------------------------
#ifndef L1ScalesProducers_ScaleRecordHelper_h
#define L1ScalesProducers_ScaleRecordHelper_h

// system include files
#include <memory>
#include <vector>
#include <string>

#include "CondTools/L1Trigger/interface/OMDSReader.h"

class ScaleRecordHelper {
public:
  ScaleRecordHelper(const std::string& binPrefix, unsigned int maxBin);

  void pushColumnNames(std::vector<std::string>& columns);

  void extractScales(l1t::OMDSReader::QueryResults& record, std::vector<double>& destScales);

protected:
  const std::string columnName(unsigned int bin);

private:
  std::string binPrefix_;
  unsigned int maxBin_;
};

#endif
