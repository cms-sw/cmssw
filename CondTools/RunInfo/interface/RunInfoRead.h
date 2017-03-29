#ifndef CondTools_RunInfo_RunInfoRead_h
#define CondTools_RunInfo_RunInfoRead_h

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

class RunInfoRead {
 public:
  RunInfoRead(const std::string& connectionString,
	      const edm::ParameterSet& connectionPset);
  ~RunInfoRead();
  RunInfo readData(const std::string& runinfo_schema, const std::string& dcsenv_schema, const int r_number);
 private:
  std::string m_connectionString;
  edm::ParameterSet m_connectionPset;
};

#endif
