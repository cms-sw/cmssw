#ifndef CondTools_RunInfo_RunInfoRead_h
#define CondTools_RunInfo_RunInfoRead_h

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include <string>

class RunInfoRead {
 public:
  RunInfoRead(const std::string& connectionString);
  ~RunInfoRead();
  RunInfo readData(const std::string& table, const std::string& column, const int r_number);
 private:
  std::string m_tableToRead;
  std::string m_columnToRead;
  std::string m_connectionString;
};

#endif
