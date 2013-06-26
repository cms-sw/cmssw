#ifndef CondTools_RunInfo_RunInfoRead_h
#define CondTools_RunInfo_RunInfoRead_h

#include "CondTools/RunInfo/interface/TestBase.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include <string>

class RunInfoRead : virtual public TestBase {
 public:
  RunInfoRead(const std::string& connectionString,
	      const std::string& user,
	      const std::string& pass);
  virtual ~RunInfoRead();
  void run();
  RunInfo readData(const std::string& table, const std::string& column, const int r_number);
 private:
  std::string m_tableToRead;
  std::string m_columnToRead;
  std::string m_connectionString;
  std::string m_user;
  std::string m_pass;
};

#endif
