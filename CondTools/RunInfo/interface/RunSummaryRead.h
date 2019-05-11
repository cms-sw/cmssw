#ifndef RUNSUMMARYREAD_H
#define RUNSUMMARYREAD_H

#include "TestBase.h"
#include "CoralBase/TimeStamp.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"

class RunSummaryRead : virtual public TestBase {
public:
  RunSummaryRead(const std::string& connectionString, const std::string& user, const std::string& pass);

  ~RunSummaryRead() override;
  void run() override;

  RunSummary readData(const std::string& table, const std::string& column, const int r_number);

private:
  std::string m_tableToRead;
  std::string m_columnToRead;
  std::string m_connectionString;
  std::string m_user;
  std::string m_pass;
};

#endif
