#ifndef L1TRIGGERSCALEREAD_H
#define L1TRIGGERSCALEREAD_H

#include "TestBase.h"
#include "CoralBase/TimeStamp.h"
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"

class L1TriggerScalerRead : virtual public TestBase {
public:
  L1TriggerScalerRead(const std::string& connectionString, const std::string& user, const std::string& pass);

  ~L1TriggerScalerRead() override;
  void run() override;

  void dropTable(const std::string& table);

  std::vector<L1TriggerScaler::Lumi> readData(const int r_number);

private:
  std::string m_tableToDrop;
  //  std::string m_tableToRead;
  // std::string m_columnToRead;
  std::string m_connectionString;
  std::string m_user;
  std::string m_pass;
};

#endif
