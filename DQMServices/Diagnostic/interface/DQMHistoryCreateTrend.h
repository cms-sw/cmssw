#ifndef DQMHISTORYCREATETREND_H
#define DQMHISTORYCREATETREND_H

#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"
#include "DQMServices/Diagnostic/interface/DQMHistoryTrendsConfig.h"
#include "DQMServices/Diagnostic/interface/HDQMInspector.h"

#include <string>
#include <boost/shared_ptr.hpp>

using namespace std;

/**
 * Simple functor used to create trends.
 */
class DQMHistoryCreateTrend
{
 public:
  inline DQMHistoryCreateTrend(const HDQMInspectorConfigBase * HDQMconfig) { inspector_.reset(new HDQMInspector(HDQMconfig)); }
  /**
   * At this time it does the first-last OR the nRuns trends, NOT both. <br>
   * This is because we do not want to mix them for some misconfiguration.
   */
  void operator()(const DQMHistoryTrendsConfig & trend);
  inline void setDB(string dbName, string dbTag, string dbUser="", string dbPassword="", string dbBlob="") {
    inspector_->setDB(dbName, dbTag, dbUser, dbPassword, dbBlob);
  }
  inline void setDebug(const int i) {
    inspector_->setDebug(i);
  }
  inline void setDoStat(const int i) {
    inspector_->setDoStat(i);
  }
  inline void setBlackList(const string & listItems) {
    inspector_->setBlackList(listItems);
  }
  inline void closeFile() {
    inspector_->closeFile();
  }
  inline void setSkip99s(bool const in) {
    inspector_->setSkip99s(in);
  }
  inline void setSkip0s(bool const in) {
    inspector_->setSkip0s(in);
  }
  inline void setSeparator(std::string const in) {
    inspector_->setSeparator(in);
  }

 protected:
  // Do not use auto_ptr if you want to use the object with stl algorithms.
  boost::shared_ptr<HDQMInspector> inspector_;
};

#endif
