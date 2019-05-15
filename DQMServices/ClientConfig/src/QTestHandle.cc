/** \file
 *
 *  Implementation of  QTestHandle
 *
 *  \author Ilaria Segoni
 */

#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "DQMServices/ClientConfig/interface/QTestConfigurationParser.h"
#include "DQMServices/ClientConfig/interface/QTestConfigure.h"
#include "DQMServices/ClientConfig/interface/QTestStatusChecker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/scoped_ptr.hpp"
#include <atomic>

namespace {
  std::atomic<bool> firstTime{true};
}

QTestHandle::QTestHandle() {
  qtParser = new QTestConfigurationParser;
  qtConfigurer = new QTestConfigure;
  qtChecker = new QTestStatusChecker;

  testsConfigured = false;
}

QTestHandle::~QTestHandle() {
  delete qtParser;
  delete qtConfigurer;
  delete qtChecker;
}

bool QTestHandle::configureTests(const std::string &configFile, DQMStore *bei, bool UseDB) {
  //In case of UseDB==true the configFile is the content of the xml file itself
  //In case of UseDB==false (default) configFile is just the name of the file
  if (testsConfigured)
    qtParser->getNewDocument(configFile, UseDB);
  else {
    qtParser->getDocument(configFile, UseDB);
    testsConfigured = true;
  }

  if (!qtParser->parseQTestsConfiguration()) {
    std::map<std::string, std::map<std::string, std::string> > testsONList = qtParser->testsList();

    if (qtConfigurer->enableTests(testsONList, bei))
      return true;
  } else
    return true;

  return false;
}

void QTestHandle::attachTests(DQMStore *bei, bool verboseQT) {
  std::map<std::string, std::vector<std::string> > mapMeToTests = qtParser->meToTestsList();

  //If firstTime is true, then firstCaller will be true
  bool expected = true;
  const bool firstCaller = firstTime.compare_exchange_strong(expected, false);

  for (std::map<std::string, std::vector<std::string> >::iterator itr = mapMeToTests.begin(); itr != mapMeToTests.end();
       ++itr) {
    const std::string &meName = itr->first;
    const std::vector<std::string> &tests = itr->second;

    for (std::vector<std::string>::const_iterator testsItr = tests.begin(); testsItr != tests.end(); ++testsItr) {
      int cases = bei->useQTestByMatch(meName, *testsItr);
      if (firstCaller && verboseQT && cases == 0)
        edm::LogWarning("QTestHandle::attachTests")
            << " ==>> Invalid qtest xml: Link '" << meName << "', QTest '" << *testsItr << "'  - no matching ME! <<== ";
    }
  }
}

std::pair<std::string, std::string> QTestHandle::checkGlobalQTStatus(DQMStore *bei) const {
  return qtChecker->checkGlobalStatus(bei);
}

std::map<std::string, std::vector<std::string> > QTestHandle::checkDetailedQTStatus(DQMStore *bei) const {
  return qtChecker->checkDetailedStatus(bei);
}
