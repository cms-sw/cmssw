/** \file
 *
 *  Implementation of  QTestHandle
 *
 *  $Date: 2008/04/14 14:40:38 $
 *  $Revision: 1.9.2.3 $
 *  \author Ilaria Segoni
 */


#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "DQMServices/ClientConfig/interface/QTestConfigurationParser.h"
#include "DQMServices/ClientConfig/interface/QTestConfigure.h"
#include "DQMServices/ClientConfig/interface/QTestStatusChecker.h"

#include "DQMServices/Core/interface/DQMStore.h"

QTestHandle::QTestHandle()
{
  qtParser     = new QTestConfigurationParser;
  qtConfigurer = new QTestConfigure;
  qtChecker    = new QTestStatusChecker;

  testsConfigured = false;
}

QTestHandle::~QTestHandle()
{
  delete qtParser;
  delete qtConfigurer;
  delete qtChecker;
}

bool QTestHandle::configureTests(const std::string &configFile, DQMStore *bei)
{
  if (testsConfigured)
    qtParser->getNewDocument(configFile);
  else
  {
    qtParser->getDocument(configFile);
    testsConfigured=true;
  }

  if (! qtParser->parseQTestsConfiguration())
  {
    std::map<std::string, std::map<std::string, std::string> > testsONList
      = qtParser->testsList();

    if (qtConfigurer->enableTests(testsONList,bei)) 
      return true;
  }
  else
    return true;

  return false;
}

void QTestHandle::attachTests(DQMStore *bei)
{
  std::map<std::string, std::vector<std::string> > mapMeToTests
    = qtParser->meToTestsList();

  for (std::map<std::string, std::vector<std::string> >::iterator itr = mapMeToTests.begin();
       itr != mapMeToTests.end();
       ++itr)
  {
    const std::string &meName = itr->first;
    const std::vector<std::string> &tests = itr->second;
    for (std::vector<std::string>::const_iterator testsItr = tests.begin();
	 testsItr != tests.end(); ++testsItr)
      bei->useQTestByMatch(meName, *testsItr);
  }
}

std::pair<std::string,std::string>
QTestHandle::checkGlobalQTStatus(DQMStore *bei) const
{
  return qtChecker->checkGlobalStatus(bei);
}

std::map< std::string, std::vector<std::string> >
QTestHandle::checkDetailedQTStatus(DQMStore *bei) const
{
  return qtChecker->checkDetailedStatus(bei);
}
