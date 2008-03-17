/*
 * \file QualityTester.cc
 *
 * $Date: 2008/02/21 03:26:49 $
 * $Revision: 1.10 $
 * \author M. Zanetti - CERN PH
 *
 */

#include "DQMServices/Components/interface/QualityTester.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <FWCore/ParameterSet/interface/FileInPath.h>
#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

QualityTester::QualityTester(const ParameterSet& ps)
{
  prescaleFactor = ps.getUntrackedParameter<int>("prescaleFactor", 1);
  getQualityTestsFromFile = ps.getUntrackedParameter<bool>("getQualityTestsFromFile", true);
  reportThreshold = ps.getUntrackedParameter<string>("reportThreshold", "");

  bei = &*edm::Service<DQMStore>();

  qtHandler=new QTestHandle;

  // if you use this module, it's non-sense not to provide the QualityTests.xml
  if (getQualityTestsFromFile) {
    edm::FileInPath qtlist = ps.getUntrackedParameter<edm::FileInPath>("qtList");
    qtHandler->configureTests(FileInPath(qtlist).fullPath(), bei);
  }
}


QualityTester::~QualityTester()
{
  delete qtHandler;
}

void
QualityTester::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context)
{
  if (getQualityTestsFromFile
      && lumiSeg.id().luminosityBlock() % prescaleFactor == 0)
  {
    // done here because new ME can appear while processing data
    qtHandler->attachTests(bei);

    edm::LogVerbatim ("QualityTester") << "Running the Quality Test";

    bei->runQTests();

    if (reportThreshold.size() != 0)
    {
      std::map< std::string, std::vector<std::string> > theAlarms
	= qtHandler->checkDetailedQTStatus(bei);

      for (std::map<std::string,std::vector<std::string> >::iterator itr = theAlarms.begin();
	   itr != theAlarms.end(); ++itr)
      {
        const std::string &alarmType = itr->first;
        const std::vector<std::string> &msgs = itr->second;
        if ((reportThreshold == "black")
	    || (reportThreshold == "orange" && (alarmType == "orange" || alarmType == "red"))
	    || (reportThreshold == "red" && alarmType == "red"))
	{
          std::cout << std::endl;
          std::cout << "Error Type: " << alarmType << std::endl;
          for (std::vector<std::string>::const_iterator msg = msgs.begin();
	       msg != msgs.end(); ++msg)
            std::cout << *msg << std::endl;
        }
      }
      std::cout << std::endl;
    }
  }
}
