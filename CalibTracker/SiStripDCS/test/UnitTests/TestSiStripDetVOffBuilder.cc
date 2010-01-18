#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>

#include <vector>

// Make everything public to access all methods
#define protected public
#define private public
#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"
#undef protected
#undef private

std::vector<int> vectorDate(const int year, const int month,
			    const int day, const int hour,
			    const int minute, const int second,
			    const int microsecond)
{
  std::vector<int> timeVector;
  timeVector.push_back(year);
  timeVector.push_back(month);
  timeVector.push_back(day);
  timeVector.push_back(hour);
  timeVector.push_back(minute);
  timeVector.push_back(second);
  timeVector.push_back(microsecond);
  return timeVector;
}

class TestSiStripDetVOffBuilder : public CppUnit::TestFixture { 
public: 
  TestSiStripDetVOffBuilder() {}

  void setUp()
  {
    edm::ParameterSet pset;
    // Must set the string type explicitly or it will take it as bool
    pset.addParameter("onlineDB", std::string("onlineDBString"));
    pset.addParameter("authPath", std::string("authPathString"));
    pset.addParameter("queryType", std::string("STATUSCHANGE"));
    pset.addParameter("lastValueFile", std::string("lastValueFileString"));
    pset.addParameter("lastValueFromFile", false);
    pset.addParameter("debugModeOn", false);
    pset.addParameter("Tmin", vectorDate(2009, 12, 7,  12,  0, 0, 000));
    pset.addParameter("Tmax", vectorDate(2009, 12, 8, 9, 0, 0, 000));
    pset.addParameter("TSetMin", vectorDate(2007, 11, 26, 0, 0, 0, 0));
    pset.addParameter("DetIdListFile", std::string("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
    pset.addParameter("HighVoltageOnThreshold", 0.97);

    object = new SiStripDetVOffBuilder(pset, edm::ActivityRegistry());
  }
  void tearDown()
  {
    delete object;
  }
  void testConstructor()
  {
    CPPUNIT_ASSERT( object->highVoltageOnThreshold_ == 0.97 );
    CPPUNIT_ASSERT( object->whichTable == "STATUSCHANGE" );
  }


  // Factorize the methods in SiStripDetVOffBuilder, do not call
  // coralInterface, but provide a list of modules built here.


  void testStatusChange()
  {
    object->coralInterface.reset( new SiStripCoralIface(object->onlineDbConnectionString,object->authenticationPath) );
    SiStripDetVOffBuilder::TimesAndValues tStruct;
    object->statusChange( object->lastStoredCondObj.second, tStruct );
    CPPUNIT_ASSERT(tStruct.actualStatus.size() != 0);
  }
  void testBuildDetVOffObj()
  {
    // CPPUNIT_ASSERT( object->BuildDetVOffObj() );
  }

  // Declare and build the test suite
  CPPUNIT_TEST_SUITE( TestSiStripDetVOffBuilder );
  CPPUNIT_TEST( testConstructor );
  CPPUNIT_TEST( testStatusChange );
  CPPUNIT_TEST_SUITE_END();

  SiStripDetVOffBuilder * object;
};

// Register the test suite in the registry.
// This way we will have to only pass the registry to the runner
// and it will contain all the registered test suites.
CPPUNIT_TEST_SUITE_REGISTRATION( TestSiStripDetVOffBuilder );
