#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestRunner.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TextTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>

#include <vector>

#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"

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

    pset.addParameter("debugModeOn", true);

    pset.addParameter("Tmin", vectorDate(2009, 12, 7,  12,  0, 0, 000));
    pset.addParameter("Tmax", vectorDate(2009, 12, 8, 9, 0, 0, 000));
    pset.addParameter("TSetMin", vectorDate(2007, 11, 26, 0, 0, 0, 0));
    pset.addParameter("DetIdListFile", std::string("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"));
    pset.addParameter("ExcludedDetIdListFile", std::string(""));
    pset.addParameter("HighVoltageOnThreshold", 0.97);
    pset.addParameter("PsuDetIdMapFile", std::string("CalibTracker/SiStripDCS/data/PsuDetIdMap.dat"));

    object = new SiStripDetVOffBuilder(pset, edm::ActivityRegistry());
    detVoff = new SiStripDetVOff;
  }

  void tearDown()
  {
    delete object;
    delete detVoff;
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
    object->coralInterface.reset( new SiStripCoralIface(object->onlineDbConnectionString,object->authenticationPath,false) );
    SiStripDetVOffBuilder::TimesAndValues tStruct;
    object->statusChange( object->lastStoredCondObj.second, tStruct );
    CPPUNIT_ASSERT(tStruct.actualStatus.size() != 0);
  }
  void testBuildDetVOffObj()
  {
    // CPPUNIT_ASSERT( object->BuildDetVOffObj() );
  }

  void testReduction()
  {
    fillModulesOff();

    cout << "number of IOVs before reduction = " << object->modulesOff.size() << endl;

    object->reduction(1, 1000);

    cout << "number of IOVs after reduction = " << object->modulesOff.size() << endl;
    vector<pair<SiStripDetVOff *, cond::Time_t> >::const_iterator iovContent = object->modulesOff.begin();
    for( ; iovContent != object->modulesOff.end(); ++iovContent ) {
      coral::TimeStamp coralTime(object->getCoralTime(iovContent->second));
      cout << "iov seconds = " << coralTime.second() << ", nanoseconds = " << coralTime.nanosecond();
      cout << ", number of modules with HV off = " << iovContent->first->getHVoffCounts() << endl;
      cout << ", number of modules with LV off = " << iovContent->first->getLVoffCounts() << endl;
    }

    CPPUNIT_ASSERT(object->modulesOff.size() == 5);
  }

  void fillModulesOff()
  {
    // Initialization: all off
    fillModule( 1,  1, 1,   0, 0 );
    fillModule( 2,  1, 1,   0, 1000 );
    fillModule( 3,  1, 1,   0, 2000 );

    // Ramping up phase: LV going on
    fillModule( 1,  1, 0,   5, 0 );
    fillModule( 2,  1, 0,   5, 1000 );
    fillModule( 3,  1, 0,   5, 2000 );
    // HV going on
    fillModule( 1,  0, 0,  10, 0 );
    fillModule( 2,  0, 0,  10, 1000 );
    fillModule( 3,  0, 0,  10, 2000 );

    // Wait some time, then switch off HV
    fillModule( 1,  1, 0,  15, 0 );
    fillModule( 2,  1, 0,  15, 1000 );
    fillModule( 3,  1, 0,  15, 2000 );
    // LV off
    fillModule( 1,  1, 1,  20, 0 );
    fillModule( 2,  1, 1,  20, 1000 );
    fillModule( 3,  1, 1,  20, 2000 );

    // fillModule( 3,  1, 0,  25, 0 );
  }

  void fillModule(const unsigned int detId, const unsigned int HVoff, const unsigned int LVoff, const unsigned int seconds, const unsigned int nanoseconds)
  {
    // Build the cond time from the 
    cond::Time_t condTime = object->getCondTime( coral::TimeStamp(2009, 12, 1, 1, 0, seconds, nanoseconds) );
    // Carefull, there is a memory leakage here. Fine if the program is kept simple.
    detVoff->put(detId, HVoff, LVoff);
    SiStripDetVOff * localDetVoff = new SiStripDetVOff(*detVoff);
    object->modulesOff.push_back( make_pair(localDetVoff, condTime) );
  }


  // Declare and build the test suite
  CPPUNIT_TEST_SUITE( TestSiStripDetVOffBuilder );
  CPPUNIT_TEST( testConstructor );
  // CPPUNIT_TEST( testStatusChange );
  CPPUNIT_TEST( testReduction );
  CPPUNIT_TEST_SUITE_END();


  SiStripDetVOffBuilder * object;
  SiStripDetVOff * detVoff;
};

// Register the test suite in the registry.
// This way we will have to only pass the registry to the runner
// and it will contain all the registered test suites.
CPPUNIT_TEST_SUITE_REGISTRATION( TestSiStripDetVOffBuilder );
