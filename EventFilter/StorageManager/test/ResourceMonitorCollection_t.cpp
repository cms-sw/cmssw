#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/test/MockAlarmHandler.h"

using namespace stor;

class stor::testResourceMonitorCollection : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testResourceMonitorCollection);

  CPPUNIT_TEST(sataBeasts);
  //CPPUNIT_TEST(sataBeastsOnSpecialNode);  // can only be used on specially prepared SATA beast host

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void sataBeasts();
  void sataBeastsOnSpecialNode();

private:

  boost::shared_ptr<MockAlarmHandler> _ah;
  ResourceMonitorCollection* _rmc;

};

void
testResourceMonitorCollection::setUp()
{
  _ah.reset(new MockAlarmHandler());
  _rmc = new ResourceMonitorCollection(1, _ah);
}

void
testResourceMonitorCollection::tearDown()
{
  delete _rmc;
}


void
testResourceMonitorCollection::sataBeasts()
{
  _rmc->checkSataBeasts();
  CPPUNIT_ASSERT( _rmc->_sataBeastStatus == 0 );
  CPPUNIT_ASSERT( _ah->noAlarmSet() );
}


void
testResourceMonitorCollection::sataBeastsOnSpecialNode()
{
  ResourceMonitorCollection::SATABeasts sataBeasts;
  bool foundSataBeasts =
    _rmc->getSataBeasts(sataBeasts);
  CPPUNIT_ASSERT( foundSataBeasts );
  CPPUNIT_ASSERT( sataBeasts.size() == 1 );
  std::string sataBeast = *(sataBeasts.begin());
  CPPUNIT_ASSERT( sataBeast == "satab-c2c07-06" );

  CPPUNIT_ASSERT(! _rmc->checkSataDisks(sataBeast,"-00.cms") );
  CPPUNIT_ASSERT( _rmc->checkSataDisks(sataBeast,"-10.cms") );

  CPPUNIT_ASSERT( _rmc->_sataBeastStatus == 101 );

  std::vector<MockAlarmHandler::Alarms> alarms;
  bool alarmsAreSet = _ah->getActiveAlarms(sataBeast, alarms);
  CPPUNIT_ASSERT( alarmsAreSet );

  std::cout << "\nActive alarms for " << sataBeast << std::endl;
  for (std::vector<MockAlarmHandler::Alarms>::iterator it = alarms.begin(),
         itEnd = alarms.end();
       it != itEnd;
       ++it)
  {
    std::cout << "   " << it->first << "\t" << it->second.message() << std::endl;
  }
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testResourceMonitorCollection);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
