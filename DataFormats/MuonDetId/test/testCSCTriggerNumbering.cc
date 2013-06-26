/**
   \file
   test file for CSCTriggerNumbering

   \author Lindsey GRAY
   \version $Id: testCSCTriggerNumbering.cc,v 1.3 2006/02/22 23:34:02 lgray Exp $
   \date 27 Jul 2005
*/

static const char CVSId[] = "$Id: testCSCTriggerNumbering.cc,v 1.3 2006/02/22 23:34:02 lgray Exp $";

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <iostream>
using namespace std;

class testCSCTriggerNumbering: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testCSCTriggerNumbering);

  CPPUNIT_TEST(testNumbering);
  CPPUNIT_TEST(testFail);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}
  void testNumbering();
  void testFail();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCTriggerNumbering);

// run a self consistency check
void testCSCTriggerNumbering::testNumbering()
{
  for (int endcap = CSCDetId::minEndcapId();
       endcap <= CSCDetId::maxEndcapId(); ++endcap)
    for(int station = CSCDetId::minStationId();
	station <= CSCDetId::maxStationId(); ++station)
      for(int sector = CSCTriggerNumbering::minTriggerSectorId();
	  sector <= CSCTriggerNumbering::maxTriggerSectorId(); ++sector)
	for(int cscid = CSCTriggerNumbering::minTriggerCscId();
	    cscid <= CSCTriggerNumbering::maxTriggerCscId(); ++cscid)
	  {
	    if(station == 1)
	      for(int subs = CSCTriggerNumbering::minTriggerSubSectorId();
		  subs <= CSCTriggerNumbering::maxTriggerSubSectorId(); subs++)
		{
		  int tcscid = CSCTriggerNumbering::triggerCscIdFromLabels(station, 
									   CSCTriggerNumbering::ringFromTriggerLabels(station,cscid),
									   CSCTriggerNumbering::chamberFromTriggerLabels(sector,subs,station,cscid));
		  CPPUNIT_ASSERT(tcscid == cscid);

		  int tsubs = CSCTriggerNumbering::triggerSubSectorFromLabels(station,
									      CSCTriggerNumbering::chamberFromTriggerLabels(sector,subs,station,cscid));

		  CPPUNIT_ASSERT(tsubs == subs);	  

		  int tsector = CSCTriggerNumbering::triggerSectorFromLabels(station, 
									     CSCTriggerNumbering::ringFromTriggerLabels(station,cscid),
									     CSCTriggerNumbering::chamberFromTriggerLabels(sector,subs,station,cscid));
		  CPPUNIT_ASSERT(tsector == sector);		  	  
		}
	    else
	      {
		int tcscid = CSCTriggerNumbering::triggerCscIdFromLabels(station, 
									 CSCTriggerNumbering::ringFromTriggerLabels(station,cscid),
									 CSCTriggerNumbering::chamberFromTriggerLabels(sector,0,station,cscid));
		CPPUNIT_ASSERT(tcscid == cscid);

		int tsector = CSCTriggerNumbering::triggerSectorFromLabels(station, 
									   CSCTriggerNumbering::ringFromTriggerLabels(station,cscid),
									   CSCTriggerNumbering::chamberFromTriggerLabels(sector,0,station,cscid));
		CPPUNIT_ASSERT(tsector == sector);		
	      }	    
	  }
}

void testCSCTriggerNumbering::testFail()
{
  // Give triggerSectorFromLabels a bad input
  try 
    {
      CSCTriggerNumbering::triggerSectorFromLabels(0,1,2);
      CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    } 
  catch (cms::Exception& e) 
    { } 
  catch (...) 
    {
      CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
    }

  // Give triggerSubSectorFromLabels a bad input

  try
    {
      CSCTriggerNumbering::triggerSubSectorFromLabels(0, 34);
      CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    }
  catch(cms::Exception& e)
    { }
  catch(...)
    {
      CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
    }

  // Give triggerCscIdFromLabels a bad input

  try 
    {
      CSCTriggerNumbering::triggerCscIdFromLabels(0,1,2);
      CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    } 
  catch (cms::Exception& e) 
    { } 
  catch (...) 
    {
      CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
    }

  // Give ringFromTriggerLabels a bad input
  try
    {
      CSCTriggerNumbering::ringFromTriggerLabels(0,2);
      CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    }
  catch (cms::Exception& e)
    { }
  catch(...)
    {
      CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
    }

  // Give ringFromTriggerLabels a bad input
  try
    {
      CSCTriggerNumbering::chamberFromTriggerLabels(0,1,2,9);
      CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    }
  catch (cms::Exception& e)
    { }
  catch(...)
    {
      CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
    } 
}
