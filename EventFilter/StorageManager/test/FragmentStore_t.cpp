#include <iostream>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"

#include "EventFilter/StorageManager/test/TestHelper.h"


/////////////////////////////////////////////////////////////
//
// This test exercises the FragmentStore class
//
/////////////////////////////////////////////////////////////

using namespace stor;

using stor::testhelper::outstanding_bytes;
using stor::testhelper::allocate_frame_with_basic_header;

class testFragmentStore : public CppUnit::TestFixture
{
  typedef toolbox::mem::Reference Reference;
  CPPUNIT_TEST_SUITE(testFragmentStore);
  CPPUNIT_TEST(testSingleFragment);
  CPPUNIT_TEST(testMultipleFragments);
  CPPUNIT_TEST(testConcurrentFragments);
  CPPUNIT_TEST(testStaleEvent);
  CPPUNIT_TEST(testSingleIncompleteEvent);
  CPPUNIT_TEST(testOneOfManyIncompleteEvent);
  CPPUNIT_TEST(testClear);
  CPPUNIT_TEST(testFull);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void testSingleFragment();
  void testMultipleFragments();
  void testConcurrentFragments();
  void testStaleEvent();
  void testSingleIncompleteEvent();
  void testOneOfManyIncompleteEvent();
  void testClear();
  void testFull();

private:
  unsigned int getEventNumber
  (
    const unsigned int eventCount
  );

  stor::I2OChain getFragment
  (
    const unsigned int eventCount,
    const unsigned int fragmentCount,
    const unsigned int totalFragments
  );

  boost::scoped_ptr<FragmentStore> fragmentStore_;

};

void testFragmentStore::setUp()
{
  fragmentStore_.reset(new FragmentStore(1));
}

unsigned int testFragmentStore::getEventNumber
(
  const unsigned int eventCount
)
{
  unsigned int eventNumber = 0xb4b4e1e1;
  eventNumber <<= eventCount;
  return eventNumber;
}

stor::I2OChain testFragmentStore::getFragment
(
  const unsigned int eventCount,
  const unsigned int fragmentCount,
  const unsigned int totalFragments
)
{
  Reference* ref = 
    allocate_frame_with_basic_header(I2O_SM_DATA, fragmentCount, totalFragments);
  I2O_SM_DATA_MESSAGE_FRAME *i2oMsg =
    (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
  i2oMsg->rbBufferID = 2;
  i2oMsg->runID = 0xa5a5d2d2;
  i2oMsg->eventID = getEventNumber(eventCount);
  i2oMsg->outModID = 0xc3c3f0f0;
  i2oMsg->fuProcID = 0x01234567;
  i2oMsg->fuGUID = 0x89abcdef;
  stor::I2OChain frag(ref);
  
  return frag;
}

void testFragmentStore::testSingleFragment()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    stor::I2OChain frag = getFragment(1,0,1);
    CPPUNIT_ASSERT(frag.complete());
    bool complete = fragmentStore_->addFragment(frag);
    CPPUNIT_ASSERT(complete);
    CPPUNIT_ASSERT(frag.complete());
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void testFragmentStore::testMultipleFragments()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);

  const unsigned int totalFragments = 3;

  for (unsigned int i = 0; i < totalFragments; ++i)
  {
    stor::I2OChain frag = getFragment(1,i,totalFragments);
    CPPUNIT_ASSERT(!frag.complete());
    bool complete = fragmentStore_->addFragment(frag);
    if ( i < totalFragments-1 )
    {
      CPPUNIT_ASSERT(!complete);
      CPPUNIT_ASSERT(frag.empty());
    }
    else
    {
      CPPUNIT_ASSERT(complete);
      CPPUNIT_ASSERT(!frag.empty());
      CPPUNIT_ASSERT(frag.complete());
      CPPUNIT_ASSERT(!frag.faulty());
      CPPUNIT_ASSERT(frag.fragmentCount() == totalFragments);
    }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}


void testFragmentStore::testConcurrentFragments()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);

  const unsigned int totalFragments = 5;
  const unsigned int totalEvents = 15;

  for (unsigned int fragNum = 0; fragNum < totalFragments; ++fragNum)
  {
    for (unsigned int eventNum = 0; eventNum < totalEvents; ++eventNum)
    {
      stor::I2OChain frag = getFragment(eventNum,fragNum,totalFragments);
      CPPUNIT_ASSERT(!frag.complete());
      bool complete = fragmentStore_->addFragment(frag);
      if ( fragNum < totalFragments-1 )
      {
        CPPUNIT_ASSERT(!complete);
        CPPUNIT_ASSERT(frag.empty());
      }
      else
      {
        CPPUNIT_ASSERT(complete);
        CPPUNIT_ASSERT(frag.complete());
        CPPUNIT_ASSERT(!frag.faulty());
        CPPUNIT_ASSERT(frag.fragmentCount() == totalFragments);

        stor::FragKey fragmentKey = frag.fragmentKey();
        CPPUNIT_ASSERT(fragmentKey.event_ == getEventNumber(eventNum));
      }
    }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void testFragmentStore::testStaleEvent()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    stor::I2OChain frag = getFragment(1,0,1);
    CPPUNIT_ASSERT(frag.complete());
    bool hasStaleEvent = fragmentStore_->getStaleEvent(frag, boost::posix_time::seconds(0));
    CPPUNIT_ASSERT(hasStaleEvent == false);
    CPPUNIT_ASSERT(frag.empty());
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void testFragmentStore::testSingleIncompleteEvent()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    const unsigned int totalFragments = 5;
    stor::I2OChain frag;
    bool complete;
    
    for (unsigned int i = 0; i < totalFragments-1; ++i)
    {
      frag = getFragment(1,i,totalFragments);
      CPPUNIT_ASSERT(!frag.complete());
      complete = fragmentStore_->addFragment(frag);
    }
    CPPUNIT_ASSERT(frag.empty());
    
    CPPUNIT_ASSERT(fragmentStore_->getStaleEvent(frag, boost::posix_time::seconds(0)) == true);
    CPPUNIT_ASSERT(!frag.complete());
    CPPUNIT_ASSERT(frag.faulty());
    CPPUNIT_ASSERT(frag.fragmentCount() == totalFragments-1);
    
    stor::FragKey fragmentKey = frag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.event_ == getEventNumber(1));
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}


void testFragmentStore::testOneOfManyIncompleteEvent()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    const unsigned int totalFragments = 5;
    const unsigned int totalEvents = 15;
    stor::I2OChain frag;
    bool complete;
    
    // Fill one event with a missing fragment
    for (unsigned int fragNum = 0; fragNum < totalFragments-1; ++fragNum)
    {
      frag = getFragment(0,fragNum,totalFragments);
      CPPUNIT_ASSERT(!frag.complete());
      complete = fragmentStore_->addFragment(frag);
    }
    CPPUNIT_ASSERT(!complete);
    CPPUNIT_ASSERT(frag.empty());
    
    // Sleep for a second
    sleep(1);
    
    // Fill more events with missing fragmentes
    for (unsigned int eventNum = 1; eventNum < totalEvents; ++eventNum)
    {
      for (unsigned int fragNum = 1; fragNum < totalFragments; ++fragNum)
      {
        frag = getFragment(eventNum,fragNum,totalFragments);
        CPPUNIT_ASSERT(!frag.complete());
        complete = fragmentStore_->addFragment(frag);
      }
      CPPUNIT_ASSERT(!complete);
      CPPUNIT_ASSERT(frag.empty());
    }

    // Get the first event which should be stale by now
    CPPUNIT_ASSERT(fragmentStore_->getStaleEvent(frag, boost::posix_time::seconds(1)) == true);
    CPPUNIT_ASSERT(!frag.complete());
    CPPUNIT_ASSERT(frag.faulty());
    CPPUNIT_ASSERT(frag.fragmentCount() == totalFragments-1);
    
    stor::FragKey fragmentKey = frag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.event_ == getEventNumber(0));
    
    // Finish the other events
    for (unsigned int eventNum = 1; eventNum < totalEvents; ++eventNum)
    {
      frag = getFragment(eventNum,0,totalFragments);
      CPPUNIT_ASSERT(!frag.complete());
      complete = fragmentStore_->addFragment(frag);
      CPPUNIT_ASSERT(complete);
      CPPUNIT_ASSERT(frag.complete());
      CPPUNIT_ASSERT(!frag.faulty());
      CPPUNIT_ASSERT(frag.fragmentCount() == totalFragments);
    
      stor::FragKey fragmentKey = frag.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.event_ == getEventNumber(eventNum));
    }
    CPPUNIT_ASSERT(fragmentStore_->getStaleEvent(frag, boost::posix_time::seconds(0)) == false);
    CPPUNIT_ASSERT(frag.empty());
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}


void testFragmentStore::testClear()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    const unsigned int totalFragments = 5;
    const unsigned int totalEvents = 15;
    stor::I2OChain frag;
    bool complete;
    
    for (unsigned int eventNum = 0; eventNum < totalEvents; ++eventNum)
    {
      for (unsigned int fragNum = 0; fragNum < totalFragments-1; ++fragNum)
      {
        frag = getFragment(eventNum,fragNum,totalFragments);
        CPPUNIT_ASSERT(!frag.complete());
        complete = fragmentStore_->addFragment(frag);
      }
      CPPUNIT_ASSERT(!complete);
      CPPUNIT_ASSERT(frag.empty());
    }
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
    fragmentStore_->clear();
    CPPUNIT_ASSERT(outstanding_bytes() == 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}


void testFragmentStore::testFull()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    stor::I2OChain frag = getFragment(1,0,1);
    CPPUNIT_ASSERT(fragmentStore_->addFragment(frag) == true);
    const size_t fragSize = frag.memoryUsed();
    const unsigned int maxFragCount = 1024*1024 / fragSize;
    for (unsigned int i = 0; i < maxFragCount-1; ++i)
    {
      frag = getFragment(2,i,maxFragCount);
      CPPUNIT_ASSERT(fragmentStore_->addFragment(frag) == false);
      CPPUNIT_ASSERT(!fragmentStore_->full());
    }
    frag = getFragment(3,0,2);
    CPPUNIT_ASSERT(fragmentStore_->addFragment(frag) == false);
    CPPUNIT_ASSERT(fragmentStore_->full());
    frag = getFragment(3,1,2);
    CPPUNIT_ASSERT(fragmentStore_->addFragment(frag) == true);
    CPPUNIT_ASSERT(!fragmentStore_->full());
    frag = getFragment(2,maxFragCount-1,maxFragCount);
    CPPUNIT_ASSERT(fragmentStore_->addFragment(frag) == true);
    CPPUNIT_ASSERT(fragmentStore_->empty());
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testFragmentStore);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
