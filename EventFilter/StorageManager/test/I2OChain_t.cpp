#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <assert.h>
#include <vector>
#include "zlib.h"
#include <cstring>

#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/StreamID.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "EventFilter/StorageManager/test/TestHelper.h"

#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"

#include "FWCore/Utilities/interface/Adler32Calculator.h"

using stor::testhelper::outstanding_bytes;
using stor::testhelper::allocate_frame;
using stor::testhelper::allocate_frame_with_basic_header;
using stor::testhelper::allocate_frame_with_sample_header;


class testI2OChain : public CppUnit::TestFixture
{
  typedef toolbox::mem::Reference Reference;
  CPPUNIT_TEST_SUITE(testI2OChain);
  // CPPUNIT_TEST(default_chain);
  CPPUNIT_TEST(null_reference);
  CPPUNIT_TEST(nonempty_chain_cleans_up_nice);
  CPPUNIT_TEST(copy_chain);
  CPPUNIT_TEST(assign_chain);
  CPPUNIT_TEST(swap_chain);
  CPPUNIT_TEST(copying_does_not_exhaust_buffer);
  CPPUNIT_TEST(release_chain);
  CPPUNIT_TEST(release_default_chain);
  CPPUNIT_TEST(invalid_fragment);
  CPPUNIT_TEST(populate_i2o_header);
  CPPUNIT_TEST(copy_with_valid_header);
  CPPUNIT_TEST(assign_with_valid_header);
  CPPUNIT_TEST(swap_with_valid_header);
  CPPUNIT_TEST(release_with_valid_header);
  CPPUNIT_TEST(add_fragment);
  CPPUNIT_TEST(chained_references);
  CPPUNIT_TEST(fragkey_mismatches);
  CPPUNIT_TEST(multipart_msg_header);
  CPPUNIT_TEST(init_msg_header);
  CPPUNIT_TEST(event_msg_header);
  CPPUNIT_TEST(error_event_msg_header);
  CPPUNIT_TEST(stream_and_queue_tags);
  CPPUNIT_TEST(split_init_header);
  CPPUNIT_TEST(split_event_header);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void default_chain();
  void null_reference();
  void nonempty_chain_cleans_up_nice();
  void copy_chain();
  void assign_chain();
  void swap_chain();
  void copying_does_not_exhaust_buffer();
  void release_chain();
  void release_default_chain();
  void invalid_fragment();
  void populate_i2o_header();
  void copy_with_valid_header();
  void assign_with_valid_header();
  void swap_with_valid_header();
  void release_with_valid_header();
  void add_fragment();
  void chained_references();
  void fragkey_mismatches();
  void multipart_msg_header();
  void init_msg_header();
  void event_msg_header();
  void error_event_msg_header();
  void stream_and_queue_tags();
  void split_init_header();
  void split_event_header();

private:

};

void
testI2OChain::setUp()
{ 
  CPPUNIT_ASSERT(g_factory);
  CPPUNIT_ASSERT(g_alloc);
  CPPUNIT_ASSERT(g_pool);
}

void
testI2OChain::tearDown()
{ 
}

void 
testI2OChain::default_chain()
{
  stor::I2OChain frag;
  CPPUNIT_ASSERT(frag.empty());
  CPPUNIT_ASSERT(!frag.complete());
  CPPUNIT_ASSERT(!frag.faulty());
  CPPUNIT_ASSERT(frag.messageCode() == Header::INVALID);
  CPPUNIT_ASSERT(frag.fragmentCount() == 0);
  CPPUNIT_ASSERT(frag.rbBufferId() == 0);
  CPPUNIT_ASSERT(frag.fuProcessId() == 0);
  CPPUNIT_ASSERT(frag.fuGuid() == 0);
  CPPUNIT_ASSERT(frag.creationTime() == boost::posix_time::not_a_date_time);
  CPPUNIT_ASSERT(frag.lastFragmentTime() == boost::posix_time::not_a_date_time);
  //CPPUNIT_ASSERT(!frag.getTotalDataSize() == 0);
  size_t memory_consumed_by_zero_frames = outstanding_bytes();
  CPPUNIT_ASSERT(memory_consumed_by_zero_frames == 0);  
}

void 
testI2OChain::null_reference()
{
  stor::I2OChain frag(0);
  CPPUNIT_ASSERT(frag.empty());
  CPPUNIT_ASSERT(!frag.complete());
  CPPUNIT_ASSERT(!frag.faulty());
  CPPUNIT_ASSERT(frag.messageCode() == Header::INVALID);
  CPPUNIT_ASSERT(frag.fragmentCount() == 0);
  CPPUNIT_ASSERT(frag.rbBufferId() == 0);
  CPPUNIT_ASSERT(frag.fuProcessId() == 0);
  CPPUNIT_ASSERT(frag.fuGuid() == 0);
  CPPUNIT_ASSERT(frag.creationTime() == boost::posix_time::not_a_date_time);
  CPPUNIT_ASSERT(frag.lastFragmentTime() == boost::posix_time::not_a_date_time);
  //CPPUNIT_ASSERT(!frag.getTotalDataSize() == 0);
  size_t memory_consumed_by_zero_frames = outstanding_bytes();
  CPPUNIT_ASSERT(memory_consumed_by_zero_frames == 0);  
}

void
testI2OChain::nonempty_chain_cleans_up_nice()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    stor::I2OChain frag(allocate_frame());
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
    CPPUNIT_ASSERT(!frag.empty());
    CPPUNIT_ASSERT(!frag.complete());
    //frag.markComplete();
    //CPPUNIT_ASSERT(frag.complete());
    CPPUNIT_ASSERT(frag.faulty());  // because the buffer is empty
    frag.markFaulty();
    CPPUNIT_ASSERT(frag.faulty());
    CPPUNIT_ASSERT(frag.messageCode() == Header::INVALID);
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
    CPPUNIT_ASSERT(frag.rbBufferId() == 0);
    CPPUNIT_ASSERT(frag.fuProcessId() == 0);
    CPPUNIT_ASSERT(frag.fuGuid() == 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}


void
testI2OChain::copy_chain()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    stor::I2OChain frag(allocate_frame());
    stor::utils::TimePoint_t creationTime = frag.creationTime();
    stor::utils::TimePoint_t lastFragmentTime = frag.lastFragmentTime();
    size_t memory_consumed_by_one_frame = outstanding_bytes();
    CPPUNIT_ASSERT(memory_consumed_by_one_frame != 0);
    {
      stor::I2OChain copy(frag);
      size_t memory_consumed_after_copy = outstanding_bytes();
      CPPUNIT_ASSERT(memory_consumed_after_copy != 0);
      CPPUNIT_ASSERT(memory_consumed_after_copy ==
                     memory_consumed_by_one_frame);
      CPPUNIT_ASSERT(copy.getBufferData() == frag.getBufferData());
      CPPUNIT_ASSERT(copy.creationTime() == creationTime); 
      CPPUNIT_ASSERT(copy.lastFragmentTime() == lastFragmentTime); 
    }
    // Here the copy is gone, but the original remains; we should not
    // have released the resources.
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::assign_chain()
{
  stor::I2OChain frag1(allocate_frame());
  size_t memory_consumed_by_one_frame = outstanding_bytes();
  CPPUNIT_ASSERT(memory_consumed_by_one_frame != 0);
  
  stor::I2OChain frag2(allocate_frame());
  size_t memory_consumed_by_two_frames = outstanding_bytes();
  CPPUNIT_ASSERT(memory_consumed_by_two_frames > memory_consumed_by_one_frame);

  stor::I2OChain no_frags;
  CPPUNIT_ASSERT(no_frags.empty());

  // Assigning to frag2 should release the resources associated with frag2.
  frag2 = no_frags;
  CPPUNIT_ASSERT(outstanding_bytes() == memory_consumed_by_one_frame);  
  CPPUNIT_ASSERT(frag2.empty());

  // Assigning frag1 to frag2 should consume no new resources
  frag2 = frag1;
  CPPUNIT_ASSERT(outstanding_bytes() == memory_consumed_by_one_frame);  
  CPPUNIT_ASSERT(!frag2.empty());
  CPPUNIT_ASSERT(frag2.getBufferData() == frag1.getBufferData());  
  CPPUNIT_ASSERT(frag2.creationTime() == frag1.creationTime());  
  CPPUNIT_ASSERT(frag2.lastFragmentTime() == frag1.lastFragmentTime());  

  // Assigning no_frags to frag1 and frag2 should release all resources.
  CPPUNIT_ASSERT(no_frags.empty());
  frag1 = no_frags;
  frag2 = no_frags;
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::swap_chain()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    stor::I2OChain frag(allocate_frame());
    size_t memory_consumed_by_one_frame = outstanding_bytes();
    CPPUNIT_ASSERT(memory_consumed_by_one_frame != 0);
    CPPUNIT_ASSERT(!frag.empty());
    unsigned long* data_location = frag.getBufferData();
    stor::utils::TimePoint_t creationTime = frag.creationTime();
    stor::utils::TimePoint_t lastFragmentTime = frag.lastFragmentTime();

    stor::I2OChain no_frags;
    CPPUNIT_ASSERT(no_frags.empty());
    CPPUNIT_ASSERT(outstanding_bytes() == memory_consumed_by_one_frame);
    
    // Swapping should not change the amount of allocated memory, but
    // it should reverse the roles: no_frags should be non-empty, and
    // frags should be empty.
    std::swap(no_frags, frag);
    CPPUNIT_ASSERT(outstanding_bytes() == memory_consumed_by_one_frame);
    CPPUNIT_ASSERT(frag.empty());
    CPPUNIT_ASSERT(!no_frags.empty());
    CPPUNIT_ASSERT(no_frags.getBufferData() == data_location);
    CPPUNIT_ASSERT(no_frags.creationTime() == creationTime);
    CPPUNIT_ASSERT(no_frags.lastFragmentTime() == lastFragmentTime);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}


void
testI2OChain::copying_does_not_exhaust_buffer()
{
  stor::I2OChain frag(allocate_frame());
  size_t memory_consumed_by_one_frame = outstanding_bytes();
  CPPUNIT_ASSERT(memory_consumed_by_one_frame != 0);

  // Now make many copies.
  size_t copies_to_make = 1000*1000UL;
  std::vector<stor::I2OChain> copies(copies_to_make, frag);
  CPPUNIT_ASSERT(copies.size() == copies_to_make);

  // Make sure we haven't consumed any more buffer space...
  CPPUNIT_ASSERT(outstanding_bytes() ==
                 memory_consumed_by_one_frame);

  // Make sure they all manage the identical buffer...  Using
  // std::find_if would be briefer, but more obscure to many
  // maintainers. If you disagree, please replace this look with the
  // appropriate predicate (using lambda) and use of find_if.
  unsigned long* expected_data_location = frag.getBufferData();
  for (std::vector<stor::I2OChain>::iterator 
         i = copies.begin(),
         e = copies.end();
       i != e;
       ++i)
    {
      CPPUNIT_ASSERT( i->getBufferData() == expected_data_location);
    }
         
  
  // Now destroy the copies.
  copies.clear();
  CPPUNIT_ASSERT(outstanding_bytes() ==
                 memory_consumed_by_one_frame);                 
}

void
testI2OChain::release_chain()
{
  CPPUNIT_ASSERT(outstanding_bytes() ==0);
  {
    stor::I2OChain frag(allocate_frame());
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
    frag.release();
    CPPUNIT_ASSERT(frag.empty());
    CPPUNIT_ASSERT(frag.getBufferData() == 0);
    CPPUNIT_ASSERT(frag.creationTime() == boost::posix_time::not_a_date_time);
    CPPUNIT_ASSERT(frag.lastFragmentTime() == boost::posix_time::not_a_date_time);
    CPPUNIT_ASSERT(outstanding_bytes() == 0);
    
  }
  CPPUNIT_ASSERT(outstanding_bytes() ==0);
}

void
testI2OChain::release_default_chain()
{
  CPPUNIT_ASSERT(outstanding_bytes() ==0);
  {
    stor::I2OChain empty;
    CPPUNIT_ASSERT(outstanding_bytes() == 0);
    empty.release();
    CPPUNIT_ASSERT(empty.empty());
    CPPUNIT_ASSERT(empty.getBufferData() == 0);
    CPPUNIT_ASSERT(outstanding_bytes() == 0);    
  }
  CPPUNIT_ASSERT(outstanding_bytes() ==0);
}

void 
testI2OChain::invalid_fragment()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    Reference* ref = allocate_frame();
    stor::I2OChain frag(ref);
    CPPUNIT_ASSERT(!frag.empty());
    CPPUNIT_ASSERT(!frag.complete());
    CPPUNIT_ASSERT(frag.faulty());
    CPPUNIT_ASSERT(frag.messageCode() == Header::INVALID);
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
    CPPUNIT_ASSERT(frag.rbBufferId() == 0);
    CPPUNIT_ASSERT(frag.fuProcessId() == 0);
    CPPUNIT_ASSERT(frag.fuGuid() == 0);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    Reference* ref = allocate_frame_with_basic_header(0xff, 0, 0);
    stor::I2OChain frag(ref);
    CPPUNIT_ASSERT(!frag.empty());
    CPPUNIT_ASSERT(!frag.complete());
    CPPUNIT_ASSERT(frag.faulty());
    CPPUNIT_ASSERT(frag.messageCode() == Header::INVALID);
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
    CPPUNIT_ASSERT(frag.rbBufferId() == 0);
    CPPUNIT_ASSERT(frag.fuProcessId() == 0);
    CPPUNIT_ASSERT(frag.fuGuid() == 0);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    Reference* ref = allocate_frame_with_basic_header(I2O_SM_ERROR, 0, 0);
    stor::I2OChain frag(ref);
    CPPUNIT_ASSERT(!frag.empty());
    CPPUNIT_ASSERT(!frag.complete());
    CPPUNIT_ASSERT(frag.faulty());
    CPPUNIT_ASSERT(frag.messageCode() == Header::ERROR_EVENT);
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
    CPPUNIT_ASSERT(frag.rbBufferId() == 0);
    CPPUNIT_ASSERT(frag.fuProcessId() == 0);
    CPPUNIT_ASSERT(frag.fuGuid() == 0);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  {
    Reference* ref = allocate_frame_with_basic_header(I2O_SM_ERROR, 3, 3);
    stor::I2OChain frag(ref);
    CPPUNIT_ASSERT(!frag.empty());
    CPPUNIT_ASSERT(!frag.complete());
    CPPUNIT_ASSERT(frag.faulty());
    CPPUNIT_ASSERT(frag.messageCode() == Header::ERROR_EVENT);
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
    CPPUNIT_ASSERT(frag.rbBufferId() == 0);
    CPPUNIT_ASSERT(frag.fuProcessId() == 0);
    CPPUNIT_ASSERT(frag.fuGuid() == 0);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    Reference* ref = allocate_frame_with_basic_header(I2O_SM_ERROR, 100, 2);
    stor::I2OChain frag(ref);
    CPPUNIT_ASSERT(!frag.empty());
    CPPUNIT_ASSERT(!frag.complete());
    CPPUNIT_ASSERT(frag.faulty());
    CPPUNIT_ASSERT(frag.messageCode() == Header::ERROR_EVENT);
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
    CPPUNIT_ASSERT(frag.rbBufferId() == 0);
    CPPUNIT_ASSERT(frag.fuProcessId() == 0);
    CPPUNIT_ASSERT(frag.fuGuid() == 0);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    Reference* ref = allocate_frame_with_basic_header(I2O_SM_ERROR, 0, 1);
    stor::I2OChain frag(ref);
    CPPUNIT_ASSERT(!frag.empty());
    CPPUNIT_ASSERT(frag.complete());
    CPPUNIT_ASSERT(!frag.faulty());
    CPPUNIT_ASSERT(frag.messageCode() == Header::ERROR_EVENT);
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
    CPPUNIT_ASSERT(frag.rbBufferId() == 0);
    CPPUNIT_ASSERT(frag.fuProcessId() == 0);
    CPPUNIT_ASSERT(frag.fuGuid() == 0);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    Reference* ref = allocate_frame_with_basic_header(I2O_SM_ERROR, 1, 2);
    stor::I2OChain frag(ref);
    CPPUNIT_ASSERT(!frag.empty());
    CPPUNIT_ASSERT(!frag.complete());
    CPPUNIT_ASSERT(!frag.faulty());
    CPPUNIT_ASSERT(frag.messageCode() == Header::ERROR_EVENT);
    CPPUNIT_ASSERT(frag.fragmentCount() == 1);
    CPPUNIT_ASSERT(frag.rbBufferId() == 0);
    CPPUNIT_ASSERT(frag.fuProcessId() == 0);
    CPPUNIT_ASSERT(frag.fuGuid() == 0);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::populate_i2o_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 0x12345678;
    unsigned int value5 = 6;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_PREAMBLE, 0, 1);
    I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
      (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->hltTid = value1;
    smMsg->rbBufferID = 2;
    smMsg->outModID = value2;
    smMsg->fuProcID = value3;
    smMsg->fuGUID = value4;
    smMsg->nExpectedEPs = value5;

    stor::I2OChain initMsgFrag(ref);
    CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);
    stor::FragKey fragmentKey = initMsgFrag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == Header::INIT);
    CPPUNIT_ASSERT(fragmentKey.run_ == 0);
    CPPUNIT_ASSERT(fragmentKey.event_ == value1);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value2);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value3);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value4);
    CPPUNIT_ASSERT(initMsgFrag.rbBufferId() == 2);
    CPPUNIT_ASSERT(initMsgFrag.fuProcessId() == value3);
    CPPUNIT_ASSERT(initMsgFrag.fuGuid() == value4);
    CPPUNIT_ASSERT(initMsgFrag.nExpectedEPs() == value5);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::copy_with_valid_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 0x01234567;
    unsigned int value5 = 0x89abcdef;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_DATA, 0, 1);
    I2O_SM_DATA_MESSAGE_FRAME *smMsg =
      (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->rbBufferID = 2;
    smMsg->runID = value1;
    smMsg->eventID = value2;
    smMsg->outModID = value3;
    smMsg->fuProcID = value4;
    smMsg->fuGUID = value5;

    stor::I2OChain eventMsgFrag(ref);
    stor::I2OChain copy(eventMsgFrag);

    {
      CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);
      stor::FragKey fragmentKey = eventMsgFrag.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.code_ == Header::EVENT);
      CPPUNIT_ASSERT(fragmentKey.run_ == value1);
      CPPUNIT_ASSERT(fragmentKey.event_ == value2);
      CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value3);
      CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value4);
      CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value5);
      CPPUNIT_ASSERT(eventMsgFrag.rbBufferId() == 2);
      CPPUNIT_ASSERT(eventMsgFrag.fuProcessId() == value4);
      CPPUNIT_ASSERT(eventMsgFrag.fuGuid() == value5);
    }

    {
      CPPUNIT_ASSERT(copy.messageCode() == Header::EVENT);
      stor::FragKey fragmentKey = copy.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.code_ == Header::EVENT);
      CPPUNIT_ASSERT(fragmentKey.run_ == value1);
      CPPUNIT_ASSERT(fragmentKey.event_ == value2);
      CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value3);
      CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value4);
      CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value5);
      CPPUNIT_ASSERT(copy.rbBufferId() == 2);
      CPPUNIT_ASSERT(copy.fuProcessId() == value4);
      CPPUNIT_ASSERT(copy.fuGuid() == value5);
    }

    {
      CPPUNIT_ASSERT(eventMsgFrag.creationTime() == copy.creationTime());
      CPPUNIT_ASSERT(eventMsgFrag.lastFragmentTime() == copy.lastFragmentTime());
    }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::assign_with_valid_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 0x01234567;
    unsigned int value5 = 0x89abcdef;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_ERROR, 0, 1);
    I2O_SM_DATA_MESSAGE_FRAME *smMsg =
      (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->rbBufferID = 2;
    smMsg->runID = value1;
    smMsg->eventID = value2;
    smMsg->outModID = value3;
    smMsg->fuProcID = value4;
    smMsg->fuGUID = value5;

    stor::I2OChain eventMsgFrag(ref);
    stor::I2OChain copy = eventMsgFrag;

    {
      CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::ERROR_EVENT);
      stor::FragKey fragmentKey = eventMsgFrag.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.code_ == Header::ERROR_EVENT);
      CPPUNIT_ASSERT(fragmentKey.run_ == value1);
      CPPUNIT_ASSERT(fragmentKey.event_ == value2);
      CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value3);
      CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value4);
      CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value5);
      CPPUNIT_ASSERT(eventMsgFrag.rbBufferId() == 2);
      CPPUNIT_ASSERT(eventMsgFrag.fuProcessId() == value4);
      CPPUNIT_ASSERT(eventMsgFrag.fuGuid() == value5);
    }

    {
      CPPUNIT_ASSERT(copy.messageCode() == Header::ERROR_EVENT);
      stor::FragKey fragmentKey = copy.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.code_ == Header::ERROR_EVENT);
      CPPUNIT_ASSERT(fragmentKey.run_ == value1);
      CPPUNIT_ASSERT(fragmentKey.event_ == value2);
      CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value3);
      CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value4);
      CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value5);
      CPPUNIT_ASSERT(copy.rbBufferId() == 2);
      CPPUNIT_ASSERT(copy.fuProcessId() == value4);
      CPPUNIT_ASSERT(copy.fuGuid() == value5);
    }

    {
      CPPUNIT_ASSERT(eventMsgFrag.creationTime() == copy.creationTime());
      CPPUNIT_ASSERT(eventMsgFrag.lastFragmentTime() == copy.lastFragmentTime());
    }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::swap_with_valid_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 0x01234567;
    unsigned int value5 = 0x89abcdef;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_DQM, 0, 1);
    I2O_SM_DQM_MESSAGE_FRAME *smMsg =
      (I2O_SM_DQM_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->rbBufferID = 2;
    smMsg->runID = value1;
    smMsg->eventAtUpdateID = value2;
    smMsg->folderID = value3;
    smMsg->fuProcID = value4;
    smMsg->fuGUID = value5;

    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();
    stor::utils::TimePoint_t lastFragmentTime1 = frag1.lastFragmentTime();
 
    ref = allocate_frame_with_basic_header(I2O_SM_DQM, 0, 1);
    smMsg = (I2O_SM_DQM_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->rbBufferID = 3;
    smMsg->runID = value5;
    smMsg->eventAtUpdateID = value4;
    smMsg->folderID = value3;
    smMsg->fuProcID = value2;
    smMsg->fuGUID = value1;

    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t creationTime2 = frag2.creationTime();
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();

    {
      CPPUNIT_ASSERT(frag1.messageCode() == Header::DQM_EVENT);
      stor::FragKey fragmentKey = frag1.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.code_ == Header::DQM_EVENT);
      CPPUNIT_ASSERT(fragmentKey.run_ == value1);
      CPPUNIT_ASSERT(fragmentKey.event_ == value2);
      CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value3);
      CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value4);
      CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value5);
      CPPUNIT_ASSERT(frag1.rbBufferId() == 2);
      CPPUNIT_ASSERT(frag1.fuProcessId() == value4);
      CPPUNIT_ASSERT(frag1.fuGuid() == value5);
    }

    {
      CPPUNIT_ASSERT(frag2.messageCode() == Header::DQM_EVENT);
      stor::FragKey fragmentKey = frag2.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.code_ == Header::DQM_EVENT);
      CPPUNIT_ASSERT(fragmentKey.run_ == value5);
      CPPUNIT_ASSERT(fragmentKey.event_ == value4);
      CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value3);
      CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value2);
      CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value1);
      CPPUNIT_ASSERT(frag2.rbBufferId() == 3);
      CPPUNIT_ASSERT(frag2.fuProcessId() == value2);
      CPPUNIT_ASSERT(frag2.fuGuid() == value1);
    }

    std::swap(frag1, frag2);

    {
      CPPUNIT_ASSERT(frag1.messageCode() == Header::DQM_EVENT);
      stor::FragKey fragmentKey = frag1.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.code_ == Header::DQM_EVENT);
      CPPUNIT_ASSERT(fragmentKey.run_ == value5);
      CPPUNIT_ASSERT(fragmentKey.event_ == value4);
      CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value3);
      CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value2);
      CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value1);
      CPPUNIT_ASSERT(frag1.rbBufferId() == 3);
      CPPUNIT_ASSERT(frag1.fuProcessId() == value2);
      CPPUNIT_ASSERT(frag1.fuGuid() == value1);
    }

    {
      CPPUNIT_ASSERT(frag2.messageCode() == Header::DQM_EVENT);
      stor::FragKey fragmentKey = frag2.fragmentKey();
      CPPUNIT_ASSERT(fragmentKey.code_ == Header::DQM_EVENT);
      CPPUNIT_ASSERT(fragmentKey.run_ == value1);
      CPPUNIT_ASSERT(fragmentKey.event_ == value2);
      CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value3);
      CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value4);
      CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value5);
      CPPUNIT_ASSERT(frag2.rbBufferId() == 2);
      CPPUNIT_ASSERT(frag2.fuProcessId() == value4);
      CPPUNIT_ASSERT(frag2.fuGuid() == value5);
    }

    {
      CPPUNIT_ASSERT(frag1.creationTime() == creationTime2);
      CPPUNIT_ASSERT(frag2.creationTime() == creationTime1);
      CPPUNIT_ASSERT(frag1.lastFragmentTime() == lastFragmentTime2);
      CPPUNIT_ASSERT(frag2.lastFragmentTime() == lastFragmentTime1);
    }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::release_with_valid_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 0x12345678;
    unsigned int value5 = 32;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_PREAMBLE, 0, 1);
    I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
      (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->hltTid = value1;
    smMsg->rbBufferID = 2;
    smMsg->outModID = value2;
    smMsg->fuProcID = value3;
    smMsg->fuGUID = value4;
    smMsg->nExpectedEPs = value5;

    stor::I2OChain initMsgFrag(ref);
    CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);
    stor::FragKey fragmentKey = initMsgFrag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == Header::INIT);
    CPPUNIT_ASSERT(fragmentKey.run_ == 0);
    CPPUNIT_ASSERT(fragmentKey.event_ == value1);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == value2);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value3);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value4);
    CPPUNIT_ASSERT(initMsgFrag.rbBufferId() == 2);
    CPPUNIT_ASSERT(initMsgFrag.fuProcessId() == value3);
    CPPUNIT_ASSERT(initMsgFrag.fuGuid() == value4);
    CPPUNIT_ASSERT(initMsgFrag.nExpectedEPs() == value5);

    initMsgFrag.release();
    CPPUNIT_ASSERT(initMsgFrag.messageCode() == 0);
    CPPUNIT_ASSERT(initMsgFrag.creationTime() == boost::posix_time::not_a_date_time);
    CPPUNIT_ASSERT(initMsgFrag.lastFragmentTime() == boost::posix_time::not_a_date_time);
    fragmentKey = initMsgFrag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == 0);
    CPPUNIT_ASSERT(fragmentKey.run_ == 0);
    CPPUNIT_ASSERT(fragmentKey.event_ == 0);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == 0);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == 0);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == 0);
    CPPUNIT_ASSERT(initMsgFrag.rbBufferId() == 0);
    CPPUNIT_ASSERT(initMsgFrag.fuProcessId() == 0);
    CPPUNIT_ASSERT(initMsgFrag.fuGuid() == 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::add_fragment()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add two fragments together (normal order)

    Reference* ref = allocate_frame_with_sample_header(0, 2, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    ref = allocate_frame_with_sample_header(1, 2, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.fragmentCount() == 2);
    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add two fragments together (reverse order)

    Reference* ref = allocate_frame_with_sample_header(0, 2, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t creationTime2 = frag2.creationTime();

    ref = allocate_frame_with_sample_header(1, 2, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t lastFragmentTime1 = frag1.lastFragmentTime();
    ::usleep((unsigned int) 50);

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.fragmentCount() == 2);
    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime2);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime1);

    // verify that adding a fragment to a complete chain throws an exception

    ref = allocate_frame_with_sample_header(0, 2, 2);
    stor::I2OChain frag3(ref);
    stor::utils::TimePoint_t creationTime3 = frag3.creationTime();
    stor::utils::TimePoint_t lastFragmentTime3 = frag3.lastFragmentTime();

    try
      {
        frag1.addToChain(frag3);
        CPPUNIT_ASSERT(false);
      }
    catch (stor::exception::I2OChain& excpt)
      {
      }
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag3.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag3.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag3.faulty());
    CPPUNIT_ASSERT(frag1.fragmentCount() == 2);
    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime2);
    CPPUNIT_ASSERT(frag3.creationTime() == creationTime3);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime1);
    CPPUNIT_ASSERT(frag3.lastFragmentTime() == lastFragmentTime3);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add three fragments together (normal order)

    Reference* ref = allocate_frame_with_sample_header(0, 3, 1);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    ref = allocate_frame_with_sample_header(1, 3, 1);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();

    ref = allocate_frame_with_sample_header(2, 3, 1);
    stor::I2OChain frag3(ref);
    stor::utils::TimePoint_t lastFragmentTime3 = frag3.lastFragmentTime();
    ::usleep((unsigned int) 50);

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);

    frag1.addToChain(frag3);

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag3.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag3.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag3.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);
    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 1);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime3);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add three fragments together (reverse order)

    Reference* ref = allocate_frame_with_sample_header(0, 3, 2);
    stor::I2OChain frag3(ref);
    stor::utils::TimePoint_t creationTime3 = frag3.creationTime();
    stor::utils::TimePoint_t lastFragmentTime3 = frag3.lastFragmentTime();

    ref = allocate_frame_with_sample_header(1, 3, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t creationTime2 = frag2.creationTime();

    ref = allocate_frame_with_sample_header(2, 3, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t lastFragmentTime1 = frag1.lastFragmentTime();
    ::usleep((unsigned int) 50);

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime2);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime1);

    frag1.addToChain(frag3);

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag3.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag3.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag3.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime3);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime3);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add three fragments together (mixed order)

    Reference* ref = allocate_frame_with_sample_header(2, 3, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    ref = allocate_frame_with_sample_header(0, 3, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();

    ref = allocate_frame_with_sample_header(1, 3, 2);
    stor::I2OChain frag3(ref);
    stor::utils::TimePoint_t lastFragmentTime3 = frag3.lastFragmentTime();
    ::usleep((unsigned int) 50);

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);

    frag1.addToChain(frag3);

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag3.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag3.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag3.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime3);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // verify that adding duplicate frames makes a chain faulty

    Reference* ref = allocate_frame_with_sample_header(1, 3, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    ref = allocate_frame_with_sample_header(1, 3, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);

    // verify that adding a fragment to a faulty chain works

    ref = allocate_frame_with_sample_header(0, 3, 2);
    stor::I2OChain frag3(ref);
    stor::utils::TimePoint_t lastFragmentTime3 = frag3.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(frag3.messageCode() != Header::INVALID);

    frag1.addToChain(frag3);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag3.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag3.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(!frag3.faulty());

    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 1);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime3);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // verify that adding a fragment to an empty chain throws an exception

    Reference* ref = allocate_frame_with_sample_header(0, 3, 2);
    stor::I2OChain frag1(ref);
    stor::I2OChain frag2;

    try
      {
        frag2.addToChain(frag1);
        CPPUNIT_ASSERT(false);
      }
    catch (stor::exception::I2OChain& excpt)
      {
      }
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());
    CPPUNIT_ASSERT(frag2.fragmentCount() == 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // verify that adding a faulty chain to a chain works

    Reference* ref = allocate_frame_with_sample_header(0, 2, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    ref = allocate_frame_with_sample_header(1, 2, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);
    frag2.markFaulty();
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(frag2.faulty());
    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());
    CPPUNIT_ASSERT(frag1.fragmentCount() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::chained_references()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // chained references in correct order

    Reference* ref1 = allocate_frame_with_sample_header(0, 2, 2);
    Reference* ref2 = allocate_frame_with_sample_header(1, 2, 2);
    ref1->setNextReference(ref2);

    stor::I2OChain frag1(ref1);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();
    stor::utils::TimePoint_t lastFragmentTime1 = frag1.lastFragmentTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.fragmentCount() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() == lastFragmentTime1);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // chained references in incorrect order

    Reference* ref1 = allocate_frame_with_sample_header(1, 2, 2);
    Reference* ref2 = allocate_frame_with_sample_header(0, 2, 2);
    ref1->setNextReference(ref2);

    stor::I2OChain frag1(ref1);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();
    stor::utils::TimePoint_t lastFragmentTime1 = frag1.lastFragmentTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 0);
    CPPUNIT_ASSERT(frag1.fragmentCount() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() == lastFragmentTime1);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // chained references with invalid indexes

    Reference* ref1 = allocate_frame_with_sample_header(1, 2, 2);
    Reference* ref2 = allocate_frame_with_sample_header(2, 2, 2);
    ref1->setNextReference(ref2);

    stor::I2OChain frag1(ref1);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();
    stor::utils::TimePoint_t lastFragmentTime1 = frag1.lastFragmentTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 2);
    CPPUNIT_ASSERT(frag1.fragmentCount() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() == lastFragmentTime1);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add fragment to I2OChain created from reference chain

    Reference* ref1 = allocate_frame_with_sample_header(0, 3, 2);
    Reference* ref2 = allocate_frame_with_sample_header(1, 3, 2);
    ref1->setNextReference(ref2);
    stor::I2OChain frag1(ref1);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());

    Reference* ref = allocate_frame_with_sample_header(2, 3, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag2.faulty());

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add reference chain to I2OChain

    Reference* ref = allocate_frame_with_sample_header(0, 3, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());

    Reference* ref1 = allocate_frame_with_sample_header(1, 3, 2);
    Reference* ref2 = allocate_frame_with_sample_header(2, 3, 2);
    ref1->setNextReference(ref2);
    stor::I2OChain frag2(ref1);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag2.faulty());

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add fragment to I2OChain created from reference chain

    Reference* ref1 = allocate_frame_with_sample_header(1, 3, 2);
    Reference* ref2 = allocate_frame_with_sample_header(2, 3, 2);
    ref1->setNextReference(ref2);
    stor::I2OChain frag1(ref1);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());

    Reference* ref = allocate_frame_with_sample_header(0, 3, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag2.faulty());

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add reference chain to I2OChain

    Reference* ref = allocate_frame_with_sample_header(1, 3, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());

    Reference* ref1 = allocate_frame_with_sample_header(0, 3, 2);
    Reference* ref2 = allocate_frame_with_sample_header(2, 3, 2);
    ref1->setNextReference(ref2);
    stor::I2OChain frag2(ref1);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag2.faulty());

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add fragment to I2OChain created from faulty reference chain

    Reference* ref1 = allocate_frame_with_sample_header(1, 3, 2);
    Reference* ref2 = allocate_frame_with_sample_header(0, 3, 2);
    ref1->setNextReference(ref2);
    stor::I2OChain frag1(ref1);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(frag1.faulty());

    Reference* ref = allocate_frame_with_sample_header(2, 3, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag2.faulty());

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add faulty reference chain to I2OChain

    Reference* ref = allocate_frame_with_sample_header(0, 3, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());

    Reference* ref1 = allocate_frame_with_sample_header(2, 3, 2);
    Reference* ref2 = allocate_frame_with_sample_header(1, 3, 2);
    ref1->setNextReference(ref2);
    stor::I2OChain frag2(ref1);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag2.faulty());

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 1);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);
    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add fragment to I2OChain created from faulty reference chain

    Reference* ref1 = allocate_frame_with_sample_header(0, 3, 2);
    Reference* ref2 = allocate_frame_with_sample_header(0, 3, 2);
    ref1->setNextReference(ref2);
    stor::I2OChain frag1(ref1);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(frag1.faulty());

    Reference* ref = allocate_frame_with_sample_header(2, 3, 2);
    stor::I2OChain frag2(ref);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(!frag2.faulty());

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);
    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // add faulty reference chain to I2OChain

    Reference* ref = allocate_frame_with_sample_header(0, 3, 2);
    stor::I2OChain frag1(ref);
    stor::utils::TimePoint_t creationTime1 = frag1.creationTime();

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());

    Reference* ref1 = allocate_frame_with_sample_header(2, 3, 2);
    Reference* ref2 = allocate_frame_with_sample_header(2, 3, 2);
    ref1->setNextReference(ref2);
    stor::I2OChain frag2(ref1);
    stor::utils::TimePoint_t lastFragmentTime2 = frag2.lastFragmentTime();
    ::usleep((unsigned int) 50);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag2.faulty());

    frag1.addToChain(frag2);
    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(frag2.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag1.faulty());
    CPPUNIT_ASSERT(!frag2.faulty());

    CPPUNIT_ASSERT(frag1.fragmentCount() == 3);
    CPPUNIT_ASSERT(frag1.getFragmentID(0) == 0);
    CPPUNIT_ASSERT(frag1.getFragmentID(1) == 2);
    CPPUNIT_ASSERT(frag1.getFragmentID(2) == 2);
    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 2);

    CPPUNIT_ASSERT(frag1.creationTime() == creationTime1);
    CPPUNIT_ASSERT(frag1.lastFragmentTime() > lastFragmentTime2);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}


void
testI2OChain::fragkey_mismatches()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // verify that we can't add an unparsable chain
    // to a normal chain (because the unparsable one has
    // a different fragKey)

    Reference* ref = allocate_frame_with_sample_header(0, 3, 0);
    stor::I2OChain frag1(ref);

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(frag1.messageCode() == Header::INIT);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 0);

    Reference* ref1 = allocate_frame_with_sample_header(100, 3, 0);
    Reference* ref2 = allocate_frame_with_sample_header( 99, 3, 0);
    ref1->setNextReference(ref2);
    stor::I2OChain frag2(ref1);

    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag2.faulty());
    CPPUNIT_ASSERT(frag2.messageCode() == Header::INIT);
    CPPUNIT_ASSERT(frag2.rbBufferId() == 0);

    try
      {
        frag1.addToChain(frag2);
        CPPUNIT_ASSERT(false);
      }
    catch (stor::exception::I2OChain& excpt)
      {
      }

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(!frag1.faulty());
    CPPUNIT_ASSERT(!frag2.empty());
    CPPUNIT_ASSERT(!frag2.complete());
    CPPUNIT_ASSERT(frag2.faulty());
    CPPUNIT_ASSERT(frag1.messageCode() != Header::INVALID);
    CPPUNIT_ASSERT(frag1.rbBufferId() == 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    // verify that we can't add unparsable chains together
    // forever (this could cause the chain to grow forever
    // in the FragmentStore and never become stale)

    Reference* ref = allocate_frame_with_sample_header(100, 3, 2);
    stor::I2OChain frag1(ref);

    CPPUNIT_ASSERT(!frag1.empty());
    CPPUNIT_ASSERT(!frag1.complete());
    CPPUNIT_ASSERT(frag1.faulty());

    // We test this by verifying that the addToChain method
    // throws an exception when we try to add two chains with
    // different fragKeys.  This should happen at some point
    // over the course of several seconds worth of unparsable chains.

    try
      {
        for (int idx = 0; idx < 100; ++idx)
          {
            ref = allocate_frame_with_sample_header(100, 3, 2);
            stor::I2OChain frag2(ref);

            CPPUNIT_ASSERT(!frag2.empty());
            CPPUNIT_ASSERT(!frag2.complete());
            CPPUNIT_ASSERT(frag2.faulty());

            frag1.addToChain(frag2);

            CPPUNIT_ASSERT(!frag1.empty());
            CPPUNIT_ASSERT(!frag1.complete());
            CPPUNIT_ASSERT(frag1.faulty());
            CPPUNIT_ASSERT(frag2.empty());
            CPPUNIT_ASSERT(!frag2.complete());
            CPPUNIT_ASSERT(!frag2.faulty());

            ::usleep(30000);
          }

        CPPUNIT_ASSERT(false);
      }
    catch (stor::exception::I2OChain& excpt)
      {
      }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::multipart_msg_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    std::string outputModuleLabel = "HLTOutput";
    std::string hltURL = "http://cmswn1340.fnal.gov:51985";
    std::string hltClass = "evf::FUResourceBroker";

    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
    unsigned int outputModuleId = crc32(crc,crcbuf,outputModuleLabel.length());

    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 0xdeadbeef;
    unsigned int value5 = 0x01234567;
    unsigned int value6 = 0x89abcdef;
    unsigned int value7 = 22;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_PREAMBLE, 0, 1);
    I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
      (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->rbBufferID = 2;
    smMsg->outModID = outputModuleId;
    smMsg->fuProcID = value2;
    smMsg->fuGUID = value3;
    smMsg->nExpectedEPs = value7;

    std::strcpy(smMsg->hltURL, hltURL.c_str());
    std::strcpy(smMsg->hltClassName, hltClass.c_str());
    smMsg->hltLocalId = value4;
    smMsg->hltInstance = value5;
    smMsg->hltTid = value6;


    stor::I2OChain initMsgFrag(ref);
    CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

    stor::FragKey fragmentKey = initMsgFrag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == Header::INIT);
    CPPUNIT_ASSERT(fragmentKey.run_ == 0);
    CPPUNIT_ASSERT(fragmentKey.event_ == value6);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == outputModuleId);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value2);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value3);

    CPPUNIT_ASSERT(initMsgFrag.hltLocalId() == value4);
    CPPUNIT_ASSERT(initMsgFrag.hltInstance() == value5);
    CPPUNIT_ASSERT(initMsgFrag.hltTid() == value6);
    CPPUNIT_ASSERT(initMsgFrag.hltURL() == hltURL);
    CPPUNIT_ASSERT(initMsgFrag.hltClassName() == hltClass);
    CPPUNIT_ASSERT(initMsgFrag.rbBufferId() == 2);
    CPPUNIT_ASSERT(initMsgFrag.fuProcessId() == value2);
    CPPUNIT_ASSERT(initMsgFrag.fuGuid() == value3);
    CPPUNIT_ASSERT(initMsgFrag.nExpectedEPs() == value7);


    stor::I2OChain initMsgFrag2;
    CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INVALID);
    CPPUNIT_ASSERT(initMsgFrag2.hltLocalId() == 0);
    CPPUNIT_ASSERT(initMsgFrag2.hltInstance() == 0);
    CPPUNIT_ASSERT(initMsgFrag2.hltTid() == 0);
    CPPUNIT_ASSERT(initMsgFrag2.hltURL() == "");
    CPPUNIT_ASSERT(initMsgFrag2.hltClassName() == "");
    CPPUNIT_ASSERT(initMsgFrag2.rbBufferId() == 0);
    CPPUNIT_ASSERT(initMsgFrag2.fuProcessId() == 0);
    CPPUNIT_ASSERT(initMsgFrag2.fuGuid() == 0);

    std::swap(initMsgFrag, initMsgFrag2);

    CPPUNIT_ASSERT(initMsgFrag2.messageCode() == Header::INIT);
    CPPUNIT_ASSERT(initMsgFrag2.hltLocalId() == value4);
    CPPUNIT_ASSERT(initMsgFrag2.hltInstance() == value5);
    CPPUNIT_ASSERT(initMsgFrag2.hltTid() == value6);
    CPPUNIT_ASSERT(initMsgFrag2.hltURL() == hltURL);
    CPPUNIT_ASSERT(initMsgFrag2.hltClassName() == hltClass);
    CPPUNIT_ASSERT(initMsgFrag2.rbBufferId() == 2);
    CPPUNIT_ASSERT(initMsgFrag2.fuProcessId() == value2);
    CPPUNIT_ASSERT(initMsgFrag2.fuGuid() == value3);
    CPPUNIT_ASSERT(initMsgFrag2.nExpectedEPs() == value7);

    CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INVALID);
    CPPUNIT_ASSERT(initMsgFrag.hltLocalId() == 0);
    CPPUNIT_ASSERT(initMsgFrag.hltInstance() == 0);
    CPPUNIT_ASSERT(initMsgFrag.hltTid() == 0);
    CPPUNIT_ASSERT(initMsgFrag.hltURL() == "");
    CPPUNIT_ASSERT(initMsgFrag.hltClassName() == "");
    CPPUNIT_ASSERT(initMsgFrag.rbBufferId() == 0);
    CPPUNIT_ASSERT(initMsgFrag.fuProcessId() == 0);
    CPPUNIT_ASSERT(initMsgFrag.fuGuid() == 0);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    std::string outputModuleLabel = "HLTOutput";
    std::string hltClass =
      "evf::FUResourceBroker012345678901234567890123456789012345678901234567";

    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
    unsigned int outputModuleId = crc32(crc,crcbuf,outputModuleLabel.length());

    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 0xdeadbeef;
    unsigned int value5 = 22;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_PREAMBLE, 0, 1);
    I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
      (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->hltTid = value1;
    smMsg->rbBufferID = 2;
    smMsg->outModID = outputModuleId;
    smMsg->fuProcID = value2;
    smMsg->fuGUID = value3;
    smMsg->nExpectedEPs = value5;
    smMsg->nExpectedEPs = value5;

    std::strncpy(smMsg->hltClassName, hltClass.c_str(), MAX_I2O_SM_URLCHARS);
    smMsg->hltInstance = value4;

    stor::I2OChain initMsgFrag(ref);
    CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

    stor::FragKey fragmentKey = initMsgFrag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == Header::INIT);
    CPPUNIT_ASSERT(fragmentKey.run_ == 0);
    CPPUNIT_ASSERT(fragmentKey.event_ == value1);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == outputModuleId);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value2);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value3);

    CPPUNIT_ASSERT(initMsgFrag.hltInstance() == value4);
    CPPUNIT_ASSERT(initMsgFrag.hltClassName() ==
                   hltClass.substr(0, MAX_I2O_SM_URLCHARS));
    CPPUNIT_ASSERT(initMsgFrag.rbBufferId() == 2);
    CPPUNIT_ASSERT(initMsgFrag.fuProcessId() == value2);
    CPPUNIT_ASSERT(initMsgFrag.fuGuid() == value3);
    CPPUNIT_ASSERT(initMsgFrag.nExpectedEPs() == value5);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::init_msg_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    char psetid[] = "1234567890123456";
    Strings hlt_names;
    Strings hlt_selections;
    Strings l1_names;

    hlt_names.push_back("a");  hlt_names.push_back("b");
    hlt_names.push_back("c");  hlt_names.push_back("d");
    hlt_names.push_back("e");  hlt_names.push_back("f");
    hlt_names.push_back("g");  hlt_names.push_back("h");
    hlt_names.push_back("i");

    hlt_selections.push_back("a");
    hlt_selections.push_back("c");
    hlt_selections.push_back("e");
    hlt_selections.push_back("g");
    hlt_selections.push_back("i");

    l1_names.push_back("t10");  l1_names.push_back("t11");
    l1_names.push_back("t12");  l1_names.push_back("t13");
    l1_names.push_back("t14");  l1_names.push_back("t15");
    l1_names.push_back("t16");  l1_names.push_back("t17");
    l1_names.push_back("t18");  l1_names.push_back("t19");
    l1_names.push_back("t20");

    char reltag[]="CMSSW_3_0_0_pre7";
    std::string processName = "HLT";
    std::string outputModuleLabel = "HLTOutput";

    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
    unsigned int outputModuleId = crc32(crc,crcbuf,outputModuleLabel.length());

    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 1;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_PREAMBLE, 0, 1);
    I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
      (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->hltTid = value1;
    smMsg->rbBufferID = 2;
    smMsg->outModID = outputModuleId;
    smMsg->fuProcID = value2;
    smMsg->fuGUID = value3;
    smMsg->nExpectedEPs = value4;

    char test_value[] = "This is a test, This is a";
    uint32_t adler32_chksum = (uint32_t)cms::Adler32((char*)&test_value[0], sizeof(test_value));
    char host_name[255];
    gethostname(host_name, 255);

    InitMsgBuilder
      initBuilder(smMsg->dataPtr(), smMsg->dataSize, 100,
                  Version((const uint8*)psetid), (const char*) reltag,
                  processName.c_str(), outputModuleLabel.c_str(),
                  outputModuleId, hlt_names, hlt_selections, l1_names,
                  adler32_chksum, host_name);

    initBuilder.setDataLength(sizeof(test_value));
    std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
              initBuilder.dataAddress());
    smMsg->dataSize = initBuilder.headerSize() + sizeof(test_value);

    stor::I2OChain initMsgFrag(ref);
    CPPUNIT_ASSERT(initMsgFrag.messageCode() == Header::INIT);

    stor::FragKey fragmentKey = initMsgFrag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == Header::INIT);
    CPPUNIT_ASSERT(fragmentKey.run_ == 0);
    CPPUNIT_ASSERT(fragmentKey.event_ == value1);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == outputModuleId);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value2);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value3);

    CPPUNIT_ASSERT(initMsgFrag.outputModuleLabel() == outputModuleLabel);
    CPPUNIT_ASSERT(initMsgFrag.outputModuleId() == outputModuleId);

    CPPUNIT_ASSERT(initMsgFrag.rbBufferId() == 2);
    CPPUNIT_ASSERT(initMsgFrag.fuProcessId() == value2);
    CPPUNIT_ASSERT(initMsgFrag.fuGuid() == value3);
    CPPUNIT_ASSERT(initMsgFrag.nExpectedEPs() == value4);

    Strings outNames;
    outNames.clear();
    initMsgFrag.hltTriggerNames(outNames);
    for (uint32_t idx = 0; idx < hlt_names.size(); ++idx)
      {
        CPPUNIT_ASSERT(outNames[idx] == hlt_names[idx]);
      }
    outNames.clear();
    initMsgFrag.hltTriggerSelections(outNames);
    for (uint32_t idx = 0; idx < hlt_selections.size(); ++idx)
      {
        CPPUNIT_ASSERT(outNames[idx] == hlt_selections[idx]);
      }
    outNames.clear();
    initMsgFrag.l1TriggerNames(outNames);
    for (uint32_t idx = 0; idx < l1_names.size(); ++idx)
      {
        CPPUNIT_ASSERT(outNames[idx] == l1_names[idx]);
      }

    CPPUNIT_ASSERT(initMsgFrag.headerSize() == initBuilder.headerSize());
    CPPUNIT_ASSERT(initMsgFrag.headerLocation() ==
                   initMsgFrag.dataLocation(0));
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::event_msg_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    std::vector<bool> l1Bits;
    l1Bits.push_back(true);
    l1Bits.push_back(true);
    l1Bits.push_back(false);
    l1Bits.push_back(true);
    l1Bits.push_back(true);
    l1Bits.push_back(false);
    l1Bits.push_back(false);
    l1Bits.push_back(true);
    l1Bits.push_back(false);
    l1Bits.push_back(true);

    uint32_t hltBitCount = 21;
    std::vector<unsigned char> hltBits;
    hltBits.resize(1 + (hltBitCount-1)/4);
    for (uint32_t idx = 0; idx < hltBits.size(); ++idx) {
      hltBits[idx] = 0x3 << idx;
      // should mask off bits for trig num GT hltBitCount...
    }

    std::string outputModuleLabel = "HLTOutput";
    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
    unsigned int outputModuleId = crc32(crc,crcbuf,outputModuleLabel.length());

    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int runNumber = 100;
    unsigned int eventNumber = 42;
    unsigned int lumiNumber = 777;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_DATA, 0, 1);
    I2O_SM_DATA_MESSAGE_FRAME *smMsg =
      (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->hltTid = value1;
    smMsg->rbBufferID = 3;
    smMsg->runID = runNumber;
    smMsg->eventID = eventNumber;
    smMsg->outModID = outputModuleId;
    smMsg->fuProcID = value2;
    smMsg->fuGUID = value3;

    char test_value_event[] = "This is a test Event, This is a";
    uint32_t adler32_chksum = (uint32_t)cms::Adler32((char*)&test_value_event[0], sizeof(test_value_event));
    char host_name[255];
    gethostname(host_name, 255);

    EventMsgBuilder
      eventBuilder(smMsg->dataPtr(), smMsg->dataSize, runNumber,
                   eventNumber, lumiNumber, outputModuleId, 0,
                   l1Bits, &hltBits[0], hltBitCount, adler32_chksum, host_name);

    eventBuilder.setOrigDataSize(78);
    eventBuilder.setEventLength(sizeof(test_value_event));
    std::copy(&test_value_event[0],&test_value_event[0]+sizeof(test_value_event),
              eventBuilder.eventAddr());
    smMsg->dataSize = eventBuilder.headerSize() + sizeof(test_value_event);

    stor::I2OChain eventMsgFrag(ref);
    CPPUNIT_ASSERT(eventMsgFrag.messageCode() == Header::EVENT);
    CPPUNIT_ASSERT(eventMsgFrag.runNumber() == runNumber);
    CPPUNIT_ASSERT(eventMsgFrag.lumiSection() == lumiNumber);
    CPPUNIT_ASSERT(eventMsgFrag.eventNumber() == eventNumber);

    stor::FragKey fragmentKey = eventMsgFrag.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == Header::EVENT);
    CPPUNIT_ASSERT(fragmentKey.run_ == runNumber);
    CPPUNIT_ASSERT(fragmentKey.event_ == eventNumber);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == outputModuleId);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value2);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value3);

    CPPUNIT_ASSERT(eventMsgFrag.outputModuleId() == outputModuleId);
    CPPUNIT_ASSERT(eventMsgFrag.hltTriggerCount() == hltBitCount);

    CPPUNIT_ASSERT(eventMsgFrag.rbBufferId() == 3);
    CPPUNIT_ASSERT(eventMsgFrag.fuProcessId() == value2);
    CPPUNIT_ASSERT(eventMsgFrag.fuGuid() == value3);

    std::vector<unsigned char> hltBits2;
    CPPUNIT_ASSERT(hltBits2.size() == 0);
    eventMsgFrag.hltTriggerBits(hltBits2);
    CPPUNIT_ASSERT(hltBits2.size() == hltBits.size());
    hltBits2.resize(100);
    CPPUNIT_ASSERT(hltBits2.size() == 100);
    eventMsgFrag.hltTriggerBits(hltBits2);
    CPPUNIT_ASSERT(hltBits2.size() == hltBits.size());

    uint32_t trigIndex = 0;
    for (uint32_t idx = 0; idx < hltBits.size(); ++idx) {
      for (uint32_t jdx = 0; jdx < 4; ++jdx) {
        uint32_t indexMod = (trigIndex % 4);
        uint32_t trigMask = 0;
        switch (indexMod) {
        case 0:
          {
            trigMask = 0x03;
            break;
          }
        case 1:
          {
            trigMask = 0x0c;
            break;
          }
        case 2:
          {
            trigMask = 0x30;
            break;
          }
        case 3:
          {
            trigMask = 0xc0;
            break;
          }
        }
        CPPUNIT_ASSERT((hltBits2[idx] & trigMask) ==
                       (hltBits[idx] & trigMask));
        ++trigIndex;
      }
    }

    CPPUNIT_ASSERT(eventMsgFrag.headerSize() == eventBuilder.headerSize());
    CPPUNIT_ASSERT(eventMsgFrag.headerLocation() ==
                   eventMsgFrag.dataLocation(0));
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::error_event_msg_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int runNumber = 100;
    unsigned int eventNumber = 42;
    unsigned int lumiNumber = 777;

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_ERROR, 0, 1);
    I2O_SM_DATA_MESSAGE_FRAME *smMsg =
      (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->hltTid = value1;
    smMsg->rbBufferID = 3;
    smMsg->runID = runNumber;
    smMsg->eventID = eventNumber;
    smMsg->outModID = 0xffffffff;
    smMsg->fuProcID = value2;
    smMsg->fuGUID = value3;

    uint32_t* dataPtr = (uint32_t*) smMsg->dataPtr();
    *dataPtr++ = 2;  // version number
    *dataPtr++ = runNumber;
    *dataPtr++ = lumiNumber;
    *dataPtr++ = eventNumber;

    stor::I2OChain errorMsgFrag(ref);
    CPPUNIT_ASSERT(errorMsgFrag.messageCode() == Header::ERROR_EVENT);
    CPPUNIT_ASSERT(errorMsgFrag.runNumber() == runNumber);
    CPPUNIT_ASSERT(errorMsgFrag.lumiSection() == lumiNumber);
    CPPUNIT_ASSERT(errorMsgFrag.eventNumber() == eventNumber);

    CPPUNIT_ASSERT(errorMsgFrag.headerSize() == sizeof(FRDEventHeader_V2));
    CPPUNIT_ASSERT(errorMsgFrag.headerLocation() ==
                   errorMsgFrag.dataLocation(0));

    CPPUNIT_ASSERT(errorMsgFrag.rbBufferId() == 3);
    CPPUNIT_ASSERT(errorMsgFrag.fuProcessId() == value2);
    CPPUNIT_ASSERT(errorMsgFrag.fuGuid() == value3);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::stream_and_queue_tags()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    stor::StreamID streamA =  2;
    stor::StreamID streamB =  5;
    stor::StreamID streamC = 17;

    stor::QueueID evtQueueA(stor::enquing_policy::DiscardOld, 101);
    stor::QueueID evtQueueB(stor::enquing_policy::DiscardOld, 102);
    stor::QueueID evtQueueC(stor::enquing_policy::DiscardOld, 104);
    stor::QueueID evtQueueD(stor::enquing_policy::DiscardOld, 103);

    stor::QueueID dqmQueueA(stor::enquing_policy::DiscardNew, 0xdeadbeef);

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_DATA, 0, 1);
    stor::I2OChain eventMsgFrag(ref);

    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

    eventMsgFrag.tagForStream(streamA);
    eventMsgFrag.tagForStream(streamB);
    eventMsgFrag.tagForStream(streamC);

    eventMsgFrag.tagForEventConsumer(evtQueueA);
    eventMsgFrag.tagForEventConsumer(evtQueueB);
    eventMsgFrag.tagForEventConsumer(evtQueueC);
    eventMsgFrag.tagForEventConsumer(evtQueueD);

    eventMsgFrag.tagForDQMEventConsumer(dqmQueueA);

    CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(eventMsgFrag.isTaggedForAnyDQMEventConsumer());

    CPPUNIT_ASSERT(eventMsgFrag.getStreamTags().size() == 3);
    CPPUNIT_ASSERT(eventMsgFrag.getEventConsumerTags().size() == 4);
    CPPUNIT_ASSERT(eventMsgFrag.getDQMEventConsumerTags().size() == 1);

    std::vector<stor::StreamID> streamTags = eventMsgFrag.getStreamTags();
    CPPUNIT_ASSERT(std::count(streamTags.begin(),streamTags.end(),streamA) == 1);
    CPPUNIT_ASSERT(std::count(streamTags.begin(),streamTags.end(),streamB) == 1);
    CPPUNIT_ASSERT(std::count(streamTags.begin(),streamTags.end(),streamC) == 1);
    
    streamTags.push_back(999);
    CPPUNIT_ASSERT(eventMsgFrag.getStreamTags().size() == 3);
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    stor::I2OChain eventMsgFrag;

    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyStream());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyEventConsumer());
    CPPUNIT_ASSERT(!eventMsgFrag.isTaggedForAnyDQMEventConsumer());

    CPPUNIT_ASSERT(eventMsgFrag.getStreamTags().size() == 0);
    CPPUNIT_ASSERT(eventMsgFrag.getEventConsumerTags().size() == 0);
    CPPUNIT_ASSERT(eventMsgFrag.getDQMEventConsumerTags().size() == 0);

    try
      {
        eventMsgFrag.tagForStream(100);
        CPPUNIT_ASSERT(false);
      }
    catch (stor::exception::I2OChain& excpt)
      {
      }

    stor::QueueID nonExistingQueueId(stor::enquing_policy::DiscardOld, 100);
    try
      {
        eventMsgFrag.tagForEventConsumer(nonExistingQueueId);
        CPPUNIT_ASSERT(false);
      }
    catch (stor::exception::I2OChain& excpt)
      {
      }
    try
      {
        eventMsgFrag.tagForDQMEventConsumer(nonExistingQueueId);
        CPPUNIT_ASSERT(false);
      }
    catch (stor::exception::I2OChain& excpt)
      {
      }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::split_init_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    char psetid[] = "1234567890123456";
    Strings hlt_names;
    Strings hlt_selections;
    Strings l1_names;

    hlt_names.push_back("a");  hlt_names.push_back("b");
    hlt_names.push_back("c");  hlt_names.push_back("d");
    hlt_names.push_back("e");  hlt_names.push_back("f");
    hlt_names.push_back("g");  hlt_names.push_back("h");
    hlt_names.push_back("i");

    hlt_selections.push_back("a");
    hlt_selections.push_back("c");
    hlt_selections.push_back("e");
    hlt_selections.push_back("g");
    hlt_selections.push_back("i");

    l1_names.push_back("t10");  l1_names.push_back("t11");
    l1_names.push_back("t12");  l1_names.push_back("t13");
    l1_names.push_back("t14");  l1_names.push_back("t15");
    l1_names.push_back("t16");  l1_names.push_back("t17");
    l1_names.push_back("t18");  l1_names.push_back("t19");
    l1_names.push_back("t20");

    char reltag[]="CMSSW_3_0_0_pre7";
    std::string processName = "HLT";
    std::string outputModuleLabel = "HLTOutput";

    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
    unsigned int outputModuleId = crc32(crc,crcbuf,outputModuleLabel.length());

    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int value4 = 34;

    int bufferSize = 2000;
    std::vector<unsigned char> tmpBuffer;
    tmpBuffer.resize(bufferSize);

    char test_value[] = "This is a test, This is a";
    uint32_t adler32_chksum = (uint32_t)cms::Adler32((char*)&test_value[0], sizeof(test_value));
    char host_name[255];
    gethostname(host_name, 255);

    InitMsgBuilder
      initBuilder(&tmpBuffer[0], bufferSize, 100,
                  Version((const uint8*)psetid), (const char*) reltag,
                  processName.c_str(), outputModuleLabel.c_str(),
                  outputModuleId, hlt_names, hlt_selections, l1_names,
                  adler32_chksum, host_name);

    initBuilder.setDataLength(sizeof(test_value));
    std::copy(&test_value[0],&test_value[0]+sizeof(test_value),
              initBuilder.dataAddress());

    //std::cout << "Size = " << initBuilder.size() << std::endl;

    uint32_t fragmentSize = 50;
    uint32_t msgSize = initBuilder.size();
    uint32_t fragmentCount = 1 + ((uint32_t) (msgSize - 1) / fragmentSize);

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_PREAMBLE, 0,
                                                      fragmentCount);
    I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
      (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->hltTid = value1;
    smMsg->rbBufferID = 2;
    smMsg->outModID = outputModuleId;
    smMsg->fuProcID = value2;
    smMsg->fuGUID = value3;
    smMsg->nExpectedEPs = value4;

    unsigned char* sourceLoc = &tmpBuffer[0];
    unsigned long sourceSize = fragmentSize;
    if (msgSize < fragmentSize) sourceSize = msgSize;
    unsigned char* targetLoc = (unsigned char*) smMsg->dataPtr();;
    std::copy(sourceLoc, sourceLoc+sourceSize, targetLoc);
    smMsg->dataSize = sourceSize;

    unsigned long dataSize = sourceSize + sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME);
    ref->setDataSize(dataSize);
    smMsg->PvtMessageFrame.StdMessageFrame.MessageSize = dataSize / 4;

    stor::I2OChain initMsgChain(ref);

    CPPUNIT_ASSERT(!initMsgChain.empty());
    CPPUNIT_ASSERT(!initMsgChain.complete());
    CPPUNIT_ASSERT(!initMsgChain.faulty());

    for (uint32_t idx = 1; idx < fragmentCount; ++idx)
      {
        ref = allocate_frame_with_basic_header(I2O_SM_PREAMBLE, idx,
                                               fragmentCount);
        smMsg = (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref->getDataLocation();
        smMsg->hltTid = value1;
        smMsg->rbBufferID = 2;
        smMsg->outModID = outputModuleId;
        smMsg->fuProcID = value2;
        smMsg->fuGUID = value3;
        smMsg->nExpectedEPs = value4;

        sourceLoc = &tmpBuffer[idx*fragmentSize];
        sourceSize = fragmentSize;
        if ((msgSize - idx*fragmentSize) < fragmentSize)
          {
            sourceSize = msgSize - idx*fragmentSize;
          }
        targetLoc = (unsigned char*) smMsg->dataPtr();;
        std::copy(sourceLoc, sourceLoc+sourceSize, targetLoc);
        smMsg->dataSize = sourceSize;

        dataSize = sourceSize + sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME);
        ref->setDataSize(dataSize);
        smMsg->PvtMessageFrame.StdMessageFrame.MessageSize = dataSize / 4;

        stor::I2OChain initMsgFrag(ref);

        CPPUNIT_ASSERT(!initMsgFrag.empty());
        CPPUNIT_ASSERT(!initMsgFrag.complete());
        CPPUNIT_ASSERT(!initMsgFrag.faulty());

        initMsgChain.addToChain(initMsgFrag);

        CPPUNIT_ASSERT(!initMsgChain.empty());
        if (idx == (fragmentCount-1))
          {
            CPPUNIT_ASSERT(initMsgChain.complete());
          }
        else
          {
            CPPUNIT_ASSERT(!initMsgChain.complete());
          }
        CPPUNIT_ASSERT(!initMsgChain.faulty());
      }

    std::vector<unsigned char> bufferCopy;
    initMsgChain.copyFragmentsIntoBuffer(bufferCopy);
    for (uint32_t idx = 0; idx < msgSize; ++idx)
      {
        CPPUNIT_ASSERT(bufferCopy[idx] == tmpBuffer[idx]);
      }

    CPPUNIT_ASSERT(initMsgChain.messageCode() == Header::INIT);

    stor::FragKey fragmentKey = initMsgChain.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == Header::INIT);
    CPPUNIT_ASSERT(fragmentKey.run_ == 0);
    CPPUNIT_ASSERT(fragmentKey.event_ == value1);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == outputModuleId);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value2);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value3);

    CPPUNIT_ASSERT(initMsgChain.outputModuleLabel() == outputModuleLabel);
    CPPUNIT_ASSERT(initMsgChain.outputModuleId() == outputModuleId);

    Strings outNames;
    outNames.clear();
    initMsgChain.hltTriggerNames(outNames);
    for (uint32_t idx = 0; idx < hlt_names.size(); ++idx)
      {
        CPPUNIT_ASSERT(outNames[idx] == hlt_names[idx]);
      }
    outNames.clear();
    initMsgChain.hltTriggerSelections(outNames);
    for (uint32_t idx = 0; idx < hlt_selections.size(); ++idx)
      {
        CPPUNIT_ASSERT(outNames[idx] == hlt_selections[idx]);
      }
    outNames.clear();
    initMsgChain.l1TriggerNames(outNames);
    for (uint32_t idx = 0; idx < l1_names.size(); ++idx)
      {
        CPPUNIT_ASSERT(outNames[idx] == l1_names[idx]);
      }

    CPPUNIT_ASSERT(initMsgChain.headerSize() == initBuilder.headerSize());
    CPPUNIT_ASSERT(initMsgChain.headerLocation() !=
                   initMsgChain.dataLocation(0));

    unsigned char* headerLoc = initMsgChain.headerLocation();
    for (uint32_t idx = 0; idx < initMsgChain.headerSize(); ++idx)
      {
        CPPUNIT_ASSERT(headerLoc[idx] == tmpBuffer[idx]);
      }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testI2OChain::split_event_header()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  {
    std::vector<bool> l1Bits;
    l1Bits.push_back(true);
    l1Bits.push_back(true);
    l1Bits.push_back(false);
    l1Bits.push_back(true);
    l1Bits.push_back(true);
    l1Bits.push_back(false);
    l1Bits.push_back(false);
    l1Bits.push_back(true);
    l1Bits.push_back(false);
    l1Bits.push_back(true);

    uint32_t hltBitCount = 21;
    std::vector<unsigned char> hltBits;
    hltBits.resize(1 + (hltBitCount-1)/4);
    for (uint32_t idx = 0; idx < hltBits.size(); ++idx) {
      hltBits[idx] = 0x3 << idx;
      // should mask off bits for trig num GT hltBitCount...
    }

    std::string outputModuleLabel = "HLTOutput";
    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
    unsigned int outputModuleId = crc32(crc,crcbuf,outputModuleLabel.length());

    unsigned int value1 = 0xa5a5d2d2;
    unsigned int value2 = 0xb4b4e1e1;
    unsigned int value3 = 0xc3c3f0f0;
    unsigned int runNumber = 100;
    unsigned int eventNumber = 42;
    unsigned int lumiNumber = 777;

    int bufferSize = 2000;
    std::vector<unsigned char> tmpBuffer;
    tmpBuffer.resize(bufferSize);

    char test_value_event[] = "This is a test Event, This is a";
    uint32_t adler32_chksum = (uint32_t)cms::Adler32((char*)&test_value_event[0], sizeof(test_value_event));
    char host_name[255];
    gethostname(host_name, 255);

    EventMsgBuilder
      eventBuilder(&tmpBuffer[0], bufferSize, runNumber,
                   eventNumber, lumiNumber, outputModuleId, 0,
                   l1Bits, &hltBits[0], hltBitCount, adler32_chksum, host_name);

    eventBuilder.setOrigDataSize(78);
    eventBuilder.setEventLength(sizeof(test_value_event));
    std::copy(&test_value_event[0],&test_value_event[0]+sizeof(test_value_event),
              eventBuilder.eventAddr());

    //std::cout << "Size = " << eventBuilder.size() << std::endl;

    uint32_t fragmentSize = 10;
    uint32_t msgSize = eventBuilder.size();
    uint32_t fragmentCount = 1 + ((uint32_t) (msgSize - 1) / fragmentSize);

    Reference* ref = allocate_frame_with_basic_header(I2O_SM_DATA, 0,
                                                      fragmentCount);
    I2O_SM_DATA_MESSAGE_FRAME *smMsg =
      (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
    smMsg->hltTid = value1;
    smMsg->rbBufferID = 3;
    smMsg->runID = runNumber;
    smMsg->eventID = eventNumber;
    smMsg->outModID = outputModuleId;
    smMsg->fuProcID = value2;
    smMsg->fuGUID = value3;

    unsigned char* sourceLoc = &tmpBuffer[0];
    unsigned long sourceSize = fragmentSize;
    if (msgSize < fragmentSize) sourceSize = msgSize;
    unsigned char* targetLoc = (unsigned char*) smMsg->dataPtr();;
    std::copy(sourceLoc, sourceLoc+sourceSize, targetLoc);
    smMsg->dataSize = sourceSize;

    unsigned long dataSize = sourceSize + sizeof(I2O_SM_DATA_MESSAGE_FRAME);
    ref->setDataSize(dataSize);
    smMsg->PvtMessageFrame.StdMessageFrame.MessageSize = dataSize / 4;

    stor::I2OChain eventMsgChain(ref);

    CPPUNIT_ASSERT(!eventMsgChain.empty());
    CPPUNIT_ASSERT(!eventMsgChain.complete());
    CPPUNIT_ASSERT(!eventMsgChain.faulty());

    for (uint32_t idx = 1; idx < fragmentCount; ++idx)
      {
        ref = allocate_frame_with_basic_header(I2O_SM_DATA, idx,
                                               fragmentCount);
        smMsg = (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
        smMsg->hltTid = value1;
        smMsg->rbBufferID = 3;
        smMsg->runID = runNumber;
        smMsg->eventID = eventNumber;
        smMsg->outModID = outputModuleId;
        smMsg->fuProcID = value2;
        smMsg->fuGUID = value3;

        sourceLoc = &tmpBuffer[idx*fragmentSize];
        sourceSize = fragmentSize;
        if ((msgSize - idx*fragmentSize) < fragmentSize)
          {
            sourceSize = msgSize - idx*fragmentSize;
          }
        targetLoc = (unsigned char*) smMsg->dataPtr();;
        std::copy(sourceLoc, sourceLoc+sourceSize, targetLoc);
        smMsg->dataSize = sourceSize;

        dataSize = sourceSize + sizeof(I2O_SM_DATA_MESSAGE_FRAME);
        ref->setDataSize(dataSize);
        smMsg->PvtMessageFrame.StdMessageFrame.MessageSize = dataSize / 4;

        stor::I2OChain eventMsgFrag(ref);

        CPPUNIT_ASSERT(!eventMsgFrag.empty());
        CPPUNIT_ASSERT(!eventMsgFrag.complete());
        CPPUNIT_ASSERT(!eventMsgFrag.faulty());

        eventMsgChain.addToChain(eventMsgFrag);

        CPPUNIT_ASSERT(!eventMsgChain.empty());
        if (idx == (fragmentCount-1))
          {
            CPPUNIT_ASSERT(eventMsgChain.complete());
          }
        else
          {
            CPPUNIT_ASSERT(!eventMsgChain.complete());
          }
        CPPUNIT_ASSERT(!eventMsgChain.faulty());
      }

    std::vector<unsigned char> bufferCopy;
    eventMsgChain.copyFragmentsIntoBuffer(bufferCopy);
    for (uint32_t idx = 0; idx < msgSize; ++idx)
      {
        CPPUNIT_ASSERT(bufferCopy[idx] == tmpBuffer[idx]);
      }

    CPPUNIT_ASSERT(eventMsgChain.messageCode() == Header::EVENT);
    CPPUNIT_ASSERT(eventMsgChain.runNumber() == runNumber);
    CPPUNIT_ASSERT(eventMsgChain.lumiSection() == lumiNumber);
    CPPUNIT_ASSERT(eventMsgChain.eventNumber() == eventNumber);

    stor::FragKey fragmentKey = eventMsgChain.fragmentKey();
    CPPUNIT_ASSERT(fragmentKey.code_ == Header::EVENT);
    CPPUNIT_ASSERT(fragmentKey.run_ == runNumber);
    CPPUNIT_ASSERT(fragmentKey.event_ == eventNumber);
    CPPUNIT_ASSERT(fragmentKey.secondaryId_ == outputModuleId);
    CPPUNIT_ASSERT(fragmentKey.originatorPid_ == value2);
    CPPUNIT_ASSERT(fragmentKey.originatorGuid_ == value3);

    CPPUNIT_ASSERT(eventMsgChain.outputModuleId() == outputModuleId);
    CPPUNIT_ASSERT(eventMsgChain.hltTriggerCount() == hltBitCount);

    std::vector<unsigned char> hltBits2;
    eventMsgChain.hltTriggerBits(hltBits2);
    CPPUNIT_ASSERT(hltBits2.size() == hltBits.size());

    uint32_t trigIndex = 0;
    for (uint32_t idx = 0; idx < hltBits.size(); ++idx)
      {
        for (uint32_t jdx = 0; jdx < 4; ++jdx)
          {
            uint32_t indexMod = (trigIndex % 4);
            uint32_t trigMask = 0;
            switch (indexMod)
              {
              case 0:
                {
                  trigMask = 0x03;
                  break;
                }
              case 1:
                {
                  trigMask = 0x0c;
                  break;
                }
              case 2:
                {
                  trigMask = 0x30;
                  break;
                }
              case 3:
                {
                  trigMask = 0xc0;
                  break;
                }
              }
            CPPUNIT_ASSERT((hltBits2[idx] & trigMask) ==
                           (hltBits[idx] & trigMask));
            ++trigIndex;
          }
      }

    CPPUNIT_ASSERT(eventMsgChain.headerSize() == eventBuilder.headerSize());
    CPPUNIT_ASSERT(eventMsgChain.headerLocation() !=
                   eventMsgChain.dataLocation(0));

    unsigned char* headerLoc = eventMsgChain.headerLocation();
    for (uint32_t idx = 0; idx < eventMsgChain.headerSize(); ++idx)
      {
        CPPUNIT_ASSERT(headerLoc[idx] == tmpBuffer[idx]);
      }
  }
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testI2OChain);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
