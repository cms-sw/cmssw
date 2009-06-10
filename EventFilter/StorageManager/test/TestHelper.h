// $Id$

#ifndef StorageManager_TestHelper_h
#define StorageManager_TestHelper_h

///////////////////////////////////////////////////
// Collection of helper function for test suites //
///////////////////////////////////////////////////

#include "DataFormats/Common/interface/HLTenums.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"

#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/exception/Exception.h"

#include "zlib.h"

using toolbox::mem::Pool;
using toolbox::mem::MemoryPoolFactory;
using toolbox::mem::getMemoryPoolFactory;
using toolbox::mem::HeapAllocator;
using toolbox::mem::Reference;

// There seems to be no sane way to create and destroy some of these
// toolbox entities, so we use globals. valgrind complains about
// leaked resources, but attempts to clean up resources (and to not
// use global resources) does not remove the leaks.
namespace
{
  MemoryPoolFactory*  g_factory(getMemoryPoolFactory());
  toolbox::net::URN   g_urn("toolbox-mem-pool","myPool");
  HeapAllocator* g_alloc(new HeapAllocator);
  Pool*          g_pool(g_factory->createPool(g_urn, g_alloc));
}


namespace stor
{
  namespace testhelper
  {

    // Allocate a new frame from the (global) Pool.
    Reference*
    allocate_frame()
    {
      const int bufferSize = 1024;
      Reference* temp = g_factory->getFrame(g_pool, bufferSize);
      assert(temp);
      
      unsigned char* tmpPtr = static_cast<unsigned char*>(temp->getDataLocation());
      for (int idx = 0; idx < bufferSize; ++idx)
      {
        tmpPtr[idx] = 0;
      }
      
      return temp;
    }
    
    
    Reference*
    allocate_frame_with_basic_header
    (
      unsigned short code,
      unsigned int frameIndex,
      unsigned int totalFrameCount
    )
    {
      const int bufferSize = 1024;
      Reference* temp = g_factory->getFrame(g_pool, bufferSize);
      assert(temp);
      
      unsigned char* tmpPtr = static_cast<unsigned char*>(temp->getDataLocation());
      for (int idx = 0; idx < bufferSize; ++idx)
      {
        tmpPtr[idx] = 0;
      }
      
      I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
        (I2O_PRIVATE_MESSAGE_FRAME*) temp->getDataLocation();
      I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
        (I2O_SM_MULTIPART_MESSAGE_FRAME*) pvtMsg;
      pvtMsg->StdMessageFrame.MessageSize = bufferSize / 4;
      pvtMsg->XFunctionCode = code;
      smMsg->numFrames = totalFrameCount;
      smMsg->frameCount = frameIndex;
      smMsg->dataSize = bufferSize - sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME);
      
      return temp;
    }
    
    Reference*
    allocate_frame_with_sample_header
    (
      unsigned int frameIndex,
      unsigned int totalFrameCount,
      unsigned int rbBufferId
    )
    {
      unsigned int value1 = 0xa5a5d2d2;
      unsigned int value2 = 0xb4b4e1e1;
      unsigned int value3 = 0xc3c3f0f0;
      unsigned int value4 = 0x12345678;
      
      const int bufferSize = 1024;
      Reference* temp = g_factory->getFrame(g_pool, bufferSize);
      assert(temp);
      
      unsigned char* tmpPtr = static_cast<unsigned char*>(temp->getDataLocation());
      for (int idx = 0; idx < bufferSize; ++idx)
      {
        tmpPtr[idx] = 0;
      }
      
      I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
        (I2O_PRIVATE_MESSAGE_FRAME*) temp->getDataLocation();
      I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
        (I2O_SM_PREAMBLE_MESSAGE_FRAME*) pvtMsg;
      pvtMsg->StdMessageFrame.MessageSize = bufferSize / 4;
      pvtMsg->XFunctionCode = I2O_SM_PREAMBLE;
      smMsg->numFrames = totalFrameCount;
      smMsg->frameCount = frameIndex;
      smMsg->hltTid = value1;
      smMsg->rbBufferID = rbBufferId;
      smMsg->outModID = value2;
      smMsg->fuProcID = value3;
      smMsg->fuGUID = value4;
      smMsg->dataSize = bufferSize - sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME);
      
      return temp;
    }
    

    Reference*
    allocate_frame_with_init_msg
    (
      std::string requestedOMLabel
    )
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

      char reltag[]="CMSSW_3_0_0_pre7";
      std::string processName = "HLT";
      std::string outputModuleLabel = requestedOMLabel;

      uLong crc = crc32(0L, Z_NULL, 0);
      Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
      unsigned int outputModuleId =
        crc32(crc,crcbuf,outputModuleLabel.length());

      unsigned int value1 = 0xa5a5d2d2;
      unsigned int value2 = 0xb4b4e1e1;
      unsigned int value3 = 0xc3c3f0f0;

      Reference* ref = allocate_frame_with_basic_header(I2O_SM_PREAMBLE, 0, 1);
      I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
        (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref->getDataLocation();
      smMsg->hltTid = value1;
      smMsg->rbBufferID = 2;
      smMsg->outModID = outputModuleId;
      smMsg->fuProcID = value2;
      smMsg->fuGUID = value3;

      InitMsgBuilder
        initBuilder(smMsg->dataPtr(), smMsg->dataSize, 100,
                    Version(7,(const uint8*)psetid), (const char*) reltag,
                    processName.c_str(), outputModuleLabel.c_str(),
                    outputModuleId, hlt_names, hlt_selections, l1_names);

      return ref;
    }
    

    Reference*
    allocate_frame_with_event_msg
    (
      std::string requestedOMLabel,
      std::vector<unsigned char>& hltBits,
      unsigned int hltBitCount,
      unsigned int eventNumber
    )
    {
      std::vector<bool> l1Bits;
      l1Bits.push_back(true);
      l1Bits.push_back(false);

      unsigned int runNumber = 100;
      unsigned int lumiNumber = 1;

      std::string outputModuleLabel = requestedOMLabel;
      uLong crc = crc32(0L, Z_NULL, 0);
      Bytef* crcbuf = (Bytef*) outputModuleLabel.data();
      unsigned int outputModuleId =
        crc32(crc,crcbuf,outputModuleLabel.length());

      unsigned int value1 = 0xa5a5d2d2;
      unsigned int value2 = 0xb4b4e1e1;
      unsigned int value3 = 0xc3c3f0f0;

      Reference* ref = allocate_frame_with_basic_header(I2O_SM_DATA, 0, 1);
      I2O_SM_DATA_MESSAGE_FRAME *smEventMsg =
        (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
      smEventMsg->hltTid = value1;
      smEventMsg->rbBufferID = 3;
      smEventMsg->runID = runNumber;
      smEventMsg->eventID = eventNumber;
      smEventMsg->outModID = outputModuleId;
      smEventMsg->fuProcID = value2;
      smEventMsg->fuGUID = value3;

      EventMsgBuilder
        eventBuilder(smEventMsg->dataPtr(), smEventMsg->dataSize, runNumber,
                     eventNumber, lumiNumber, outputModuleId,
                     l1Bits, &hltBits[0], hltBitCount);

      return ref;
    }


    Reference*
    allocate_frame_with_error_msg
    (
      unsigned int eventNumber
    )
    {
      unsigned int runNumber = 100;

      unsigned int value1 = 0xa5a5d2d2;
      unsigned int value2 = 0xb4b4e1e1;
      unsigned int value3 = 0xc3c3f0f0;

      Reference* ref = allocate_frame_with_basic_header(I2O_SM_ERROR, 0, 1);
      I2O_SM_DATA_MESSAGE_FRAME *smEventMsg =
        (I2O_SM_DATA_MESSAGE_FRAME*) ref->getDataLocation();
      smEventMsg->hltTid = value1;
      smEventMsg->rbBufferID = 3;
      smEventMsg->runID = runNumber;
      smEventMsg->eventID = eventNumber;
      smEventMsg->outModID = 0xffffffff;
      smEventMsg->fuProcID = value2;
      smEventMsg->fuGUID = value3;

      return ref;
    }


    Reference* allocate_frame_with_dqm_msg( unsigned int eventNumber,
                                            const std::string& topFolder )
    {

      unsigned int run = 1111;
      edm::Timestamp ts;
      unsigned int lumi_section = 1;
      unsigned int update_number = 1;
      std::string release_tag( "v00" );
      DQMEvent::TObjectTable mon_elts;

      Reference* ref = allocate_frame_with_basic_header( I2O_SM_DQM, 0, 1 );

      I2O_SM_DQM_MESSAGE_FRAME* msg = (I2O_SM_DQM_MESSAGE_FRAME*)ref->getDataLocation();

      DQMEventMsgBuilder b( (void*)(msg->dataPtr()), msg->dataSize, run, eventNumber,
                            ts,
                            lumi_section, update_number,
                            release_tag,
                            topFolder,
                            mon_elts );

      return ref;

    }


    void
    set_trigger_bit
    (
      std::vector<unsigned char>& hltBits,
      uint32 bitIndex,
      edm::hlt::HLTState pathStatus
    )
    {
      // ensure that bit vector is large enough
      uint32 minBitCount = bitIndex + 1;
      uint32 minSize = 1 + ((minBitCount - 1) / 4);
      if (hltBits.size() < minSize) hltBits.resize(minSize);

      uint32 vectorIndex = (uint32) (bitIndex / 4);
      uint32 shiftCount = 2 * (bitIndex % 4);

      uint32 clearMask = 0xff;
      clearMask ^= 0x3 << shiftCount;

      hltBits[vectorIndex] &= clearMask;
      hltBits[vectorIndex] |= pathStatus << shiftCount;
    }


    void
    clear_trigger_bits
    (
      std::vector<unsigned char>& hltBits
    )
    {
      for (unsigned int idx = 0; idx < hltBits.size(); ++idx)
        {
          hltBits[idx] = 0;
        }
    }


    // Return the number of bytes currently allocated out of the
    // (global) Pool.
    size_t
    outstanding_bytes()
    {
      return g_pool->getMemoryUsage().getUsed();
    }

  } // namespace testhelper
} // namespace stor

#endif //StorageManager_TestHelper_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
