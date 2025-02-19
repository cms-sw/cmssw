// $Id: TestHelper.h,v 1.10 2012/04/05 12:38:38 mommsen Exp $

#ifndef StorageManager_TestHelper_h
#define StorageManager_TestHelper_h

///////////////////////////////////////////////////
// Collection of helper function for test suites //
///////////////////////////////////////////////////

#include <fstream>
#include <sstream>
#include <string>

#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/exception/Exception.h"

#include "zlib.h"


// There seems to be no sane way to create and destroy some of these
// toolbox entities, so we use globals. valgrind complains about
// leaked resources, but attempts to clean up resources (and to not
// use global resources) does not remove the leaks.
namespace
{
  using toolbox::mem::Pool;
  using toolbox::mem::MemoryPoolFactory;
  using toolbox::mem::getMemoryPoolFactory;
  using toolbox::mem::HeapAllocator;
  MemoryPoolFactory*  g_factory(getMemoryPoolFactory());
  toolbox::net::URN   g_urn("toolbox-mem-pool","myPool");
  HeapAllocator* g_alloc(new HeapAllocator);
  Pool*          g_pool(g_factory->createPool(g_urn, g_alloc));
}


namespace stor
{
  namespace testhelper
  {
    using toolbox::mem::Reference;

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

      switch(code)
      {
        case I2O_SM_PREAMBLE:
        {
          char psetid[] = "1234567890123456";
          I2O_SM_PREAMBLE_MESSAGE_FRAME* initMsg =
            (I2O_SM_PREAMBLE_MESSAGE_FRAME*) smMsg;
          InitHeader* h = (InitHeader*)initMsg->dataPtr();
          new (&h->version_) Version((const uint8*)psetid);
          h->header_.code_ = Header::INIT;
          break;
        }

        case I2O_SM_DATA:
        {
          I2O_SM_DATA_MESSAGE_FRAME* eventMsg =
            (I2O_SM_DATA_MESSAGE_FRAME*) smMsg;
          EventHeader* h = (EventHeader*)eventMsg->dataPtr();
          h->protocolVersion_ = 9;
          h->header_.code_ = Header::EVENT;
          break;
        }

        case I2O_SM_DQM:
        {
          I2O_SM_DQM_MESSAGE_FRAME* dqmMsg =
            (I2O_SM_DQM_MESSAGE_FRAME*) smMsg;
          DQMEventHeader* h = (DQMEventHeader*)dqmMsg->dataPtr();
          convert(static_cast<uint16>(3), h->protocolVersion_);
          h->header_.code_ = Header::DQM_EVENT;
          break;
        }
      }

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
      char psetid[] = "1234567890123456";
      InitHeader* h = (InitHeader*)smMsg->dataPtr();
      new (&h->version_) Version((const uint8*)psetid);
      h->header_.code_ = Header::INIT;
      
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

      char test_value[] = "This is a test, This is a";
      uint32_t adler32_chksum = (uint32_t)cms::Adler32((char*)&test_value[0], sizeof(test_value));
      char host_name[255];
      gethostname(host_name, sizeof(host_name));

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

      char test_value_event[] = "This is a test Event, This is a";
      uint32_t adler32_chksum = (uint32_t)cms::Adler32((char*)&test_value_event[0], sizeof(test_value_event));
      char host_name[255];
      gethostname(host_name, sizeof(host_name));

      EventMsgBuilder
        eventBuilder(smEventMsg->dataPtr(), smEventMsg->dataSize, runNumber,
                     eventNumber, lumiNumber, outputModuleId, 0,
                     l1Bits, &hltBits[0], hltBitCount, adler32_chksum, host_name);

      eventBuilder.setOrigDataSize(78); // no compression
      eventBuilder.setEventLength(sizeof(test_value_event));
      std::copy(&test_value_event[0],&test_value_event[0]+sizeof(test_value_event),
                eventBuilder.eventAddr());
      smEventMsg->dataSize = eventBuilder.headerSize() + sizeof(test_value_event);

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

      // no data yet to get a checksum (not needed for test)
      uint32_t adler32_chksum = 0;
      char host_name[255];
      gethostname(host_name, sizeof(host_name));

      DQMEventMsgBuilder b( (void*)(msg->dataPtr()), msg->dataSize, run, eventNumber,
                            ts,
                            lumi_section, update_number,
                            adler32_chksum,
                            host_name,
                            release_tag,
                            topFolder,
                            mon_elts );

      return ref;

    }


    void
    set_trigger_bit
    (
      std::vector<unsigned char>& hltBits,
      uint32_t bitIndex,
      edm::hlt::HLTState pathStatus
    )
    {
      // ensure that bit vector is large enough
      uint32_t minBitCount = bitIndex + 1;
      uint32_t minSize = 1 + ((minBitCount - 1) / 4);
      if (hltBits.size() < minSize) hltBits.resize(minSize);

      uint32_t vectorIndex = (uint32_t) (bitIndex / 4);
      uint32_t shiftCount = 2 * (bitIndex % 4);

      uint32_t clearMask = 0xff;
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


    // Fills the string with content of file.
    // Returns false if an error occurred
    bool
    read_file(const std::string& filename, std::string& content)
    {
      std::ifstream in(filename.c_str());
      if (!in.is_open())
        return false;
      
      std::string line;
      while (std::getline(in, line))
        content.append(line);

      if (!in.eof())
        return false;

      return true;
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
