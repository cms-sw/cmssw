// $Id: StreamerFileWriter.cc,v 1.17.14.2 2009/04/01 20:09:41 biery Exp $

#include "IOPool/Streamer/src/StreamerFileWriter.h"

namespace edm
{
StreamerFileWriter::StreamerFileWriter(edm::ParameterSet const& ps):
  stream_writer_(new StreamerOutputFile(
                    ps.getUntrackedParameter<std::string>("fileName","teststreamfile.dat"))),
  index_writer_(new StreamerOutputIndexFile(
                    ps.getUntrackedParameter<std::string>("indexFileName","testindexfile.ind"))),
  hltCount_(0),
  index_eof_size_(0),
  stream_eof_size_(0)
  {
  }


StreamerFileWriter::StreamerFileWriter(std::string const& fileName, std::string const& indexFileName):
  stream_writer_(new StreamerOutputFile(fileName)),
  index_writer_(new StreamerOutputIndexFile(indexFileName)),
  hltCount_(0),
  index_eof_size_(0),
  stream_eof_size_(0)
  {
  }


StreamerFileWriter::~StreamerFileWriter()
  {
  }

void StreamerFileWriter::stop() 
  {

    // User code of this class MUST call method

    //Write the EOF Record Both at the end of Streamer file and Index file
    uint32 dummyStatusCode = 1234;
    
    index_eof_size_ = index_writer_->writeEOF(dummyStatusCode, hltStats_);
    stream_eof_size_ = stream_writer_->writeEOF(dummyStatusCode, hltStats_);
    
  }

void StreamerFileWriter::doOutputHeader(InitMsgBuilder const& init_message)
  {

    //Let us turn it into a View
    InitMsgView view(init_message.startAddress());
    doOutputHeader(view);

  }
void StreamerFileWriter::doOutputHeader(InitMsgView const& init_message)
  {

    //Write the Init Message to Streamer file
    stream_writer_->write(init_message);

    uint32 magic = 22;
    uint64 reserved = 666;
    index_writer_->writeIndexFileHeader(magic, reserved);
    index_writer_->write(init_message);

    //HLT Count
    hltCount_ = init_message.get_hlt_bit_cnt();

    //Initialize the HLT Stat vector with all ZEROs
    for(uint32 i = 0; i != hltCount_; ++i)
       hltStats_.push_back(0);
  }

void StreamerFileWriter::
doOutputHeaderFragment(StreamerFileWriterHeaderParams const& hdrParams)
  {
    //Write the Init Message to Streamer file
    stream_writer_->writeInitFragment(hdrParams.fragmentIndex,
                                      hdrParams.fragmentCount,
                                      hdrParams.dataPtr,
                                      hdrParams.dataSize);

    if (hdrParams.fragmentIndex == 0)
    {
      uint32 magic = 22;
      uint64 reserved = 666;
      index_writer_->writeIndexFileHeader(magic, reserved);
      index_writer_->writeInit(hdrParams.runNumber, hdrParams.headerPtr,
                               hdrParams.headerSize);

      //HLT Count
      hltCount_ = hdrParams.hltCount;

      //Initialize the HLT Stat vector with all ZEROs
      for(uint32 i = 0; i != hltCount_; ++i)
        hltStats_.push_back(0);
    }
  }


void StreamerFileWriter::doOutputEvent(EventMsgView const& msg)
  {
    //Write the Event Message to Streamer file
    uint64 event_offset = stream_writer_->write(msg);
    
    index_writer_->write(msg, event_offset);
    
    // Lets update HLT Stat, know how many 
    // Events for which Trigger are being written
    
    //get the HLT Packed bytes
    std::vector<uint8> packedHlt;
    uint32 hlt_sz = 0;
    if (hltCount_ != 0) hlt_sz = 1 + ((hltCount_-1)/4); 
    packedHlt.resize(hlt_sz);
    msg.hltTriggerBits(&packedHlt[0]);
    updateHLTStats(packedHlt); 
  }


void StreamerFileWriter::doOutputEvent(EventMsgBuilder const& msg)
  {

    EventMsgView eview(msg.startAddress());
    doOutputEvent(eview);
 
  }

void StreamerFileWriter::
doOutputEventFragment(StreamerFileWriterEventParams const& evtParams)
  {
    //Write the Event Message to Streamer file
    uint64 event_offset =
      stream_writer_->writeEventFragment(evtParams.fragmentIndex,
                                         evtParams.fragmentCount,
                                         evtParams.dataPtr,
                                         evtParams.dataSize);

    if (evtParams.fragmentIndex == 0)
    {
      index_writer_->writeEvent(evtParams.headerPtr, evtParams.headerSize,
                                event_offset);

      // Lets update HLT Stat, know how many 
      // Events for which Trigger are being written

      updateHLTStats(evtParams.hltBits);
    }
  }


void StreamerFileWriter::updateHLTStats(std::vector<uint8> const& packedHlt)
   {
    unsigned int packInOneByte = 4;
    unsigned char testAgaint = 0x01;
    for(unsigned int i = 0; i != hltCount_; ++i)
    {
       unsigned int whichByte = i/packInOneByte;
       unsigned int indxWithinByte = i % packInOneByte;
       if ((testAgaint << (2 * indxWithinByte)) & (packedHlt.at(whichByte))) {
           ++hltStats_[i];
       }
       //else  std::cout <<"Bit "<<i<<" is not set"<< std::endl;
    }
   }
} //namespace edm



