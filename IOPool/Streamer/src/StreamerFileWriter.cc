#include "IOPool/Streamer/src/StreamerFileWriter.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "IOPool/Streamer/interface/InitMessage.h"

#include <iostream>
#include <vector>
#include <string>

using namespace edm;
using namespace std;

namespace edm
{
StreamerFileWriter::StreamerFileWriter(edm::ParameterSet const& ps):
  stream_writer_(new StreamerOutputFile(ps.template getParameter<std::string>("fileName"))),
  index_writer_(new StreamerOutputIndexFile(
                    ps.template getParameter<std::string>("indexFileName"))),
  hltCount_(0)
  {
  }

StreamerFileWriter::~StreamerFileWriter()
  {
    //Write the EOF Record Both at the end of Streamer file and Index file
    uint32 dummyStatusCode = 1234;

    stream_writer_->writeEOF(dummyStatusCode, hltStats_);
    index_writer_->writeEOF(dummyStatusCode, hltStats_);
  }

void StreamerFileWriter::stop() 
  {
    // call method in stream_writer_ and index_writer_ to
    // close their respective files here instead of in
    // destructor in case they need to throw?
  }

void StreamerFileWriter::doOutputHeader(std::auto_ptr<InitMsgBuilder> init_message)
  {

    //Write the Init Message to Streamer file
    stream_writer_->write(*init_message); 

    uint32 magic = 22;
    uint64 reserved = 666;
    index_writer_->writeIndexFileHeader(magic, reserved);
    index_writer_->write(*init_message);

    //Let us get some information for HLT
    //This will be required for HLT Stats
    InitMsgView view(init_message->startAddress()); 
    //HLT Count
    hltCount_ = view.get_hlt_bit_cnt(); 

    //Initialize the HLT Stat vector with all ZEROs
    for(uint32 i = 0; i != hltCount_; ++i ) 
       hltStats_.push_back(0);
  }

void StreamerFileWriter::doOutputEvent(std::auto_ptr<EventMsgBuilder> msg)
  {
    //Write the Event Message to Streamer file
    uint64 event_offset = stream_writer_->write(*msg);
    
    index_writer_->write(*msg, event_offset);
    
    // Lets update HLT Stat, know how many 
    // Events for which Trigger are being written
    
    //get the HLT Packed bytes

    EventMsgView eview(msg->startAddress()); 
    std::vector<uint8> packedHlt;
    uint32 hlt_sz=0;
    if (hltCount_ != 0) hlt_sz = 1+ ((hltCount_-1)/4); 
    packedHlt.resize(hlt_sz);
    eview.hltTriggerBits(&packedHlt[0]);
    updateHLTStats(packedHlt); 
  }


void StreamerFileWriter::updateHLTStats(std::vector<uint8> const& packedHlt)
   {
    unsigned int packInOneByte = 4;
    unsigned char testAgaint = 0x01;
    for(unsigned int i=0; i !=hltCount_; ++i)
    {
       unsigned int whichByte = i/packInOneByte;
       unsigned int indxWithinByte = i % packInOneByte;
       if ( ( testAgaint << (2 * indxWithinByte)) & (packedHlt.at(whichByte)) )
          {
           hltStats_[i]++;
          }
       //else  cout <<"Bit "<<i<<" is not set"<<endl;
    }
   }
} //namespace edm



