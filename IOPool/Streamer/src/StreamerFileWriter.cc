#include "IOPool/Streamer/src/StreamerFileWriter.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  StreamerFileWriter::StreamerFileWriter(edm::ParameterSet const& ps) :
    stream_writer_(new StreamerOutputFile(
                      ps.getUntrackedParameter<std::string>("fileName"))),
    hltCount_(0),
    stream_eof_size_(0) {
  }

  StreamerFileWriter::StreamerFileWriter(std::string const& fileName) :
    stream_writer_(new StreamerOutputFile(fileName)),
    hltCount_(0),
    stream_eof_size_(0) {
  }

  StreamerFileWriter::~StreamerFileWriter() {
  }

  void StreamerFileWriter::stop() {

    // User code of this class MUST call method

    //Write the EOF Record Both at the end of Streamer file
    uint32 const dummyStatusCode = 1234;

    stream_eof_size_ = stream_writer_->writeEOF(dummyStatusCode, hltStats_);
  }

  void StreamerFileWriter::doOutputHeader(InitMsgBuilder const& init_message) {
    //Let us turn it into a View
    InitMsgView view(init_message.startAddress());
    doOutputHeader(view);
  }

  void StreamerFileWriter::doOutputHeader(InitMsgView const& init_message) {
    //Write the Init Message to Streamer file
    stream_writer_->write(init_message);

    //HLT Count
    hltCount_ = init_message.get_hlt_bit_cnt();

    //Initialize the HLT Stat vector with all ZEROs
    for(uint32 i = 0; i != hltCount_; ++i) {
       hltStats_.push_back(0);
    }
  }

  void StreamerFileWriter::doOutputHeaderFragment(StreamerFileWriterHeaderParams const& hdrParams) {
    //Write the Init Message to Streamer file
    stream_writer_->writeInitFragment(hdrParams.fragmentIndex,
                                        hdrParams.fragmentCount,
                                        hdrParams.dataPtr,
                                        hdrParams.dataSize);

    if (hdrParams.fragmentIndex == 0) {
      //HLT Count
      hltCount_ = hdrParams.hltCount;

      //Initialize the HLT Stat vector with all ZEROs
      for(uint32 i = 0; i != hltCount_; ++i) {
        hltStats_.push_back(0);
      }
    }
  }

  void StreamerFileWriter::doOutputEvent(EventMsgView const& msg) {
    //Write the Event Message to Streamer file
    stream_writer_->write(msg);

    // Lets update HLT Stat, know how many
    // Events for which Trigger are being written

    //get the HLT Packed bytes
    std::vector<uint8> packedHlt;
    uint32 const hlt_sz = (hltCount_ != 0 ? 1 + ((hltCount_ - 1) / 4) : 0);
    packedHlt.resize(hlt_sz);
    msg.hltTriggerBits(&packedHlt[0]);
    updateHLTStats(packedHlt);
  }


  void StreamerFileWriter::doOutputEvent(EventMsgBuilder const& msg) {
    EventMsgView eview(msg.startAddress());
    doOutputEvent(eview);
  }

  void StreamerFileWriter::doOutputEventFragment(StreamerFileWriterEventParams const& evtParams) {
    //Write the Event Message to Streamer file
    stream_writer_->writeEventFragment(evtParams.fragmentIndex,
                                       evtParams.fragmentCount,
                                       evtParams.dataPtr,
                                       evtParams.dataSize);
    if (evtParams.fragmentIndex == 0) {
      // Lets update HLT Stat, know how many
      // Events for which Trigger are being written
      updateHLTStats(evtParams.hltBits);
    }
  }

  void StreamerFileWriter::updateHLTStats(std::vector<uint8> const& packedHlt) {
    unsigned int const packInOneByte = 4;
    unsigned char const testAgaint = 0x01;
    for(unsigned int i = 0; i != hltCount_; ++i) {
      unsigned int const whichByte = i/packInOneByte;
      unsigned int const indxWithinByte = i % packInOneByte;
      if ((testAgaint << (2 * indxWithinByte)) & (packedHlt.at(whichByte))) {
         ++hltStats_[i];
      }
      //else  std::cout <<"Bit "<<i<<" is not set"<< std::endl;
    }
  }

  void StreamerFileWriter::fillDescription(ParameterSetDescription& desc) {
    desc.addUntracked<std::string>("fileName", "teststreamfile.dat")->setComment("Name of output file.");
  }
} //namespace edm
