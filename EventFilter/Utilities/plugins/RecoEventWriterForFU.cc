#include "RecoEventWriterForFU.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace evf {
  RecoEventWriterForFU::RecoEventWriterForFU(edm::ParameterSet const& ps) :
    stream_writer_preamble_(0),
    stream_writer_postamble_(0),
    stream_writer_events_(0),
    hltCount_(0),
    stream_eof_size_(0) {
  }

  RecoEventWriterForFU::~RecoEventWriterForFU() {
  }

  void RecoEventWriterForFU::stop() {
    // User code of this class MUST call method

    //Write the EOF Record Both at the end of Streamer file
    uint32 const dummyStatusCode = 1234;

    stream_eof_size_ = stream_writer_postamble_->writeEOF(dummyStatusCode, hltStats_);
  }

  void RecoEventWriterForFU::doOutputHeader(InitMsgBuilder const& init_message) {
    //Let us turn it into a View
    InitMsgView view(init_message.startAddress());
    doOutputHeader(view);
  }

  void RecoEventWriterForFU::doOutputHeader(InitMsgView const& init_message) {
    //Write the Init Message to Streamer file
    stream_writer_preamble_->write(init_message);

    //HLT Count
    hltCount_ = init_message.get_hlt_bit_cnt();

    //Initialize the HLT Stat vector with all ZEROs
    for(uint32 i = 0; i != hltCount_; ++i) {
       hltStats_.push_back(0);
    }
  }

  void RecoEventWriterForFU::doOutputHeaderFragment(RecoEventWriterForFUHeaderParams const& hdrParams) {
    //Write the Init Message to Streamer file
    stream_writer_preamble_->writeInitFragment(hdrParams.fragmentIndex,
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

  void RecoEventWriterForFU::doOutputEvent(EventMsgView const& msg) {
    //Write the Event Message to Streamer file
    stream_writer_events_->write(msg);

    // Lets update HLT Stat, know how many
    // Events for which Trigger are being written

    //get the HLT Packed bytes
    std::vector<uint8> packedHlt;
    uint32 const hlt_sz = (hltCount_ != 0 ? 1 + ((hltCount_ - 1) / 4) : 0);
    packedHlt.resize(hlt_sz);
    msg.hltTriggerBits(&packedHlt[0]);
    updateHLTStats(packedHlt);
  }

  void RecoEventWriterForFU::doOutputEvent(EventMsgBuilder const& msg) {
    EventMsgView eview(msg.startAddress());
    doOutputEvent(eview);
  }

  void RecoEventWriterForFU::doOutputEventFragment(RecoEventWriterForFUEventParams const& evtParams) {
    //Write the Event Message to Streamer file
    stream_writer_events_->writeEventFragment(evtParams.fragmentIndex,
					      evtParams.fragmentCount,
					      evtParams.dataPtr,
					      evtParams.dataSize);
    if (evtParams.fragmentIndex == 0) {
      // Lets update HLT Stat, know how many
      // Events for which Trigger are being written
      updateHLTStats(evtParams.hltBits);
    }
  }

  void RecoEventWriterForFU::updateHLTStats(std::vector<uint8> const& packedHlt) {
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

  void RecoEventWriterForFU::fillDescription(edm::ParameterSetDescription& desc) {
    desc.setComment("Writes events into a streamer output file.");
    desc.addUntracked<std::string>("fileName", "teststreamfile.dat")->setComment("Name of output file.");
  }
  void RecoEventWriterForFU::setOutputFiles(std::string &init, std::string &eof){
    stream_writer_preamble_.reset(new StreamerOutputFile(init));
    stream_writer_postamble_.reset(new StreamerOutputFile(eof));
    //    stream_writer_events_.reset(new StreamerOutputFile(events));
  }
  void RecoEventWriterForFU::setOutputFile(std::string &events){
    stream_writer_events_.reset(new StreamerOutputFile(events));
  }
} //namespace edm
