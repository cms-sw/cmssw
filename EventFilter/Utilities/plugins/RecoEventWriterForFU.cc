#include "RecoEventWriterForFU.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace evf {
  RecoEventWriterForFU::RecoEventWriterForFU(edm::ParameterSet const& ps) {
  }

  RecoEventWriterForFU::~RecoEventWriterForFU() {
  }

  void RecoEventWriterForFU::doOutputHeader(InitMsgBuilder const& init_message) {
    //Let us turn it into a View
    InitMsgView view(init_message.startAddress());
    doOutputHeader(view);
  }

  void RecoEventWriterForFU::doOutputHeader(InitMsgView const& init_message) {
    //Write the Init Message to init file and close it
    if ( stream_writer_preamble_.get() ) {
      stream_writer_preamble_->write(init_message);
      preamble_adler32_ = stream_writer_preamble_->adler32();
      stream_writer_preamble_.reset();
    }
  }

  void RecoEventWriterForFU::doOutputEvent(EventMsgView const& msg) {
    //Write the Event Message to Streamer file
    stream_writer_events_->write(msg);
  }

  void RecoEventWriterForFU::doOutputEvent(EventMsgBuilder const& msg) {
    EventMsgView eview(msg.startAddress());
    doOutputEvent(eview);
  }

  void RecoEventWriterForFU::fillDescription(edm::ParameterSetDescription& desc) {
    desc.setComment("Writes events into a streamer output file.");
    desc.addUntracked<std::string>("fileName", "teststreamfile.dat")->setComment("Name of output file.");
  }

  void RecoEventWriterForFU::setInitMessageFile(std::string const& init){
    stream_writer_preamble_.reset(new StreamerOutputFile(init));
    preamble_adler32_ = 1;
  }

  void RecoEventWriterForFU::setOutputFile(std::string const& events){
    stream_writer_events_.reset(new StreamerOutputFile(events));
  }

  void RecoEventWriterForFU::closeOutputFile(){
    stream_writer_events_.reset();
  }

} //namespace edm
