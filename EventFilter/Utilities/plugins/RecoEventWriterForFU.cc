#include "RecoEventWriterForFU.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace evf {
  RecoEventWriterForFU::RecoEventWriterForFU(edm::ParameterSet const& ps) :
    stream_writer_preamble_(0),
    stream_writer_events_(0)
  {
  }

  RecoEventWriterForFU::~RecoEventWriterForFU() {
  }

  void RecoEventWriterForFU::doOutputHeader(InitMsgBuilder const& init_message) {
    //Let us turn it into a View
    InitMsgView view(init_message.startAddress());
    doOutputHeader(view);
  }

  void RecoEventWriterForFU::doOutputHeader(InitMsgView const& init_message) {
    //Write the Init Message to Streamer file
    stream_writer_preamble_->write(init_message);
  }

  void RecoEventWriterForFU::doOutputHeaderFragment(RecoEventWriterForFUHeaderParams const& hdrParams) {
    //Write the Init Message to Streamer file
    stream_writer_preamble_->writeInitFragment(hdrParams.fragmentIndex,
					       hdrParams.fragmentCount,
					       hdrParams.dataPtr,
					       hdrParams.dataSize);
  }

  void RecoEventWriterForFU::doOutputEvent(EventMsgView const& msg) {
    //Write the Event Message to Streamer file
    stream_writer_events_->write(msg);
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
  }

  void RecoEventWriterForFU::fillDescription(edm::ParameterSetDescription& desc) {
    desc.setComment("Writes events into a streamer output file.");
    desc.addUntracked<std::string>("fileName", "teststreamfile.dat")->setComment("Name of output file.");
  }
  void RecoEventWriterForFU::setOutputFiles(std::string &init, std::string &eof){
    stream_writer_preamble_.reset(new StreamerOutputFile(init));
    //    stream_writer_events_.reset(new StreamerOutputFile(events));
  }
  void RecoEventWriterForFU::setOutputFile(std::string &events){
    stream_writer_events_.reset(new StreamerOutputFile(events));
  }
} //namespace edm
