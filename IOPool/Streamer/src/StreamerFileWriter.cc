#include "IOPool/Streamer/src/StreamerFileWriter.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  StreamerFileWriter::StreamerFileWriter(edm::ParameterSet const& ps) :
    stream_writer_(new StreamerOutputFile(
                      ps.getUntrackedParameter<std::string>("fileName")))
  {
  }

  StreamerFileWriter::StreamerFileWriter(std::string const& fileName) :
    stream_writer_(new StreamerOutputFile(fileName))
  {
  }

  StreamerFileWriter::~StreamerFileWriter() {
  }

  void StreamerFileWriter::doOutputHeader(InitMsgBuilder const& init_message) {
    //Let us turn it into a View
    InitMsgView view(init_message.startAddress());
    doOutputHeader(view);
  }

  void StreamerFileWriter::doOutputHeader(InitMsgView const& init_message) {
    //Write the Init Message to Streamer file
    stream_writer_->write(init_message);
  }

  void StreamerFileWriter::doOutputEvent(EventMsgView const& msg) {
    //Write the Event Message to Streamer file
    stream_writer_->write(msg);
  }

  void StreamerFileWriter::doOutputEvent(EventMsgBuilder const& msg) {
    EventMsgView eview(msg.startAddress());
    doOutputEvent(eview);
  }

  void StreamerFileWriter::fillDescription(ParameterSetDescription& desc) {
    desc.setComment("Writes events into a streamer output file.");
    desc.addUntracked<std::string>("fileName", "teststreamfile.dat")->setComment("Name of output file.");
  }
} //namespace edm
