#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "FWCore/Utilities/interface/Exception.h"

StreamerOutputFile::~StreamerOutputFile() {}

StreamerOutputFile::StreamerOutputFile(const std::string& name, uint32 padding)
    : streamerfile_(std::make_shared<edm::streamer::OutputFile>(name, padding)) {
  streamerfile_->set_do_adler(true);
}

uint64 StreamerOutputFile::write(const EventMsgBuilder& ineview) {
  EventMsgView tmpView(ineview.startAddress());
  return write(tmpView);
}

uint64 StreamerOutputFile::write(const EventMsgView& ineview) {
  /** Offset where current event starts */
  uint64 offset_to_return = streamerfile_->current_offset();

  writeEventHeader(ineview);
  bool ret = streamerfile_->write((const char*)ineview.eventData(), ineview.size() - ineview.headerSize(), true);
  if (ret) {
    throw cms::Exception("OutputFile", "write(EventMsgView)")
        << "Error writing streamer event data to " << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
  }
  return offset_to_return;
}

uint64 StreamerOutputFile::writeEventFragment(uint32 fragIndex, uint32 fragCount, const char* dataPtr, uint32 dataSize) {
  /** Offset where current event starts */
  uint64 offset_to_return = streamerfile_->current_offset();

  bool ret = streamerfile_->write(dataPtr, dataSize);
  if (ret) {
    throw cms::Exception("OutputFile", "writeEventFragment()")
        << "Error writing streamer event data to " << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
  }
  return offset_to_return;
}

void StreamerOutputFile::writeEventHeader(const EventMsgView& ineview) {
  bool ret = streamerfile_->write((const char*)ineview.startAddress(), ineview.headerSize());
  if (ret) {
    throw cms::Exception("OutputFile", "writeEventHeader")
        << "Error writing streamer event data to " << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
  }
}

void StreamerOutputFile::write(const InitMsgBuilder& inview) {
  InitMsgView tmpView(inview.startAddress());
  return write(tmpView);
}

void StreamerOutputFile::write(const InitMsgView& inview) {
  writeStart(inview);
  bool ret = streamerfile_->write((const char*)inview.descData(), inview.size() - inview.headerSize(), true);
  if (ret) {
    throw cms::Exception("OutputFile", "write(InitMsgView)")
        << "Error writing streamer header data to " << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
  }
}

void StreamerOutputFile::writeInitFragment(uint32 fragIndex, uint32 fragCount, const char* dataPtr, uint32 dataSize) {
  bool ret = streamerfile_->write((const char*)dataPtr, dataSize);
  if (ret) {
    throw cms::Exception("OutputFile", "writeInitFragment()")
        << "Error writing streamer header data to " << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
  }
}

void StreamerOutputFile::writeStart(const InitMsgView& inview) {
  bool ret = streamerfile_->write((const char*)inview.startAddress(), inview.headerSize());
  if (ret) {
    throw cms::Exception("OutputFile", "writeStart")
        << "Error writing streamer header data to " << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
  }
}
