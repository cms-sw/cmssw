#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

namespace edm::streamer {
  OutputFile::OutputFile(const std::string& name, uint32 padding)
      : current_offset_(1),
        do_adler_(false),
        adlera_(1),
        adlerb_(0),
        padding_(padding),
        ost_(new std::ofstream(name.c_str(), std::ios_base::binary | std::ios_base::out)),
        filename_(name) {
    if (!ost_->is_open()) {
      throw cms::Exception("OutputFile", "OutputFile") << "Error Opening Output File: " << name << "\n";
    }
    ost_->rdbuf()->pubsetbuf(nullptr, 0);
    if (padding_) {
      paddingBuf_ = std::make_unique<char[]>(padding_);
      memset(paddingBuf_.get(), Header::PADDING, padding_);
    }
  }

  OutputFile::~OutputFile() { ost_->close(); }

  bool OutputFile::write(const char* ptr, size_t n, bool doPadding) {
    ost_->write(ptr, n);
    if (!ost_->fail()) {
      current_offset_ += (uint64)(n);
      if (do_adler_)
        cms::Adler32(ptr, n, adlera_, adlerb_);
      if (doPadding && padding_) {
        return writePadding();
      }
      return false;
    }
    return true;
  }

  bool OutputFile::writePadding() {
    uint64 mod = ost_->tellp() % padding_;
    if (mod) {
      uint32 rem = padding_ - (uint32)(mod % padding_);
      bool ret = write(paddingBuf_.get(), rem, false);
      return ret;
    }
    return false;
  }

  void OutputFile::close() {
    if (padding_)
      if (writePadding())
        throw cms::Exception("OutputFile", "OutputFile")
            << "Error writing padding to the output file: " << filename_ << ": " << std::strerror(errno);
    ost_->close();
  }
}  // namespace edm::streamer
