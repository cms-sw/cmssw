#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include <fstream>
#include <iostream>
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"

  OutputFile::OutputFile(const std::string& name):
    current_offset_(1), 
    do_adler_(0),
    adlera_(1),
    adlerb_(0),
    ost_(new std::ofstream(name.c_str(), std::ios_base::binary | std::ios_base::out)),
    filename_(name)
  {
    if(!ost_->is_open()) {
      throw cms::Exception("OutputFile","OutputFile")
        << "Error Opening Output File: "<<name<<"\n";
    }
    ost_->rdbuf()->pubsetbuf(0,0);
  }

  OutputFile::~OutputFile() 
  {
    ost_->close();
  }

  bool OutputFile::write(const char *ptr, size_t n) 
  {
    ost_->write(ptr,n);
    if(!ost_->fail()) {
      current_offset_ += (uint64)(n);
      if (do_adler_)
        cms::Adler32(ptr,n,adlera_,adlerb_);
      return 0;
    }
    return 1;
  }

