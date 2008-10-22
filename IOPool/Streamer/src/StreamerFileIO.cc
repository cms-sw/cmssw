#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include <fstream>
#include <iostream>
#include "FWCore/Utilities/interface/Exception.h"

  OutputFile::OutputFile(const std::string& name):
    current_offset_(1), 
    first_event_offset_(0),
    last_event_offset_(0),
    events_(0),
    run_(0),
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
    delete ost_;
  }

  bool OutputFile::write(const char *ptr, size_t n) 
  {
    ost_->write(ptr,n);
    if(!ost_->fail()) {
      current_offset_ += (uint64)(n);
      if (do_adler_)
        adler32(ptr,n,adlera_,adlerb_);
      return 0;
    }
    return 1;
  }

//-------------------------------------------------------
// the following is adapted from 
// http://en.wikipedia.org/wiki/Adler-32
//-------------------------------------------------------

void OutputFile::adler32(const char *data, size_t len, uint32 &a, uint32 &b)
{
 /* data: Pointer to the data to be summed; len is in bytes */

  #define MOD_ADLER 65521
 
  const unsigned char *ptr = (const unsigned char *)data;
  while (len > 0) 
  {
    size_t tlen = len > 5552 ? 5552 : len;
    len -= tlen;
    do 
    {
      a += *ptr++;
      b += a;
    } while (--tlen);
    
    a %= MOD_ADLER;
    b %= MOD_ADLER;
  }

  #undef MOD_ADLER
}

uint32 OutputFile::adler32(const char *data, size_t len)
{
 /* data: Pointer to the data to be summed; len is in bytes */

  uint32_t a = 1, b = 0;
  adler32(data,len,a,b);
  return (b << 16) | a;
}
