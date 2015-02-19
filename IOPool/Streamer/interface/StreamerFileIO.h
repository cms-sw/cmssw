#ifndef IOPool_Streamer_StreamerFileIO_h
#define IOPool_Streamer_StreamerFileIO_h

/**
This file contains Class definitions for the 
Class representing Output (Streamer) file.
*/

#include "IOPool/Streamer/interface/MsgTools.h"
#include <memory>
#include <iosfwd>
#include <string>

//-------------------------------------------------------
  
class OutputFile 
  /**
  Class representing Output (Streamer) file.
  */
  {
  public:
     explicit OutputFile(const std::string& name);
     /**
      CTOR, takes file path name as argument
     */
     ~OutputFile();

     bool write(const char *ptr, size_t n);

     std::string fileName()      const { return filename_; }
     uint64 current_offset()     const { return current_offset_; }
     uint32 adler32()            const { return (adlerb_ << 16) | adlera_; }

     void set_do_adler(bool v)             { do_adler_ = v; }
     void set_current_offset(uint64 v)     { current_offset_ = v; }

  private:
     uint64 current_offset_;  /** Location of current ioptr */

     bool   do_adler_;
     uint32 adlera_;
     uint32 adlerb_;

     std::shared_ptr<std::ofstream> ost_;
     std::string filename_; 
  };

#endif
