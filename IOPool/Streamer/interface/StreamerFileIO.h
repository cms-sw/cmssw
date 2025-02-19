#ifndef IOPool_Streamer_StreamerFileIO_h
#define IOPool_Streamer_StreamerFileIO_h

/**
This file contains Class definitions for the 
Class representing Output (Streamer) file.
*/

#include "IOPool/Streamer/interface/MsgTools.h"
#include "boost/shared_ptr.hpp"
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
     uint64 first_event_offset() const { return first_event_offset_; }
     uint64 last_event_offset()  const { return last_event_offset_; }
     uint64 events()             const { return events_; }
     uint64 run()                const { return run_; }
     uint32 adler32()            const { return (adlerb_ << 16) | adlera_; }

     void set_do_adler(bool v)             { do_adler_ = v; }
     void set_current_offset(uint64 v)     { current_offset_ = v; }
     void set_first_event_offset(uint64 v) { first_event_offset_ = v; }
     void set_last_event_offset(uint64 v)  { last_event_offset_ = v; }
     void set_events(uint64 v)             { events_ = v; }
     void inc_events()                     { ++events_; }
     void set_run(uint64 v)                { run_ = v; }

  private:
     uint64 current_offset_;  /** Location of current ioptr */
     uint64 first_event_offset_;
     uint64 last_event_offset_;
     uint32 events_;
     uint32 run_;

     bool   do_adler_;
     uint32 adlera_;
     uint32 adlerb_;

     boost::shared_ptr<std::ofstream> ost_;
     std::string filename_; 
  };

#endif
