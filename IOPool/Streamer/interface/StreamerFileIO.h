/**
This file contains Class definitions for the 
Class representing Output (Streamer/Index) file.
*/

#ifndef _StreamerFileIO_h
#define _StreamerFileIO_h

#include "IOPool/Streamer/interface/MsgTools.h"

#include <iosfwd>
#include<string>

//-------------------------------------------------------
  
  class OutputFile 
  /**
  Class representing Output (Streamer/Index) file.
  */
  {
  public:
     explicit OutputFile(const std::string& name);
     /**
      CTOR, takes file path name as argument
     */
     ~OutputFile();

      std::ofstream* ost() {return ost_;}
      std::string fileName() const {return filename_;}

      uint64 current_offset_;  /** Location of current ioptr */
      uint64 first_event_offset_;
      uint64 last_event_offset_;
      uint32 events_;
      uint32 run_;

   private:
     std::ofstream* ost_;
     std::string filename_; 
  };

#endif

