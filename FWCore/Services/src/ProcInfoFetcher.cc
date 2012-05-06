// -*- C++ -*-
//
// Package:     Services
// Class  :     ProcInfoFetcher
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Sun May  6 11:14:31 CDT 2012
// $Id$
//

// system include files

// user include files

#include <cstring>
#include <cassert>
#include <iostream>
#ifdef __linux__
#include <malloc.h>
#endif
#include <sstream>
//#include <stdio.h>
#include <string>
#include <boost/lexical_cast.hpp>

#include <fcntl.h>
#include <unistd.h>

#include "FWCore/Services/src/ProcInfoFetcher.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

namespace { 
  struct linux_proc {
    int pid; // %d
    std::string comm;
    char state; // %c
    int ppid; // %d
    int pgrp; // %d
    int session; // %d
    int tty; // %d
    int tpgid; // %d
    unsigned long flags; // %lu
    unsigned long minflt; // %lu
    unsigned long cminflt; // %lu
    unsigned long majflt; // %lu
    unsigned long cmajflt; // %lu
    unsigned long utime; // %lu
    unsigned long stime; // %lu
    long cutime; // %ld
    long cstime; // %ld
    long priority; // %ld
    long nice; // %ld
    long num_threads; // %ld
    long itrealvalue; // %ld
    int starttime; // %d
    unsigned long vsize; // %lu
    long rss; // %ld
    unsigned long rlim; // %lu
    unsigned long startcode; // %lu
    unsigned long endcode; // %lu
    unsigned long startstack; // %lu
    unsigned long kstkesp; // %lu
    unsigned long kstkeip; // %lu
    unsigned long signal; // %lu
    unsigned long blocked; // %lu
    unsigned long sigignore; // %lu
    unsigned long sigcatch; // %lu
    unsigned long wchan; // %lu
  };
  
  class Fetcher {
  public:
    friend Fetcher& operator>>(Fetcher&, int&);
    friend Fetcher& operator>>(Fetcher&, long&);
    friend Fetcher& operator>>(Fetcher&, unsigned int&);
    friend Fetcher& operator>>(Fetcher&, unsigned long&);
    friend Fetcher& operator>>(Fetcher&, char&);
    friend Fetcher& operator>>(Fetcher&, std::string&);
    
    explicit Fetcher(char* buffer) : 
    buffer_(buffer),
    save_(0),
    delims_(" \t\n\f\v\r") {
    }
  private:
    int getInt() {
      const char* t = getItem();
      //std::cout <<"int '"<<t <<"'"<<std::endl;
      return boost::lexical_cast<int>(t);
    }
    long getLong() {
      const char* t = getItem();
      //std::cout <<"long '"<<t <<"'"<<std::endl;
      return boost::lexical_cast<long>(t);
    }
    unsigned int getUInt() {
      const char* t = getItem();
      //std::cout <<"uint '"<<t <<"'"<<std::endl;
      return boost::lexical_cast<unsigned int>(t);
    }
    unsigned long getULong() {
      const char* t = getItem();
      //std::cout <<"ulong '"<<t <<"'"<<std::endl;
      return boost::lexical_cast<unsigned long>(t);
    }
    char getChar() {
      return *getItem();
    }
    std::string getString() {
      return std::string(getItem());
    }
    char* getItem() {
      char* item = strtok_r(buffer_, delims_, &save_); 
      assert(item);
      buffer_ = 0; // Null for subsequent strtok_r calls.
      return item;
    }
    char* buffer_;
    char* save_;
    char const* const delims_; 
  };
  
  Fetcher& operator>>(Fetcher& iFetch, int& oValue) {
    oValue = iFetch.getInt();
    return iFetch;
  }
  Fetcher& operator>>(Fetcher& iFetch, long& oValue) {
    oValue = iFetch.getLong();
    return iFetch;
  }
  Fetcher& operator>>(Fetcher& iFetch, unsigned int& oValue) {
    oValue = iFetch.getUInt();
    return iFetch;      
  }
  Fetcher& operator>>(Fetcher& iFetch, unsigned long& oValue) {
    oValue = iFetch.getULong();
    return iFetch;      
  }
  Fetcher& operator>>(Fetcher& iFetch, char& oValue) {
    oValue = iFetch.getChar();
    return iFetch;
  }
  Fetcher& operator>>(Fetcher& iFetch, std::string& oValue) {
    oValue = iFetch.getString();
    return iFetch;
  }
}

namespace edm {
  namespace service {
  
    ProcInfoFetcher::ProcInfoFetcher():
    pg_size_(sysconf(_SC_PAGESIZE)) {
#ifdef __linux__
      std::ostringstream ost;
      ost << "/proc/" << getpid() << "/stat";
      
      if((fd_ = open(ost.str().c_str(), O_RDONLY)) < 0) {
        throw Exception(errors::Configuration)
        << "Failed to open " << ost.str() << std::endl;
      }
#endif
    }
    ProcInfoFetcher::~ProcInfoFetcher() {
#ifdef LINUX
      close(fd_);
#endif      
    }
    ProcInfo ProcInfoFetcher::fetch() const {
      ProcInfo ret;
      
#ifdef __linux__
      double pr_size = 0.0, pr_rssize = 0.0;
      
      linux_proc pinfo;
      int cnt;
      
      lseek(fd_, 0, SEEK_SET);
      
      if((cnt = read(fd_, buf_, sizeof(buf_) - 1)) < 0) {
        perror("Read of Proc file failed:");
        return ProcInfo();
      }
      
      if(cnt > 0) {
        buf_[cnt] = '\0';
        
        
        try {
          Fetcher fetcher(buf_);
          fetcher >> pinfo.pid
          >> pinfo.comm
          >> pinfo.state
          >> pinfo.ppid
          >> pinfo.pgrp
          >> pinfo.session
          >> pinfo.tty
          >> pinfo.tpgid
          >> pinfo.flags
          >> pinfo.minflt
          >> pinfo.cminflt
          >> pinfo.majflt
          >> pinfo.cmajflt
          >> pinfo.utime
          >> pinfo.stime
          >> pinfo.cutime
          >> pinfo.cstime
          >> pinfo.priority
          >> pinfo.nice
          >> pinfo.num_threads
          >> pinfo.itrealvalue
          >> pinfo.starttime
          >> pinfo.vsize
          >> pinfo.rss
          >> pinfo.rlim
          >> pinfo.startcode
          >> pinfo.endcode
          >> pinfo.startstack
          >> pinfo.kstkesp
          >> pinfo.kstkeip
          >> pinfo.signal
          >> pinfo.blocked
          >> pinfo.sigignore
          >> pinfo.sigcatch
          >> pinfo.wchan;
        } catch (boost::bad_lexical_cast& iE) {
          LogWarning("ProcInfoFetcher")<<"Parsing of Prof file failed:"<<iE.what()<<std::endl;
          return ProcInfo();
        }
        
        // resident set size in pages
        pr_size = (double)pinfo.vsize;
        pr_rssize = (double)pinfo.rss;
        
        ret.vsize = pr_size / (1024.0*1024.0);
        ret.rss   = (pr_rssize * pg_size_) / (1024.0*1024.0);
      }
#else
      ret.vsize = 0;
      ret.rss = 0;
#endif
      return ret;
    }
  }
}