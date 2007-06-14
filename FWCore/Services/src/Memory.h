#ifndef FWCore_Services_Memory_h
#define FWCore_Services_Memory_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     SimpleMemoryCheck
// 
//
// Original Author:  Jim Kowalkowski
// $Id: Memory.h,v 1.1 2006/01/30 05:09:24 jbk Exp $
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  class EventID;
  class Timestamp;
  class Event;
  class EventSetup;
  class ModuleDescription;

  namespace service {
    struct procInfo
    {
      procInfo():vsize(),rss() {}
      procInfo(double sz, double rss_sz): vsize(sz),rss(rss_sz) {}
	
      bool operator==(const procInfo& p) const
      { return vsize==p.vsize && rss==p.rss; }

      bool operator>(const procInfo& p) const
      { return vsize>p.vsize || rss>p.rss; }

      // see proc(4) man pages for units and a description
      double vsize;   // in MB (used to be in pages?)
      double rss;     // in MB (used to be in pages?)
    };

    class SimpleMemoryCheck
    {
    public:

      SimpleMemoryCheck(const ParameterSet&,ActivityRegistry&);
      ~SimpleMemoryCheck();
      
      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const Event&, const EventSetup&);
      
      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);

    private:
      procInfo fetch();
      double pageSize() const { return pg_size_; }
	
      procInfo a_;
      procInfo b_;
      procInfo max_;
      procInfo* current_;
      procInfo* previous_;
      
      char buf_[500];
      int fd_;
      std::string fname_;
      double pg_size_;
      int num_to_skip_;
      int count_;
    };
  }
}



#endif
