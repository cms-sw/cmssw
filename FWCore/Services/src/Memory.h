#ifndef FWCore_Services_Memory_h
#define FWCore_Services_Memory_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     SimpleMemoryCheck
// 
//
// Original Author:  Jim Kowalkowski
// $Id: Memory.h,v 1.5 2008/04/24 22:28:29 fischler Exp $
//
// Change Log
//
// 1 - Apr 25, 2008 M. Fischler
//	Data structures for Event summary statistics, 
//


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"

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
      
      void preSourceConstruction(const ModuleDescription&);
      void postSourceConstruction(const ModuleDescription&);
      void postSource();

      void postBeginJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const Event&, const EventSetup&);
      
      void postModuleBeginJob(const ModuleDescription&);
      void postModuleConstruction(const ModuleDescription&);

      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);

      void postEndJob();

    private:
      procInfo fetch();
      double pageSize() const { return pg_size_; }
      void update();
      void updateMax();
      void andPrint(const std::string& type,
                const std::string& mdlabel, const std::string& mdname) const;
      void updateAndPrint(const std::string& type,
                        const std::string& mdlabel, const std::string& mdname);

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
      //options
      bool showMallocInfo;
      bool oncePerEventMode;
      int count_;

      // Event summary statistics 				changeLog 1
      struct SignificantEvent {
        int count;
	double vsize;
	double deltaVsize;
	double rss;
	double deltaRss;
	edm::EventID event;
	SignificantEvent() : count(0), vsize(0), deltaVsize(0), 
			     rss(0), deltaRss(0), event() {}
	void set (double deltaV, double deltaR, 
		  edm::EventID const & e, SimpleMemoryCheck *t)
	{ count = t->count_;
	  vsize = t->current_->vsize;
	  deltaVsize = deltaV;
	  rss = t->current_->rss;
	  deltaRss = deltaR;
	  event = e;
	}
      }; // SignificantEvent
      friend class SignificantEvent;
      friend std::ostream & operator<< (std::ostream & os, 
    		SimpleMemoryCheck::SignificantEvent const & se); 
      SignificantEvent eventM_;  
      SignificantEvent eventL1_; 
      SignificantEvent eventL2_; 
      SignificantEvent eventR1_; 
      SignificantEvent eventR2_; 
      SignificantEvent eventT1_; 
      SignificantEvent eventT2_; 
      SignificantEvent eventT3_; 
      void updateEventStats(edm::EventID const & e);
      std::string eventStatOutput(std::string title, 
      				  SignificantEvent const& e) const;
      void eventStatOutput(std::string title, 
    			   SignificantEvent const& e,
			   std::map<std::string, double> &m) const;
      std::string mallOutput(std::string title, size_t const& n) const;

      // Module summary statistices
      struct SignificantModule {
        int    postEarlyCount;
	double totalDeltaVsize;
	double maxDeltaVsize;
	edm::EventID eventMaxDeltaV;
	double totalEarlyVsize;
	double maxEarlyVsize;
	SignificantModule() : postEarlyCount  (0)
			    , totalDeltaVsize (0)
			    , maxDeltaVsize   (0)
			    , eventMaxDeltaV  ()
			    , totalEarlyVsize (0)
			    , maxEarlyVsize   (0)     {}
	void set (double deltaV, bool early);
      }; // SignificantModule
      friend class SignificantModule;
      friend std::ostream & operator<< (std::ostream & os, 
    		SimpleMemoryCheck::SignificantModule const & se); 
      bool moduleSummaryRequested;
      typedef std::map<std::string,SignificantModule> SignificantModulesMap;
      SignificantModulesMap modules_;      
      double moduleEntryVsize_;
      edm::EventID currentEventID_;
      void updateModuleMemoryStats(SignificantModule & m, double dv);
                 
    }; // SimpleMemoryCheck
    
    std::ostream & 
    operator<< (std::ostream & os, 
    		SimpleMemoryCheck::SignificantEvent const & se); 
    
    std::ostream & 
    operator<< (std::ostream & os, 
    		SimpleMemoryCheck::SignificantModule const & se); 
    
  }
}



#endif
