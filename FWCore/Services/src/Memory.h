#ifndef FWCore_Services_Memory_h
#define FWCore_Services_Memory_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     SimpleMemoryCheck
// 
//
// Original Author:  Jim Kowalkowski
// $Id: Memory.h,v 1.10 2011/07/22 20:42:38 chrjones Exp $
//
// Change Log
//
// 1 - Apr 25, 2008 M. Fischler
//	Data structures for Event summary statistics, 
//
// 2 - Jan 14, 2009 Natalia Garcia Nebot
//      Added:  - Average rate of growth in RSS and peak value attained.
//              - Average rate of growth in VSize over time, Peak VSize


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include <cstdio>

namespace edm {
  class EventID;
  class Timestamp;
  class Event;
  class EventSetup;
  class ModuleDescription;
  class ConfigurationDescriptions;

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

    struct smapsInfo
    {
      smapsInfo():private_(),pss_() {}
      smapsInfo(double private_sz, double pss_sz): private_(private_sz),pss_(pss_sz) {}
      
      bool operator==(const smapsInfo& p) const
      { return private_==p.private_ && pss_==p.pss_; }
      
      bool operator>(const smapsInfo& p) const
      { return private_>p.private_ || pss_>p.pss_; }
      
      double private_;   // in MB
      double pss_;     // in MB
    };

    
    class SimpleMemoryCheck
    {
    public:

      SimpleMemoryCheck(const ParameterSet&,ActivityRegistry&);
      ~SimpleMemoryCheck();

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

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

      void postFork(unsigned int, unsigned int);
    private:
      procInfo fetch();
      smapsInfo fetchSmaps();
      double pageSize() const { return pg_size_; }
      double averageGrowthRate(double current, double past, int count);
      void update();
      void updateMax();
      void andPrint(const std::string& type,
                const std::string& mdlabel, const std::string& mdname) const;
      void updateAndPrint(const std::string& type,
                        const std::string& mdlabel, const std::string& mdname);
      void openFiles();
      
      procInfo a_;
      procInfo b_;
      procInfo max_;
      procInfo* current_;
      procInfo* previous_;
      
      smapsInfo currentSmaps_;
      
      char buf_[500];
      int fd_;
      std::string fname_;
      double pg_size_;
      int num_to_skip_;
      //options
      bool showMallocInfo_;
      bool oncePerEventMode_;
      bool jobReportOutputOnly_;
      bool monitorPssAndPrivate_;
      int count_;

      //smaps
      FILE* smapsFile_;
      char* smapsLineBuffer_;
      size_t smapsLineBufferLen_;

      
      //Rates of growth
      double growthRateVsize_;
      double growthRateRss_;

      // Event summary statistics 				changeLog 1
      struct SignificantEvent {
        int count;
        double vsize;
        double deltaVsize;
        double rss;
        double deltaRss;
        bool monitorPssAndPrivate;
        double privateSize;
        double pss;
        edm::EventID event;
        SignificantEvent() : count(0), vsize(0), deltaVsize(0), 
          rss(0), deltaRss(0), monitorPssAndPrivate(false), privateSize(0), pss(0),event() {}
        void set (double deltaV, double deltaR, 
                  edm::EventID const & e, SimpleMemoryCheck *t)
        { count = t->count_;
          vsize = t->current_->vsize;
          deltaVsize = deltaV;
          rss = t->current_->rss;
          deltaRss = deltaR;
          monitorPssAndPrivate = t->monitorPssAndPrivate_;
          if (monitorPssAndPrivate) {
            privateSize = t->currentSmaps_.private_;
            pss = t->currentSmaps_.pss_;
          }
          event = e;
        }
      }; // SignificantEvent
      friend struct SignificantEvent;
      friend std::ostream & operator<< (std::ostream & os, 
    		SimpleMemoryCheck::SignificantEvent const & se); 

      /* 
        Significative events for deltaVsize:
	- eventM_: Event which makes the biggest value for deltaVsize
	- eventL1_: Event which makes the second biggest value for deltaVsize
	- eventL2_: Event which makes the third biggest value for deltaVsize
	- eventR1_: Event which makes the second biggest value for deltaVsize
	- eventR2_: Event which makes the third biggest value for deltaVsize
	M>L1>L2 and M>R1>R2
		Unknown relation between Ls and Rs events ???????
        Significative events for vsize:
	- eventT1_: Event whith the biggest value for vsize
        - eventT2_: Event whith the second biggest value for vsize
        - eventT3_: Event whith the third biggest value for vsize
	T1>T2>T3
       */
      SignificantEvent eventM_;  
      SignificantEvent eventL1_; 
      SignificantEvent eventL2_; 
      SignificantEvent eventR1_; 
      SignificantEvent eventR2_; 
      SignificantEvent eventT1_; 
      SignificantEvent eventT2_; 
      SignificantEvent eventT3_; 

      /*
	Significative event for deltaRss:
        - eventRssT1_: Event whith the biggest value for rss
        - eventRssT2_: Event whith the second biggest value for rss
        - eventRssT3_: Event whith the third biggest value for rss
	T1>T2>T3
	Significative events for deltaRss:
        - eventDeltaRssT1_: Event whith the biggest value for deltaRss
        - eventDeltaRssT2_: Event whith the second biggest value for deltaRss
        - eventDeltaRssT3_: Event whith the third biggest value for deltaRss
        T1>T2>T3
       */
      SignificantEvent eventRssT1_;
      SignificantEvent eventRssT2_;
      SignificantEvent eventRssT3_;
      SignificantEvent eventDeltaRssT1_;
      SignificantEvent eventDeltaRssT2_;
      SignificantEvent eventDeltaRssT3_;


      void updateEventStats(edm::EventID const & e);
      std::string eventStatOutput(std::string title, 
      				  SignificantEvent const& e) const;
      void eventStatOutput(std::string title, 
    			   SignificantEvent const& e,
			   std::map<std::string, std::string> &m) const;
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
      friend struct SignificantModule;
      friend std::ostream & operator<< (std::ostream & os, 
    		SimpleMemoryCheck::SignificantModule const & se); 
      bool moduleSummaryRequested_;
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
