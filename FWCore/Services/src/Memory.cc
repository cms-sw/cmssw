// -*- C++ -*-
//
// Package:     Services
// Class  :     Memory
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
//
// Change Log
//
// 1 - Apr 25, 2008 M. Fischler
//        Collect event summary information and output to XML file and logger
//        at the end of the job.  Involves split-up of updateAndPrint method.
//
// 2 - May 7, 2008 M. Fischler
//      Collect module summary information and output to XML file and logger
//        at the end of the job.
//
// 3 - Jan 14, 2009 Natalia Garcia Nebot
//        Added:        - Average rate of growth in RSS and peak value attained.
//                - Average rate of growth in VSize over time, Peak VSize
//
//

#include "FWCore/Services/src/Memory.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/MallocOpts.h"

#include <cstring>
#include <iostream>
#ifdef __linux__
#include <malloc.h>
#endif
#include <sstream>
//#include <stdio.h>
#include <string>
//#include <string.h>
#include <boost/lexical_cast.hpp>

#ifdef __linux__
#define LINUX 1
#endif

#include <fcntl.h>
#include <unistd.h>

namespace edm {
  namespace service {

    static std::string d2str(double d) {
      std::ostringstream t;
      t << d;
      return t.str();
    }

    static std::string i2str(int i) {
      std::ostringstream t;
      t << i;
      return t.str();
    }

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


    procInfo SimpleMemoryCheck::fetch() {
      procInfo ret;

#ifdef LINUX
      double pr_size = 0.0, pr_rssize = 0.0;

      linux_proc pinfo;
      int cnt;

      lseek(fd_, 0, SEEK_SET);

      if((cnt = read(fd_, buf_, sizeof(buf_) - 1)) < 0) {
        perror("Read of Proc file failed:");
        return procInfo();
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
          LogWarning("MemoryCheck")<<"Parsing of Prof file failed:"<<iE.what()<<std::endl;
          return procInfo();
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
    
    smapsInfo SimpleMemoryCheck::fetchSmaps() {
      smapsInfo ret;
      ret.private_ = 0;
      ret.pss_ = 0;
#ifdef LINUX
      fseek(smapsFile_, 0, SEEK_SET);
      ssize_t read;
      
      /*
       The format of the report is
       Private_Clean:        0 kB
       Private_Dirty:       72 kB
       Swap:                 0 kB
       Pss:                 72 kB
       */
      
      while ((read = getline(&smapsLineBuffer_, &smapsLineBufferLen_, smapsFile_)) != -1) {
        if(read > 14) {
          //Private
          if(0==strncmp("Private_",smapsLineBuffer_,8)) {
            unsigned int value = atoi(smapsLineBuffer_+14);
            //Convert from kB to MB
            ret.private_ += static_cast<double>(value)/1024.;
          } else if(0==strncmp("Pss:",smapsLineBuffer_,4)) {
            unsigned int value = atoi(smapsLineBuffer_+4);
            //Convert from kB to MB
            ret.pss_ += static_cast<double>(value)/1024.;            
          }
        }
      }
#endif
      return ret;
    }

    double SimpleMemoryCheck::averageGrowthRate(double current, double past, int count) {
      return(current-past)/(double)count;
    }

    SimpleMemoryCheck::SimpleMemoryCheck(ParameterSet const& iPS,
                                         ActivityRegistry&iReg)
    : a_()
    , b_()
    , current_(&a_)
    , previous_(&b_)
    , pg_size_(sysconf(_SC_PAGESIZE)) // getpagesize()
    , num_to_skip_(iPS.getUntrackedParameter<int>("ignoreTotal"))
    , showMallocInfo_(iPS.getUntrackedParameter<bool>("showMallocInfo"))
    , oncePerEventMode_(iPS.getUntrackedParameter<bool>("oncePerEventMode"))
    , jobReportOutputOnly_(iPS.getUntrackedParameter<bool>("jobReportOutputOnly"))
    , count_()
    , smapsFile_(0)
    , smapsLineBuffer_(NULL)
    , smapsLineBufferLen_(0)
    , growthRateVsize_()
    , growthRateRss_()
    , moduleSummaryRequested_(iPS.getUntrackedParameter<bool>("moduleMemorySummary")) {
                                                                // changelog 2
      // pg_size = (double)getpagesize();
      std::ostringstream ost;

      openFiles();
      iReg.watchPostForkReacquireResources(this,&SimpleMemoryCheck::postFork);
      
      if(!oncePerEventMode_) { // default, prints on increases
        iReg.watchPreSourceConstruction(this, &SimpleMemoryCheck::preSourceConstruction);
        iReg.watchPostSourceConstruction(this, &SimpleMemoryCheck::postSourceConstruction);
        iReg.watchPostSource(this, &SimpleMemoryCheck::postSource);
        iReg.watchPostModuleConstruction(this, &SimpleMemoryCheck::postModuleConstruction);
        iReg.watchPostModuleBeginJob(this, &SimpleMemoryCheck::postModuleBeginJob);
        iReg.watchPostProcessEvent(this, &SimpleMemoryCheck::postEventProcessing);
        iReg.watchPostModule(this, &SimpleMemoryCheck::postModule);
        iReg.watchPostBeginJob(this, &SimpleMemoryCheck::postBeginJob);
        iReg.watchPostEndJob(this, &SimpleMemoryCheck::postEndJob);
      } else {
        iReg.watchPostProcessEvent(this, &SimpleMemoryCheck::postEventProcessing);
        iReg.watchPostEndJob(this, &SimpleMemoryCheck::postEndJob);
      }
      if(moduleSummaryRequested_) {                                // changelog 2
        iReg.watchPreProcessEvent(this, &SimpleMemoryCheck::preEventProcessing);
        iReg.watchPreModule(this, &SimpleMemoryCheck::preModule);
        if(oncePerEventMode_) {
          iReg.watchPostModule(this, &SimpleMemoryCheck::postModule);
        }
      }

      // The following are not currenty used/implemented below for either
      // of the print modes (but are left here for reference)
      //  iReg.watchPostBeginJob(this,
      //       &SimpleMemoryCheck::postBeginJob);
      //  iReg.watchPreProcessEvent(this,
      //       &SimpleMemoryCheck::preEventProcessing);
      //  iReg.watchPreModule(this,
      //       &SimpleMemoryCheck::preModule);

      typedef MallocOpts::opt_type opt_type;
      MallocOptionSetter& mopts = getGlobalOptionSetter();

      opt_type
      p_mmap_max = iPS.getUntrackedParameter<int>("M_MMAP_MAX"),
      p_trim_thr = iPS.getUntrackedParameter<int>("M_TRIM_THRESHOLD"),
      p_top_pad = iPS.getUntrackedParameter<int>("M_TOP_PAD"),
      p_mmap_thr = iPS.getUntrackedParameter<int>("M_MMAP_THRESHOLD");

      if(p_mmap_max >= 0) mopts.set_mmap_max(p_mmap_max);
      if(p_trim_thr >= 0) mopts.set_trim_thr(p_trim_thr);
      if(p_top_pad >= 0) mopts.set_top_pad(p_top_pad);
      if(p_mmap_thr >= 0) mopts.set_mmap_thr(p_mmap_thr);

      mopts.adjustMallocParams();

      if(mopts.hasErrors()) {
        LogWarning("MemoryCheck")
        << "ERROR: Problem with setting malloc options\n"
        << mopts.error_message();
      }

      if(iPS.getUntrackedParameter<bool>("dump") == true) {
        MallocOpts mo = mopts.get();
        LogWarning("MemoryCheck")
        << "Malloc options: " << mo << "\n";
      }
    }

    SimpleMemoryCheck::~SimpleMemoryCheck() {
#ifdef LINUX
      close(fd_);
      if(0 != smapsFile_) {
        fclose(smapsFile_);
      }
#endif
      if (smapsLineBuffer_) {
        //getline will create the memory using malloc
        free(smapsLineBuffer_);
      }
    }

    void SimpleMemoryCheck::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<int>("ignoreTotal", 1);
      desc.addUntracked<bool>("showMallocInfo", false);
      desc.addUntracked<bool>("oncePerEventMode", false);
      desc.addUntracked<bool>("jobReportOutputOnly", false);
      desc.addUntracked<bool>("moduleMemorySummary", false);
      desc.addUntracked<int>("M_MMAP_MAX", -1);
      desc.addUntracked<int>("M_TRIM_THRESHOLD", -1);
      desc.addUntracked<int>("M_TOP_PAD", -1);
      desc.addUntracked<int>("M_MMAP_THRESHOLD", -1);
      desc.addUntracked<bool>("dump", false);
      descriptions.add("SimpleMemoryCheck", desc);
    }

    void SimpleMemoryCheck::openFiles() {
#ifdef LINUX
      std::ostringstream ost;
      ost << "/proc/" << getpid() << "/stat";
      fname_ = ost.str();
      
      if((fd_ = open(ost.str().c_str(), O_RDONLY)) < 0) {
        throw Exception(errors::Configuration)
        << "Memory checker server: Failed to open " << ost.str() << std::endl;
      }
      std::ostringstream smapsNameOst;
      smapsNameOst <<"/proc/"<<getpid()<<"/smaps";
      if((smapsFile_ =fopen(smapsNameOst.str().c_str(), "r"))==0) {
        throw Exception(errors::Configuration) <<"Failed to open smaps file "<<smapsNameOst.str()<<std::endl;
      }
#endif
    }
    
    void SimpleMemoryCheck::postBeginJob() {
        growthRateVsize_ = current_->vsize;
        growthRateRss_ = current_->rss;
    }

    void SimpleMemoryCheck::preSourceConstruction(ModuleDescription const& md) {
      updateAndPrint("pre-ctor", md.moduleLabel(), md.moduleName());
    }


    void SimpleMemoryCheck::postSourceConstruction(ModuleDescription const& md) {
      updateAndPrint("ctor", md.moduleLabel(), md.moduleName());
    }

    void SimpleMemoryCheck::postSource() {
      updateAndPrint("module", "source", "source");
    }

    void SimpleMemoryCheck::postModuleConstruction(ModuleDescription const& md) {
      updateAndPrint("ctor", md.moduleLabel(), md.moduleName());
    }

    void SimpleMemoryCheck::postModuleBeginJob(ModuleDescription const& md) {
      updateAndPrint("beginJob", md.moduleLabel(), md.moduleName());
    }

    void SimpleMemoryCheck::postEndJob() {
      if(not jobReportOutputOnly_) {
        LogAbsolute("MemoryReport")                                 // changelog 1
        << "MemoryReport> Peak virtual size " << eventT1_.vsize << " Mbytes"
        << "\n"
        << " Key events increasing vsize: \n"
        << eventL2_ << "\n"
        << eventL1_ << "\n"
        << eventM_  << "\n"
        << eventR1_ << "\n"
        << eventR2_ << "\n"
        << eventT3_ << "\n"
        << eventT2_ << "\n"
        << eventT1_ ;
      }
      if(moduleSummaryRequested_ and not jobReportOutputOnly_) {                                // changelog 1
        LogAbsolute mmr("ModuleMemoryReport"); // at end of if block, mmr
                                                    // is destructed, causing
                                                    // message to be logged
        mmr << "ModuleMemoryReport> Each line has module label and: \n";
        mmr << "  (after early ignored events) \n";
        mmr <<
        "    count of times module executed; average increase in vsize \n";
        mmr <<
        "    maximum increase in vsize; event on which maximum occurred \n";
        mmr << "  (during early ignored events) \n";
        mmr << "    total and maximum vsize increases \n \n";
        for(SignificantModulesMap::iterator im = modules_.begin();
            im != modules_.end(); ++im) {
          SignificantModule const& m = im->second;
          if(m.totalDeltaVsize == 0 && m.totalEarlyVsize == 0) continue;
          mmr << im->first << ": ";
          mmr << "n = " << m.postEarlyCount;
          if(m.postEarlyCount > 0) {
            mmr << " avg = " << m.totalDeltaVsize/m.postEarlyCount;
          }
          mmr <<  " max = " << m.maxDeltaVsize << " " << m.eventMaxDeltaV;
          if(m.totalEarlyVsize > 0) {
            mmr << " early total: " << m.totalEarlyVsize;
            mmr << " max: " << m.maxEarlyVsize;
          }
          mmr << "\n";
        }
      } // end of if; mmr goes out of scope; log message is queued

      Service<JobReport> reportSvc;
                                                                // changelog 1
#define SIMPLE_MEMORY_CHECK_ORIGINAL_XML_OUTPUT
#ifdef  SIMPLE_MEMORY_CHECK_ORIGINAL_XML_OUTPUT
//     std::map<std::string, double> reportData;
      std::map<std::string, std::string> reportData;

      if(eventL2_.vsize > 0)
              eventStatOutput("LargeVsizeIncreaseEventL2", eventL2_, reportData);
      if(eventL1_.vsize > 0)
              eventStatOutput("LargeVsizeIncreaseEventL1", eventL1_, reportData);
      if(eventM_.vsize > 0)
              eventStatOutput("LargestVsizeIncreaseEvent", eventM_,  reportData);
      if(eventR1_.vsize > 0)
              eventStatOutput("LargeVsizeIncreaseEventR1", eventR1_, reportData);
      if(eventR2_.vsize > 0)
              eventStatOutput("LargeVsizeIncreaseEventR2", eventR2_, reportData);
      if(eventT3_.vsize > 0)
              eventStatOutput("ThirdLargestVsizeEventT3",  eventT3_, reportData);
      if(eventT2_.vsize > 0)
              eventStatOutput("SecondLargestVsizeEventT2", eventT2_, reportData);
      if(eventT1_.vsize > 0)
              eventStatOutput("LargestVsizeEventT1",       eventT1_, reportData);

      if(eventRssT3_.rss > 0)
        eventStatOutput("ThirdLargestRssEvent", eventRssT3_, reportData);
      if(eventRssT2_.rss > 0)
        eventStatOutput("SecondLargestRssEvent", eventRssT2_, reportData);
      if(eventRssT1_.rss > 0)
        eventStatOutput("LargestRssEvent", eventRssT1_, reportData);
      if(eventDeltaRssT3_.deltaRss > 0)
        eventStatOutput("ThirdLargestIncreaseRssEvent", eventDeltaRssT3_, reportData);
      if(eventDeltaRssT2_.deltaRss > 0)
        eventStatOutput("SecondLargestIncreaseRssEvent", eventDeltaRssT2_, reportData);
      if(eventDeltaRssT1_.deltaRss > 0)
        eventStatOutput("LargestIncreaseRssEvent", eventDeltaRssT1_, reportData);

#ifdef __linux__
      struct mallinfo minfo = mallinfo();
      reportData.insert(
        std::make_pair("HEAP_ARENA_SIZE_BYTES", i2str(minfo.arena)));
      reportData.insert(
        std::make_pair("HEAP_ARENA_N_UNUSED_CHUNKS", i2str(minfo.ordblks)));
      reportData.insert(
        std::make_pair("HEAP_TOP_FREE_BYTES", i2str(minfo.keepcost)));
      reportData.insert(
        std::make_pair("HEAP_MAPPED_SIZE_BYTES", i2str(minfo.hblkhd)));
      reportData.insert(
        std::make_pair("HEAP_MAPPED_N_CHUNKS", i2str(minfo.hblks)));
      reportData.insert(
        std::make_pair("HEAP_USED_BYTES", i2str(minfo.uordblks)));
      reportData.insert(
        std::make_pair("HEAP_UNUSED_BYTES", i2str(minfo.fordblks)));
#endif

      // Report Growth rates for VSize and Rss
      reportData.insert(
        std::make_pair("AverageGrowthRateVsize", d2str(averageGrowthRate(current_->vsize, growthRateVsize_, count_))));
      reportData.insert(
        std::make_pair("PeakValueVsize", d2str(eventT1_.vsize)));
      reportData.insert(
        std::make_pair("AverageGrowthRateRss", d2str(averageGrowthRate(current_->rss, growthRateRss_, count_))));
      reportData.insert(
        std::make_pair("PeakValueRss", d2str(eventRssT1_.rss)));

      if(moduleSummaryRequested_) {                                // changelog 2
        for(SignificantModulesMap::iterator im = modules_.begin();
            im != modules_.end(); ++im) {
          SignificantModule const& m = im->second;
          if(m.totalDeltaVsize == 0 && m.totalEarlyVsize == 0) continue;
          std::string label = im->first+":";
          reportData.insert(std::make_pair(label+"PostEarlyCount", i2str(m.postEarlyCount)));
          if(m.postEarlyCount > 0) {
            reportData.insert(std::make_pair(label+"AverageDeltaVsize",
                                             d2str(m.totalDeltaVsize/m.postEarlyCount)));
          }
                reportData.insert(std::make_pair(label+"MaxDeltaVsize", d2str(m.maxDeltaVsize)));
          if(m.totalEarlyVsize > 0) {
            reportData.insert(std::make_pair(label+"TotalEarlyVsize", d2str(m.totalEarlyVsize)));
            reportData.insert(std::make_pair(label+"MaxEarlyDeltaVsize", d2str(m.maxEarlyVsize)));
          }
        }
      }

      std::map<std::string, std::string> reportMemoryProperties;

      if(FILE* fmeminfo = fopen("/proc/meminfo", "r")) {
        char buf[128];
        char space[] = " ";
        size_t value;

        while(fgets(buf, sizeof(buf), fmeminfo)) {
          char* token = NULL;
          token = strtok(buf, space);
          if(token != NULL) {
            value = atol(strtok(NULL, space));
            std::string category = token;
            reportMemoryProperties.insert(std::make_pair(category.substr(0, strlen(token)-1), i2str(value)));
          }
        }

        fclose(fmeminfo);
      }

//      reportSvc->reportMemoryInfo(reportData, reportMemoryProperties);
      reportSvc->reportPerformanceSummary("ApplicationMemory", reportData);
      reportSvc->reportPerformanceSummary("SystemMemory", reportMemoryProperties);
#endif

#ifdef SIMPLE_MEMORY_CHECK_DIFFERENT_XML_OUTPUT
      std::vector<std::string> reportData;

      if(eventL2_.vsize > 0) reportData.push_back(
              eventStatOutput("LargeVsizeIncreaseEventL2", eventL2_));
      if(eventL1_.vsize > 0) reportData.push_back(
              eventStatOutput("LargeVsizeIncreaseEventL1", eventL1_));
      if(eventM_.vsize > 0) reportData.push_back(
              eventStatOutput("LargestVsizeIncreaseEvent", eventM_));
      if(eventR1_.vsize > 0) reportData.push_back(
              eventStatOutput("LargeVsizeIncreaseEventR1", eventR1_));
      if(eventR2_.vsize > 0) reportData.push_back(
              eventStatOutput("LargeVsizeIncreaseEventR2", eventR2_));
      if(eventT3_.vsize > 0) reportData.push_back(
              eventStatOutput("ThirdLargestVsizeEventT3", eventT3_));
      if(eventT2_.vsize > 0) reportData.push_back(
              eventStatOutput("SecondLargestVsizeEventT2", eventT2_));
      if(eventT1_.vsize > 0) reportData.push_back(
              eventStatOutput("LargestVsizeEventT1", eventT1_));

      if(eventRssT3_.rss > 0)
        eventStatOutput("ThirdLargestRssEvent", eventRssT3_, reportData);
      if(eventRssT2_.rss > 0)
        eventStatOutput("SecondLargestRssEvent", eventRssT2_, reportData);
      if(eventRssT1_.rss > 0)
        eventStatOutput("LargestRssEvent", eventRssT1_, reportData);
      if(eventDeltaRssT3_.deltaRss > 0)
        eventStatOutput("ThirdLargestIncreaseRssEvent", eventDeltaRssT3_, reportData);
      if(eventDeltaRssT2_.deltaRss > 0)
        eventStatOutput("SecondLargestIncreaseRssEvent", eventDeltaRssT2_, reportData);
      if(eventDeltaRssT1_.deltaRss > 0)
        eventStatOutput("LargestIncreaseRssEvent", eventDeltaRssT1_, reportData);

      struct mallinfo minfo = mallinfo();
      reportData.push_back(
        mallOutput("HEAP_ARENA_SIZE_BYTES", minfo.arena));
      reportData.push_back(
        mallOutput("HEAP_ARENA_N_UNUSED_CHUNKS", minfo.ordblks));
      reportData.push_back(
        mallOutput("HEAP_TOP_FREE_BYTES", minfo.keepcost));
      reportData.push_back(
        mallOutput("HEAP_MAPPED_SIZE_BYTES", minfo.hblkhd));
      reportData.push_back(
        mallOutput("HEAP_MAPPED_N_CHUNKS", minfo.hblks));
      reportData.push_back(
        mallOutput("HEAP_USED_BYTES", minfo.uordblks));
      reportData.push_back(
        mallOutput("HEAP_UNUSED_BYTES", minfo.fordblks));

      // Report Growth rates for VSize and Rss
      reportData.insert(
        std::make_pair("AverageGrowthRateVsize", d2str(averageGrowthRate(current_->vsize, growthRateVsize_, count_))));
      reportData.insert(
        std::make_pair("PeakValueVsize", d2str(eventT1_.vsize)));
      reportData.insert(
        std::make_pair("AverageGrowthRateRss", d2str(averageGrowthRate(current_->rss, growthRateRss_, count_))));
      reportData.insert(
        std::make_pair("PeakValueRss", d2str(eventRssT1_.rss)));

      reportSvc->reportMemoryInfo(reportData);
      // This is a form of reportMemoryInfo taking s vector, not a map
#endif
    } // postEndJob

    void SimpleMemoryCheck::preEventProcessing(EventID const& iID, Timestamp const&) {
      currentEventID_ = iID;                                        // changelog 2
    }

    void SimpleMemoryCheck::postEventProcessing(Event const& e, EventSetup const&) {
      ++count_;
      update();
      currentSmaps_ = fetchSmaps();
      updateEventStats(e.id());
      if(oncePerEventMode_) {
        // should probably use be Run:Event or count_ for the label and name
        updateMax();
        andPrint("event", "", "");
      }
    }

    void SimpleMemoryCheck::preModule(ModuleDescription const&) {
      update();                                                        // changelog 2
      moduleEntryVsize_ = current_->vsize;
    }

    void SimpleMemoryCheck::postModule(ModuleDescription const& md) {
      if(!oncePerEventMode_) {
        updateAndPrint("module", md.moduleLabel(), md.moduleName());
      } else if(moduleSummaryRequested_) {                        // changelog 2
        update();
      }
      if(moduleSummaryRequested_) {                                // changelog 2
        double dv = current_->vsize - moduleEntryVsize_;
        std::string label =  md.moduleLabel();
        updateModuleMemoryStats (modules_[label], dv);
      }
    }

    void SimpleMemoryCheck::postFork(unsigned int, unsigned int) {
#ifdef LINUX
      close(fd_);
      fclose(smapsFile_);
      openFiles();
#endif      
    }


    void SimpleMemoryCheck::update() {
      std::swap(current_, previous_);
      *current_ = fetch();
    }

    void SimpleMemoryCheck::updateMax() {
      if((*current_ > max_) || oncePerEventMode_) {
          if(count_ >= num_to_skip_) {
          }
          max_ = *current_;
        }
    }

    void SimpleMemoryCheck::updateEventStats(EventID const& e) {
      if(count_ < num_to_skip_) return;
      if(count_ == num_to_skip_) {
        eventT1_.set(0, 0, e, this);
        eventM_.set (0, 0, e, this);
        eventRssT1_.set(0, 0, e, this);
        eventDeltaRssT1_.set(0, 0, e, this);
        return;
      }
      double vsize = current_->vsize;
      double deltaVsize = vsize - eventT1_.vsize;

      // Update significative events for Vsize
      if(vsize > eventT1_.vsize) {
        double deltaRss = current_->rss - eventT1_.rss;
        eventT3_ = eventT2_;
        eventT2_ = eventT1_;
        eventT1_.set(deltaVsize, deltaRss, e, this);
      } else if(vsize > eventT2_.vsize) {
        double deltaRss = current_->rss - eventT1_.rss;
        eventT3_ = eventT2_;
        eventT2_.set(deltaVsize, deltaRss, e, this);
      } else if(vsize > eventT3_.vsize) {
        double deltaRss = current_->rss - eventT1_.rss;
        eventT3_.set(deltaVsize, deltaRss, e, this);
      }

      if(deltaVsize > eventM_.deltaVsize) {
        double deltaRss = current_->rss - eventM_.rss;
        if(eventL1_.deltaVsize >= eventR1_.deltaVsize) {
          eventL2_ = eventL1_;
        } else {
          eventL2_ = eventR1_;
        }
        eventL1_ = eventM_;
        eventM_.set(deltaVsize, deltaRss, e, this);
        eventR1_ = SignificantEvent();
        eventR2_ = SignificantEvent();
      } else if(deltaVsize > eventR1_.deltaVsize) {
        double deltaRss = current_->rss - eventM_.rss;
        eventR2_ = eventR1_;
        eventR1_.set(deltaVsize, deltaRss, e, this);
      } else if(deltaVsize > eventR2_.deltaVsize) {
        double deltaRss = current_->rss - eventR1_.rss;
        eventR2_.set(deltaVsize, deltaRss, e, this);
      }

      // Update significative events for Rss
      double rss = current_->rss;
      double deltaRss = rss - eventRssT1_.rss;

      if(rss > eventRssT1_.rss) {
        eventRssT3_ = eventRssT2_;
        eventRssT2_ = eventRssT1_;
        eventRssT1_.set(deltaVsize, deltaRss, e, this);
      } else if(rss > eventRssT2_.rss) {
        eventRssT3_ = eventRssT2_;
        eventRssT2_.set(deltaVsize, deltaRss, e, this);
      } else if(rss > eventRssT3_.rss) {
        eventRssT3_.set(deltaVsize, deltaRss, e, this);
      }
      if(deltaRss > eventDeltaRssT1_.deltaRss) {
        eventDeltaRssT3_ = eventDeltaRssT2_;
        eventDeltaRssT2_ = eventDeltaRssT1_;
        eventDeltaRssT1_.set(deltaVsize, deltaRss, e, this);
      } else if(deltaRss > eventDeltaRssT2_.deltaRss) {
        eventDeltaRssT3_ = eventDeltaRssT2_;
        eventDeltaRssT2_.set(deltaVsize, deltaRss, e, this);
      } else if(deltaRss > eventDeltaRssT3_.deltaRss) {
        eventDeltaRssT3_.set(deltaVsize, deltaRss, e, this);
      }
    } // updateEventStats

    void SimpleMemoryCheck::andPrint(std::string const& type,
                    std::string const& mdlabel, std::string const& mdname) const {
      if(not jobReportOutputOnly_ && ((*current_ > max_) || oncePerEventMode_)) {
        if(count_ >= num_to_skip_) {
          double deltaVSIZE = current_->vsize - max_.vsize;
          double deltaRSS   = current_->rss - max_.rss;
          if(!showMallocInfo_) {  // default
            LogWarning("MemoryCheck")
            << "MemoryCheck: " << type << " "
            << mdname << ":" << mdlabel
            << " VSIZE " << current_->vsize << " " << deltaVSIZE
            << " RSS " << current_->rss << " " << deltaRSS
            << "\n";
          } else {
#ifdef __linux__
            struct mallinfo minfo = mallinfo();
#endif
            LogWarning("MemoryCheck")
            << "MemoryCheck: " << type << " "
            << mdname << ":" << mdlabel
            << " VSIZE " << current_->vsize << " " << deltaVSIZE
            << " RSS " << current_->rss << " " << deltaRSS
#ifdef __linux__
            << " HEAP-ARENA [ SIZE-BYTES " << minfo.arena
            << " N-UNUSED-CHUNKS " << minfo.ordblks
            << " TOP-FREE-BYTES " << minfo.keepcost << " ]"
            << " HEAP-MAPPED [ SIZE-BYTES " << minfo.hblkhd
            << " N-CHUNKS " << minfo.hblks << " ]"
            << " HEAP-USED-BYTES " << minfo.uordblks
            << " HEAP-UNUSED-BYTES " << minfo.fordblks
#endif
            << "\n";
          }
        }
      }
    }

    void SimpleMemoryCheck::updateAndPrint(std::string const& type,
                    std::string const& mdlabel, std::string const& mdname) {
      update();
      andPrint(type, mdlabel, mdname);
      updateMax();
    }

#ifdef SIMPLE_MEMORY_CHECK_ORIGINAL_XML_OUTPUT
    void
    SimpleMemoryCheck::eventStatOutput(std::string title,
                                       SignificantEvent const& e,
                                       std::map<std::string, std::string>& m) const {
      { std::ostringstream os;
        os << title << "-a-COUNT";
        m.insert(std::make_pair(os.str(), i2str(e.count))); }
      { std::ostringstream os;
        os << title << "-b-RUN";
        m.insert(std::make_pair(os.str(), d2str(static_cast<double>(e.event.run())))); }
      { std::ostringstream os;
        os << title << "-c-EVENT";
        m.insert(std::make_pair(os.str(), d2str(static_cast<double>(e.event.event())))); }
      { std::ostringstream os;
        os << title << "-d-VSIZE";
        m.insert(std::make_pair(os.str(), d2str(e.vsize))); }
      { std::ostringstream os;
        os << title << "-e-DELTV";
        m.insert(std::make_pair(os.str(), d2str(e.deltaVsize))); }
      { std::ostringstream os;
        os << title << "-f-RSS";
        m.insert(std::make_pair(os.str(), d2str(e.rss))); }
      { std::ostringstream os;
        os << title << "-g-PRIVATE";
        m.insert(std::make_pair(os.str(), d2str(e.privateSize))); }
      { std::ostringstream os;
        os << title << "-h-PSS";
        m.insert(std::make_pair(os.str(), d2str(e.pss))); }
    } // eventStatOutput
#endif

#ifdef SIMPLE_MEMORY_CHECK_DIFFERENT_XML_OUTPUT
    std::string
    SimpleMemoryCheck::eventStatOutput(std::string title,
                                           SignificantEvent const& e) const {
      std::ostringstream os;
      os << "  <" << title << ">\n";
      os << "    " << e.count << ": " << e.event;
      os << " vsize " << e.vsize-e.deltaVsize << " + " << e.deltaVsize
                                              << " = " << e.vsize;
      os << "  rss: " << e.rss << "\n";
      os << "  </" << title << ">\n";
      return os.str();
    } // eventStatOutput

    std::string
    SimpleMemoryCheck::mallOutput(std::string title, size_t const& n) const {
      std::ostringstream os;
      os << "  <" << title << ">\n";
      os << "    " << n << "\n";
      os << "  </" << title << ">\n";
      return os.str();
    }
#endif
                                                                // changelog 2
    void
    SimpleMemoryCheck::updateModuleMemoryStats(SignificantModule& m,
                                               double dv) {
      if(count_ < num_to_skip_) {
        m.totalEarlyVsize += dv;
        if(dv > m.maxEarlyVsize)  m.maxEarlyVsize = dv;
      } else {
        ++m.postEarlyCount;
        m.totalDeltaVsize += dv;
        if(dv > m.maxDeltaVsize) {
          m.maxDeltaVsize = dv;
          m.eventMaxDeltaV = currentEventID_;
        }
      }
    } //updateModuleMemoryStats

    std::ostream&
    operator<< (std::ostream& os,
                SimpleMemoryCheck::SignificantEvent const& se) {
      os << "[" << se.count << "] "
      << se.event << "  vsize = " << se.vsize
      << " deltaVsize = " << se.deltaVsize
      << " rss = " << se.rss << " delta " << se.deltaRss <<" private = "<<se.privateSize<<" pss = "<<se.pss;
      return os;
    }

    std::ostream&
    operator<< (std::ostream& os,
                SimpleMemoryCheck::SignificantModule const& sm) {
      if(sm.postEarlyCount > 0) {
        os << "\nPost Early Events:  TotalDeltaVsize: " << sm.totalDeltaVsize
        << " (avg: " << sm.totalDeltaVsize/sm.postEarlyCount
        << "; max: " << sm.maxDeltaVsize
        << " during " << sm.eventMaxDeltaV << ")";
      }
      if(sm.totalEarlyVsize > 0) {
        os << "\n     Early Events:  TotalDeltaVsize: " << sm.totalEarlyVsize
        << " (max: " << sm.maxEarlyVsize << ")";
      }

      return os;
    }

  } // end namespace service
} // end namespace edm

