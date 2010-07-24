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
//	Collect event summary information and output to XML file and logger
//	at the end of the job.  Involves split-up of updateAndPrint method.
//
// 2 - May 7, 2008 M. Fischler
//      Collect module summary information and output to XML file and logger
//	at the end of the job.
//
// 3 - Jan 14, 2009 Natalia Garcia Nebot
//	Added:	- Average rate of growth in RSS and peak value attained.
//		- Average rate of growth in VSize over time, Peak VSize
//
//

#include "FWCore/Services/src/Memory.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/MallocOpts.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <malloc.h>
#include <sstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <string.h>
#include <cstring>

#ifdef __linux__
#define LINUX 1
#endif

#include <unistd.h>
#include <fcntl.h>

namespace edm {
  namespace service {

    static std::string d2str(double d){
	std::ostringstream t;
	t << d;
	return t.str();
    }

    static std::string i2str(int i){
	std::ostringstream t;
	t << i;
	return t.str();
    }

    struct linux_proc {
      int pid; // %d
      char comm[400]; // %s
      char state; // %c
      int ppid; // %d
      int pgrp; // %d
      int session; // %d
      int tty; // %d
      int tpgid; // %d
      unsigned int flags; // %u
      unsigned int minflt; // %u
      unsigned int cminflt; // %u
      unsigned int majflt; // %u
      unsigned int cmajflt; // %u
      int utime; // %d
      int stime; // %d
      int cutime; // %d
      int cstime; // %d
      int counter; // %d
      int priority; // %d
      unsigned int timeout; // %u
      unsigned int itrealvalue; // %u
      int starttime; // %d
      unsigned int vsize; // %u
      unsigned int rss; // %u
      unsigned int rlim; // %u
      unsigned int startcode; // %u
      unsigned int endcode; // %u
      unsigned int startstack; // %u
      unsigned int kstkesp; // %u
      unsigned int kstkeip; // %u
      int signal; // %d
      int blocked; // %d
      int sigignore; // %d
      int sigcatch; // %d
      unsigned int wchan; // %u
    };
      
    procInfo SimpleMemoryCheck::fetch()
    {
      procInfo ret;
      double pr_size=0.0, pr_rssize=0.0;
      
#ifdef LINUX
      linux_proc pinfo;
      int cnt;

      lseek(fd_,0,SEEK_SET);
    
      if((cnt=read(fd_,buf_,sizeof(buf_)))<0)
	{
	  perror("Read of Proc file failed:");
	  return procInfo();
	}
    
      if(cnt>0)
	{
	  buf_[cnt]='\0';
	  
	  sscanf(buf_,
		 "%d %s %c %d %d %d %d %d %u %u %u %u %u %d %d %d %d %d %d %u %u %d %u %u %u %u %u %u %u %u %d %d %d %d %u",
		 &pinfo.pid, // %d
		 pinfo.comm, // %s
		 &pinfo.state, // %c
		 &pinfo.ppid, // %d
		 &pinfo.pgrp, // %d
		 &pinfo.session, // %d
		 &pinfo.tty, // %d
		 &pinfo.tpgid, // %d
		 &pinfo.flags, // %u
		 &pinfo.minflt, // %u
		 &pinfo.cminflt, // %u
		 &pinfo.majflt, // %u
		 &pinfo.cmajflt, // %u
		 &pinfo.utime, // %d
		 &pinfo.stime, // %d
		 &pinfo.cutime, // %d
		 &pinfo.cstime, // %d
		 &pinfo.counter, // %d
		 &pinfo.priority, // %d
		 &pinfo.timeout, // %u
		 &pinfo.itrealvalue, // %u
		 &pinfo.starttime, // %d
		 &pinfo.vsize, // %u
		 &pinfo.rss, // %u
		 &pinfo.rlim, // %u
		 &pinfo.startcode, // %u
		 &pinfo.endcode, // %u
		 &pinfo.startstack, // %u
		 &pinfo.kstkesp, // %u
		 &pinfo.kstkeip, // %u
		 &pinfo.signal, // %d
		 &pinfo.blocked, // %d
		 &pinfo.sigignore, // %d
		 &pinfo.sigcatch, // %d
		 &pinfo.wchan // %u
		 );

	  // resident set size in pages
	  pr_size = (double)pinfo.vsize;
	  pr_rssize = (double)pinfo.rss;
	  
	  ret.vsize = pr_size / (1024.0*1024.0);
	  ret.rss   = (pr_rssize * pg_size_) / (1024.0*1024.0);
	}
#else
      ret.vsize=0;
      ret.rss=0;
#endif
      return ret;
    }


    double SimpleMemoryCheck::averageGrowthRate(double current, double past, int count) {
	return (current-past)/(double)count;
    }
    
    SimpleMemoryCheck::SimpleMemoryCheck(const ParameterSet& iPS,
					 ActivityRegistry&iReg)
    : a_()
    , b_()
    , current_(&a_)
    , previous_(&b_)
    , pg_size_(sysconf(_SC_PAGESIZE)) // getpagesize()
    , num_to_skip_(iPS.getUntrackedParameter<int>("ignoreTotal"))
    , showMallocInfo(iPS.getUntrackedParameter<bool>("showMallocInfo"))
    , oncePerEventMode
      	(iPS.getUntrackedParameter<bool>("oncePerEventMode"))
    , count_()
    , growthRateVsize_()
    , growthRateRss_()
    , moduleSummaryRequested
        (iPS.getUntrackedParameter<bool>("moduleMemorySummary"))
								// changelog 2
    {
      // pg_size = (double)getpagesize();
      std::ostringstream ost;
	
#ifdef LINUX
      ost << "/proc/" << getpid() << "/stat";
      fname_ = ost.str();
      
      if((fd_=open(ost.str().c_str(),O_RDONLY))<0)
	{
	  throw edm::Exception(errors::Configuration)
	    << "Memory checker server: Failed to open " << ost.str() << std::endl;
	}
#endif
      if (!oncePerEventMode) { // default, prints on increases
        iReg.watchPreSourceConstruction(this,
             &SimpleMemoryCheck::preSourceConstruction);
        iReg.watchPostSourceConstruction(this,
             &SimpleMemoryCheck::postSourceConstruction);
        iReg.watchPostSource(this,
             &SimpleMemoryCheck::postSource);
        iReg.watchPostModuleConstruction(this,
             &SimpleMemoryCheck::postModuleConstruction);
        iReg.watchPostModuleBeginJob(this,
             &SimpleMemoryCheck::postModuleBeginJob);
        iReg.watchPostProcessEvent(this,
             &SimpleMemoryCheck::postEventProcessing);
        iReg.watchPostModule(this,
             &SimpleMemoryCheck::postModule);
        iReg.watchPostBeginJob(this,
             &SimpleMemoryCheck::postBeginJob);
        iReg.watchPostEndJob(this,
             &SimpleMemoryCheck::postEndJob);
      } else { 
        iReg.watchPostProcessEvent(this,
             &SimpleMemoryCheck::postEventProcessing);
        iReg.watchPostEndJob(this,
             &SimpleMemoryCheck::postEndJob);
      }
      if (moduleSummaryRequested) {				// changelog 2
        iReg.watchPreProcessEvent(this,
             &SimpleMemoryCheck::preEventProcessing);
        iReg.watchPreModule(this,
             &SimpleMemoryCheck::preModule);
        if (oncePerEventMode) {
        iReg.watchPostModule(this,
             &SimpleMemoryCheck::postModule);
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

      typedef edm::MallocOpts::opt_type opt_type;
      edm::MallocOptionSetter& mopts = edm::getGlobalOptionSetter();
      
      opt_type 
	p_mmap_max=iPS.getUntrackedParameter<int>("M_MMAP_MAX"),
	p_trim_thr=iPS.getUntrackedParameter<int>("M_TRIM_THRESHOLD"),
	p_top_pad=iPS.getUntrackedParameter<int>("M_TOP_PAD"),
	p_mmap_thr=iPS.getUntrackedParameter<int>("M_MMAP_THRESHOLD");
	  
      if(p_mmap_max>=0) mopts.set_mmap_max(p_mmap_max);
      if(p_trim_thr>=0) mopts.set_trim_thr(p_trim_thr);
      if(p_top_pad>=0) mopts.set_top_pad(p_top_pad);
      if(p_mmap_thr>=0) mopts.set_mmap_thr(p_mmap_thr);

      mopts.adjustMallocParams();

      if(mopts.hasErrors())
	{
	  LogWarning("MemoryCheck") 
	    << "ERROR: Problem with setting malloc options\n"
	    << mopts.error_message(); 
	}

      if(iPS.getUntrackedParameter<bool>("dump")==true)
        {
	  edm::MallocOpts mo = mopts.get();
	  LogWarning("MemoryCheck") 
	    << "Malloc options: " << mo << "\n";
        }
    }

    SimpleMemoryCheck::~SimpleMemoryCheck()
    {
#ifdef LINUX
      close(fd_);
#endif
    }

    void SimpleMemoryCheck::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<int>("ignoreTotal",1);
      desc.addUntracked<bool>("showMallocInfo",false);
      desc.addUntracked<bool>("oncePerEventMode",false);
      desc.addUntracked<bool>("moduleMemorySummary",false);
      desc.addUntracked<int>("M_MMAP_MAX",-1);
      desc.addUntracked<int>("M_TRIM_THRESHOLD",-1);
      desc.addUntracked<int>("M_TOP_PAD",-1);
      desc.addUntracked<int>("M_MMAP_THRESHOLD",-1);
      desc.addUntracked<bool>("dump",false);
      descriptions.add("SimpleMemoryCheck", desc);
    }

    void SimpleMemoryCheck::postBeginJob()
    {
        growthRateVsize_ = current_->vsize;
        growthRateRss_ = current_->rss;
    }
 
    void SimpleMemoryCheck::preSourceConstruction(const ModuleDescription& md) 
    {
      updateAndPrint("pre-ctor", md.moduleLabel(), md.moduleName());
    }
 
 
    void SimpleMemoryCheck::postSourceConstruction(const ModuleDescription& md)
    {
      updateAndPrint("ctor", md.moduleLabel(), md.moduleName());
    }
 
    void SimpleMemoryCheck::postSource() 
    {
      updateAndPrint("module", "source", "source");
    }
 
    void SimpleMemoryCheck::postModuleConstruction(const ModuleDescription& md)
    {
      updateAndPrint("ctor", md.moduleLabel(), md.moduleName());
    }
 
    void SimpleMemoryCheck::postModuleBeginJob(const ModuleDescription& md) 
    {
      updateAndPrint("beginJob", md.moduleLabel(), md.moduleName());
    }
 
    void SimpleMemoryCheck::postEndJob() 
    {
      edm::LogAbsolute("MemoryReport") 				// changelog 1
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
    
      if (moduleSummaryRequested) {				// changelog 1
        edm::LogAbsolute mmr("ModuleMemoryReport"); // at end of if block, mmr
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
	for (SignificantModulesMap::iterator im=modules_.begin(); 
	     im != modules_.end(); ++im) {
	  SignificantModule const& m = im->second;
	  if ( m.totalDeltaVsize == 0 && m.totalEarlyVsize == 0 ) continue;
	  mmr << im->first << ": ";
	  mmr << "n = " << m.postEarlyCount;
	  if ( m.postEarlyCount > 0 ) mmr << " avg = " 
	  				  << m.totalDeltaVsize/m.postEarlyCount;
	  mmr <<  " max = " << m.maxDeltaVsize << " " << m.eventMaxDeltaV;
	  if ( m.totalEarlyVsize > 0 ) {
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

      if (eventL2_.vsize > 0) 
      	eventStatOutput("LargeVsizeIncreaseEventL2", eventL2_, reportData);
      if (eventL1_.vsize > 0) 
      	eventStatOutput("LargeVsizeIncreaseEventL1", eventL1_, reportData);
      if (eventM_.vsize > 0) 
      	eventStatOutput("LargestVsizeIncreaseEvent", eventM_,  reportData);
      if (eventR1_.vsize > 0) 
      	eventStatOutput("LargeVsizeIncreaseEventR1", eventR1_, reportData);
      if (eventR2_.vsize > 0)
      	eventStatOutput("LargeVsizeIncreaseEventR2", eventR2_, reportData);
      if (eventT3_.vsize > 0) 
      	eventStatOutput("ThirdLargestVsizeEventT3",  eventT3_, reportData);
      if (eventT2_.vsize > 0) 
      	eventStatOutput("SecondLargestVsizeEventT2", eventT2_, reportData);
      if (eventT1_.vsize > 0)
      	eventStatOutput("LargestVsizeEventT1",       eventT1_, reportData);

      if (eventRssT3_.rss > 0)
        eventStatOutput("ThirdLargestRssEvent", eventRssT3_, reportData);
      if (eventRssT2_.rss > 0)
        eventStatOutput("SecondLargestRssEvent", eventRssT2_, reportData);
      if (eventRssT1_.rss > 0)
        eventStatOutput("LargestRssEvent", eventRssT1_, reportData);
      if (eventDeltaRssT3_.deltaRss > 0)
        eventStatOutput("ThirdLargestIncreaseRssEvent", eventDeltaRssT3_, reportData);
      if (eventDeltaRssT2_.deltaRss > 0)
        eventStatOutput("SecondLargestIncreaseRssEvent", eventDeltaRssT2_, reportData);
      if (eventDeltaRssT1_.deltaRss > 0)
        eventStatOutput("LargestIncreaseRssEvent", eventDeltaRssT1_, reportData);
     
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

      // Report Growth rates for VSize and Rss
      reportData.insert(
        std::make_pair("AverageGrowthRateVsize", d2str(averageGrowthRate(current_->vsize, growthRateVsize_, count_))));     
      reportData.insert(
        std::make_pair("PeakValueVsize", d2str(eventT1_.vsize)));
      reportData.insert(
        std::make_pair("AverageGrowthRateRss", d2str(averageGrowthRate(current_->rss, growthRateRss_, count_))));
      reportData.insert(
        std::make_pair("PeakValueRss", d2str(eventRssT1_.rss)));

      if (moduleSummaryRequested) {				// changelog 2
	for (SignificantModulesMap::iterator im=modules_.begin(); 
	     im != modules_.end(); ++im) {
	  SignificantModule const& m = im->second;
	  if ( m.totalDeltaVsize == 0 && m.totalEarlyVsize == 0 ) continue;
	  std::string label = im->first+":";
      	  reportData.insert(
            std::make_pair(label+"PostEarlyCount", i2str(m.postEarlyCount)));  
	  if ( m.postEarlyCount > 0 ) {
      	    reportData.insert(
              std::make_pair(label+"AverageDeltaVsize", 
	      d2str(m.totalDeltaVsize/m.postEarlyCount)));  
	  }
      	  reportData.insert(
              std::make_pair(label+"MaxDeltaVsize", d2str(m.maxDeltaVsize)));  
	  if ( m.totalEarlyVsize > 0 ) {
      	    reportData.insert(
              std::make_pair(label+"TotalEarlyVsize", d2str(m.totalEarlyVsize)));  
      	    reportData.insert(
              std::make_pair(label+"MaxEarlyDeltaVsize", d2str(m.maxEarlyVsize)));  
	  }
	}
      } 

      std::map<std::string, std::string> reportMemoryProperties;

      if (FILE *fmeminfo = fopen("/proc/meminfo", "r")){
	char buf[128];
	char space[] = " ";
	size_t value;

	while (fgets(buf, sizeof(buf), fmeminfo)){
	  char *token = NULL;
	  token = strtok(buf, space);
	  if (token != NULL){
	    value = atol(strtok(NULL, space));
	    std::string category = token;
	    reportMemoryProperties.insert(std::make_pair(category.substr(0,strlen(token)-1), i2str(value)));
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

      if (eventL2_.vsize > 0) reportData.push_back(
      	eventStatOutput("LargeVsizeIncreaseEventL2", eventL2_));
      if (eventL1_.vsize > 0) reportData.push_back(
      	eventStatOutput("LargeVsizeIncreaseEventL1", eventL1_));
      if (eventM_.vsize > 0) reportData.push_back(
      	eventStatOutput("LargestVsizeIncreaseEvent", eventM_));
      if (eventR1_.vsize > 0) reportData.push_back(
      	eventStatOutput("LargeVsizeIncreaseEventR1", eventR1_));
      if (eventR2_.vsize > 0) reportData.push_back(
      	eventStatOutput("LargeVsizeIncreaseEventR2", eventR2_));
      if (eventT3_.vsize > 0) reportData.push_back(
      	eventStatOutput("ThirdLargestVsizeEventT3", eventT3_));
      if (eventT2_.vsize > 0) reportData.push_back(
      	eventStatOutput("SecondLargestVsizeEventT2", eventT2_));
      if (eventT1_.vsize > 0) reportData.push_back(
      	eventStatOutput("LargestVsizeEventT1", eventT1_));

      if (eventRssT3_.rss > 0)
        eventStatOutput("ThirdLargestRssEvent", eventRssT3_, reportData);
      if (eventRssT2_.rss > 0)
        eventStatOutput("SecondLargestRssEvent", eventRssT2_, reportData);
      if (eventRssT1_.rss > 0)
        eventStatOutput("LargestRssEvent", eventRssT1_, reportData);
      if (eventDeltaRssT3_.deltaRss > 0)
        eventStatOutput("ThirdLargestIncreaseRssEvent", eventDeltaRssT3_, reportData);
      if (eventDeltaRssT2_.deltaRss > 0)
        eventStatOutput("SecondLargestIncreaseRssEvent", eventDeltaRssT2_, reportData);
      if (eventDeltaRssT1_.deltaRss > 0)
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
 
    void SimpleMemoryCheck::preEventProcessing(const edm::EventID& iID,
          				       const edm::Timestamp& iTime) 
    {
      currentEventID_ = iID;					// changelog 2
    }

    void SimpleMemoryCheck::postEventProcessing(const Event& e,
          					const EventSetup&) 
    {
      ++count_;
      update();
      updateEventStats( e.id() );
      if (oncePerEventMode) {
        // should probably use be Run:Event or count_ for the label and name
        updateMax();
	andPrint("event", "", ""); 
      } 
    }
 
    void SimpleMemoryCheck::preModule(const ModuleDescription& md) { 
      update();							// changelog 2
      moduleEntryVsize_ = current_->vsize;
    }
 
    void SimpleMemoryCheck::postModule(const ModuleDescription& md) {
      if (!oncePerEventMode) {
        updateAndPrint("module", md.moduleLabel(), md.moduleName());
      } else if (moduleSummaryRequested) {			// changelog 2
        update();
      }
      if (moduleSummaryRequested) {				// changelog 2
	double dv = current_->vsize - moduleEntryVsize_;
	std::string label =  md.moduleLabel();
	updateModuleMemoryStats (modules_[label],dv);
      }
    }
 
 
    void SimpleMemoryCheck::update() 
    {
      std::swap(current_,previous_);
      *current_ = fetch();
    }

    void SimpleMemoryCheck::updateMax() 
    {
      if ((*current_ > max_) || oncePerEventMode)
        {
          if(count_ >= num_to_skip_) {
          }
          max_ = *current_;
        }
    }

    void SimpleMemoryCheck::updateEventStats(edm::EventID const & e) {
      if (count_ < num_to_skip_) return;
      if (count_ == num_to_skip_) {
	eventT1_.set(0, 0, e, this);
	eventM_.set (0, 0, e, this);
	eventRssT1_.set(0, 0, e, this);
	eventDeltaRssT1_.set(0, 0, e, this);
        return;
      }
      double vsize = current_->vsize;
      double deltaVsize = vsize - eventT1_.vsize;

      // Update significative events for Vsize
      if (vsize > eventT1_.vsize) {
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

      if (deltaVsize > eventM_.deltaVsize) {
	double deltaRss = current_->rss - eventM_.rss;
        if (eventL1_.deltaVsize >= eventR1_.deltaVsize) {
	  eventL2_ = eventL1_; 
	} else {
	  eventL2_ = eventR1_; 
        }
	eventL1_ = eventM_;
	eventM_.set(deltaVsize, deltaRss, e, this);
	eventR1_ = SignificantEvent();
	eventR2_ = SignificantEvent();
      } else if (deltaVsize > eventR1_.deltaVsize) {
	double deltaRss = current_->rss - eventM_.rss;
        eventR2_ = eventR1_;
	eventR1_.set(deltaVsize, deltaRss, e, this);
      } else if (deltaVsize > eventR2_.deltaVsize) {
	double deltaRss = current_->rss - eventR1_.rss;
	eventR2_.set(deltaVsize, deltaRss, e, this);
      }

      // Update significative events for Rss
      double rss = current_->rss;
      double deltaRss = rss - eventRssT1_.rss;

      if(rss > eventRssT1_.rss){
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
    }	// updateEventStats
      
    void SimpleMemoryCheck::andPrint(const std::string& type, 
                    const std::string& mdlabel, const std::string& mdname) const
    {
      if ((*current_ > max_) || oncePerEventMode)
        {
          if(count_ >= num_to_skip_) {
            double deltaVSIZE = current_->vsize - max_.vsize;
            double deltaRSS   = current_->rss - max_.rss;
            if (!showMallocInfo) {  // default
              LogWarning("MemoryCheck")
              << "MemoryCheck: " << type << " "
              << mdname << ":" << mdlabel 
              << " VSIZE " << current_->vsize << " " << deltaVSIZE
              << " RSS " << current_->rss << " " << deltaRSS
              << "\n";
            } else {
              struct mallinfo minfo = mallinfo();
              LogWarning("MemoryCheck")
              << "MemoryCheck: " << type << " "
              << mdname << ":" << mdlabel 
              << " VSIZE " << current_->vsize << " " << deltaVSIZE
              << " RSS " << current_->rss << " " << deltaRSS
              << " HEAP-ARENA [ SIZE-BYTES " << minfo.arena
              << " N-UNUSED-CHUNKS " << minfo.ordblks
              << " TOP-FREE-BYTES " << minfo.keepcost << " ]"
              << " HEAP-MAPPED [ SIZE-BYTES " << minfo.hblkhd
              << " N-CHUNKS " << minfo.hblks << " ]"
              << " HEAP-USED-BYTES " << minfo.uordblks
              << " HEAP-UNUSED-BYTES " << minfo.fordblks
              << "\n";
            }
          }
        }
    }

    void SimpleMemoryCheck::updateAndPrint(const std::string& type, 
                    const std::string& mdlabel, const std::string& mdname) 
    {
      update();
      andPrint(type, mdlabel, mdname);
      updateMax();
    }

#ifdef SIMPLE_MEMORY_CHECK_ORIGINAL_XML_OUTPUT
    void
    SimpleMemoryCheck::eventStatOutput(std::string title, 
    				       SignificantEvent const& e,
				       std::map<std::string, std::string> &m) const
    {
      { std::ostringstream os;
        os << title << "-a-COUNT";
        m.insert(std::make_pair(os.str(), i2str(e.count))); }
      { std::ostringstream os;
        os << title << "-b-RUN";
        m.insert(std::make_pair(os.str(), d2str(static_cast<double>(e.event.run())) )); }
      { std::ostringstream os;
        os << title << "-c-EVENT";
        m.insert(std::make_pair(os.str(), d2str(static_cast<double>(e.event.event())) )); }
      { std::ostringstream os;
        os << title << "-d-VSIZE";
        m.insert(std::make_pair(os.str(), d2str(e.vsize))); }
      { std::ostringstream os;
        os << title << "-e-DELTV";
        m.insert(std::make_pair(os.str(), d2str(e.deltaVsize))); }
      { std::ostringstream os;
        os << title << "-f-RSS";
        m.insert(std::make_pair(os.str(), d2str(e.rss))); }
    } // eventStatOutput
#endif

 
#ifdef SIMPLE_MEMORY_CHECK_DIFFERENT_XML_OUTPUT
    std::string 
    SimpleMemoryCheck::eventStatOutput(std::string title, 
    				       SignificantEvent const& e) const
    {
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
    SimpleMemoryCheck::updateModuleMemoryStats(SignificantModule & m, 
    					       double dv) {
      if(count_ < num_to_skip_) {
	m.totalEarlyVsize += dv;
	if (dv > m.maxEarlyVsize)  m.maxEarlyVsize = dv;     	
      } else {
	++m.postEarlyCount;
	m.totalDeltaVsize += dv;
	if (dv > m.maxDeltaVsize)  {
	  m.maxDeltaVsize = dv;    
	  m.eventMaxDeltaV = currentEventID_;
	}	
      }
    } //updateModuleMemoryStats
 


    std::ostream & 
    operator<< (std::ostream & os, 
    		SimpleMemoryCheck::SignificantEvent const & se) {
      os << "[" << se.count << "] "
         << se.event << "  vsize = " << se.vsize 
	 << " deltaVsize = " << se.deltaVsize 
         << " rss = " << se.rss << " delta " << se.deltaRss;
      return os;
    }

    std::ostream & 
    operator<< (std::ostream & os, 
    		SimpleMemoryCheck::SignificantModule const & sm) {
      if ( sm.postEarlyCount > 0 ) {
        os << "\nPost Early Events:  TotalDeltaVsize: " << sm.totalDeltaVsize
           << " (avg: " << sm.totalDeltaVsize/sm.postEarlyCount 
	   << "; max: " << sm.maxDeltaVsize  
	   << " during " << sm.eventMaxDeltaV << ")";
      }
      if ( sm.totalEarlyVsize > 0 ) {
        os << "\n     Early Events:  TotalDeltaVsize: " << sm.totalEarlyVsize
           << " (max: " << sm.maxEarlyVsize << ")";
      }
	
      return os;
    }

  } // end namespace service
} // end namespace edm

