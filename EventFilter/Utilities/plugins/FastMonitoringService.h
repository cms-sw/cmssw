#ifndef EvFFastMonitoringService_H
#define EvFFastMonitoringService_H 1


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include "boost/filesystem.hpp"

#include "EventFilter/Utilities/interface/MicroStateService.h"
#include "FastMonitoringThread.h"

#include <string>
#include <vector>

#include <sstream>
#include <unordered_map>

/*Description
  this is an evolution of the MicroStateService intended to be run standalone in cmsRun or similar
  As such, it has to independently create a monitoring thread and run it in each forked process, which needs 
  to be arranged following the standard CMSSW procedure.
  We try to use boost threads for uniformity with the rest of the framework, even if they suck a bit.
  A legenda for use by the monitoring process in the DAQ needs to be generated as soon as convenient - since 
  no access to the EventProcessor is granted, this needs to wait until after beginJob is executed.
  At the same time, we try to spare time in the monitoring by avoiding even a single string lookup and using the 
  moduledesc pointer to key into the map instead.
  As a bonus, we can now add to the monitored status the current path (and possibly associate modules to a path...)
  this intermediate info will be called "ministate" :D
  The general counters and status variables (event number, number of processed events, number of passed and stored 
  events, luminosity section etc. are also monitored here.

  NOTA BENE!!! with respect to the MicroStateService, no string or string pointers are used for the microstates.
  NOTA BENE!!! the state of the edm::EventProcessor cannot be monitored directly from within a service, so a 
  different solution must be identified for that (especially one needs to identify error states). 
  NOTA BENE!!! to keep backward compatibility with the MicroStateService, a common base class with abstract interface,
  exposing the single  method to be used by all other packages (except EventFilter/Processor, 
  which should continue to use the concrete class interface) will be defined 

*/

namespace evf{

  class FastMonitoringService : public MicroStateService
    {
      struct Encoding
      {
	Encoding(unsigned int res): reserved_(res), current_(reserved_), currentReserved_(0)
	{
	  dummiesForReserved_ = new edm::ModuleDescription[reserved_];
	  //	  completeReservedWithDummies();
	}
	//trick: only encode state when sending it over (i.e. every sec)
	int encode(const void *add){
	  std::map<const void *, int>::const_iterator it=quickReference_.find(add);
	  return (it!=quickReference_.end()) ? (*it).second : 0;
	}
	const void* decode(unsigned int index){return decoder_[index];}
	void fillReserved(void* add, unsigned int i){
	  //	  translation_[*name]=current_; 
	  quickReference_[add]=i; 
	  if(decoder_.size()<=i)
	    decoder_.push_back(add);
	  else
	    decoder_[currentReserved_] = add;
	}
	void updateReserved(void* add){
	  fillReserved(add,currentReserved_);
	  currentReserved_++;
	}
	void completeReservedWithDummies()
	{
	  for(unsigned int i = currentReserved_; i<reserved_; i++)
	    fillReserved(dummiesForReserved_+i,i);
	}
	void update(void* add){
	  //	  translation_[*name]=current_; 
	  quickReference_[add]=current_; 
	  decoder_.push_back(add); 
	  current_++;
	}
	std::map<const void *,int> quickReference_;
	std::vector<const void *> decoder_;
	unsigned int reserved_;
	int current_;
	int currentReserved_;
	edm::ModuleDescription *dummiesForReserved_;
      };
    public:


      // the names of the states - some of them are never reached in an online app
      static const std::string macroStateNames[FastMonitoringThread::MCOUNT];
      // Reserved names for microstates
      // moved into base class in EventFilter/Utilities for compatibility with MicroStateServiceClassic
      static const std::string nopath_;
      FastMonitoringService(const edm::ParameterSet&,edm::ActivityRegistry&);
      ~FastMonitoringService();
     
      std::string makePathLegenda();
      std::string makeModuleLegenda();

      void preallocate(edm::service::SystemBounds const&);
      void jobFailure();
      void preModuleBeginJob(edm::ModuleDescription const&);
      void postBeginJob();
      void postEndJob();
      void prePathBeginRun(const std::string& pathName);//!

      void postGlobalBeginRun(edm::GlobalContext const&);
      void preGlobalBeginLumi(edm::GlobalContext const&);
      void preGlobalEndLumi(edm::GlobalContext const&);
      void postGlobalEndLumi(edm::GlobalContext const&);

      void preStreamBeginLumi(edm::Streamcontext const&);
      void preStreamEndLumi(edm::Streamcontext const&);
      void prePathEvent(edm::StreamContext const&, const edm::PathContext const&);
      void preEvent(edm::StreamContext const&);
      void postEvent(edm::StreamContext const&);
      void preSourceEvent(edm::StreamID);
      void postSourceEvent(edm::StreamID);
      void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
      void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);

      //OBSOLETE
      void setMicroState(Microstate); // this is still needed for use in special functions like DQM which are in turn framework services.
      void setMicroState(edm::StreamID, Microstate);

      void accumulateFileSize(unsigned int lumi, unsigned long fileSize);
      void startedLookingForFile();
      void stoppedLookingForFile(unsigned int lumi);
      unsigned int getEventsProcessedForLumi(unsigned int lumi);
      std::string getOutputDefPath() const { return outputDefPath_; }
      std::string getRunDirName() const { return runDirectory_.stem().string(); }

    private:

      void doSnapshot(bool outputCSV, unsigned int forLumi) {

	// update monitored content
	fmt_.m_data.fastMacrostateJ_ = macrostate_;

	//update following vars unless we are in the middle of lumi transition (todo:be able to collect despite)
	if (!isGlobalLumiTransition) {
	  //these are stored maps, try if there's element for last globalLumi
	  auto itd = throughput_.find(foLumi);
	  if (itd!=std::map:end) {
	    fmt_.m_data.fastThroughputJ_ = *it;//throughput_[lastGlobalLumi_];
	    else fmt_.m_data.fastThroughputJ_=0.;
	  }

	  itd = avgLeadTime_.find(forLumi);
	  if (itd != std::map:end) {
	    fmt_.m_data.fastAvgLeadTimeJ_ = *it;//avgLeadTime_[lastGlobalLumi_];
	    else fmt_.m_data.fastAvgLeadTimeJ_=0.;
	  }

	  auto iti = filesProcessed_.find(forLumi);
	  if (iti != std::map:end) {
	    fmt_.m_data.fastFilesProcessedJ_ = *it;//filesProcessed_[lastGlobalLumi_];
	    else fmt_.m_data.fastFilesProcessedJ_=0;
	  }
	}
	else return; //skip snapshot if it happens in lumi transition

	//decode mini/microstate using what is latest stored per stream()
	for (unsigned int i=0;i<nStreams;i++) {
	  fmt_.m_data.ministateDecoded_[i] = 0;//not supported for now
	  //fmt_.m_data.ministateDecoded_[i] = encPath_.encode(fmt_.m_data.ministate_[i]);
	  fmt_.m_data.microstateDecoded_[i] = encModule_.encode(fmt_.m_data.microstate_[i]);
	}

	//do a snapshot, also output fast CSV
	fmt_.jsonMonitor_->snap(outputCSV, fastPath_,forLumi);
      }

      void dowork() { // the function to be called in the thread. Thread completes when function returns.
	while (!fmt_.m_stoprequest) {
	  std::cout << "Current states: Ms=" << fmt_.m_data.macrostate_;
	  if (nStreams_==1)//only makes sense for 1 thread
	    std::cout << " ms=" << encPath_.encode(fmt_.m_data.ministate_)
	      << " us=" << encModule_.encode(fmt_.m_data.microstate_)
	  std::cout << std::endl;

	  // lock the monitor
	  fmt_.monlock_.lock();
	  //do a snapshot, also output fast CSV
          doSnapshot(true,lastGlobalLumi_);
	  fmt_.monlock_.unlock();

	  ::sleep(sleepTime_);
	}
      }

      //the actual monitoring thread is held by a separate class object for ease of maintenance
      FastMonitoringThread fmt_;
      Encoding encModule_;
      Encoding encPath_;

      unsigned int nStreams_;
      int sleepTime_;
      std::string /*rootDirectory_,*/ microstateDefPath_, outputDefPath_;
      std::string fastName_, fastPath_, slowName_;

      //variables that are used by/monitored by FastMonitoringThread / FastMonitor

      std::map<unsigned int, timeval> lumiStartTime_// ,lumiStopTime_;//needed for multiplexed begin/end lumis
      timeval fileLookStart_, fileLookStop_;//this stuff should be better calculated by input source

      std::atomic<unsigned int> lastGlobalLumi_;
      std::queue<unsigned int> lastGlobalLumisClosed_;
      std::atomic<bool> isGlobalLumiTransition_;
      unsigned int lumiFromSource_;//possibly atomic

      //global state
      Macrostate macrostate_;

      //per stream
      std::vector<const void*> ministate_;
      std::vector<const void*> microstate_;

      //variables measuring source statistics (global)
      std::map<unsigned int, double> throughput_;
      std::map<unsigned int, double> avgLeadTime_;
      std::map<unsigned int, unsigned int> filesProcessedDuringLumi_;
      //helpers for source statistics:
      std::map<unsigned int, unsigned long> accuSize_;
      std::vector<double> leadTimes_;

      //for output module
      //std::unordered_map<unsigned int, int> processedEventsPerLumi_;
      std::map<unsigned int, int> processedEventsPerLumi_;


      
      boost::mutex initPathsLock_;
      unsigned long firstEventId_ = 0;
      std::atomic<bool> collectedPathList_ = false;
      std::vector<bool> pathNamesReady_;

      boost::filesystem::path workingDirectory_, runDirectory_;
    };

}

#endif
