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
	  return (it!=quickReference_.end()) ? (*it).second : -1;
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
      void preModuleBeginJob(const edm::ModuleDescription& desc);

      void prePathBeginRun(const std::string& pathName);
      void postBeginRun(edm::Run const&, edm::EventSetup const&);

      void postBeginJob();
      void postEndJob();

      void preBeginLumi(edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime);
      void preEndLumi(edm::LuminosityBlockID const& iID, edm::Timestamp const& iTime);
      void preProcessPath(const std::string& pathName);
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const edm::Event&, const edm::EventSetup&);
      
      void preSource();
      void postSource();
      
      void preModule(const edm::ModuleDescription&);
      void postModule(const edm::ModuleDescription&);

      void jobFailure();

      void setMicroState(Microstate); // this is still needed for use in special functions like DQM which are in turn framework services. - the string pointer needs to be interpreted at each call.
      void accummulateFileSize(unsigned long fileSize);
      void startedLookingForFile();
      void stoppedLookingForFile();
      unsigned int getEventsProcessedForLumi(unsigned int lumi);
      std::string getOutputDefPath() const { return outputDefPath_; }
      std::string getRunDirName() const { return runDirectory_.stem().string(); }
      
    private:
      void dowork() { // the function to be called in the thread. Thread completes when function returns.
		while (!fmt_.m_stoprequest) {
			std::cout << "Current states: Ms=" << fmt_.m_data.macrostate_
					<< " ms=" << encPath_.encode(fmt_.m_data.ministate_)
					<< " us=" << encModule_.encode(fmt_.m_data.microstate_)
					<< std::endl;

			// lock the monitor
			fmt_.monlock_.lock();
			fmt_.m_data.macrostateJ_ = fmt_.m_data.macrostate_;
			fmt_.m_data.ministateJ_ = encPath_.encode(fmt_.m_data.ministate_);
			fmt_.m_data.microstateJ_ = encModule_.encode(
					fmt_.m_data.microstate_);

			fmt_.m_data.jsonMonitor_->snap(true, fastPath_);
			fmt_.monlock_.unlock();

			//::sleep(1);
			::sleep(sleepTime_);
		}
      }

      //the actual monitoring thread is held by a separate class object for ease of maintenance
      FastMonitoringThread fmt_;
      Encoding encModule_;
      Encoding encPath_;

      int sleepTime_;
      string /*rootDirectory_,*/ microstateDefPath_, outputDefPath_;
      string fastName_, fastPath_, slowName_;
      timeval lumiStartTime_, lumiStopTime_;
      timeval fileLookStart_, fileLookStop_;
      std::vector<double> leadTimes_;
      std::unordered_map<unsigned int, int> processedEventsPerLumi_;
      boost::filesystem::path workingDirectory_, runDirectory_;
    };

}

#endif
