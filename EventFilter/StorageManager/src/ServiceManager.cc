// $Id$

#include <EventFilter/StorageManager/interface/ServiceManager.h>
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/ParameterSet/interface/PythonProcessDesc.h"
#include <FWCore/Utilities/interface/Exception.h>
#include <typeinfo>

using namespace std;
using namespace edm;
using boost::shared_ptr;


ServiceManager::ServiceManager(stor::DiskWritingParams dwParams):
  outModPSets_(0),
  currentlumi_(0),
  timeouttime_(0),
  lasttimechecked_(0),
  errorStreamPSetIndex_(-1),
  errorStreamCreated_(false),
  samples_(1000),
  period4samples_(10),
  diskWritingParams_(dwParams)
{
  collectStreamerPSets(dwParams._streamConfiguration);
  pmeter_ = new stor::SMPerformanceMeter();
  pmeter_->init(samples_, period4samples_);
} 


ServiceManager::~ServiceManager()
{ 
  delete pmeter_;
}


void ServiceManager::start()
{
  currentlumi_ = 0;
  timeouttime_ = 0;
  lasttimechecked_ = 0;
  errorStreamCreated_ = false;
  pmeter_->init(samples_, period4samples_);
}


void ServiceManager::stop()
{
}


/**
 * Returns a map of the trigger selection strings for each output stream.
 */
std::map<std::string, Strings> ServiceManager::getStreamSelectionTable()
{
  std::map<std::string, Strings> selTable;
  int psetIdx = -1;
  for(std::vector<ParameterSet>::iterator it = outModPSets_.begin();
      it != outModPSets_.end(); ++it) {
    ++psetIdx;
    if (psetIdx == errorStreamPSetIndex_) continue;

    std::string streamLabel = it->getParameter<string> ("streamLabel");
    if (streamLabel.size() > 0) {
      selTable[streamLabel] = EventSelector::getEventSelectionVString(*it);
    }
  }
  return selTable;
}

//
// *** wrote similar example code in IOPool/Streamer/test/ParamSetWalker_t.cpp 
// *** this method is diluted version of same code.
// *** if more items needs to be extracted for config, refer to example code
//
void ServiceManager::collectStreamerPSets(const std::string& config)
{

     try{
       
       PythonProcessDesc py_pdesc(config.c_str());
       boost::shared_ptr<ProcessDesc> pdesc = py_pdesc.processDesc();

       boost::shared_ptr<ParameterSet> procPset = pdesc->getProcessPSet();
       
        ParameterSet allTrigPaths = procPset->
	 getParameter<ParameterSet>("@trigger_paths");
       
       if (allTrigPaths.empty())
         throw cms::Exception("collectStreamerPSets","ServiceManager")
	   << "No Trigger or End Path Found in the Config File" <<endl;
       
       std::vector<std::string> allEndPaths = 
	 procPset->getParameter<std::vector<std::string> >("@end_paths");
       
       if (allEndPaths.empty())
	 throw cms::Exception("collectStreamerPSets","ServiceManager")
	   << "No End Path Found in the Config File" <<endl;
       
       for(std::vector<std::string>::iterator it = allEndPaths.begin(), itEnd = allEndPaths.end();
	   it != itEnd;
	   ++it) {
	   std::vector<std::string> anEndPath = procPset->getParameter<std::vector<std::string> >((*it));
	   for(std::vector<std::string>::iterator i = anEndPath.begin(), iEnd = anEndPath.end();
	       i != iEnd; ++i) {
	       ParameterSet aModInEndPathPset = 
		 procPset->getParameter<ParameterSet>((*i));
	       if (aModInEndPathPset.empty())
		 throw cms::Exception("collectStreamerPSets","ServiceManager")
		   << "Empty End Path Found in the Config File" <<endl;
	      
	       std::string mod_type = aModInEndPathPset.getParameter<std::string> ("@module_type");
	       if (mod_type == "EventStreamFileWriter") {
		 outModPSets_.push_back(aModInEndPathPset);
               }
               else if (mod_type == "ErrorStreamFileWriter" ||
                        mod_type == "FRDStreamFileWriter") {
                 errorStreamPSetIndex_ = outModPSets_.size();
                 outModPSets_.push_back(aModInEndPathPset);
               }
	   }
       }
     } catch (cms::Exception & e) {
       std::cerr << "cms::Exception: " << e.explainSelf() << std::endl;
       std::cerr << "std::Exception: " << e.what() << std::endl;
       throw cms::Exception("collectStreamerPSets") << e.explainSelf() << std::endl;
     }
}

boost::shared_ptr<stor::SMOnlyStats> ServiceManager::get_stats()
{ 
// Copy measurements for a different thread potentially
// TODO create each time or use a data member?
    boost::shared_ptr<stor::SMOnlyStats> outstats(new stor::SMOnlyStats() );

    if ( pmeter_->getStats().shortTermCounter_->hasValidResult() )
    {
      stor::SMPerfStats stats = pmeter_->getStats();

      outstats->instantBandwidth_= stats.shortTermCounter_->getValueRate();
      outstats->instantRate_     = stats.shortTermCounter_->getSampleRate();
      outstats->instantLatency_  = 1000000.0 / outstats->instantRate_;

      double now = stor::ForeverCounter::getCurrentTime();
      outstats->totalSamples_    = stats.longTermCounter_->getSampleCount();
      outstats->duration_        = stats.longTermCounter_->getDuration(now);
      outstats->meanBandwidth_   = stats.longTermCounter_->getValueRate(now);
      outstats->meanRate_        = stats.longTermCounter_->getSampleRate(now);
      outstats->meanLatency_     = 1000000.0 / outstats->meanRate_;

      outstats->maxBandwidth_    = stats.maxBandwidth_;
      outstats->minBandwidth_    = stats.minBandwidth_;
    }

    // for time period bandwidth performance measurements
    if ( pmeter_->getStats().shortPeriodCounter_->hasValidResult() )
    {
      stor::SMPerfStats stats = pmeter_->getStats();

      outstats->instantBandwidth2_= stats.shortPeriodCounter_->getValueRate();
      outstats->instantRate2_     = stats.shortPeriodCounter_->getSampleRate();
      outstats->instantLatency2_  = 1000000.0 / outstats->instantRate2_;

      double now = stor::ForeverCounter::getCurrentTime();
      outstats->totalSamples2_    = stats.longTermCounter_->getSampleCount();
      outstats->duration2_        = stats.longTermCounter_->getDuration(now);
      outstats->meanBandwidth2_   = stats.longTermCounter_->getValueRate(now);
      outstats->meanRate2_        = stats.longTermCounter_->getSampleRate(now);
      outstats->meanLatency2_     = 1000000.0 / outstats->meanRate2_;

      outstats->maxBandwidth2_    = stats.maxBandwidth2_;
      outstats->minBandwidth2_    = stats.minBandwidth2_;
    }
    outstats->receivedVolume_ = pmeter_->totalvolumemb();
    outstats->samples_ = samples_;
    outstats->period4samples_ = period4samples_;
    return outstats;
}
