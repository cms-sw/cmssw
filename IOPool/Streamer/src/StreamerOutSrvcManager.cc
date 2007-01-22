// $Id: StreamerOutSrvcManager.cc,v 1.15 2007/01/07 18:06:15 klute Exp $

#include "IOPool/Streamer/interface/StreamerOutSrvcManager.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace edm;
using boost::shared_ptr;


StreamerOutSrvcManager::StreamerOutSrvcManager(const std::string& config):
  outModPSets_(0),
  managedOutputs_(0)
{
  collectStreamerPSets(config);
} 


StreamerOutSrvcManager::~StreamerOutSrvcManager()
{ 
  managedOutputs_.clear();
}


void StreamerOutSrvcManager::stop()
{
  for(StreamsIterator  it = managedOutputs_.begin(), itEnd = managedOutputs_.end();
      it != itEnd; ++it) {
      (*it)->stop();
  }
}


void StreamerOutSrvcManager::manageInitMsg(std::string catalog, uint32 disks, std::string sourceId, InitMsgView& view)
{
  for(std::vector<ParameterSet>::iterator it = outModPSets_.begin(), itEnd = outModPSets_.end();
      it != itEnd; ++it) {
      shared_ptr<StreamService> stream = shared_ptr<StreamService>(new StreamService((*it),view));
      stream->setCatalog(catalog);
      stream->setNumberOfFileSystems(disks);
      stream->setSourceId(sourceId);
      managedOutputs_.push_back(stream);
      stream->report(cout,3);
  }
}


void StreamerOutSrvcManager::manageEventMsg(EventMsgView& msg)
{
  bool eventAccepted = false;
  for(StreamsIterator  it = managedOutputs_.begin(), itEnd = managedOutputs_.end(); it != itEnd; ++it)
    eventAccepted = (*it)->nextEvent(msg) || eventAccepted;

}


//
// *** get all files from all streams
//
std::list<std::string>& StreamerOutSrvcManager::get_filelist() 
{ 
  filelist_.clear();
  for(StreamsIterator it = managedOutputs_.begin(), itEnd = managedOutputs_.end();
      it != itEnd; ++it) {
      std::list<std::string> sub_list = (*it)->getFileList();
      if(sub_list.size() > 0)
	filelist_.insert(filelist_.end(), sub_list.begin(), sub_list.end());
  } 
  return filelist_; 
}


//
// *** get all current files from all streams
//
std::list<std::string>& StreamerOutSrvcManager::get_currfiles()
{ 
  currfiles_.clear();
  for(StreamsIterator it = managedOutputs_.begin(), itEnd = managedOutputs_.end();
      it != itEnd; ++it) {
      std::list<std::string> sub_list = (*it)->getCurrentFileList();
      if(sub_list.size() > 0)
	filelist_.insert(filelist_.end(), sub_list.begin(), sub_list.end());
  }
  return currfiles_;  
}


//
// *** wrote similar example code in IOPool/Streamer/test/ParamSetWalker_t.cpp 
// *** this method is diluted version of same code.
// *** if more items needs to be extracted for config, refer to example code
//
void StreamerOutSrvcManager::collectStreamerPSets(const std::string& config)
{

     try{
       
       ProcessDesc  pdesc(config.c_str());
       
       boost::shared_ptr<ParameterSet> procPset = pdesc.getProcessPSet();
       
        ParameterSet allTrigPaths = procPset->
	 getUntrackedParameter<ParameterSet>("@trigger_paths");
       
       if (allTrigPaths.empty())
         throw cms::Exception("collectStreamerPSets","StreamerOutSrvcManager")
	   << "No Trigger or End Path Found in the Config File" <<endl;
       
       std::vector<std::string> allEndPaths = 
	 allTrigPaths.getParameter<std::vector<std::string> >("@end_paths");
       
       if (allEndPaths.empty())
	 throw cms::Exception("collectStreamerPSets","StreamerOutSrvcManager")
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
		 throw cms::Exception("collectStreamerPSets","StreamerOutSrvcManager")
		   << "Empty End Path Found in the Config File" <<endl;
	      
	       std::string mod_type = aModInEndPathPset.getParameter<std::string> ("@module_type");
	       if (mod_type == "EventStreamFileWriter") 
		 outModPSets_.push_back(aModInEndPathPset);
	   }
       }
     } catch (cms::Exception & e) {
       std::cerr << "cms::Exception: " << e.explainSelf() << std::endl;
       std::cerr << "std::Exception: " << e.what() << std::endl;
     }
}

