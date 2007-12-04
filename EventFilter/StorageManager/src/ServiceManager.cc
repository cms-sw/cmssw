// $Id: ServiceManager.cc,v 1.3 2007/08/15 23:16:20 hcheung Exp $

#include <EventFilter/StorageManager/interface/ServiceManager.h>
#include "EventFilter/StorageManager/interface/Configurator.h"
#include <FWCore/Utilities/interface/Exception.h>

using namespace std;
using namespace edm;
using boost::shared_ptr;


ServiceManager::ServiceManager(const std::string& config):
  outModPSets_(0),
  managedOutputs_(0)
{
  collectStreamerPSets(config);
} 


ServiceManager::~ServiceManager()
{ 
  managedOutputs_.clear();
}


void ServiceManager::stop()
{
  for(StreamsIterator  it = managedOutputs_.begin(), itEnd = managedOutputs_.end();
      it != itEnd; ++it) {
      (*it)->stop();
  }
}


void ServiceManager::manageInitMsg(std::string catalog, uint32 disks, std::string sourceId, InitMsgView& view)
{
  boost::shared_ptr<stor::Parameter> smParameter_ = stor::Configurator::instance()->getParameter();
  for(std::vector<ParameterSet>::iterator it = outModPSets_.begin(), itEnd = outModPSets_.end();
      it != itEnd; ++it) {
      shared_ptr<StreamService> stream = shared_ptr<StreamService>(new StreamService((*it),view));
      stream->setCatalog(catalog);
      stream->setNumberOfFileSystems(disks);
      stream->setSourceId(sourceId);
      stream->setFileName(smParameter_ -> fileName());
      stream->setFilePath(smParameter_ -> filePath());
      stream->setMathBoxPath(smParameter_ -> mailboxPath());
      stream->setSetupLabel(smParameter_ -> setupLabel());
      stream->setHighWaterMark(smParameter_ -> highWaterMark());
      stream->setLumiSectionTimeOut(smParameter_ -> lumiSectionTimeOut());
      managedOutputs_.push_back(stream);
      stream->report(cout,3);
  }
}


void ServiceManager::manageEventMsg(EventMsgView& msg)
{
  bool eventAccepted = false;
  for(StreamsIterator  it = managedOutputs_.begin(), itEnd = managedOutputs_.end(); it != itEnd; ++it)
    eventAccepted = (*it)->nextEvent(msg) || eventAccepted;

}


//
// *** get all files from all streams
//
std::list<std::string>& ServiceManager::get_filelist() 
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
std::list<std::string>& ServiceManager::get_currfiles()
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
void ServiceManager::collectStreamerPSets(const std::string& config)
{

     try{
       
       ProcessDesc  pdesc(config.c_str());
       
       boost::shared_ptr<ParameterSet> procPset = pdesc.getProcessPSet();
       
        ParameterSet allTrigPaths = procPset->
	 getUntrackedParameter<ParameterSet>("@trigger_paths");
       
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
	       if (mod_type == "EventStreamFileWriter") 
		 outModPSets_.push_back(aModInEndPathPset);
	   }
       }
     } catch (cms::Exception & e) {
       std::cerr << "cms::Exception: " << e.explainSelf() << std::endl;
       std::cerr << "std::Exception: " << e.what() << std::endl;
       throw cms::Exception("collectStreamerPSets") << e.explainSelf() << std::endl;
     }
}

