#include "IOPool/Streamer/interface/StreamerOutSrvcManager.h"
#include "IOPool/Streamer/interface/StreamerOutputService.h"

//using namespace edm;
using namespace std;

namespace edm {

StreamerOutSrvcManager::StreamerOutSrvcManager(const std::string& config):
  outModPSets_(0),
  managedOutputs_(0)
   {
      //cout << "CONFIG RECEIVED IS: "<<config<<endl;
      collectStreamerPSets(config);
   } 

StreamerOutSrvcManager::~StreamerOutSrvcManager()
{
   for(std::vector<StreamerOutputService*>::iterator it = managedOutputs_.begin();
       it != managedOutputs_.end(); ++it)
       { 
          delete (*it);
       }
}

void StreamerOutSrvcManager::stop()
{
       // Received a stop(), Pass it ON to each outputFile
       // outputFile(service) will close files and write EOF Message
       
       // Not caling stop() from destructor, as its not a preffered way
       // for some mysterious reason. Hope to figure that out soon. 
       // Though it seems OK to call it from destructor here
       // instead of explicitly calling in FragCollector

       for(std::vector<StreamerOutputService*>::iterator it = managedOutputs_.begin();
          it != managedOutputs_.end(); ++it)
        {
            (*it)->stop();
        }
}

void StreamerOutSrvcManager::collectStreamerPSets(const std::string& config)
{
     // wrote similar example code in IOPool/Streamer/test/ParamSetWalker_t.cpp 
     // this method is diluted version of same code.
     // if more items needs to be extracted for config, refer to example code

     try{
       
       ProcessDesc  pdesc(config.c_str());
       
       boost::shared_ptr<ParameterSet> procPset = pdesc.getProcessPSet();
       //std::cout<<"Process PSet:"<<procPset->toString()<<endl;
       
       ParameterSet allTrigPaths = procPset->
	 getUntrackedParameter<ParameterSet>("@trigger_paths");
       //std::cout <<"Found  Trig Path :"<<allTrigPaths.toString()<<endl;
       
       if (allTrigPaths.empty())
         throw cms::Exception("collectStreamerPSets","StreamerOutSrvcManager")
	   << "No Trigger or End Path Found in the Config File" <<endl;
       
       std::vector<std::string> allEndPaths = 
	 allTrigPaths.getParameter<std::vector<std::string> >("@end_paths");
       
       if (allEndPaths.empty())
	 throw cms::Exception("collectStreamerPSets","StreamerOutSrvcManager")
	   << "No End Path Found in the Config File" <<endl;
       
       for(std::vector<std::string>::iterator it = allEndPaths.begin();
	   it != allEndPaths.end();
	   ++it)
	 {
	   //std::cout <<"Found an end Path :"<<(*it)<<std::endl;
	   //Lets try to get this PSet from the Process PSet
	   std::vector<std::string> anEndPath = procPset->getParameter<std::vector<std::string> >((*it));
	   for(std::vector<std::string>::iterator it = anEndPath.begin();
	       it != anEndPath.end(); ++it) 
	     {
	       //std::cout <<"Found a end Path PSet :"<<(*it)<<endl;
	       //Lets Check this Module if its a EventStreamFileWriter type
	      ParameterSet aModInEndPathPset = 
		procPset->getParameter<ParameterSet>((*it));
	      if (aModInEndPathPset.empty())
		throw cms::Exception("collectStreamerPSets","StreamerOutSrvcManager")
		  << "Empty End Path Found in the Config File" <<endl;
	      //std::cout <<"This Module PSet is: "<<aModInEndPathPset.toString()<<std::endl;
	      std::string mod_type = aModInEndPathPset.getParameter<std::string> ("@module_type");
	      //std::cout <<"Type of This Module is: "<<mod_type<<endl;
	      if (mod_type == "EventStreamFileWriter") 
	      //if (mod_type == "I2OStreamConsumer") 
		{
		  //cout<<"FOUND WHAT WAS LOOKING FOR:::"<<std::endl;
		  outModPSets_.push_back(aModInEndPathPset);
		}
	     }
	 }
     }catch (cms::Exception & e) {
       std::cerr << "cms::Exception: " << e.explainSelf() << std::endl;
       std::cerr << "std::Exception: " << e.what() << std::endl;
     }
}

void StreamerOutSrvcManager::manageInitMsg(std::string fileName, unsigned long maxFileSize, double highWaterMark,
		std::string path, std::string mpath, std::string catalog, uint32 disks, InitMsgView& init_message)
     {

      //received file name is ignored for now, and later we can remove it, not understood if its required

      // An INIT Message has arrived, and we need to open 
      // StreamerOutputService (output file)
      // for each of outModPSets_

      for(std::vector<ParameterSet>::iterator it = outModPSets_.begin();
          it != outModPSets_.end(); ++it)
        {
	   //Get the filename
           std::string fileNameLocal = (*it).getParameter<string> ("fileName");
          
           //Other parameters can also be pulled here, if provided in config file. 
           
           StreamerOutputService* outputFile = new StreamerOutputService((*it));
           //invoke its init, later StreamerOutputService CTOR will call its own init 
           //it should take a SelectEvents PSet too
           outputFile->init(fileNameLocal, maxFileSize, highWaterMark,
                             path, mpath, catalog, disks, init_message);
          
           //Stor it in list of managed outputFiles
           managedOutputs_.push_back(outputFile);
        }
     }

void StreamerOutSrvcManager::manageEventMsg(EventMsgView& msg)
    {
       //Received an Event Message, Pass it ON to each outputFile
       // outputFile(service) will decide to write or Pass it
       for(std::vector<StreamerOutputService*>::iterator it = managedOutputs_.begin();
          it != managedOutputs_.end(); ++it)
        {
            (*it)->writeEvent(msg);
        }
    }


std::list<std::string>& StreamerOutSrvcManager::get_filelist() 
    { 
     filelist_.clear();
     for(std::vector<StreamerOutputService*>::iterator it = managedOutputs_.begin();
          it != managedOutputs_.end(); ++it)
        {
            std::list<std::string>& sub_list = (*it)->get_filelist();
            filelist_.assign(sub_list.begin(), sub_list.end() );
        } 
     return filelist_; 
    }


std::list<std::string>& StreamerOutSrvcManager::get_currfiles()
    { 
      currfiles_.clear();
      for(std::vector<StreamerOutputService*>::iterator it = managedOutputs_.begin();
          it != managedOutputs_.end(); ++it)
        {
            currfiles_.push_back((*it)->get_currfile());
        }
      return currfiles_;  
    }
} //emd-namespace

