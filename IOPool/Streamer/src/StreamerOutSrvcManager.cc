#include "IOPool/Streamer/interface/StreamerOutSrvcManager.h"
#include "IOPool/Streamer/interface/StreamerOutputService.h"

//using namespace edm;
using namespace std;

namespace edm {

StreamerOutSrvcManager::StreamerOutSrvcManager(const std::string& config):
  outModPSets_(0),
  managedOutputs_(0)
   {
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
     try{
       
       ProcessDesc  pdesc(config.c_str());
       
       boost::shared_ptr<ParameterSet> procPset = pdesc.getProcessPSet();
       std::cout<<"Process PSet:"<<procPset->toString()<<endl;
       
       //cout << "Module Label: " << procPset->getParameter<string>("@module_label")
       //     << std::endl;
       
       ParameterSet allTrigPaths = procPset->
	 getUntrackedParameter<ParameterSet>("@trigger_paths");
       std::cout <<"Found  Trig Path :"<<allTrigPaths.toString()<<endl;
       
       if (allTrigPaths.empty())
         throw cms::Exception("ParamSetWalker","ParamSetWalker")
	   << "No Trigger or End Path Found in the Config File" <<endl;
       
       std::vector<std::string> allEndPaths = 
	 allTrigPaths.getParameter<std::vector<std::string> >("@end_paths");
       
       if (allEndPaths.empty())
	 throw cms::Exception("ParamSetWalker","ParamSetWalker")
	   << "No End Path Found in the Config File" <<endl;
       
       for(std::vector<std::string>::iterator it = allEndPaths.begin();
	   it != allEndPaths.end();
	   ++it)
	 {
	   std::cout <<"Found an end Path :"<<(*it)<<std::endl;
	   //Lets try to get this PSet from the Process PSet
	   std::vector<std::string> anEndPath = procPset->getParameter<std::vector<std::string> >((*it));
	   for(std::vector<std::string>::iterator it = anEndPath.begin();
	       it != anEndPath.end(); ++it) 
	     {
	       std::cout <<"Found a end Path PSet :"<<(*it)<<endl;
	       //Lets Check this Module if its a EventStreamFileWriter type
	      ParameterSet aModInEndPathPset = 
		procPset->getParameter<ParameterSet>((*it));
	      if (aModInEndPathPset.empty())
		throw cms::Exception("ParamSetWalker","ParamSetWalker")
		  << "Empty End Path Found in the Config File" <<endl;
	      std::cout <<"This Module PSet is: "<<aModInEndPathPset.toString()<<std::endl;
	      std::string mod_type = aModInEndPathPset.getParameter<std::string> ("@module_type");
	      std::cout <<"Type of This Module is: "<<mod_type<<endl;
	      if (mod_type == "EventStreamFileWriter") 
		{
		  cout<<"FOUND WHAT WAS LOOKING FOR:::"<<std::endl;
		  outModPSets_.push_back(aModInEndPathPset);
		  
		  //std::string fileName = aModInEndPathPset.getParameter<string> ("fileName");
		  //std::cout <<"Streamer File Name:"<<fileName<<endl;
		  //std::string indexFileName = aModInEndPathPset.getParameter<string> ("indexFileName");
		  //std::cout <<"Index File Name:"<<indexFileName<<endl;
		  //ParameterSet selectEventsPSet = 
		  //  aModInEndPathPset.getUntrackedParameter<ParameterSet>("SelectEvents");
		  //if ( !selectEventsPSet.empty() ) {
		  //  std::cout <<"SelectEvents: "<<selectEventsPSet.toString()<<std::endl;
		  //}
		}
	     }
	 }
     }catch (cms::Exception & e) {
       std::cerr << "cms::Exception: " << e.explainSelf() << std::endl;
       std::cerr << "std::Exception: " << e.what() << std::endl;
     }
}

void StreamerOutSrvcManager::manageInitMsg(unsigned long maxFileSize, double highWaterMark,
		std::string path, std::string mpath, InitMsgView& init_message)
     {
      // An INIT Message has arrived, and we need to open 
      // StreamerOutputService (output file)
      // for each of outModPSets_

      for(std::vector<ParameterSet>::iterator it = outModPSets_.begin();
          it != outModPSets_.end(); ++it)
        {
	   //Get the filename
           std::string fileName = (*it).getParameter<string> ("fileName");
           //ParameterSet selectEventsPSet = 
           //     (*it).getUntrackedParameter<ParameterSet>("SelectEvents");
           //if ( !selectEventsPSet.empty() ) {
           //    std::cout <<"SelectEvents: "<<selectEventsPSet.toString()<<std::endl;
           //}
           //else {
           //  std::cout << "Should i make up a Select All PSet ?" <<endl;
           //}   
           
           StreamerOutputService* outputFile = new StreamerOutputService((*it));
           //invoke its init, later StreamerOutputService CTOR will call its own init 
           //it should take a SelectEvents PSet too
           outputFile->init(fileName, maxFileSize, highWaterMark,
                             path, mpath, init_message);
          
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

} //emd-namespace

