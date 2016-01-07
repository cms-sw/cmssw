#ifndef SISTRIPPOPCON_DB_HANDLER_H
#define SISTRIPPOPCON_DB_HANDLER_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

namespace popcon{
  
  template <typename T, typename U>
    class SiStripPopConDbObjHandler : public popcon::PopConSourceHandler<T>{
    public:

    SiStripPopConDbObjHandler(const edm::ParameterSet& pset):
      m_name(pset.getUntrackedParameter<std::string>("name","SiStripPopConDbObjHandler")),
      m_since(pset.getUntrackedParameter<uint32_t>("since",5)),
      m_debugMode(pset.getUntrackedParameter<bool>("debug",false)){}; 

    //---------------------------------------
    //
    ~SiStripPopConDbObjHandler(){}; 

    //---------------------------------------
    //
    void getNewObjects(){
      edm::LogInfo   ("SiStripPopConDbObjHandler") << "[SiStripPopConDbObjHandler::getNewObjects] for PopCon application " << m_name;
     
      if (m_debugMode){
	std::stringstream ss;
	ss << "\n\n------- " << m_name 
	   << " - > getNewObjects\n"; 
	if (this->tagInfo().size){
	  //check whats already inside of database
	  ss << "\ngot offlineInfo"<< this->tagInfo().name 
	     << "\n size " << this->tagInfo().size 
	     << "\n" << this->tagInfo().token 
	     << "\n last object valid since " << this->tagInfo().lastInterval.first 
	     << "\n token " << this->tagInfo().lastPayloadToken 
	     << "\n UserText " << this->userTextLog()
	     << "\n LogDBEntry \n" 
	     << this->logDBEntry().logId<< "\n"
	     << this->logDBEntry().destinationDB<< "\n"   
	     << this->logDBEntry().provenance<< "\n"
	     << this->logDBEntry().usertext<< "\n"
	     << this->logDBEntry().iovtag<< "\n"
	     << this->logDBEntry().iovtimetype<< "\n"
	     << this->logDBEntry().payloadIdx<< "\n"
	     << this->logDBEntry().payloadClass<< "\n"
	     << this->logDBEntry().payloadToken<< "\n"
	     << this->logDBEntry().exectime<< "\n"
	     << this->logDBEntry().execmessage<< "\n";
	  if(this->logDBEntry().usertext!="")
	    ss<< "\n-- user text " << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@")) ;
	  
	} else {
	  ss << " First object for this tag ";
	}
	edm::LogInfo   ("SiStripPopConDbObjHandler") << ss.str();
      }
      
      condObjBuilder->initialize(); 
      
      if (isTransferNeeded())
	setForTransfer();
  
      edm::LogInfo   ("SiStripPopConDbObjHandler") << "[SiStripPopConDbObjHandler::getNewObjects] for PopCon application " << m_name << " Done\n--------------\n";
    }


    //---------------------------------------
    //
    std::string id() const { return m_name;}

    private:
    //methods
    
    std::string getDataType(){return typeid(T).name();}
    

    
    //---------------------------------------
    //
    bool isTransferNeeded(){

      edm::LogInfo   ("SiStripPopConDbObjHandler") << "[SiStripPopConDbObjHandler::isTransferNeeded] checking for transfer " << std::endl;

      if(m_since<=this->tagInfo().lastInterval.first){
	edm::LogInfo   ("SiStripPopConDbObjHandler") 
	  << "[SiStripPopConDbObjHandler::isTransferNeeded] \nthe current starting iov " << m_since
	  << "\nis not compatible with the last iov ("  
	  << this->tagInfo().lastInterval.first << ") open for the object " 
	  << this->logDBEntry().payloadClass << " \nin the db " 
	  << this->logDBEntry().destinationDB << " \n NO TRANSFER NEEDED";
	return false;
      }
      
      std::stringstream ss_logdb, ss;
      
      //get log information from previous upload
      if (this->logDBEntry().usertext!="")
	ss_logdb << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@")+2);


      condObjBuilder->getMetaDataString(ss);
      if(condObjBuilder->checkForCompatibility(ss_logdb.str())){
	
	this->m_userTextLog = "@ " + ss.str();
	
	edm::LogInfo   ("SiStripPopConDbObjHandler") 
	  << "[SiStripPopConDbObjHandler::isTransferNeeded] \nthe selected conditions will be uploaded: " << ss.str()
	  << "\n Current MetaData - "<< ss.str()  << "\n Last Uploaded MetaData- " << ss_logdb.str() << "\n Fine";
	
	return true;
      } else {
	edm::LogInfo   ("SiStripPopConDbObjHandler") 
	  << "[SiStripPopConDbObjHandler::isTransferNeeded] \nthe current MetaData conditions " << ss.str() 
	  << "\nare not compatible with the MetaData Conditions of the last iov ("  
	  << this->tagInfo().lastInterval.first << ") open for the object " 
	  << this->logDBEntry().payloadClass << " \nin the db " 
	  << this->logDBEntry().destinationDB << " \nConditions: "  << ss_logdb.str() << "\n NO TRANSFER NEEDED";
	return false;
      }	
    }


    //---------------------------------------
    //
    void setForTransfer(){
      edm::LogInfo   ("SiStripPopConDbObjHandler") << "[SiStripPopConDbObjHandler::setForTransfer] " << m_name << " getting data to be transferred "  << std::endl;
      
      T *obj=0; 
      condObjBuilder->getObj(obj);
 
      if(!this->tagInfo().size)
	m_since=1;
      else
	if (m_debugMode)
	  m_since=this->tagInfo().lastInterval.first+1; 

      if (obj!=0){

	edm::LogInfo   ("SiStripPopConDbObjHandler") <<"setting since = "<< m_since <<std::endl;
	this->m_to_transfer.push_back(std::make_pair(obj,m_since));
      }else{
	edm::LogError   ("SiStripPopConDbObjHandler") <<"[SiStripPopConDbObjHandler::setForTransfer] " << m_name << "  : NULL pointer of obj " << typeid(T).name() << " reported by SiStripCondObjBuilderFromDb\n Transfer aborted"<<std::endl;
      }
    }

    private: 
    // data members
    std::string m_name;
    unsigned long long m_since;
    bool m_debugMode;
    edm::Service<U> condObjBuilder;
  };
}

#endif //SISTRIPPOPCON_DB_HANDLER_H
