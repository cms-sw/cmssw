#ifndef SISTRIPPOPCON_CONFIGDB_HANDLER_H
#define SISTRIPPOPCON_CONFIGDB_HANDLER_H

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/DBOutputService/interface/TagInfo.h"
#include "CondCore/DBOutputService/interface/LogDBEntry.h"

#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"

#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

namespace popcon{
  
  template <typename T>
    class SiStripPopConConfigDbObjHandler : public popcon::PopConSourceHandler<T>{
    public:

    //---------------------------------------
    //
    SiStripPopConConfigDbObjHandler(const edm::ParameterSet& pset):
      m_name(pset.getUntrackedParameter<std::string>("name","SiStripPopPopConConfigDbObjHandler")),
      m_since(pset.getUntrackedParameter<uint32_t>("since",5)),
      m_debugMode(pset.getUntrackedParameter<bool>("debug",false)){}; 

    //---------------------------------------
    //
    ~SiStripPopConConfigDbObjHandler(){}; 

    //---------------------------------------
    //
    void getNewObjects(){
      edm::LogInfo   ("SiStripPopPopConConfigDbObjHandler") << "[getNewObjects] for PopCon application " << m_name;
     
      if (m_debugMode){
	std::stringstream ss;
	ss << "\n\n------- " << m_name 
	   << " - > getNewObjects\n"; 
	if (this->tagInfo().size){
	  //check whats already inside of database
	  ss << "got offlineInfo"<<
	    this->tagInfo().name << ", size " << this->tagInfo().size << " " << this->tagInfo().token 
	     << " , last object valid since " 
	     << this->tagInfo().lastInterval.first << " token "   
	     << this->tagInfo().lastPayloadToken << "\n\n UserText " << this->userTextLog() 
	     << "\n LogDBEntry \n" 
	     << this->logDBEntry().logId<< "\n"
	     << this->logDBEntry().destinationDB<< "\n"   
	     << this->logDBEntry().provenance<< "\n"
	     << this->logDBEntry().usertext<< "\n"
	     << this->logDBEntry().iovtag<< "\n"
	     << this->logDBEntry().iovtimetype<< "\n"
	     << this->logDBEntry().payloadIdx<< "\n"
	     << this->logDBEntry().payloadName<< "\n"
	     << this->logDBEntry().payloadToken<< "\n"
	     << this->logDBEntry().payloadContainer<< "\n"
	     << this->logDBEntry().exectime<< "\n"
	     << this->logDBEntry().execmessage<< "\n"
	     << "\n\n-- user text " << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@")) ;
	} else {
	  ss << " First object for this tag ";
	}
	edm::LogInfo   ("SiStripPopPopConConfigDbObjHandler") << ss.str();
      }
      if (isTransferNeeded())
	setForTransfer();
  
      edm::LogInfo   ("SiStripPopPopConConfigDbObjHandler") << "[getNewObjects] for PopCon application " << m_name << " Done";
    }


    //---------------------------------------
    //
    std::string id() const { return m_name;}
    
    private:
    //methods
    
    //---------------------------------------
    //
    bool isTransferNeeded(){


      edm::LogInfo   ("SiStripPopPopConConfigDbObjHandler") << "[isTransferNeeded] checking for transfer"  << std::endl;
      std::stringstream ss_logdb, ss;
      std::stringstream ss1; 

      //get log information from previous upload
      if (this->tagInfo().size)
	ss_logdb << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@"));
      else
	ss_logdb << "";
      
      //get current config DB parameter
      const SiStripDbParams& dbParams = condObjBuilder->dbParams();
      
      SiStripDbParams::const_iterator_range partitionsRange = dbParams.partitions(); 

      SiStripDbParams::SiStripPartitions::const_iterator ipart = partitionsRange.begin();
      SiStripDbParams::SiStripPartitions::const_iterator ipartEnd = partitionsRange.end();
      for ( ; ipart != ipartEnd; ++ipart ) { 
	SiStripPartition partition=ipart->second;
	partition.print(ss1,true);
	ss  << "@ "
	    << " Partition " << partition.partitionName() 
	    << " CabVer "    << partition.cabVersion().first << "." << partition.cabVersion().second
	    << " FedVer "    << partition.fedVersion().first << "." << partition.fedVersion().second
	  ;
      }
  
      if (!strcmp(ss.str().c_str(),ss_logdb.str().c_str())){
	//string are equal, no need to do transfer
					  edm::LogInfo   ("SiStripPopPopConConfigDbObjHandler") 
					    << "[isTransferNeeded] the selected conditions are already uploaded in the last iov ("  
					    << this->tagInfo().lastInterval.first << ") open for the object " 
					    << this->logDBEntry().payloadName << " in the db " 
					    << this->logDBEntry().destinationDB << " parameters: "  << ss.str() << "\n NO TRANSFER NEEDED";
	return false;
      }
      this->m_userTextLog = ss.str();
      edm::LogInfo   ("SiStripPopPopConConfigDbObjHandler") 
	<< "[isTransferNeeded] the selected conditions will be uploaded: " << ss.str() 
	<< "\n A- "<< ss.str()  << "\n B- " << ss_logdb.str() << "\n Fine";

      return true;
    }


    //---------------------------------------
    //
    void setForTransfer(){
      edm::LogInfo   ("SiStripPopPopConConfigDbObjHandler") << "[setForTransfer] getting data to be transferred "  << std::endl;
      
      T *obj; 
      condObjBuilder->getValue(obj);
 
      if(!this->tagInfo().size)
	m_since=1;
      else
	if (m_debugMode)
	  m_since=this->tagInfo().lastInterval.first+1; 


      edm::LogInfo   ("SiStripPopPopConConfigDbObjHandler") <<"setting since = "<< m_since <<std::endl;
      this->m_to_transfer.push_back(std::make_pair(obj,m_since));
    }

    private: 
    // data members
    std::string m_name;
    unsigned long long m_since;
    bool m_debugMode;
    edm::Service<SiStripCondObjBuilderFromDb> condObjBuilder;
  };
}

#endif //SISTRIPPOPCON_CONFIGDB_HANDLER_H
