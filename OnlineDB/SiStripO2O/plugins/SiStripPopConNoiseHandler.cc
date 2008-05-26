#include "OnlineDB/SiStripO2O/plugins/SiStripPopConNoiseHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"

#include<iostream>
#include<sstream>
#include<vector>

popcon::SiStripPopConNoiseHandler::SiStripPopConNoiseHandler (const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","SiStripPopConNoiseHandler")),
  m_since(pset.getUntrackedParameter<uint32_t>("since",5)),
  m_debugMode(pset.getUntrackedParameter<bool>("debug",false)){
}

popcon::SiStripPopConNoiseHandler::~SiStripPopConNoiseHandler(){}


void popcon::SiStripPopConNoiseHandler::getNewObjects() {

  std::stringstream ss;
  ss << "\n\n------- " << m_name 
     << " - > getNewObjects\n"; 
  if (tagInfo().size){
      //check whats already inside of database
    ss << "got offlineInfo"<<
      tagInfo().name << ", size " << tagInfo().size << " " << tagInfo().token 
       << " , last object valid since " 
       << tagInfo().lastInterval.first << " token "   
       << tagInfo().lastPayloadToken << "\n\n UserText " << userTextLog() 
       << "\n LogDBEntry \n" 
       << logDBEntry().logId<< "\n"
       << logDBEntry().destinationDB<< "\n"   
       << logDBEntry().provenance<< "\n"
       << logDBEntry().usertext<< "\n"
       << logDBEntry().iovtag<< "\n"
       << logDBEntry().iovtimetype<< "\n"
       << logDBEntry().payloadIdx<< "\n"
       << logDBEntry().payloadName<< "\n"
       << logDBEntry().payloadToken<< "\n"
       << logDBEntry().payloadContainer<< "\n"
       << logDBEntry().exectime<< "\n"
       << logDBEntry().execmessage<< "\n"
       << "\n\n-- user text " << logDBEntry().usertext.substr(logDBEntry().usertext.find_last_of("@")) ;
  } else {
    ss << " First object for this tag ";
  }
  edm::LogInfo   ("SiStripPopConNoiseHandler") << ss.str();
  /*
    if (tagInfo().size>0) {
    Ref payload = lastPayload();
    edm::LogInfo   ("SiStripPopConNoiseHandler")<<"size of last payload  "<< 
      payload->getDataVectorEnd()-payload->getDataVectorBegin()<<std::endl;
  }
  */
  
  if (isTransferNeeded())
    setForTransfer();
  
  //edm::LogInfo   ("SiStripPopConNoiseHandler") << "\n\n------- " << m_name << " - > getNewObjects \n Usertext" << m_userTextLog << std::endl;
}

bool popcon::SiStripPopConNoiseHandler::isTransferNeeded(){

  edm::LogInfo   ("SiStripPopConNoiseHandler") << "\n\n-------\n isTransferNeeded "  << std::endl;
  std::stringstream ss_logdb, ss;
  std::stringstream ss1; 

  //get log information from previous upload
  if (tagInfo().size)
    ss_logdb << logDBEntry().usertext.substr(logDBEntry().usertext.find_last_of("@"));
  else
    ss_logdb << "";

  //get current config DB parameter
  const SiStripDbParams& dbParams = condObjBuilder->dbParams();

  SiStripDbParams::const_iterator_range partitionsRange = dbParams.partitions(); 

  SiStripDbParams::SiStripPartitions::const_iterator ipart = partitionsRange.begin();
  SiStripDbParams::SiStripPartitions::const_iterator ipartEnd = partitionsRange.end();
  for ( ; ipart != ipartEnd; ++ipart ) { 
    SiStripPartition partition=ipart->second;
    //partition.print(ss1,true);
    ss  << "@ "
	<< " Partition " << partition.partitionName() 
	<< " CabVer "    << partition.cabVersion().first << "." << partition.cabVersion().second
	<< " FedVer "    << partition.fedVersion().first << "." << partition.fedVersion().second
      ;
  }
  
  if (!strcmp(ss.str().c_str(),ss_logdb.str().c_str())){
    // string are equal, no need to do transfer
    edm::LogInfo   ("SiStripPopConNoiseHandler") << "[isTransferNeeded] the selected conditions are already uploaded in the last iov ("  << tagInfo().lastInterval.first << ") open for the object " << logDBEntry().payloadName << " in the db " << logDBEntry().destinationDB << "parameters: "  << ss.str() << "\n NO TRANSFER NEEDED";
    return false;
  }
  m_userTextLog = ss.str();
  edm::LogInfo   ("SiStripPopConNoiseHandler") << "[isTransferNeeded] the selected conditions will be uploaded: \n A- "<< ss.str()  << "\n B- " << ss_logdb.str() << "\n Fine";
  //edm::LogInfo   ("SiStripPopConNoiseHandler") << "\n\n-------\n " << ss1.str() << " " << ss.str();
  return true;
}

void popcon::SiStripPopConNoiseHandler::setForTransfer(){

  edm::LogInfo   ("SiStripPopConNoiseHandler") << "\n\n-------\n setForTransfer "  << std::endl;

  condObjBuilder->buildCondObj();
  SiStripNoises *noise;
  
  condObjBuilder->getValue(noise);

  //DEBUG//////
  std::vector<uint32_t> ndetid;
  noise->getDetIds(ndetid);
  edm::LogInfo("SiStripO2O") << " Noise Found " << ndetid.size() << " DetIds";
  SiStripNoises::RegistryIterator ireg=noise->getRegistryVectorBegin();
  SiStripNoises::RegistryIterator iregEnd=noise->getRegistryVectorEnd();
  for (; ireg!=iregEnd; ++ireg){
    
    SiStripNoises::Range range(noise->getDataVectorBegin()+ireg->ibegin,noise->getDataVectorBegin()+ireg->iend);
    int strip=0;
    edm::LogInfo("SiStripO2O")  << "NOISE detid " <<  ireg->detid << " \t"
				<< " strip " << strip << " \t"
				<< noise->getNoise(strip,range)     << " \t" 
				<< std::endl; 	    
  } 
  ////

  if (m_debugMode)
    if(tagInfo().size)
      m_since=tagInfo().lastInterval.first+1; 
    else
      m_since=1;

  edm::LogInfo   ("SiStripPopConNoiseHandler") <<"setting since = "<< m_since <<std::endl;
  
  m_to_transfer.push_back(std::make_pair((SiStripNoises*)noise,m_since));

}
