#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVHandler.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"
#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

popcon::SiStripModuleHVHandler::SiStripModuleHVHandler (const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","SiStripModuleHVHandler"))
{ }

popcon::SiStripModuleHVHandler::~SiStripModuleHVHandler() { 
  LogTrace("SiStripModuleHVHandler") << "[SiStripModuleHVHandler::" << __func__ << "] Destructing ...";
}

void popcon::SiStripModuleHVHandler::getNewObjects()
{
  std::stringstream dbstr;
  dbstr << "\n\n------- " << m_name 
	<< " - > getNewObjects\n"; 
  if (tagInfo().size){
    //check whats already inside of database
    dbstr << "got offlineInfo" << tagInfo().name << ", size " << tagInfo().size << " " << tagInfo().token 
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
    dbstr << " First object for this tag ";
  }
  edm::LogInfo   ("SiStripModuleHVHandler") << dbstr.str();
  
  // Do the transfer!
  setForTransfer();
}

void popcon::SiStripModuleHVHandler::setForTransfer() { 
  edm::LogInfo("SiStripModuleHVHandler") << "[SiStripModuleHVHandler::" << __func__ << "]" << std::endl;

  // build the object!
  resultVec.clear();
  modHVBuilder->BuildModuleHVObj();
  resultVec = modHVBuilder->getModulesVOff();
  std::vector< std::vector<uint32_t> > payloadStats = modHVBuilder->getPayloadStats();
  
  if (!resultVec.empty()){
    // assume by default that transfer is needed
    bool is_transfer_needed = true;
    
    // check if there is an existing payload and retrieve if there is
    if (tagInfo().size > 0) {
      Ref payload = lastPayload();
      // resultVec does not contain duplicates, so only need to compare payload with resultVec[0]
      SiStripDetVOff * modV = resultVec[0].first;
      if (*modV == *payload) {
	is_transfer_needed = false;
	LogTrace("SiStripModuleHVHandler") << "[SiStripModuleHVHandler::" << __func__ << "] Transfer of first element not required!";
      }
    } else {
      LogTrace("SiStripModuleHVHandler") << "[SiStripModuleHVHandler::" << __func__ << "] No previous payload";
    }
    
    std::stringstream ss;
    ss << "@@@ Number of payloads transferred " << resultVec.size() << ". "
       << "PayloadNo/Badmodules/NoAdded/NoRemoved: ";
    for (unsigned int j = 0; j < payloadStats.size(); j++) {
      ss << j << "/" << payloadStats[j][0] << "/" << payloadStats[j][1] << "/" << payloadStats[j][2] << ". ";
    }
    this->m_userTextLog = ss.str();
    
    for (unsigned int i = 0; i < resultVec.size(); i++) {
      if (i == 0 && is_transfer_needed) {
	this->m_to_transfer.push_back(resultVec[i]);
      } else if (i > 0) {
	this->m_to_transfer.push_back(resultVec[i]);
      }
    }

    LogTrace("SiStripModuleHVHandler") << "[SiStripModuleHVHandler::" << __func__ << "] " << ss.str();

  } else {
    edm::LogError("SiStripModuleHVHandler") << "[SiStripModuleHVHandler::" << __func__ << "] " 
					    << m_name << "  : NULL pointer reported by SiStripModuleHVBuilder"
					    << "\n Transfer aborted"<< std::endl;
  }
}
