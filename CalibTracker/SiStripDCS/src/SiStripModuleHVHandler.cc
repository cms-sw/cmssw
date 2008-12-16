#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVHandler.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"
#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

popcon::SiStripModuleHVHandler::SiStripModuleHVHandler (const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","SiStripModuleHVHandler")),
  m_since(pset.getUntrackedParameter<uint32_t>("since",5))
{ }

popcon::SiStripModuleHVHandler::~SiStripModuleHVHandler() { }

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
  

  if (isTransferNeeded()) {
    setForTransfer();
  }
}

void popcon::SiStripModuleHVHandler::setForTransfer() { 
  edm::LogInfo   ("SiStripModuleHVHandler") << "\n\n-------\n setForTransfer "  << std::endl;
  
  SiStripModuleHV *modHV = 0;
  modHV = modHVBuilder->getSiStripModuleHV();

  if (!this->tagInfo().size) {
    m_since=1;
  } else {
    if (modHV != 0){
      edm::LogInfo("SiStripModuleHVHandler") << "setting since = " << m_since << std::endl;
      this->m_to_transfer.push_back(std::make_pair(modHV,m_since));
    } else {
      edm::LogError("SiStripModuleHVHandler") << "[setForTransfer] " 
					      << m_name << "  : NULL pointer reported by SiStripModuleHVBuilderDb"
					      << "\n Transfer aborted"<< std::endl;
    }
  }
}

bool popcon::SiStripModuleHVHandler::isTransferNeeded(){
  edm::LogInfo("SiStripModuleHVHandler") << "[isTransferNeeded] checking for transfer"  << std::endl;
  std::stringstream ss_logdb;

  // get log information from previous upload
  if (this->tagInfo().size) {
    ss_logdb << this->logDBEntry().usertext.substr(this->logDBEntry().usertext.find_last_of("@"));
  } else {
    ss_logdb << "";
  }

  // how to decide whether a new upload is needed?

  return false;
}
