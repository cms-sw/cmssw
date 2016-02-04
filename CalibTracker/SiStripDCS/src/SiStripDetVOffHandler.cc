#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffHandler.h"

popcon::SiStripDetVOffHandler::SiStripDetVOffHandler (const edm::ParameterSet& pset) :
  name_(pset.getUntrackedParameter<std::string>("name","SiStripDetVOffHandler")),
  deltaTmin_(pset.getParameter<uint32_t>("DeltaTmin")),
  maxIOVlength_(pset.getParameter<uint32_t>("MaxIOVlength")),
  debug_(pset.getUntrackedParameter<bool>("Debug", false))
{ }

popcon::SiStripDetVOffHandler::~SiStripDetVOffHandler() { 
  LogTrace("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] Destructing ...";
}

void popcon::SiStripDetVOffHandler::getNewObjects()
{
  std::cout << "[SiStripDetVOffHandler::getNewObjects]" << std::endl;
  
  std::stringstream dbstr;
  dbstr << "\n\n---------------------\n " << name_ 
	<< " - > getNewObjects\n"; 
  if (tagInfo().size){
    //check whats already inside of database
    std::string userText("No data");
    size_t splitPoint = logDBEntry().usertext.find_last_of("@");
    if( splitPoint != std::string::npos ) {
      userText = logDBEntry().usertext.substr(splitPoint);
    }
    dbstr << "got offlineInfo" << tagInfo().name << ", size " << tagInfo().size << " " << tagInfo().token 
	  << " , last object valid since " 
	  << tagInfo().lastInterval.first << " token "   
	  << tagInfo().lastPayloadToken << "\n\n UserText " << userTextLog() 
	  << "\n LogDBEntry \n" 
	  << logDBEntry().logId            << "\n"
	  << logDBEntry().destinationDB    << "\n"
	  << logDBEntry().provenance       << "\n"
	  << logDBEntry().usertext         << "\n"
	  << logDBEntry().iovtag           << "\n"
	  << logDBEntry().iovtimetype      << "\n"
	  << logDBEntry().payloadIdx       << "\n"
	  << logDBEntry().payloadClass     << "\n"
	  << logDBEntry().payloadToken     << "\n"
	  << logDBEntry().exectime         << "\n"
	  << logDBEntry().execmessage      << "\n"
	  << "\n\n-- user text " << userText;
  } else {
    dbstr << " First object for this tag ";
  }
  dbstr << "\n-------------------------\n";
  edm::LogInfo   ("SiStripDetVOffHandler") << dbstr.str();
  
  // Do the transfer!
  std::cout << "getNewObjects setForTransfer" << std::endl;
  setForTransfer();
  std::cout << "getNewObjects setForTransfer end" << std::endl;
}

void popcon::SiStripDetVOffHandler::setForTransfer() { 
  edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::setForTransfer]" << std::endl;

  // retrieve the last object transferred
  if (tagInfo().size ) {
    Ref payload = lastPayload();
    SiStripDetVOff * lastV = new SiStripDetVOff( *payload );
    modHVBuilder->setLastSiStripDetVOff( lastV, tagInfo().lastInterval.first );
  }

  // build the object!
  resultVec.clear();
  modHVBuilder->BuildDetVOffObj();
  resultVec = modHVBuilder->getModulesVOff(deltaTmin_, maxIOVlength_);

  if (!resultVec.empty()){
    // assume by default that transfer is needed
    unsigned int firstPayload = 0;
    
    // check if there is an existing payload and retrieve if there is
    if (tagInfo().size > 0) {
      Ref payload = lastPayload();
      // resultVec does not contain duplicates, so only need to compare payload with resultVec[0]
      SiStripDetVOff * modV = resultVec[0].first;
      if (*modV == *payload) {
	edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::setForTransfer] Transfer of first element not required!";
	cout << "[SiStripDetVOffHandler::setForTransfer] Transfer of first element not required!" << endl;
	firstPayload = 1;
      }
      else {
	cout << "[SiStripDetVOffHandler::setForTransfer] Transfer of first element required" << endl;
      }
    } else {     
      edm::LogInfo("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::setForTransfer] No previous payload";
      cout << "[SiStripDetVOffHandler::setForTransfer] No previous payload" << endl;
    }
 
    setUserTextLog();
    
    for (unsigned int i = firstPayload; i < resultVec.size(); i++) {
      this->m_to_transfer.push_back(resultVec[i]);
    }
    
  } else {
    edm::LogError("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::" << __func__ << "] " 
					   << name_ << "  : NULL pointer reported by SiStripDetVOffBuilder"
					   << "\n Transfer aborted"<< std::endl;
  }
}


void popcon::SiStripDetVOffHandler::setUserTextLog()
{
  std::stringstream ss;
  ss << "@@@ Number of payloads transferred " << resultVec.size() << "." << std::endl;
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::const_iterator it = resultVec.begin();
  ss << "time \t #LV off \t #HV off" << std::endl;
  for( ; it != resultVec.end(); ++it ) {
    ss << boost::posix_time::to_iso_extended_string( cond::time::to_boost(it->second) );
    ss << "\t\t  " << it->first->getLVoffCounts() << "\t\t  " << it->first->getHVoffCounts() << std::endl;
  }

  if( debug_ ) {
    ss << "PayloadNo/Badmodules/NoAdded/NoRemoved: ";
    std::vector< std::vector<uint32_t> > payloadStats = modHVBuilder->getPayloadStats();
    for (unsigned int j = 0; j < payloadStats.size(); j++) {
      ss << j << "/" << payloadStats[j][0] << "/" << payloadStats[j][1] << "/" << payloadStats[j][2] << "\t ";
    }
  }
  
  this->m_userTextLog = ss.str();
  
  LogTrace("SiStripDetVOffHandler") << "[SiStripDetVOffHandler::setUserTextLog] " << ss.str();
    
 
}
