#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVHandler.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"
#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"
#include "CondCore/DBCommon/interface/TagInfo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

popcon::SiStripModuleHVHandler::SiStripModuleHVHandler (const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","SiStripModuleHVHandler")),
  m_deltaTmin(pset.getUntrackedParameter<uint32_t>("DeltaTmin", 1))
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

  // retrieve the last object transferred
  if (tagInfo().size ) {
    Ref payload = lastPayload();
    SiStripDetVOff * lastV = new SiStripDetVOff( *payload );
    modHVBuilder->retrieveLastSiStripDetVOff( lastV, tagInfo().lastInterval.first );
  }
  
  // build the object!
  resultVec.clear();
  modHVBuilder->BuildModuleHVObj();
  resultVec = modHVBuilder->getModulesVOff(m_deltaTmin);
  std::vector< std::vector<uint32_t> > payloadStats = modHVBuilder->getPayloadStats();

//   // Reduce the number of IOVs to be saved.
//   // we scan the full list of modules and:
//   // - if the difference in time is < deltaT and the two iovs have a number of differences
//   // less than deltaDiff, go to the next element and repeat until one of the two conditions fail.
//   // Then, if the first element of this sequence has less elements than the last it means we are
//   // going in the direction of turning off.
//   // In any case, the sequence should be removed and the only element to be kept and moved to cover
//   // the sequence must be the bigger one (we treat the transition phase as off).
//   // The resultVec is a vector<pair<SiStripDetVOff, cond::Time_t> >

//   // Go on until you find a DeltaT > deltaTmin. Then stop, compare the first of the sequence with the
//   // last and remove the intermediate steps.


//   // Used to store the initial state of a sequence of close-in-time iovs
//   std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator initialIt;
//   int count = 0;

//   if( resultVec.size() > 1 ) {
//     std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator it = resultVec.begin();
//     for( ; it != resultVec.end()-1; ++it ) {
//       unsigned long long time1 = it->second >> 32;
//       unsigned long long time2 = (it+1)->second >> 32;
//       unsigned long long deltaT = time2 - time1;
//       unsigned long long deltaTlong = it->second - (it+1)->second;
//       int deltaLVCounts = abs(it->first->getLVoffCounts() - (it+1)->first->getLVoffCounts());
//       int deltaHVCounts = abs(it->first->getHVoffCounts() - (it+1)->first->getHVoffCounts());
//       std::cout << "deltaT = " << deltaT << ", deltaTlong = " << deltaTlong <<", deltaLVCounts = " << deltaLVCounts << ", deltaHVCounts = " << deltaHVCounts << std::endl;
//       // Save the initial pair
//       if( deltaT <= 1 ) {
// 	// If we are not in a the sequence
// 	if( count == 0 ) {
// 	  initialIt = it;
// 	}
// 	// Increase the counter in any case.
// 	++count;
//       }
//       // We do it only if the sequence is bigger than two cases
//       else if( count > 1 ) {

// 	// if it was going off
// 	if( it->first->getLVoffCounts() - initialIt->first->getLVoffCounts() > 0 || it->first->getHVoffCounts() - initialIt->first->getHVoffCounts() > 0) {
// 	  // Set the time of the current (last) iov as the time of the initial iov of the sequence
// 	  it->second = initialIt->second;
// 	  // remove from the initial iov of the sequence up to the one before the current one
// 	  it = resultVec.erase(initialIt, it);
// 	}
// 	// if it was going on
// 	else if( it->first->getLVoffCounts() - initialIt->first->getLVoffCounts() <= 0 || it->first->getHVoffCounts() - initialIt->first->getHVoffCounts() <= 0 ) {
// 	  // replace the last minus one iov with the first one
// 	  initialIt->second = (it-1)->second;
// 	  it = resultVec.erase(initialIt, it);
// 	}
// 	// reset counter
// 	count = 0;
//       }
//       else {
// 	// reset counter
// 	count = 0;
//       }
//     }
//   }



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
