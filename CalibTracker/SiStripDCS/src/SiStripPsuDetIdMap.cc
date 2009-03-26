#include "CalibTracker/SiStripDCS/interface/SiStripPsuDetIdMap.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;
using namespace sistrip;

// only one constructor
SiStripPsuDetIdMap::SiStripPsuDetIdMap() { LogTrace("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] Constructing ..."; }
// destructor
SiStripPsuDetIdMap::~SiStripPsuDetIdMap() {LogTrace("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] Destructing ..."; }

// Build PSU-DETID map
void SiStripPsuDetIdMap::BuildMap() {
  // initialize the map vector
  pgMap.clear();
  detectorLocations.clear();
  dcuIds.clear();
  // first = DCU ID, second = pointer to TkDcuInfo object
  SiStripConfigDb::DcuDetIdsV dcu_detid_vector;
  // pointer to TkDcuPsuMap objects
  DcuPsuVector powerGroup;
  
  // check that the db connection is ready
  if ( db_ ) {
    // retrieve both maps, if available
    SiStripDbParams dbParams_ = db_->dbParams();
    SiStripDbParams::SiStripPartitions::const_iterator iter;
    for (iter = dbParams_.partitions().begin(); iter != dbParams_.partitions().end(); ++iter) {
      if (iter->second.psuVersion().first > 0) {
	DcuPsusRange PGrange;
	getDcuPsuMap(PGrange,iter->second.partitionName());
	if (!PGrange.empty()) {
	  DcuPsuVector nextVec( PGrange.begin(), PGrange.end() );
	  powerGroup.insert( powerGroup.end(), nextVec.begin(), nextVec.end() );
	}
      }
      
      if (iter->second.dcuVersion().first > 0) {
	SiStripConfigDb::DcuDetIdsRange range = db_->getDcuDetIds(iter->second.partitionName());
	if (!range.empty()) {
	  SiStripConfigDb::DcuDetIdsV nextVec( range.begin(), range.end() );
	  dcu_detid_vector.insert( dcu_detid_vector.end(), nextVec.begin(), nextVec.end() );
	}
      }
    }
  } else {
    edm::LogWarning("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] NULL pointer to SiStripConfigDb service returned!  Cannot build PSU <-> DETID map"; 
    return;
  }
  LogTrace("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] All information retrieved!";
  
  if (!powerGroup.empty()) {
    if (!dcu_detid_vector.empty()) {
      // Retrieve the collated information for all partitions
      // Now build the map, starting from the PSUs for power groups
      for (unsigned int psu = 0; psu < powerGroup.size(); psu++) {
	SiStripConfigDb::DcuDetIdsV::iterator iter = SiStripConfigDb::findDcuDetId( dcu_detid_vector.begin(), dcu_detid_vector.end(), powerGroup[psu]->getDcuHardId() );
	if (iter != dcu_detid_vector.end()) {
	  // check for duplicates
	  bool presentInMap = false, multiEntry = false;
	  unsigned int locInMap = 0;
	  for (unsigned int ch = 0; ch < pgMap.size(); ch++) {
	    if (pgMap[ch].first == iter->second->getDetId() && pgMap[ch].second == powerGroup[psu]->getDatapointName()) {presentInMap = true;}
	    if (pgMap[ch].first == iter->second->getDetId() && pgMap[ch].second != powerGroup[psu]->getDatapointName()) {
	      multiEntry = true;
	      locInMap = ch;
	    }
	  }
	  // if no duplicates, store it!
	  if (!presentInMap && !multiEntry) {
	    pgMap.push_back( std::make_pair( iter->second->getDetId(), powerGroup[psu]->getDatapointName() ) );
	    detectorLocations.push_back( powerGroup[psu]->getPVSSName() );
	    dcuIds.push_back( powerGroup[psu]->getDcuHardId() );
	  }
	  if (multiEntry) {
	    pgMap[locInMap].first = iter->second->getDetId();
	    pgMap[locInMap].second = powerGroup[psu]->getDatapointName();
	    detectorLocations[locInMap] = powerGroup[psu]->getPVSSName();
	    dcuIds[locInMap] = powerGroup[psu]->getDcuHardId();
	  }
	}
      }
    } else {
      edm::LogWarning("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] DCU <-> DETID mapping missing!  Cannot build PSU <-> DETID map";
    }
  } else {
    edm::LogWarning("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] DCU <-> PSU mapping missing!  Cannot build PSU <-> DETID map";
  }
  LogTrace("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "]: Size of PSU-DetID map is: " << pgMap.size();
}

// Extract DCU-PSU map from DB
void SiStripPsuDetIdMap::getDcuPsuMap(DcuPsusRange &pRange, std::string partition) {
  // initialize the DCU-PSU range
  pRange = DcuPsuMapPG_.emptyRange();
  // check that the db connection is ready
  SiStripDbParams dbParams_ = db_->dbParams();
  // devicefactory needed for DCU-PSU information
  if ( db_->deviceFactory() ) {
    // loop over all specified partitions
    SiStripDbParams::SiStripPartitions::const_iterator iter;
    for (iter = dbParams_.partitions().begin(); iter != dbParams_.partitions().end(); ++iter) {
      if ( partition == "" || partition == iter->second.partitionName() ) {
	if ( iter->second.partitionName() == SiStripPartition::defaultPartitionName_ ) { continue; }
	// Stolen from RB code - modify to store DCU PSU map instead
	// Do things the way RB does to make life easier
	DcuPsusRange rangePG = DcuPsuMapPG_.find(iter->second.partitionName());
	
	if (rangePG == DcuPsuMapPG_.emptyRange()) {
	  try {
	    db_->deviceFactory()->getDcuPsuMapPartition(iter->second.partitionName(),iter->second.psuVersion().first,iter->second.psuVersion().second);
	  } catch (... ) { db_->handleException( __func__ ); }
	  
	  // now store it locally for power groups
	  DcuPsuVector pGroup   = db_->deviceFactory()->getPowerGroupDcuPsuMaps();
	  DcuPsuVector dstPG;
	  clone(pGroup, dstPG);
	  DcuPsuMapPG_.loadNext(iter->second.partitionName(), dstPG);
	}
      } // if partition is blank or equal to the partitionName specified
    } // for loop
  } // device factory check
  
  // Create range object
  uint16_t npPG = 0, ncPG = 0;
  DcuPsusRange PGrange;
  if ( partition != "" ) { 
    PGrange = DcuPsuMapPG_.find(partition);
    npPG = 1;
    ncPG = PGrange.size();
  } else { 
    if (!DcuPsuMapPG_.empty()) {
      PGrange = DcuPsusRange( DcuPsuMapPG_.find( dbParams_.partitions().begin()->second.partitionName() ).begin(),
			      DcuPsuMapPG_.find( (--(dbParams_.partitions().end()))->second.partitionName() ).end() );
    } else {
      PGrange = DcuPsuMapPG_.emptyRange();
    }
    npPG = DcuPsuMapPG_.size();
    ncPG = PGrange.size();
  }
  
  stringstream ss; 
  ss << "Found " << ncPG << " entries for power groups in DCU-PSU map";
  if (DcuPsuMapPG_.empty()) {edm::LogWarning("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] " << ss.str();}
  else {LogTrace("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] " << ss.str();}
  
  pRange = PGrange;
}

// This method needs to be updated once HV channel mapping is known
// Currently, channel number is ignored for mapping purposes
std::vector<uint32_t> SiStripPsuDetIdMap::getDetID(std::string pvss) {
  PsuDetIdMap::iterator iter;
  std::vector<uint32_t> detids;
  
  std::string inputBoard = pvss;
  std::string::size_type loc = inputBoard.size()-3;
  inputBoard.erase(loc,3);
  
  for (iter = pgMap.begin(); iter != pgMap.end(); iter++) {
    std::string board = iter->second;
    std::string::size_type loca = board.size()-3;
    board.erase(loca,3);
    if (iter->first && inputBoard == board) {detids.push_back(iter->first);}
  }
  return detids;
}

// returns PSU channel name for a given DETID
std::string SiStripPsuDetIdMap::getPSUName(uint32_t detid) {
  PsuDetIdMap::iterator iter;
  for (iter = pgMap.begin(); iter != pgMap.end(); iter++) {
    if (iter->first && iter->first == detid) {return iter->second;}
  }
  // if we reach here, then we didn't find the detid in the map
  return "UNKNOWN";
}

// returns the PVSS name for a given DETID
std::string SiStripPsuDetIdMap::getDetectorLocation(uint32_t detid) {
  for (unsigned int i = 0; i < pgMap.size(); i++) {
    if (pgMap[i].first == detid) {return detectorLocations[i];}
  }
  return "UNKNOWN";
}

// returns the PVSS name for a given PSU channel
std::string SiStripPsuDetIdMap::getDetectorLocation(std::string pvss) {
  for (unsigned int i = 0; i < pgMap.size(); i++) {
    if (pgMap[i].second == pvss) {return detectorLocations[i];}
  }
  return "UNKNOWN";
}

// returns the DCU ID for a given PSU channel
uint32_t SiStripPsuDetIdMap::getDcuId(std::string pvss) {
  for (unsigned int i = 0; i < pgMap.size(); i++) {
    if (pgMap[i].second == pvss) {return dcuIds[i];}
  }
  return 0;
}

uint32_t SiStripPsuDetIdMap::getDcuId(uint32_t detid) {
  for (unsigned int i = 0; i < pgMap.size(); i++) {
    if (pgMap[i].first == detid) {return dcuIds[i];}
  }
  return 0;
}

std::vector< std::pair<uint32_t, std::string> > SiStripPsuDetIdMap::getPsuDetIdMap() {return pgMap;}
std::vector<std::string> SiStripPsuDetIdMap::getDetectorLocations() {return detectorLocations;}
std::vector<uint32_t> SiStripPsuDetIdMap::getDcuIds() {return dcuIds;}

// determine if a given PSU channel is HV or not
int SiStripPsuDetIdMap::IsHVChannel(std::string pvss) {
  // isHV = 0 means LV, = 1 means HV, = -1 means error
  int isHV = 0;
  std::string::size_type loc = pvss.find( "channel", 0 );
  if (loc != std::string::npos) {
    std::string chNumber = pvss.substr(loc+7,3);
    if (chNumber == "002" || chNumber == "003") {
      isHV = 1;
    } else if (chNumber == "000" || chNumber == "001") {
      isHV = 0;
    } else {
      edm::LogWarning("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] channel number of expected format, setting error flag!";
      isHV = -1;
    }
  } else {
    edm::LogWarning("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] channel number not located in PVSS name, setting error flag!";
    isHV = -1;
  }
  return isHV;
}

void SiStripPsuDetIdMap::clone(DcuPsuVector &input, DcuPsuVector &output) {
  output.clear();
  for (unsigned int i = 0; i < input.size(); i++) {
    output.push_back(new TkDcuPsuMap(*(input[i])));
  }
}

void SiStripPsuDetIdMap::printMap() {
  stringstream pg;
  pg << "Map of power supplies to DET IDs: " << std::endl
     << "-- PSU name --          -- Det Id --" << std::endl;
  for (unsigned int p = 0; p < pgMap.size(); p++) {
    pg << pgMap[p].first << "         " << pgMap[p].second << std::endl;
  }
  edm::LogInfo("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] " << pg.str();
}

std::vector< std::pair<uint32_t, std::string> > SiStripPsuDetIdMap::getDcuPsuMap() {
  if (pgMap.size() != 0) { return pgMap; }
  std::vector< std::pair<uint32_t, std::string> > emptyVec;
  return emptyVec;
}

void SiStripPsuDetIdMap::checkMapInputValues(SiStripConfigDb::DcuDetIdsV dcuDetIds_, DcuPsuVector dcuPsus_) {
  std::cout << "Number of entries in DCU-PSU map:    " << dcuPsus_.size() << std::endl;
  std::cout << "Number of entries in DCU-DETID map:  " << dcuDetIds_.size() << std::endl;
  std::cout << std::endl;
  
  std::vector<bool> ddUsed(dcuDetIds_.size(),false);
  std::vector<bool> dpUsed(dcuPsus_.size(),false);

  for (unsigned int dp = 0; dp < dcuPsus_.size(); dp++) {
    for (unsigned int dd = 0; dd < dcuDetIds_.size(); dd++) {
      if (dcuPsus_[dp]->getDcuHardId() == dcuDetIds_[dd].second->getDcuHardId()) {
	dpUsed[dp] = true;
	ddUsed[dd] = true;
      }
    }
  }
  unsigned int numDpUsed = 0, numDpNotUsed = 0;
  for (unsigned int dp = 0; dp < dpUsed.size(); dp++) {
    if (dpUsed[dp]) { numDpUsed++; }
    else { numDpNotUsed++; }
  }

  std::cout << "Number of used DCU-PSU entries:   " << numDpUsed << std::endl;
  std::cout << "Number of unused DCU-PSU entries: " << numDpNotUsed << std::endl;

  unsigned int numDdUsed = 0, numDdNotUsed = 0;
  for (unsigned int dd = 0; dd < ddUsed.size(); dd++) {
    if (ddUsed[dd]) { numDdUsed++; }
    else { numDdNotUsed++; }
  }

  std::cout << "Number of used DCU-DETID entries:   " << numDdUsed << std::endl;
  std::cout << "Number of unused DCU-DETID entries: " << numDdNotUsed << std::endl;
  std::cout << std::endl;
  std::cout << "Size of PSU-DETID map:              " << pgMap.size() << std::endl;
  std::cout << "Size of detectorLocations:          " << detectorLocations.size() << std::endl;
}
