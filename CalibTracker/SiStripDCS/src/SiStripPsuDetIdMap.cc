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

using namespace sistrip;

// only one constructor
SiStripPsuDetIdMap::SiStripPsuDetIdMap() { LogTrace("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] Constructing ..."; }
// destructor
SiStripPsuDetIdMap::~SiStripPsuDetIdMap() {LogTrace("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] Destructing ..."; }

// Build PSU-DETID map
void SiStripPsuDetIdMap::BuildMap( const std::string & mapFile, const bool debug )
{
  BuildMap(mapFile, debug, LVMap, HVMap, HVUnmapped_Map, HVCrosstalking_Map);
}

void SiStripPsuDetIdMap::BuildMap( const std::string & mapFile, std::vector<std::pair<uint32_t,std::string> > & rawmap) {
  //This method is a remnant of the old method, that provided a vector type of map, based on the 
  //raw reading of a file, with no processing.
  //FIXME:
  //This is not currently used, but I think we could slim this down to just a vector with 
  //the detIDs since the PSUChannel part of the excludedlist (if it ever is in a file) is never used!
  edm::FileInPath file(mapFile.c_str());
  std::ifstream ifs( file.fullPath().c_str() );
  string line;
  while( getline( ifs, line ) ) {
    if( line != "" ) {
      // split the line and insert in the map
      stringstream ss(line);
      string PSUChannel;
      uint32_t detId;
      ss >> detId;
      ss >> PSUChannel;
      rawmap.push_back(std::make_pair(detId, PSUChannel) );
    }
  }
}

//The following is the currently used method (called from SiStripDetVOffBuilder::buildPSUdetIdMap)
void SiStripPsuDetIdMap::BuildMap( const std::string & mapFile, const bool debug, PsuDetIdMap & LVmap, PsuDetIdMap & HVmap,PsuDetIdMap & HVUnmappedmap,PsuDetIdMap & HVCrosstalkingmap ) //Maybe it would be nicer to return the map instead of using a reference...
{
  //This method reads the map from the mapfile indicated in the cfg
  //It populates the 4 maps (private data members of the SiStripPSUDetIdMap in question) (all maps are std::map<std::string,uint32_t > ):
  //LVMap
  //HVMap
  //HVUnmapped_Map
  //HVCrosstalking_Map
  //These maps are accessed, based on the LV/HV case, to extract the detIDs connected to a given PSUChannel... 
  //see the getDetIDs method...
  edm::FileInPath file(mapFile.c_str());
  std::ifstream ifs( file.fullPath().c_str() );
  string line;
  while( getline( ifs, line ) ) {
    if( line != "" ) {
      // split the line and insert in the map
      stringstream ss(line);
      string PSUChannel;
      uint32_t detId;
      ss >> detId;
      ss >> PSUChannel;
      //Old "vector of pairs" map!
      //map.push_back( std::make_pair(detId, dpName) );//This "map" is normally the pgMap of the map of which we are executing BuildMap()...
      //Using a map to make the look-up easy and avoid lots of lookup loops.
      std::string PSU=PSUChannel.substr(0,PSUChannel.size()-10);
      std::string Channel=PSUChannel.substr(PSUChannel.size()-10);
      LVmap[PSU].push_back(detId); // LVmap uses simply the PSU since there is no channel distinction necessary
      if (Channel=="channel000") {
	HVUnmappedmap[PSU].push_back(detId); //Populate HV Unmapped map, by PSU listing all detids unmapped in that PSU (not necessarily all will be unmapped)
      }
      else if (Channel=="channel999") {
	HVCrosstalkingmap[PSU].push_back(detId); //Populate HV Crosstalking map, by PSU listing all detids crosstalking in that PSU (usually all will be unmapped)
      }
      else {
	HVmap[PSUChannel].push_back(detId); //HV map for HV mapped channels, populated by PSU channel!
      }
    }
  }
  
  //Remove duplicates for all 4 maps
  for (PsuDetIdMap::iterator psu = LVMap.begin(); psu != LVMap.end(); psu++) {
    RemoveDuplicateDetIDs(psu->second);
  }
  for (PsuDetIdMap::iterator psuchan = HVMap.begin(); psuchan != HVMap.end(); psuchan++) {
    RemoveDuplicateDetIDs(psuchan->second);
  }
  for (PsuDetIdMap::iterator psu = HVUnmapped_Map.begin(); psu != HVUnmapped_Map.end(); psu++) {
    RemoveDuplicateDetIDs(psu->second);
  }
  for (PsuDetIdMap::iterator psu = HVCrosstalking_Map.begin(); psu != HVCrosstalking_Map.end(); psu++) {
    RemoveDuplicateDetIDs(psu->second);
  }
  if (debug) {
    //Print out all the 4 maps:
    std::cout<<"Dumping the LV map"<<std::endl;
    std::cout<<"PSU->detids"<<std::endl;
    for (PsuDetIdMap::iterator psu = LVMap.begin(); psu != LVMap.end(); psu++) {
      std::cout<<psu->first<<" corresponds to following detids"<<endl;
      for (unsigned int i=0; i<psu->second.size(); i++) {
	std::cout<<"\t\t"<<psu->second[i]<<std::endl;
      }
    }
    std::cout<<"Dumping the HV map for HV mapped channels"<<std::endl;
    std::cout<<"PSUChannel->detids"<<std::endl;
    for (PsuDetIdMap::iterator psuchan = HVMap.begin(); psuchan != HVMap.end(); psuchan++) {
      std::cout<<psuchan->first<<" corresponds to following detids"<<endl;
      for (unsigned int i=0; i<psuchan->second.size(); i++) {
	std::cout<<"\t\t"<<psuchan->second[i]<<std::endl;
      }
    }
    std::cout<<"Dumping the HV map for HV UNmapped channels"<<std::endl;
    std::cout<<"PSU->detids"<<std::endl;
    for (PsuDetIdMap::iterator psu = HVUnmapped_Map.begin(); psu != HVUnmapped_Map.end(); psu++) {
      std::cout<<psu->first<<" corresponds to following detids"<<endl;
      for (unsigned int i=0; i<psu->second.size(); i++) {
	std::cout<<"\t\t"<<psu->second[i]<<std::endl;
      }
    }
    std::cout<<"Dumping the HV map for HV Crosstalking channels"<<std::endl;
    std::cout<<"PSU->detids"<<std::endl;
    for (PsuDetIdMap::iterator psu = HVCrosstalking_Map.begin(); psu != HVCrosstalking_Map.end(); psu++) {
      std::cout<<psu->first<<" corresponds to following detids"<<endl;
      for (unsigned int i=0; i<psu->second.size(); i++) {
	std::cout<<"\t\t"<<psu->second[i]<<std::endl;
      }
    }
    //Could add here consistency checks against the list of detIDs for Strip or Pixels
    //Number of total detIDs LVMapped, HV Mapped, HVunmapped, HV crosstalking... 
  }
}

void SiStripPsuDetIdMap::RemoveDuplicateDetIDs(std::vector<uint32_t> & detids) {
  //Function to remove duplicates from a vector of detids
  if (!detids.empty()) { //Leave empty vector alone ;)
    std::sort(detids.begin(),detids.end());
    std::vector<uint32_t>::iterator it = std::unique(detids.begin(),detids.end());
    detids.resize( it - detids.begin() );
  }
}

std::vector<uint32_t> SiStripPsuDetIdMap::getLvDetID(std::string PSU) {
  //Function that returns a vector with all detids associated with a PSU 
  //(no channel information is saved in the map since it is not relevant for LV!)
  if (LVMap.find(PSU)!=LVMap.end()) {
    return LVMap[PSU];
  }
  else {
    std::vector<uint32_t> detids;
    return detids;
  }
}

void SiStripPsuDetIdMap::getHvDetID(std::string PSUChannel, std::vector<uint32_t> & ids, std::vector<uint32_t> & unmapped_ids, std::vector<uint32_t> & crosstalking_ids ) {
  //Function that (via reference parameters) populates ids, unmapped_ids, crosstalking_ids vectors of detids associated with a given PSU *HV* channel.
  if (HVMap.find(PSUChannel)!=HVMap.end()) {
    ids=HVMap[PSUChannel];
  }
  //Extract the PSU to check the unmapped and crosstalking maps too corresponding to this channel
  std::string PSU = PSUChannel.substr(0,PSUChannel.size()-10);
  if (HVUnmapped_Map.find(PSU)!=HVUnmapped_Map.end()) {
    unmapped_ids=HVUnmapped_Map[PSU];
  }
  if (HVCrosstalking_Map.find(PSU)!=HVCrosstalking_Map.end()) {
    crosstalking_ids=HVCrosstalking_Map[PSU];
  }
}

// This method needs to be updated once HV channel mapping is known
// Currently, channel number is ignored for mapping purposes
// check both PG and CG as the channels should be unique

void SiStripPsuDetIdMap::getDetID(std::string PSUChannel,const bool debug,std::vector<uint32_t> & detids,std::vector<uint32_t> & unmapped_detids,std::vector<uint32_t> & crosstalking_detids ) {
  //This function takes as argument the PSUChannel (i.e. the dpname as it comes from the PVSS query, e.g. cms_trk_dcs_02:CAEN/CMS_TRACKER_SY1527_2/branchController05/easyCrate0/easyBoard12/channel001)
  //And it returns 3 vectors:
  //1-detids->all the detids positively matching the PSUChannel in question
  //2-unmapped_detids->the detids that are matching the PSU in question but that are not HV mapped
  //3-crosstalking_detids->the detids that are matching the PSU in question but exhibit the HV channel cross-talking behavior (they are ON as long as ANY of the 2 HV channels of the supply is ON, so they only go OFF when both channels are OFF)
  //The second and third vectors are only relevant for the HV case, when unmapped and cross-talking channels need further processing before being turned ON and OFF.
  
  const std::string& PSUChannelFromQuery = PSUChannel;

  //Get the channel to see if it is LV or HV, they will be treated differently
  std::string ChannelFromQuery=PSUChannelFromQuery.substr(PSUChannelFromQuery.size()-10);
  //Get the PSU from Query, to be used for LVMap and for the HVUnmapped and HVCrosstalking maps:
  std::string PSUFromQuery=PSUChannelFromQuery.substr(0,PSUChannelFromQuery.size()-10);
  if (debug) {
    //FIXME:
    //Should handle all the couts with MessageLogger!
    std::cout << "DPNAME from QUERY: "<<PSUChannelFromQuery<<", Channel: "<<ChannelFromQuery<<"PSU: "<<PSUFromQuery<<std::endl;
  }

  //First prepare the strings needed to do the matching of the PSUChannel from the query to the ones in the map

  //Handle the LV case first:
  if (ChannelFromQuery=="channel000" or ChannelFromQuery=="channel001") {
    //For LV channels we need to look for any detID that is reported either as channel000 (not HV mapped)
    //but also as channel002 and channel003 (if they are HV mapped), or as channel999 (if they are in a crosstalking PSU)
    //Get the PSU to do a PSU-only matching to get all detIDs connected to the LV channel:
    //Now loop over the map!
    //for (PsuDetIdMap::iterator iter = pgMap.begin(); iter != pgMap.end(); iter++) {
    //  std::string PSUFromMap = iter->second.substr(0,iter->second.size()-10);
    //  //Careful if you uncomment this cout: it prints 15148 lines when checking for 1 psu name match! (meant for debugging of course)
    //  //std::cout<<"Truncated DPNAME from MAP: "<<PSUFromMap<<std::endl;
    //  if (PSUFromQuery == PSUFromMap) {
    //    detids.push_back(iter->first); //And fill the detids vector with the all detids matching the PSU from the query!
    //  }
    //}
    //No need to loop over if we use an actual map!
    
    if (LVMap.find(PSUFromQuery)!=LVMap.end()) {
      detids=LVMap[PSUFromQuery];
    }
  }
  //Handle the HV case too:
  else if (ChannelFromQuery=="channel002" or ChannelFromQuery=="channel003") {
    //For the HV channel we need to look at the actual positive matching detIDs, 
    //but also to the unmapped one (channel000) and the crosstalking ones (channel999).
    //Assemble the corresponding channel000 (unmapped channels) replacing the last character in PSUChannelFromQuery:
    //  std::string ZeroedPSUChannelFromQuery= PSUChannelFromQuery;
    //  ZeroedPSUChannelFromQuery.replace(ZeroedPSUChannelFromQuery.size()-1,1,"0");
    //  //Same for channel999 for the crosstalking channels:
    //  //std::string NineNineNine='999';
    //  std::string NinedPSUChannelFromQuery= PSUChannelFromQuery;
    //  NinedPSUChannelFromQuery.replace(NinedPSUChannelFromQuery.size()-3,3,"999");
    //  //std::string NinedPSUChannelFromQuery= PSUChannelFromQuery.substr(0,PSUChannelFromQuery.size()-3);// + '999';
    //  //Now loop over the map!
    //  for (PsuDetIdMap::iterator iter = pgMap.begin(); iter != pgMap.end(); iter++) {
    //    std::string PSUChannelFromMap = iter->second;
    //    //Careful if you uncomment this cout: it prints 15148 lines when checking for 1 psu name match! (meant for debugging of course)
    //    //std::cout<<"Truncated DPNAME from MAP: "<<PSUFromMap<<std::endl;
    //    if (PSUChannelFromMap==PSUChannelFromQuery)  {
    //      detids.push_back(iter->first); //Fill the detids vector with the all detids matching the PSUChannel from the query!
    //    }
    //    if (PSUChannelFromMap==ZeroedPSUChannelFromQuery) {
    //  	unmapped_detids.push_back(iter->first); //Fill the unmapped_detids vector with the all detids matching the channel000 for the PSU from the query!
    //  	if (debug) { //BEWARE: this debug printouts can become very heavy! 1 print out per detID matched!
    //  	  std::cout<<"Matched one of the HV-UNMAPPED channels: "<<ZeroedPSUChannelFromQuery<<std::endl;
    //  	  std::cout<<"Corresponding to detID: "<<iter->first<<std::endl;
    //  	  //for (unsigned int i_nohvmap_detid=0;i_nohvmap_detid < iter->first.size();i_nohvmap_detid++) {
    //  	  //  cout<< iter->first[i_nohvmap_detid] << std::endl;
    //  	}
    //    }
    //    if (PSUChannelFromMap==NinedPSUChannelFromQuery) {
    //  	crosstalking_detids.push_back(iter->first); //Fill the crosstalking_detids vector with the all detids matching the channel999 for the PSU from the query!
    //    }
    //  }
    if (HVMap.find(PSUChannelFromQuery)!=HVMap.end()) {
      detids=HVMap[PSUChannelFromQuery];
    }
    else if (HVUnmapped_Map.find(PSUFromQuery)!=HVUnmapped_Map.end()) {
      unmapped_detids=HVUnmapped_Map[PSUFromQuery];
    }
    else if (HVCrosstalking_Map.find(PSUFromQuery)!=HVCrosstalking_Map.end()) {
      crosstalking_detids=HVCrosstalking_Map[PSUFromQuery];
    }
  }
  //  
  //  
  //  //With the new code above that makes use of the channel00X information in the map
  //  //we should no more need to remove duplicates by construction.
  //  //The following code was used when there was no channel information in the map, 
  //  //to elegantly eliminate duplicates.
  //  //We can now use it as a cross-check (still removing duplicates in case they happen, but writing a message out)
  //  
  //  // remove duplicates
  //  
  //  //First sort detIDs vector, so that duplicates will be consecutive
  //  if (!detids.empty()) {
  //    std::sort(detids.begin(),detids.end());
  //    //Then use the forward iterator unique from STD that basically removes all consecutive duplicates from the vector
  //    //and reports a forward iterator pointing to the new end of the sequence
  //    std::vector<uint32_t>::iterator it = std::unique(detids.begin(),detids.end());
  //    if (it!=detids.end()) {
  //      std::cout<<"ARGH! It seems we found duplicate detIDs in the map corresponding to this PSUChannel: "<<PSUChannelFromQuery<<std::endl;
  //      detids.resize( it - detids.begin() );
  //    }
  //    if (debug) {
  //      std::cout<<"Matched the following detIDs to PSU channel from query "<<PSUChannelFromQuery <<":"<<std::endl;
  //      for (std::vector<uint32_t>::iterator i_detid=detids.begin();i_detid!=detids.end(); i_detid++) {
  //  	std::cout<<*i_detid<<std::endl;;
  //      }
  //    }
  //  }
  //  //Same for unmapped detIDs:
  //  if (!unmapped_detids.empty()) {
  //    std::sort(unmapped_detids.begin(),unmapped_detids.end());
  //    //Then use the forward iterator unique from STD that basically removes all consecutive duplicates from the vector
  //    //and reports a forward iterator pointing to the new end of the sequence
  //    std::vector<uint32_t>::iterator it = std::unique(unmapped_detids.begin(),unmapped_detids.end());
  //    if (it!=unmapped_detids.end()) {
  //      std::cout<<"ARGH! It seems we found duplicate unmapped_detids in the map corresponding to this PSUChannel: "<<PSUChannelFromQuery<<std::endl;
  //      unmapped_detids.resize( it - unmapped_detids.begin() );
  //    }
  //    if (debug) {
  //      std::cout<<"Matched the following unmapped_detids to PSU channel from query "<<PSUChannelFromQuery <<":"<<std::endl;
  //      for (std::vector<uint32_t>::iterator i_detid=unmapped_detids.begin();i_detid!=unmapped_detids.end(); i_detid++) {
  //  	std::cout<<*i_detid<<std::endl;;
  //      }
  //    }
  //  }
  //  //Finally, same for crosstalking detIDs:
  //  if (!crosstalking_detids.empty()) {
  //    std::sort(crosstalking_detids.begin(),crosstalking_detids.end());
  //    //Then use the forward iterator unique from STD that basically removes all consecutive duplicates from the vector
  //    //and reports a forward iterator pointing to the new end of the sequence
  //    std::vector<uint32_t>::iterator it = std::unique(crosstalking_detids.begin(),crosstalking_detids.end());
  //    if (it!=crosstalking_detids.end()) {
  //      std::cout<<"ARGH! It seems we found duplicate crosstalking_detids in the map corresponding to this PSUChannel: "<<PSUChannelFromQuery<<std::endl;
  //      crosstalking_detids.resize( it - crosstalking_detids.begin() );
  //    }
  //    if (debug) {
  //      std::cout<<"Matched the following crosstalking_detids to PSU channel from query "<<PSUChannelFromQuery <<":"<<std::endl;
  //      for (std::vector<uint32_t>::iterator i_detid=crosstalking_detids.begin();i_detid!=crosstalking_detids.end(); i_detid++) {
  //  	std::cout<<*i_detid<<std::endl;;
  //      }
  //    }
  //  }
  //  
  //  //Using reference parameters since we are returning multiple objects.
  //  //return detids;

}

// returns PSU channel name for a given DETID
std::string SiStripPsuDetIdMap::getPSUName(uint32_t detid) {
  std::vector< std::pair<uint32_t, std::string> >::iterator iter;
  for (iter = pgMap.begin(); iter != pgMap.end(); iter++) {
    if (iter->first && iter->first == detid) {return iter->second;}
  }
  // if we reach here, then we didn't find the detid in the map
  return "UNKNOWN";
}

std::string SiStripPsuDetIdMap::getPSUName(uint32_t detid, std::string group) {
  std::vector< std::pair<uint32_t, std::string> >::iterator iter;
  if (group == "PG") {
    for (iter = pgMap.begin(); iter != pgMap.end(); iter++) {
      if (iter->first && iter->first == detid) {return iter->second;}
    }
  }
  if (group == "CG") {
    for (iter = cgMap.begin(); iter != cgMap.end(); iter++) {
      if (iter->first && iter->first == detid) {return iter->second;}
    }
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

// returns the PVSS name for a given DETID, depending on specified map
std::string SiStripPsuDetIdMap::getDetectorLocation(uint32_t detid, std::string group) {
  if (group == "PG") {
    for (unsigned int i = 0; i < pgMap.size(); i++) {
      if (pgMap[i].first == detid) {return detectorLocations[i];}
    }
  }
  if (group == "CG") {
    for (unsigned int i = 0; i < cgMap.size(); i++) {
      if (cgMap[i].first == detid) {return controlLocations[i];}
    }
  }
  return "UNKNOWN";
}

// returns the PVSS name for a given PSU channel
std::string SiStripPsuDetIdMap::getDetectorLocation(std::string PSUChannel) {
  for (unsigned int i = 0; i < pgMap.size(); i++) {
    if (pgMap[i].second == PSUChannel) {return detectorLocations[i];}
  }
  for (unsigned int i = 0; i < cgMap.size(); i++) {
    if (cgMap[i].second == PSUChannel) {return controlLocations[i];}
  }
  return "UNKNOWN";
}

// returns the DCU ID for a given PSU channel
uint32_t SiStripPsuDetIdMap::getDcuId(std::string PSUChannel) {
  for (unsigned int i = 0; i < pgMap.size(); i++) {
    if (pgMap[i].second == PSUChannel) {return dcuIds[i];}
  }
  for (unsigned int i = 0; i < cgMap.size(); i++) {
    if (cgMap[i].second == PSUChannel) {return cgDcuIds[i];}
  }
  return 0;
}

uint32_t SiStripPsuDetIdMap::getDcuId(uint32_t detid) {
  for (unsigned int i = 0; i < pgMap.size(); i++) {
    if (pgMap[i].first == detid) {return dcuIds[i];}
  }
  return 0;
}

// determine if a given PSU channel is HV or not
int SiStripPsuDetIdMap::IsHVChannel(std::string PSUChannel) {
  // isHV = 0 means LV, = 1 means HV, = -1 means error
  int isHV = 0;
  std::string::size_type loc = PSUChannel.find( "channel", 0 );
  if (loc != std::string::npos) {
    std::string chNumber = PSUChannel.substr(loc+7,3);
    if (chNumber == "002" || chNumber == "003") {
      isHV = 1;
    } else if (chNumber == "000" || chNumber == "001") {
      isHV = 0;
    } else {
      edm::LogWarning("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] channel number of unexpected format, setting error flag!";
      isHV = -1;
    }
  } else {
    edm::LogWarning("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] channel number not located in PSU channel name, setting error flag!";
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

void SiStripPsuDetIdMap::printControlMap() {
  stringstream cg;
  cg << "Map of control power supplies to DET IDs: " << std::endl
     << "-- PSU name --                -- Det Id --" << std::endl;
  for (unsigned int p = 0; p < cgMap.size(); p++) {
    cg << cgMap[p].first << "         " << cgMap[p].second << std::endl;
  }
  edm::LogInfo("SiStripPsuDetIdMap") << "[SiStripPsuDetIdMap::" << __func__ << "] " << cg.str();
}

std::vector< std::pair<uint32_t, std::string> > SiStripPsuDetIdMap::getDcuPsuMap() {
  if (!pgMap.empty()) { return pgMap; }
  std::vector< std::pair<uint32_t, std::string> > emptyVec;
  return emptyVec;
}

void SiStripPsuDetIdMap::checkMapInputValues(const SiStripConfigDb::DcuDetIdsV& dcuDetIds_, const DcuPsuVector& dcuPsus_) {
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

//std::vector< std::pair<uint32_t, SiStripConfigDb::DeviceAddress> > SiStripPsuDetIdMap::retrieveDcuDeviceAddresses(std::string partition) {
std::vector< std::pair< std::vector<uint16_t> , std::vector<uint32_t> > > SiStripPsuDetIdMap::retrieveDcuDeviceAddresses(std::string partition) {
  // get the DB parameters
  SiStripDbParams dbParams_ = db_->dbParams();
  SiStripDbParams::SiStripPartitions::const_iterator iter;
  
  std::vector< std::pair<uint32_t, SiStripConfigDb::DeviceAddress> > resultVec;
  
  SiStripConfigDb::DeviceDescriptionsV dcuDevices_;
  SiStripConfigDb::DeviceType device_ = DCU;
  
  for (iter = dbParams_.partitions().begin(); iter != dbParams_.partitions().end(); ++iter) {
    if ( partition == "" || partition == iter->second.partitionName() ) {
      if ( iter->second.partitionName() == SiStripPartition::defaultPartitionName_ ) { continue; }
      if (iter->second.dcuVersion().first > 0 && iter->second.fecVersion().first > 0) {
	SiStripConfigDb::DeviceDescriptionsRange range = db_->getDeviceDescriptions(device_,iter->second.partitionName());
	if (!range.empty()) {
	  SiStripConfigDb::DeviceDescriptionsV nextVec( range.begin(), range.end() );
	  for (unsigned int i = 0; i < nextVec.size(); i++) {
	    dcuDescription * desc = dynamic_cast<dcuDescription *>(nextVec[i]);
	    resultVec.push_back( std::make_pair( desc->getDcuHardId(), db_->deviceAddress(*(nextVec[i])) ) );
	  }
	}
      }
    }
  }

  std::vector< std::pair< std::vector<uint16_t> , std::vector<uint32_t> > > testVec;
  std::vector< std::pair<uint32_t, SiStripConfigDb::DeviceAddress> >::iterator reorg_iter = resultVec.begin();

  for ( ; reorg_iter != resultVec.end(); reorg_iter++) {
    std::vector<uint16_t> fecInfo(4,0);
    fecInfo[0] = reorg_iter->second.fecCrate_;
    fecInfo[1] = reorg_iter->second.fecSlot_;
    fecInfo[2] = reorg_iter->second.fecRing_;
    fecInfo[3] = reorg_iter->second.ccuAddr_;
    std::vector<uint32_t> dcuids;
    std::vector< std::pair<uint32_t, SiStripConfigDb::DeviceAddress> >::iterator jter = reorg_iter;
    for ( ; jter != resultVec.end(); jter++) {
      if (reorg_iter->second.fecCrate_ == jter->second.fecCrate_ &&
	  reorg_iter->second.fecSlot_ == jter->second.fecSlot_ &&
	  reorg_iter->second.fecRing_ == jter->second.fecRing_ &&
	  reorg_iter->second.ccuAddr_ == jter->second.ccuAddr_) {
	dcuids.push_back(jter->first);
      }
    }
    // handle duplicates
    bool isDup = false;
    for (unsigned int i = 0; i < testVec.size(); i++) {
      if (fecInfo == testVec[i].first) {
	isDup = true;
	dcuids.insert(dcuids.end(), (testVec[i].second).begin(), (testVec[i].second).end() );
	std::sort(dcuids.begin(),dcuids.end());
	std::vector<uint32_t>::iterator it = std::unique(dcuids.begin(),dcuids.end());
	dcuids.resize( it - dcuids.begin() );
	testVec[i].second = dcuids;
      }
    }
    if (!isDup) {
      std::sort(dcuids.begin(),dcuids.end());
      std::vector<uint32_t>::iterator it = std::unique(dcuids.begin(),dcuids.end());
      dcuids.resize( it - dcuids.begin() );
      testVec.push_back(std::make_pair(fecInfo,dcuids));
    }
  }
  //  return resultVec;
  return testVec;
}

std::vector<uint32_t> SiStripPsuDetIdMap::findDcuIdFromDeviceAddress(uint32_t dcuid_) {
  std::vector< std::pair< std::vector<uint16_t> , std::vector<uint32_t> > >::iterator iter = dcu_device_addr_vector.begin();
  std::vector< std::pair< std::vector<uint16_t> , std::vector<uint32_t> > >::iterator res_iter = dcu_device_addr_vector.end();
  std::vector<uint32_t> pgDcu;

  for ( ; iter != dcu_device_addr_vector.end(); iter++) {
    std::vector<uint32_t> dcuids = iter->second;
    std::vector<uint32_t>::iterator dcu_iter = std::find(dcuids.begin(),dcuids.end(),dcuid_);
    bool alreadyFound = false;
    if (res_iter != dcu_device_addr_vector.end()) {alreadyFound = true;}
    if (dcu_iter != dcuids.end()) {
      res_iter = iter;
      if (!alreadyFound) {
	for (unsigned int i = 0; i < dcuids.size(); i++) {
	  if (dcuids[i] != dcuid_) {pgDcu.push_back(dcuids[i]);}
	}
      } else {
	std::cout << "Oh oh ... we have a duplicate :-(" << std::endl;
      }
    }
  }
  return pgDcu;
}


