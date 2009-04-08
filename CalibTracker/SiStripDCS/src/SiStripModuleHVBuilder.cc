#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVBuilder.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"
#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"
#include "CalibTracker/SiStripDCS/interface/SiStripPsuDetIdMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fstream>
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time.hpp"

// constructor
SiStripModuleHVBuilder::SiStripModuleHVBuilder(const edm::ParameterSet& pset, const edm::ActivityRegistry&) : 
  onlineDbConnectionString(pset.getUntrackedParameter<std::string>("onlineDB","")),
  authenticationPath(pset.getUntrackedParameter<std::string>("authPath","../data")),
  whichTable(pset.getUntrackedParameter<std::string>("queryType","STATUSCHANGE")),
  lastValueFileName(pset.getUntrackedParameter<std::string>("lastValueFile","")),
  fromFile(pset.getUntrackedParameter<bool>("lastValueFromFile",false)),
  debug_(pset.getUntrackedParameter<bool>("debugModeOn",false)),
  tDefault(7,0)
{ 
  // set up vectors based on pset parameters (tDefault purely for initialization)
  tmin_par = pset.getUntrackedParameter< std::vector<int> >("Tmin",tDefault);
  tmax_par = pset.getUntrackedParameter< std::vector<int> >("Tmax",tDefault);
  
  // initialize the coral timestamps
  if (tmin_par != tDefault && tmax_par != tDefault) {
    // Is there a better way to do this?  TODO - investigate
    coral::TimeStamp mincpy(tmin_par[0],tmin_par[1],tmin_par[2],tmin_par[3],tmin_par[4],tmin_par[5],tmin_par[6]);
    tmin = mincpy;
    coral::TimeStamp maxcpy(tmax_par[0],tmax_par[1],tmax_par[2],tmax_par[3],tmax_par[4],tmax_par[5],tmax_par[6]);
    tmax = maxcpy;
  } else {
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "] time interval not set properly ... Returning ...";
    return;
  }
  
  if (onlineDbConnectionString == "") {
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "] DB name has not been set properly ... Returning ...";
    return;
  }
  
  if (fromFile && whichTable == "LASTVALUE" && lastValueFileName == "") {
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "] File expected for lastValue table, but filename not specified ... Returning ...";
    return;
  }
  
  // write out the parameters
  std::stringstream ss;
  ss << "[SiStripModuleHVBuilder::" << __func__ << "]" << std::endl
     << "     Parameters:" << std::endl
     << "     DB connection string: " << onlineDbConnectionString << std::endl
     << "     Authentication path: " << authenticationPath << std::endl
     << "     Table to be queried: " << whichTable << std::endl
     << "     Tmin: ";
  for (unsigned int mn = 0; mn < tmin_par.size(); mn++) {ss << tmin_par[mn] << " ";}
  ss << std::endl;
  ss << "     Tmax: ";
  for (unsigned int mx = 0; mx < tmax_par.size(); mx++) {ss << tmax_par[mx]<< " ";}
  ss <<std::endl;
  LogTrace("SiStripModuleHVBuilder") << ss.str();
}

// destructor
SiStripModuleHVBuilder::~SiStripModuleHVBuilder() { 
  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: destructing ...";
}

void SiStripModuleHVBuilder::BuildModuleHVObj() {
  // vectors for storing output from DB or text file
  std::vector<coral::TimeStamp> changeDate;    // used by both
  std::vector<std::string> dpname;             // only used by DB access, not file access
  std::vector<float> actualValue;              // only used by DB access, not file access
  std::vector<uint32_t> dpid;                  // only used by file access
  std::vector<int> actualStatus;               // filled using actualValue info
  
  // Open the PVSS DB connection
  SiStripCoralIface * cif = new SiStripCoralIface(onlineDbConnectionString,authenticationPath);
  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Query type is " << whichTable << std::endl;
  if (whichTable == "LASTVALUE") {LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Use file? " << ((fromFile) ? "TRUE" : "FALSE");}

  // access the information!
  if (whichTable == "STATUSCHANGE" || (whichTable == "LASTVALUE" && !fromFile)) {
    cif->doQuery(whichTable,tmin,tmax,changeDate,actualValue,dpname);
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: PVSS DB access complete";
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of PSU channels: " << dpname.size();
  } else if (whichTable == "LASTVALUE" && fromFile) {
    readLastValueFromFile(dpid,actualValue,changeDate);
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: File access complete";
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of values read from file: " << dpid.size();
  }

  // preset the size of the status vector
  actualStatus.resize(actualValue.size());

  // convert what you retrieve from the PVSS DB to status, depending on type of info you have retrieved
  if (whichTable == "STATUSCHANGE") {
    for (unsigned int i = 0; i < actualValue.size(); i++) { actualStatus[i] = static_cast<int>(actualValue[i]); }
  } else if (whichTable == "LASTVALUE") {
    // retrieve the channel settings from the PVSS DB
    std::vector<coral::TimeStamp> settingDate;
    std::vector<float> settingValue;
    std::vector<std::string> settingDpname;
    std::vector<uint32_t> settingDpid;
    cif->doSettingsQuery(tmin,tmax,settingDate,settingValue,settingDpname,settingDpid);
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Channel settings retrieved";
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of PSU channels: " << settingDpname.size();

    unsigned int missing = 0;
    std::stringstream ss;
    if (fromFile) {
      // need to get the PSU channel names from settings
      dpname.clear();
      dpname.resize(dpid.size());
      for (unsigned int j = 0; j < dpid.size(); j++) {
	int setting = findSetting(dpid[j],changeDate[j],settingDpid,settingDate);
	if (setting >= 0) {
	  if (actualValue[j] > (0.97*settingValue[setting])) {actualStatus[j] = 1;}
	  else {actualStatus[j] = 0;}
	  dpname[j] = settingDpname[setting];
	} else {
	  actualStatus[j] = -1;
	  dpname[j] = "UNKNOWN";
	  missing++;
	  ss << "DP ID = " << dpid[j] << " date = " <<  boost::posix_time::to_iso_extended_string(changeDate[j].time()) << std::endl;
	}
      }
    } else {
      for (unsigned int j = 0; j < dpname.size(); j++) {
	int setting = findSetting(dpname[j],changeDate[j],settingDpname,settingDate);
	if (setting >= 0) {
	  if (actualValue[j] > (0.97*settingValue[setting])) {actualStatus[j] = 1;}
	  else {actualStatus[j] = 0;}
	} else {
	  actualStatus[j] = -1;
	  missing++;
	  ss << "Channel = " << dpname[j] << " date = " << boost::posix_time::to_iso_extended_string(changeDate[j].time()) << std::endl;
	}
      }
    }
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of channels with no setting information " << missing;
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of entries in dpname vector " << dpname.size();
    //    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Channels with missing setting information ";
    //    LogTrace("SiStripModuleHVBuilder") << ss.str();
  }
  delete cif;

  // build PSU - det ID map
  SiStripPsuDetIdMap map_;
  map_.BuildMap();
  LogTrace("SiStripModuleHVBuilder") <<"[SiStripModuleHVBuilder::" << __func__ << "] DCU-DET ID map built";
  
  // use map info to build input for list of objects
  // no need to check for duplicates, as put method for SiStripModuleHV checks for you!
  DetIdTimeStampVector detidV;
  std::vector<bool> StatusGood;
  std::vector<unsigned int> isHV;

  std::stringstream ss1;
  unsigned int notMatched = 0, statusGood = 0, matched = 0;
  for (unsigned int dp = 0; dp < dpname.size(); dp++) {
    // 23/03/09 -  removed bad status requirements
    //    if (dpname[dp] != "UNKNOWN" && actualStatus[dp] != 1) {
    if (dpname[dp] != "UNKNOWN") {
      // figure out the channel
      std::string board = dpname[dp];
      std::string::size_type loc = board.size()-10;
      board.erase(0,loc);
      // now store!
      std::vector<uint32_t> ids = map_.getDetID(dpname[dp]);
      if (!ids.empty()) {
	matched++;
	if (board == "channel000" || board == "channel001") {
	  detidV.push_back( std::make_pair(ids,changeDate[dp]) );
	  if (actualStatus[dp] != 1) {StatusGood.push_back(false);}
	  else {StatusGood.push_back(true);}
	  isHV.push_back(0);
	} else if (board == "channel002" || board == "channel003") {
	  detidV.push_back( std::make_pair(ids,changeDate[dp]) );
	  if (actualStatus[dp] != 1) {StatusGood.push_back(false);}
	  else {StatusGood.push_back(true);}
	  isHV.push_back(1);
	} else {
	  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "] channel name not recognised! " << board;
	}
      } else {
	notMatched++;
	ss1 << "Channel = " << dpname[dp] << " status = " << actualStatus[dp] << std::endl;
      }
    } else {
      if (dpname[dp] != "UNKNOWN" && actualStatus[dp] == 1) {statusGood++;}
    }
  }
  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of modules with bad V channels is         " << detidV.size();
  //  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Channels with no associated Det IDs";
  //  LogTrace("SiStripModuleHVBuilder") << ss1.str();

  if (debug_) {
    std::cout << "Unprocessed data from DB..." << std::endl;
    for (unsigned int pp = 0; pp < detidV.size(); pp++) {
      std::cout << "Index = " << pp << " LV or HV = " << isHV[pp] << " Status = " << StatusGood[pp] << " Time = " << getIOVTime(detidV[pp].second) << std::endl;
      std::vector<uint32_t> detids = detidV[pp].first;
      for (unsigned int rr = 0; rr < detids.size(); rr++) {
	std::cout << detids[rr] << std::endl;
      }
    }
  }
  
  // initialize variables
  modulesOff.clear();
  cond::Time_t saveIovTime = 0;

  for (unsigned int i = 0; i < detidV.size(); i++) {
    std::vector<uint32_t> detids = detidV[i].first;
    removeDuplicates(detids);
    cond::Time_t iovtime = getIOVTime((detidV[i]).second);
    unsigned int numAdded = 0, numRemoved = 0;
    
    SiStripDetVOff *modV = 0;
    if (iovtime != saveIovTime) { // time is different, so create new object
      if (i == 0) {modV = new SiStripDetVOff();} // create completely new object
      else {modV = new SiStripDetVOff( *(modulesOff.back().first) );} // start from copy of previous object
    } else {
      modV = (modulesOff.back()).first; // modify previous object
    }
    
    std::vector<uint32_t> beforeV;
    modV->getDetIds(beforeV);
    
    int lv_off = -1, hv_off = -1;
    if (isHV[i] == 0) {lv_off = !StatusGood[i];}
    if (isHV[i] == 1) {hv_off = !StatusGood[i];}
    for (unsigned int j = 0; j < detids.size(); j++) {
      modV->put(detids[j],hv_off,lv_off);
    }

    std::vector<uint32_t> afterV;
    modV->getDetIds(afterV);
    if (afterV.size() > beforeV.size()) {
      numAdded = afterV.size() - beforeV.size();
    } else if (afterV.size() < beforeV.size()) {
      numRemoved = beforeV.size() - afterV.size();
    }
    
    // store the object if it's a new object
    if (iovtime != saveIovTime) {
      if (i == 0 || !( *modV == *(modulesOff.back().first) )) {
	modulesOff.push_back( std::make_pair(modV,iovtime) );
	// save the time of the object
	saveIovTime = iovtime;
	// save stats
	std::vector<uint32_t> stats(3,0);
	stats[0] = afterV.size();
	stats[1] = numAdded;
	stats[2] = numRemoved;
	payloadStats.push_back(stats);
      }
    }
  }
  
  if (debug_) {
    std::cout << std::endl;
    std::cout << "Size of modulesOff = " << modulesOff.size() << std::endl;
    for (unsigned int i = 0; i < modulesOff.size(); i++) {
      std::vector<uint32_t> finalids;
      (modulesOff[i].first)->getDetIds(finalids);
      std::cout << "Index = " << i << " Size of DetIds vector = " << finalids.size() << std::endl;
      std::cout << "Time = " << modulesOff[i].second << std::endl;
      for (unsigned int j = 0; j < finalids.size(); j++) {
	std::cout << "detid = " << finalids[j] << " LV off = " << (modulesOff[i].first)->IsModuleLVOff(finalids[j]) << " HV off = " 
		  << (modulesOff[i].first)->IsModuleHVOff(finalids[j]) << std::endl;
      }
    }
  }
}

int SiStripModuleHVBuilder::findSetting(uint32_t id, coral::TimeStamp changeDate, std::vector<uint32_t> settingID, std::vector<coral::TimeStamp> settingDate) {
  int setting = -1;
  // find out how many channel entries there are
  std::vector<int> locations;
  for (unsigned int i = 0; i < settingID.size(); i++) { if (settingID[i] == id) {locations.push_back((int)i);} }

  // simple cases
  if (locations.size() == 0) {setting = -1;}
  else if (locations.size() == 1) {setting = locations[0];}
  // more than one entry for this channel
  // NB.  entries ordered by date!
  else {
    for (unsigned int j = 0; j < locations.size(); j++) {
      const boost::posix_time::ptime& testSec = changeDate.time();
      const boost::posix_time::ptime& limitSec = settingDate[(unsigned int)locations[j]].time();
      if (testSec >= limitSec) {setting = locations[j];}
    }
  }
  return setting;
}

int SiStripModuleHVBuilder::findSetting(std::string dpname, coral::TimeStamp changeDate, std::vector<std::string> settingDpname, std::vector<coral::TimeStamp> settingDate) {
  int setting = -1;
  // find out how many channel entries there are
  std::vector<int> locations;
  for (unsigned int i = 0; i < settingDpname.size(); i++) { if (settingDpname[i] == dpname) {locations.push_back((int)i);} }
  
  // simple cases
  if (locations.size() == 0) {setting = -1;}
  else if (locations.size() == 1) {setting = locations[0];}
  // more than one entry for this channel
  // NB.  entries ordered by date!
  else {
    for (unsigned int j = 0; j < locations.size(); j++) {
      const boost::posix_time::ptime& testSec = changeDate.time();
      const boost::posix_time::ptime& limitSec = settingDate[(unsigned int)locations[j]].time();
      if (testSec >= limitSec) {setting = locations[j];}
    }
  }
  return setting;
}

void SiStripModuleHVBuilder::readLastValueFromFile(std::vector<uint32_t> &dpIDs, std::vector<float> &vmonValues, std::vector<coral::TimeStamp> &dateChange) {
  std::ifstream lastValueFile(lastValueFileName.c_str());
  if (lastValueFile.bad()) {
    edm::LogError("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: last Value file does not exist!";
    return;
  }

  dpIDs.clear();
  vmonValues.clear();
  dateChange.clear();
  std::vector<std::string> changeDates;

  std::string line;
  // remove the first line as it is the title line
  std::getline(lastValueFile,line);
  line.clear();
  // now extract data
  while( std::getline(lastValueFile,line) ) {
    std::istringstream ss(line);
    uint32_t dpid;
    int type;
    float vmon;
    std::string changeDate;
    ss >> std::skipws >> type >> dpid >> vmon >> changeDate;
    if (type == 1) {
      dpIDs.push_back(dpid);
      vmonValues.push_back(vmon);
      changeDates.push_back(changeDate);
    }
  }
  lastValueFile.close();  

  // Now convert dates to coral::TimeStamp
  for (unsigned int i = 0; i < changeDates.size(); i++) {
    std::string part = changeDates[i].substr(0,4);
    int year = atoi(part.c_str());
    part.clear();

    part = changeDates[i].substr(5,2);
    int month = atoi(part.c_str());
    part.clear();

    part = changeDates[i].substr(8,2);
    int day = atoi(part.c_str());
    part.clear();

    part = changeDates[i].substr(11,2);
    int hour = atoi(part.c_str());
    part.clear();

    part = changeDates[i].substr(14,2);
    int minute = atoi(part.c_str());
    part.clear();

    part = changeDates[i].substr(17,2);
    int second = atoi(part.c_str());
    part.clear();

    coral::TimeStamp date(year,month,day,hour,minute,second,0);
    dateChange.push_back(date);
  }
  if (changeDates.size() != dateChange.size()) {edm::LogError("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: date conversion failed!!";}
}

cond::Time_t SiStripModuleHVBuilder::getIOVTime(coral::TimeStamp coralTime) {
  unsigned long long coralTimeInNs = coralTime.total_nanoseconds();
  // total seconds since the Epoch
  unsigned long long iovSec = coralTimeInNs/1000000000;
  // the rest of the elapsed time since the Epoch in micro seconds
  unsigned long long iovMicroSec = (coralTimeInNs%1000000000)/1000;
  // convert!
  cond::Time_t iovtime = (iovSec << 32) + iovMicroSec;
  return iovtime;
}

void SiStripModuleHVBuilder::removeDuplicates( std::vector<uint32_t> & vec ) {
  std::sort(vec.begin(),vec.end());
  std::vector<uint32_t>::iterator it = std::unique(vec.begin(),vec.end());
  vec.resize( it - vec.begin() );
}
