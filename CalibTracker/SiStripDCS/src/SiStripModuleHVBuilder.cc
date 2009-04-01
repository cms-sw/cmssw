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
  //  //  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Channels with no associated Det IDs";
  //  //  LogTrace("SiStripModuleHVBuilder") << ss1.str();

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
  
  // Convert to DetIdCondTimeVector
  DetIdCondTimeVector resultVector;
  for (unsigned int i = 0; i < detidV.size(); i++) {
    std::vector<uint32_t> detids = detidV[i].first;
    removeDuplicates(detids);
    cond::Time_t iovtime = getIOVTime((detidV[i]).second);
    resultVector.push_back( std::make_pair(detids,iovtime) );
  }

  // storage vectors
  DetIdCondTimeVector summedVector;
  std::vector< std::vector< std::pair<bool,bool> > > flags;
  std::vector<uint32_t> sumVector;
  // first = LV, second = HV
  std::vector< std::pair< bool, bool > > Vflag;

  // saved copies for removing duplicates
  std::vector<uint32_t> saveSumVector;
  cond::Time_t saveTime = 0;
  std::vector< std::pair< bool, bool > > saveVflag;

  // Convert to summed vector
  for (unsigned int i = 0; i < resultVector.size(); i++) {
    std::vector<uint32_t> listOfDetIds = (resultVector[i]).first;
    unsigned int numAdded = 0, numRemoved = 0;
    
    // check to see if detID is present in the list already, store details if it is
    for (unsigned int j = 0; j < listOfDetIds.size(); j++) {
      bool alreadyPresent = false;
      unsigned int which = 0;
      for (unsigned int k = 0; k < sumVector.size(); k++) {
	if (listOfDetIds[j] == sumVector[k]) {
	  alreadyPresent = true;
	  which = k;
	}
      }

      if (!StatusGood[i]) {  // status is bad
	if (!alreadyPresent) {  // not in list, so store it
	  numAdded++;
	  sumVector.push_back(listOfDetIds[j]);
	  if (isHV[i] == 0) {Vflag.push_back( std::make_pair(true,false) );}
	  else if (isHV[i] == 1) {Vflag.push_back( std::make_pair(false,true) );}
	} else {  // already in list, so decide what to do with it
	  if (isHV[i] == 0) {Vflag[which].first = true;}
	  else if (isHV[i] == 1) {Vflag[which].second = true;}
	} 
      } else { // status is good
	if (alreadyPresent) {  // expect that these should already be present
	  // If LV present in list as bad, remove it
	  if ( isHV[i] == 0 && (Vflag[which].first) ) {Vflag[which].first = false;}
	  // If HV present in list as bad, remove it
	  if ( isHV[i] == 1 && (Vflag[which].second) ) {Vflag[which].second = false;}
	  
	  // If both flags are false, this entry should be removed from the list
	  std::pair<bool,bool> testpair = std::make_pair(false,false);
	  if ( !(Vflag[which].first) && !(Vflag[which].second) ) {
	    // clean up detID vector
	    std::vector<uint32_t>::iterator detIdToRemove = sumVector.end();
	    detIdToRemove = find(sumVector.begin(),sumVector.end(),listOfDetIds[j]);
	    if (detIdToRemove != sumVector.end()) {
	      sumVector.erase(detIdToRemove);
	      numRemoved++;
	    }
	    // clean up the flag vector
	    std::vector<std::pair<bool, bool> >::iterator toRemove = Vflag.end();
	    toRemove = find(Vflag.begin(),Vflag.end(),testpair);
	    if (toRemove != Vflag.end()) {Vflag.erase(toRemove);}
	  } // end of if for removal with both flags false
	} // already present in list
      } // end of status if

    } // end of loop over list of det IDs

    // duplicate removal
    bool storeThis = true;
    bool removePrevious = false;
    // decide whether to store the information - time the same
    if ( (resultVector[i].second) == saveTime) {
      if (sumVector != saveSumVector) {std::cout << "Vectors are different sizes - this is bad!" << std::endl;}
      else { // time and detID vectors the same
	if (Vflag == saveVflag) {storeThis = false;}
	else {
	  // use number of true flags as an indication of what is going on 
	  unsigned int old_count = 0, new_count = 0;
	  for (unsigned int t = 0; t < saveVflag.size(); t++) {
	    if (saveVflag[t].first) {old_count++;}
	    if (saveVflag[t].second) {old_count++;}
	  }
	  for (unsigned int p = 0; p < Vflag.size(); p++) {
	    if (Vflag[p].first) {new_count++;}
	    if (Vflag[p].second) {new_count++;}
	  }
	  if (new_count > old_count) {
	    removePrevious = true;
	  } else {
	    std::cout << "Time and detIDs are the same.  Vflag is different.  Saved vector has less true entries than current.  This is strange! " << old_count << " " << new_count << std::endl; 
	    storeThis = false;
	  }
	}
      }
    } else {  // times are different
      if (sumVector == saveSumVector && Vflag == saveVflag) {storeThis = false;}
    }
    
    // if true, remove the last entry
    if (removePrevious) {
      summedVector.pop_back();
      flags.pop_back();
      payloadStats.pop_back();
    }
    // if true, store current entry
    if (storeThis) {
      summedVector.push_back( std::make_pair(sumVector,(resultVector[i].second)) );
      flags.push_back(Vflag);

      std::vector<uint32_t> stats(3,0);
      stats[0] = sumVector.size();
      stats[1] = numAdded;            // TODO - this number is not always correct on first vector entry.  Problem comes when first object from DB is not stored ...
      stats[2] = numRemoved;
      payloadStats.push_back(stats);
    }
    
    // save what was just stored properly for comparison the next time around
    if (storeThis) {
      saveSumVector.clear();
      saveSumVector = sumVector;
      saveTime = 0;
      saveTime = resultVector[i].second;
      saveVflag.clear();
      saveVflag = Vflag;
    }
    
  } // end of main loop

  // Store in the final object
  for (unsigned int i = 0; i < summedVector.size(); i++) {
    std::vector<uint32_t> ids = summedVector[i].first;
    cond::Time_t stime = summedVector[i].second;
    std::vector< std::pair<bool,bool> > sflags = flags[i];
    std::vector<bool> lvOff, hvOff;
    for (unsigned int j = 0; j < sflags.size(); j++) {
      lvOff.push_back(sflags[j].first);
      hvOff.push_back(sflags[j].second);
    }
    SiStripDetVOff * modV = new SiStripDetVOff();
    modV->put(ids,hvOff,lvOff);
    modulesOff.push_back( std::make_pair(modV,stime) );
  }
  
  if (debug_) {
    std::cout << "Final results of object building...  Number of entries in vector = "  << modulesOff.size() << std::endl;
    for (unsigned int s = 0; s < modulesOff.size(); s++) {
      std::cout << "Index = " << s << "Time = " << modulesOff[s].second << std::endl;
      std::vector<uint32_t> ids;
      std::vector< std::pair<bool,bool> > sflags = flags[s];
      (modulesOff[s].first)->getDetIds(ids);
      for (unsigned int v = 0; v < ids.size(); v++) {
	std::cout << ids[v] << " LV = " << (modulesOff[s].first)->IsModuleLVOff(ids[v]) << " HV = " << (modulesOff[s].first)->IsModuleHVOff(ids[v]) << std::endl;
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
