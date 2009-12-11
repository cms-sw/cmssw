#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVBuilder.h"
using namespace std;

// constructor
SiStripModuleHVBuilder::SiStripModuleHVBuilder(const edm::ParameterSet& pset, const edm::ActivityRegistry&) : 
  onlineDbConnectionString(pset.getParameter<std::string>("onlineDB")),
  authenticationPath(pset.getParameter<std::string>("authPath")),
  whichTable(pset.getParameter<std::string>("queryType")),
  lastValueFileName(pset.getParameter<std::string>("lastValueFile")),
  fromFile(pset.getParameter<bool>("lastValueFromFile")),
  debug_(pset.getParameter<bool>("debugModeOn")),
  tDefault(7,0),
  tmax_par(pset.getParameter< std::vector<int> >("Tmax")),
  tmin_par(pset.getParameter< std::vector<int> >("Tmin")),
  tset_par(pset.getParameter< std::vector<int> >("TSetMin")),
  detIdListFile_(pset.getParameter< std::string >("DetIdListFile"))
{ 
  lastStoredCondObj.first = NULL;
  lastStoredCondObj.second = 0;
  
  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder" << endl;

  // set up vectors based on pset parameters (tDefault purely for initialization)
  
  
  // initialize the coral timestamps
  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder (initialize the coral timestamps)" << endl;
  // always need Tmax
  coral::TimeStamp maxcpy(tmax_par[0],tmax_par[1],tmax_par[2],tmax_par[3],tmax_par[4],tmax_par[5],tmax_par[6]);
  tmax = maxcpy;
  

  // Sometimes need Tmin
  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder (sometimes need Tmin)" << endl;
  if (whichTable == "STATUSCHANGE" || (whichTable == "LASTVALUE" && !fromFile)) {
    // Is there a better way to do this?  TODO - investigate
    coral::TimeStamp mincpy(tmin_par[0],tmin_par[1],tmin_par[2],tmin_par[3],tmin_par[4],tmin_par[5],tmin_par[6]);
    tmin = mincpy;
  }
  
  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder (whichTable)" << endl;
  if (whichTable == "LASTVALUE") {
    coral::TimeStamp setcpy(tset_par[0],tset_par[1],tset_par[2],tset_par[3],tset_par[4],tset_par[5],tset_par[6]);
    tsetmin = setcpy;
  }
  
  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder (onlineDbConnectionString)" << endl;
  if (onlineDbConnectionString == "") {
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "] DB name has not been set properly ... Returning ...";
    return;
  }
  
  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder (after)" << endl;
  if (fromFile && whichTable == "LASTVALUE" && lastValueFileName == "") {
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "] File expected for lastValue table, but filename not specified ... Returning ...";
    return;
  }
  
  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder (writing out parameters)" << endl;
  // write out the parameters
  std::stringstream ss;
  ss << "[SiStripModuleHVBuilder::" << __func__ << "]" << std::endl
     << "     Parameters:" << std::endl
     << "     DB connection string: " << onlineDbConnectionString << std::endl
     << "     Authentication path: " << authenticationPath << std::endl
     << "     Table to be queried: " << whichTable << std::endl;
  
  if (whichTable == "STATUSCHANGE" || (whichTable == "LASTVALUE" && !fromFile)) {
    ss << "     Tmin: ";
    for (unsigned int mn = 0; mn < tmin_par.size(); mn++) {ss << tmin_par[mn] << " ";}
    ss << std::endl;
  }
  
  ss << "     Tmax: ";
  for (unsigned int mx = 0; mx < tmax_par.size(); mx++) {ss << tmax_par[mx]<< " ";}
  ss <<std::endl;
  
  if (whichTable == "LASTVALUE") {
    ss << "     TSetMin: ";
    for (unsigned int se = 0; se < tset_par.size(); se++) {ss << tset_par[se]<< " ";}
    ss <<std::endl;
  }
  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder (LogTrace)" << endl;
  LogTrace("SiStripModuleHVBuilder") << ss.str();

  cout << "SiStripModuleHVBuilder::SiStripModuleHVBuilder (end)" << endl;
}

// destructor
SiStripModuleHVBuilder::~SiStripModuleHVBuilder() { 
  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: destructing ...";
}

void SiStripModuleHVBuilder::BuildModuleHVObj() {
  // vectors for storing output from DB or text file
  std::vector<coral::TimeStamp> changeDate;    // used by both
  std::vector<std::string>      dpname;        // only used by DB access, not file access
  std::vector<float>            actualValue;   // only used by DB access, not file access
  std::vector<uint32_t>         dpid;          // only used by file access
  std::vector<int>              actualStatus;  // filled using actualValue info
  cond::Time_t                  latestTime = 0;// used for timestamp when using lastValue from file
  
  cout << "SiStripModuleHVBuilder::BuildModuleHVObj (Open the PVSS DB connection)" << endl;
  // Open the PVSS DB connection
  SiStripCoralIface * cif = new SiStripCoralIface(onlineDbConnectionString,authenticationPath);
  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Query type is " << whichTable << std::endl;
  if (whichTable == "LASTVALUE") {LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Use file? " << ((fromFile) ? "TRUE" : "FALSE");}
  if (lastStoredCondObj.second > 0) {LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << " retrieved last time stamp from DB: " 
									<< lastStoredCondObj.second  << std::endl;}
  cout << "SiStripModuleHVBuilder::BuildModuleHVObj (access the information)" << endl;
  // access the information!
  if (whichTable == "STATUSCHANGE" || (whichTable == "LASTVALUE" && !fromFile)) {
    if (whichTable == "STATUSCHANGE" && lastStoredCondObj.second > 0) {
      std::cout << "lastStoredCondObj.second = " << lastStoredCondObj.second << std::endl;
      coral::TimeStamp handlerTMin(getCoralTime(lastStoredCondObj.second));

      std::cout << "handlerTMin : year = " << handlerTMin.year()
		<< ", month = " << handlerTMin.month()
		<< ", day = " << handlerTMin.day()
		<< ", hour = " << handlerTMin.hour()
		<< ", minute = " << handlerTMin.minute()
		<< ", second = " << handlerTMin.second()
		<< ", nanosecond = " << handlerTMin.nanosecond() << std::endl;


      cif->doQuery(whichTable,handlerTMin,tmax,changeDate,actualValue,dpname);
    } else {
      cif->doQuery(whichTable,tmin,tmax,changeDate,actualValue,dpname);
      // if lastvalue table, take most recent time
      if (whichTable == "LASTVALUE") {latestTime = findMostRecentTimeStamp( changeDate );}
    }
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: PVSS DB access complete";
    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of PSU channels: " << dpname.size();
  } else if (whichTable == "LASTVALUE" && fromFile) {
    readLastValueFromFile(dpid,actualValue,changeDate);
    latestTime = findMostRecentTimeStamp( changeDate );
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
    cif->doSettingsQuery(tsetmin,tmax,settingDate,settingValue,settingDpname,settingDpid);
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
	  else {
	    actualStatus[j] = 0;
	  }
	  dpname[j] = settingDpname[setting];
	} else {
	  actualStatus[j] = -1;
	  dpname[j] = "UNKNOWN";
	  missing++;
	  ss << "DP ID = " << dpid[j] << std::endl;
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
	  ss << "Channel = " << dpname[j] << std::endl;
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
  unsigned int notMatched = 0;
  std::vector<std::string> psuName;
  
  unsigned int ch0bad = 0, ch1bad = 0, ch2bad = 0, ch3bad = 0;
  std::vector<unsigned int> numLvBad, numHvBad;

  for (unsigned int dp = 0; dp < dpname.size(); dp++) {
    if (dpname[dp] != "UNKNOWN") {
      // figure out the channel
      std::string board = dpname[dp];
      std::string::size_type loc = board.size()-10;
      board.erase(0,loc);
      // now store!
      std::vector<uint32_t> ids = map_.getDetID(dpname[dp]);
      if (!ids.empty()) {
	// DCU-PSU maps only channel000 and channel000 and channel001 switch on and off together
	// so check only channel000
	//	if (board == "channel000" || board == "channel001") {
	if (board == "channel000") {
	  detidV.push_back( std::make_pair(ids,changeDate[dp]) );
	  if (actualStatus[dp] != 1) {
	    if (board == "channel000") {ch0bad++;}
	    if (board == "channel001") {ch1bad++;}
	    StatusGood.push_back(false);
	    numLvBad.insert(numLvBad.end(),ids.begin(),ids.end());
	  } else {StatusGood.push_back(true);}
	  isHV.push_back(0);
	  psuName.push_back( dpname[dp] );
	} else if (board == "channel002" || board == "channel003") {
	  detidV.push_back( std::make_pair(ids,changeDate[dp]) );
	  if (actualStatus[dp] != 1) {
	    if (board == "channel002") {ch2bad++;}
	    if (board == "channel003") {ch3bad++;}
	    StatusGood.push_back(false);
	    numHvBad.insert(numHvBad.end(),ids.begin(),ids.end());
	  } else {StatusGood.push_back(true);}
	  isHV.push_back(1);
	  psuName.push_back( dpname[dp] );
	} else {
	  if (board != "channel001") {
	    LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "] channel name not recognised! " << board;
	  }
	}
      }
    } else {
      notMatched++;
    }
  }

  removeDuplicates(numLvBad);
  removeDuplicates(numHvBad);

  // useful debugging stuff!
  /*
  std::cout << "Bad 000 = " << ch0bad << " Bad 001 = " << ch1bad << std::endl;
  std::cout << "Bad 002 = " << ch0bad << " Bad 003 = " << ch1bad << std::endl;
  std::cout << "Number of bad LV detIDs = " << numLvBad.size() << std::endl;
  std::cout << "Number of bad HV detIDs = " << numHvBad.size() << std::endl;
  
  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of PSUs retrieved from DB with map information    " << detidV.size();
  LogTrace("SiStripModuleHVBuilder") << "[SiStripModuleHVBuilder::" << __func__ << "]: Number of PSUs retrieved from DB with no map information " << notMatched;
  
  unsigned int dupCount = 0;
  for (unsigned int t = 0; t < numLvBad.size(); t++) {
    std::vector<unsigned int>::iterator iter = std::find(numHvBad.begin(),numHvBad.end(),numLvBad[t]);
    if (iter != numHvBad.end()) {dupCount++;}
  }
  std::cout << "Number of channels with LV & HV bad = " << dupCount << std::endl;
  */

  // initialize variables
  modulesOff.clear();
  cond::Time_t saveIovTime = 0;
  
  // check if there is already an object stored in the DB
  if (lastStoredCondObj.first != NULL && lastStoredCondObj.second > 0) {
    modulesOff.push_back( lastStoredCondObj );
    saveIovTime = lastStoredCondObj.second;
    std::vector<uint32_t> pStats(3,0);
    pStats[0] = 0;
    pStats[1] = 0;
    pStats[2] = 0;
    payloadStats.push_back(pStats);
  }

  for (unsigned int i = 0; i < detidV.size(); i++) {
    std::vector<uint32_t> detids = detidV[i].first;
    removeDuplicates(detids);
    // set the condition time for the transfer
    cond::Time_t iovtime = 0;
    if (whichTable == "LASTVALUE") {iovtime = latestTime;}
    else {iovtime = getCondTime((detidV[i]).second);}
    
    // decide how to initialize modV
    SiStripDetVOff *modV = 0;
    if (iovtime != saveIovTime) { // time is different, so create new object
      if (modulesOff.empty()) {
        // create completely new object and set the initial state to Tracker all off
        modV = new SiStripDetVOff();

        // Use the file
        edm::FileInPath fp(detIdListFile_);
        SiStripDetInfoFileReader reader(fp.fullPath());
        const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > detInfos  = reader.getAllData();
        for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = detInfos.begin(); it != detInfos.end(); ++it) {
          modV->put( it->first, 1, 1 );
        }

      }
      else {modV = new SiStripDetVOff( *(modulesOff.back().first) );} // start from copy of previous object
    } else {
      modV = (modulesOff.back()).first; // modify previous object
    }
    
    // extract the detID vector before modifying for stats calculation
    std::vector<uint32_t> beforeV;
    modV->getDetIds(beforeV);

    // set the LV and HV off flags ready for storing
    int lv_off = -1, hv_off = -1;
    if (isHV[i] == 0) {lv_off = !StatusGood[i];}
    if (isHV[i] == 1) {
      hv_off = !StatusGood[i];
      // temporary fix to handle the fact that we don't know which HV channel the detIDs are associated to
      if (i > 0) {
	std::string iChannel = psuName[i].substr( (psuName[i].size()-3) );
	std::string iPsu = psuName[i].substr(0, (psuName[i].size()-3) );
	if (iChannel == "002" || iChannel == "003") {
	  for (unsigned int j = 0; j < i; j++) {
	    std::string jPsu = psuName[j].substr(0, (psuName[j].size()-3) );
	    std::string jChannel = psuName[j].substr( (psuName[j].size()-3) );
	    if (iPsu == jPsu && iChannel != jChannel && (jChannel == "002" || jChannel == "003")) {
	      if (StatusGood[i] != StatusGood[j]) {hv_off = 1;}
	    }
	  }
	}
      }
    }
    
    // store the det IDs in the conditions object
    for (unsigned int j = 0; j < detids.size(); j++) {modV->put(detids[j],hv_off,lv_off);}
    
    // calculate the stats for storage
    unsigned int numAdded = 0, numRemoved = 0;
    if (iovtime == saveIovTime) {
      std::vector<uint32_t> oldStats = payloadStats.back();
      numAdded = oldStats[1];
      numRemoved = oldStats[2];
    }
    std::vector<uint32_t> afterV;
    modV->getDetIds(afterV);
    
    if ((afterV.size() - beforeV.size()) > 0) {
      numAdded += afterV.size() - beforeV.size();
    } else if ((beforeV.size() - afterV.size()) > 0) {
      numRemoved += beforeV.size() - afterV.size();
    }
    
    // store the object if it's a new object
    if (iovtime != saveIovTime) {
      SiStripDetVOff * testV = 0;
      if (!modulesOff.empty()) {testV = modulesOff.back().first;}
      if (modulesOff.empty() ||  !(*modV == *testV) ) {
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
    } else {
      (payloadStats.back())[0] = afterV.size();
      (payloadStats.back())[1] = numAdded;
      (payloadStats.back())[2] = numRemoved;
    }
  }

  // compare the first element and the last from previous transfer
  if (lastStoredCondObj.first != NULL && lastStoredCondObj.second > 0) {
    if ( lastStoredCondObj.second == modulesOff[0].second &&
	 *(lastStoredCondObj.first) == *(modulesOff[0].first) ) {
      std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator moIt = modulesOff.begin();
      modulesOff.erase(moIt);
      std::vector< std::vector<uint32_t> >::iterator plIt = payloadStats.begin();
      payloadStats.erase(plIt);
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

//   std::cout << std::endl;
//   std::cout << "Size of modulesOff = " << modulesOff.size() << std::endl;
//   for (unsigned int i = 0; i < modulesOff.size(); i++) {
//     std::vector<uint32_t> finalids;
//     (modulesOff[i].first)->getDetIds(finalids);
//     std::cout << "Index = " << i << " Size of DetIds vector = " << finalids.size() << std::endl;
//     std::cout << "Time = " << modulesOff[i].second << std::endl;
//     for (unsigned int j = 0; j < finalids.size(); j++) {
//       std::cout << "detid = " << finalids[j] << " LV off = " << (modulesOff[i].first)->IsModuleLVOff(finalids[j]) << " HV off = " 
//       	  << (modulesOff[i].first)->IsModuleHVOff(finalids[j]) << std::endl;
//     }
//   }

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
#ifdef USING_NEW_CORAL
      const boost::posix_time::ptime& testSec = changeDate.time();
      const boost::posix_time::ptime& limitSec = settingDate[(unsigned int)locations[j]].time();
#else
      long testSec = changeDate.time().ns();
      long limitSec = settingDate[(unsigned int)locations[j]].time().ns();
#endif
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
#ifdef USING_NEW_CORAL
      const boost::posix_time::ptime& testSec = changeDate.time();
      const boost::posix_time::ptime& limitSec = settingDate[(unsigned int)locations[j]].time();
#else
      long testSec = changeDate.time().ns();
      long limitSec = settingDate[(unsigned int)locations[j]].time().ns();
#endif
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
  //  std::getline(lastValueFile,line);
  //  line.clear();
  // now extract data
  while( std::getline(lastValueFile,line) ) {
    std::istringstream ss(line);
    uint32_t dpid;
    float vmon;
    std::string changeDate;
    ss >> std::skipws >> dpid >> vmon >> changeDate;
    dpIDs.push_back(dpid);
    vmonValues.push_back(vmon);
    changeDates.push_back(changeDate);
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

cond::Time_t SiStripModuleHVBuilder::getCondTime(coral::TimeStamp coralTime) {

  // const boost::posix_time::ptime& t = coralTime.time();
  cond::Time_t condTime = cond::time::from_boost(coralTime.time());

  cout << "[SiStripModuleHVBuilder::getCondTime] Converting CoralTime into CondTime: "
       << " coralTime = (coralTimeInNs) " <<  coralTime.total_nanoseconds() << " condTime " << (condTime>> 32) << " - " << (condTime & 0xFFFFFFFF) << endl;
  
  return condTime;
}

coral::TimeStamp SiStripModuleHVBuilder::getCoralTime(cond::Time_t iovTime)
{
  // This method is defined in the TimeConversions header and it does the following:
  // - takes the seconds part of the iovTime (bit-shifting of 32)
  // - adds the nanoseconds part (first 32 bits mask)
  // - adds the time0 that is the time from begin of times (boost::posix_time::from_time_t(0);)
  coral::TimeStamp coralTime(cond::time::to_boost(iovTime));

  unsigned long long iovSec = iovTime >> 32;
  uint32_t iovNanoSec = uint32_t(iovTime);
  cond::Time_t testTime=getCondTime(coralTime);
  cout << "[SiStripModuleHVBuilder::getCoralTime] Converting CondTime into CoralTime: "
       << " condTime = " <<  iovSec << " - " << iovNanoSec 
       << " getCondTime(coralTime) = " << (testTime>>32) << " - " << (testTime&0xFFFFFFFF)  << endl;
  
  return coralTime;
}

void SiStripModuleHVBuilder::removeDuplicates( std::vector<uint32_t> & vec ) {
  std::sort(vec.begin(),vec.end());
  std::vector<uint32_t>::iterator it = std::unique(vec.begin(),vec.end());
  vec.resize( it - vec.begin() );
}

void SiStripModuleHVBuilder::setLastSiStripDetVOff( SiStripDetVOff * lastPayload, cond::Time_t lastTimeStamp ) {
  lastStoredCondObj.first = lastPayload;
  lastStoredCondObj.second = lastTimeStamp;
}

cond::Time_t SiStripModuleHVBuilder::findMostRecentTimeStamp( std::vector<coral::TimeStamp> coralDate ) {
  cond::Time_t latestDate = getCondTime(coralDate[0]);
  
  std::cout << "latestDate: condTime = " 
	    << (latestDate>>32) 
	    << " - " 
	    << (latestDate&0xFFFFFFFF) 
    //<< " coralTime= " << coralDate[0] 
	    << std::endl;

  for (unsigned int i = 1; i < coralDate.size(); i++) {
    cond::Time_t testDate = getCondTime(coralDate[i]);
    if (testDate > latestDate) {
      latestDate = testDate;
    }
  }
  return latestDate;
}

void SiStripModuleHVBuilder::reduce( std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator & it,
				     std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator & initialIt,
				     std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > & resultVec,
				     const bool last )
{
  int first = 0;
  // Check if it is the first
  if( distance(resultVec.begin(), initialIt) == 0 ) {
    first = 1;
  }
  // if it was going off
  if( it->first->getLVoffCounts() - initialIt->first->getLVoffCounts() > 0 || it->first->getHVoffCounts() - initialIt->first->getHVoffCounts() > 0 ) {
    cout << "going off" << endl;
    // Set the time of the current (last) iov as the time of the initial iov of the sequence
    // replace the first iov with the last one
    it->second = (initialIt)->second;
    it = resultVec.erase(initialIt, it);

  }
  // if it was going on
  else if( it->first->getLVoffCounts() - initialIt->first->getLVoffCounts() <= 0 || it->first->getHVoffCounts() - initialIt->first->getHVoffCounts() <= 0 ) {
    cout << "going on" << endl;
    // replace the last minus one iov with the first one
    // cout << "first = " << first << endl;
    // cout << "initial->first = " << initialIt->first << ", second  = " << initialIt->second << endl;
    if( last == true ) {
      resultVec.erase(initialIt+first, it+1);
      // Minus 2 because it will be incremented at the end of the loop becoming end()-1.
      it = resultVec.end()-2;
    }
    else {
      it = resultVec.erase(initialIt+first, it);
    }
  }
}

void SiStripModuleHVBuilder::reduction(const uint32_t deltaTmin)
{
  int count = 0;
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator initialIt;

  int resultVecSize = modulesOff.size();
  int resultsIndex = 0;

  if( resultVecSize > 1 ) {
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator it = modulesOff.begin();
    for( ; it != modulesOff.end()-1; ++it, ++resultsIndex ) {
      unsigned long long time1 = it->second >> 32;
      unsigned long long time2 = (it+1)->second >> 32;
      unsigned long long deltaT = time2 - time1;
      // std::cout << "deltaT = " << deltaT << std::endl;
      // Save the initial pair
      if( deltaT <= deltaTmin ) {
	// If we are not in a the sequence
	if( count == 0 ) {
	  initialIt = it;
	}
	// Increase the counter in any case.
	// cout << "count = " << count << endl;
	++count;
      }
      // We do it only if the sequence is bigger than two cases
      else if( count > 1 ) {
	reduce(it, initialIt, modulesOff);
	// reset all
	// cout << "resetting" << endl;
	count = 0;
      }
      else {
	// cout << "resetting" << endl;
	// reset all
	count = 0;
      }
      // Border case
      // cout << "resultsIndex = " << resultsIndex << ", resultVecSize-2 = " << resultVecSize-2 << endl;
      if( resultsIndex == resultVecSize-2 && count != 0 ) {
	reduce(it, initialIt, modulesOff, true);
      }
    }
  }
}
