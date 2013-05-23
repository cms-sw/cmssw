#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"
#include "boost/foreach.hpp"

// constructor
SiStripDetVOffBuilder::SiStripDetVOffBuilder(const edm::ParameterSet& pset, const edm::ActivityRegistry&) : 
  onlineDbConnectionString(pset.getParameter<std::string>("onlineDB")),
  authenticationPath(pset.getParameter<std::string>("authPath")),
  whichTable(pset.getParameter<std::string>("queryType")),
  lastValueFileName(pset.getParameter<std::string>("lastValueFile")),
  fromFile(pset.getParameter<bool>("lastValueFromFile")),
  psuDetIdMapFile_(pset.getParameter<std::string>("PsuDetIdMapFile")),
  debug_(pset.getParameter<bool>("debugModeOn")),
  tDefault(7,0),
  tmax_par(pset.getParameter< std::vector<int> >("Tmax")),
  tmin_par(pset.getParameter< std::vector<int> >("Tmin")),
  tset_par(pset.getParameter< std::vector<int> >("TSetMin")),
  detIdListFile_(pset.getParameter< std::string >("DetIdListFile")),
  excludedDetIdListFile_(pset.getParameter< std::string >("ExcludedDetIdListFile")),
  highVoltageOnThreshold_(pset.getParameter<double>("HighVoltageOnThreshold"))
{ 
  lastStoredCondObj.first = NULL;
  lastStoredCondObj.second = 0;

  edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::SiStripDetVOffBuilder] constructor" << endl;

  // set up vectors based on pset parameters (tDefault purely for initialization)

  whichQuery=(whichTable == "STATUSCHANGE" || (whichTable == "LASTVALUE" && !fromFile));

  //Define the query interval [Tmin, Tmax]
  //where Tmax comes from the cfg
  //      Tmin comes from the cfg for the first o2o, after that it is extracted from Offline DB 

  tmax = coral::TimeStamp(tmax_par[0],tmax_par[1],tmax_par[2],tmax_par[3],tmax_par[4],tmax_par[5],tmax_par[6]);

  if (whichQuery) {
    // Is there a better way to do this?  TODO - investigate
    tmin=coral::TimeStamp(tmin_par[0],tmin_par[1],tmin_par[2],tmin_par[3],tmin_par[4],tmin_par[5],tmin_par[6]);
  }
  
  if (whichTable == "LASTVALUE") {
    tsetmin = coral::TimeStamp(tset_par[0],tset_par[1],tset_par[2],tset_par[3],tset_par[4],tset_par[5],tset_par[6]);
  }
  
  if (onlineDbConnectionString == "") {
    edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::SiStripDetVOffBuilder] DB name has not been set properly ... Returning ...";
    return;
  }
  
  if (fromFile && whichTable == "LASTVALUE" && lastValueFileName == "") {
    edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::SiStripDetVOffBuilder] File expected for lastValue table, but filename not specified ... Returning ...";
    return;
  }
  
  // write out the parameters
  std::stringstream ss;
  ss << "[SiStripDetVOffBuilder::SiStripDetVOffBuilder]\n" 
     << "     Parameters:\n" 
     << "     DB connection string: " << onlineDbConnectionString << "\n"
     << "     Authentication path: "  << authenticationPath       << "\n"
     << "     Table to be queried: "  << whichTable               << "\n";
  
  if (whichQuery){
    ss << "     Tmin: "; printPar(ss,tmin_par);  ss << std::endl;
  }
  ss << "     Tmax: "  ; printPar(ss,tmax_par);  ss << std::endl;

  if (whichTable == "LASTVALUE"){ 
    ss << "     TSetMin: "; printPar(ss,tset_par);  ss << std::endl;
  }
   edm::LogError("SiStripDetVOffBuilder") << ss.str();

}

// destructor
SiStripDetVOffBuilder::~SiStripDetVOffBuilder() { 
  edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::" << __func__ << "]: destructing ...";
}

void SiStripDetVOffBuilder::printPar(std::stringstream& ss, const std::vector<int>& par){
  BOOST_FOREACH(int val, par){
    ss << val << " ";
  }
}

void SiStripDetVOffBuilder::BuildDetVOffObj()
{
  // vectors for storing output from DB or text file
  TimesAndValues timesAndValues;

  // Open the PVSS DB connection
  coralInterface.reset( new SiStripCoralIface(onlineDbConnectionString, authenticationPath, debug_) );
  edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: Query type is " << whichTable << endl;

  if (whichTable == "LASTVALUE") {edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: Use file? " << ((fromFile) ? "TRUE" : "FALSE");}

  if (lastStoredCondObj.second > 0) {edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: retrieved last time stamp from DB: " 
									    << lastStoredCondObj.second  << endl;}
  // access the information!

  if (whichQuery) {
    if( whichTable == "STATUSCHANGE" ) {
      statusChange( lastStoredCondObj.second, timesAndValues );
    }
    if( whichTable == "LASTVALUE" ) {
      if( fromFile ) {
	lastValueFromFile(timesAndValues);
      }
      else {
	lastValue(timesAndValues);
      }
    }
  }

  DetIdListTimeAndStatus dStruct;

  // build PSU - det ID map
  buildPSUdetIdMap(timesAndValues, dStruct);


  // initialize variables
  modulesOff.clear();
  cond::Time_t saveIovTime = 0;
  

  // - If there is already an object stored in the database
  // -- store it in the modulesOff vector
  // -- set the saveIovTime as that
  // -- set the payload stats to empty
  // Successivamente:
  // - loop su tutti gli elementi del detidV, che è stato letto dal pvss (questi elementi sono pair<vettore di detid, time>)
  // -- setta il tempo dell'IOV:
  // --- LASTVALUE -> iovtime settato a latestTime
  // --- altrimenti iovtime = tempo associato al detId vector del loop


  // check if there is already an object stored in the DB
  // This happens only if you are using STATUSCHANGE
  if (lastStoredCondObj.first != NULL && lastStoredCondObj.second > 0) {
    modulesOff.push_back( lastStoredCondObj );
    saveIovTime = lastStoredCondObj.second;
    setPayloadStats(0, 0, 0);
  }


  for (unsigned int i = 0; i < dStruct.detidV.size(); i++) {

    //     std::vector<uint32_t> detids = dStruct.detidV[i].first;
    //     removeDuplicates(detids);
    std::vector<uint32_t> * detids = &(dStruct.detidV[i].first);

    // set the condition time for the transfer
    cond::Time_t iovtime = 0;

    if (whichTable == "LASTVALUE") {iovtime = timesAndValues.latestTime;}

    else {iovtime = getCondTime((dStruct.detidV[i]).second);}

    // decide how to initialize modV
    SiStripDetVOff *modV = 0;

    // When using STATUSCHANGE they are equal only for the first
    // When using LASTVALUE they are equal only if the tmin was set to tsetmin

    if (iovtime != saveIovTime) { // time is different, so create new object

      // This can be only when using LASTVALUE or with a new tag
      if (modulesOff.empty()) {
        // create completely new object and set the initial state to Tracker all off
        modV = new SiStripDetVOff();

        // Use the file
        edm::FileInPath fp(detIdListFile_);
        SiStripDetInfoFileReader reader(fp.fullPath());
        const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > detInfos  = reader.getAllData();

        // Careful: if a module is in the exclusion list it must be ignored and the initial status is set to ON.
        // These modules are expected to not be in the PSU-DetId map, so they will never get any status change from the query.
        SiStripPsuDetIdMap map;
	std::vector< std::pair<uint32_t, std::string> > excludedDetIdMap;
        if( excludedDetIdListFile_ != "" ) {
          map.BuildMap(excludedDetIdListFile_, excludedDetIdMap);
        }
        for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = detInfos.begin(); it != detInfos.end(); ++it) {
	  std::vector< std::pair<uint32_t, std::string> >::const_iterator exclIt = excludedDetIdMap.begin();
          bool excluded = false;
          for( ; exclIt != excludedDetIdMap.end(); ++exclIt ) {
            if( it->first == exclIt->first ) {
              excluded = true;
              break;
            }
          }
          if( !excluded ) {
            modV->put( it->first, 1, 1 );
          }
        }

      }
      else {modV = new SiStripDetVOff( *(modulesOff.back().first) );} // start from copy of previous object
    }
    else {
      modV = (modulesOff.back()).first; // modify previous object
    }


    
    // extract the detID vector before modifying for stats calculation
    std::vector<uint32_t> beforeV;
    modV->getDetIds(beforeV);

    std::pair<int, int> hvlv = extractDetIdVector(i, modV, dStruct);

    for (unsigned int j = 0; j < detids->size(); j++) {
      if( debug_ ) cout << "at time = " << iovtime << " detid["<<j<<"] = " << (*detids)[j] << " has hv = " << hvlv.first << " and lv = " << hvlv.second << endl;
      modV->put((*detids)[j],hvlv.first,hvlv.second);
    }

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
	setPayloadStats(afterV.size(), numAdded, numRemoved);
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
}

int SiStripDetVOffBuilder::findSetting(uint32_t id, coral::TimeStamp changeDate, std::vector<uint32_t> settingID, std::vector<coral::TimeStamp> settingDate) {
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

int SiStripDetVOffBuilder::findSetting(std::string dpname, coral::TimeStamp changeDate, std::vector<std::string> settingDpname, std::vector<coral::TimeStamp> settingDate) {
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

void SiStripDetVOffBuilder::readLastValueFromFile(std::vector<uint32_t> &dpIDs, std::vector<float> &vmonValues, std::vector<coral::TimeStamp> &dateChange) {
  std::ifstream lastValueFile(lastValueFileName.c_str());
  if (lastValueFile.bad()) {
    edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::" << __func__ << "]: last Value file does not exist!";
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

  if (changeDates.size() != dateChange.size()) {edm::LogError("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::" << __func__ << "]: date conversion failed!!";}
}

cond::Time_t SiStripDetVOffBuilder::getCondTime(coral::TimeStamp coralTime) {

  // const boost::posix_time::ptime& t = coralTime.time();
  cond::Time_t condTime = cond::time::from_boost(coralTime.time());

  // cout << "[SiStripDetVOffBuilder::getCondTime] Converting CoralTime into CondTime: "
  //      << " coralTime = (coralTimeInNs) " <<  coralTime.total_nanoseconds() << " condTime " << (condTime>> 32) << " - " << (condTime & 0xFFFFFFFF) << endl;

  return condTime;
}

coral::TimeStamp SiStripDetVOffBuilder::getCoralTime(cond::Time_t iovTime)
{
  // This method is defined in the TimeConversions header and it does the following:
  // - takes the seconds part of the iovTime (bit-shifting of 32)
  // - adds the nanoseconds part (first 32 bits mask)
  // - adds the time0 that is the time from begin of times (boost::posix_time::from_time_t(0);)
  coral::TimeStamp coralTime(cond::time::to_boost(iovTime));

  if( debug_ ) {
    unsigned long long iovSec = iovTime >> 32;
    uint32_t iovNanoSec = uint32_t(iovTime);
    cond::Time_t testTime=getCondTime(coralTime);
    cout << "[SiStripDetVOffBuilder::getCoralTime] Converting CondTime into CoralTime: "
	 << " condTime = " <<  iovSec << " - " << iovNanoSec 
	 << " getCondTime(coralTime) = " << (testTime>>32) << " - " << (testTime&0xFFFFFFFF)  << endl;
  }

  return coralTime;
}

void SiStripDetVOffBuilder::removeDuplicates( std::vector<uint32_t> & vec ) {
  std::sort(vec.begin(),vec.end());
  std::vector<uint32_t>::iterator it = std::unique(vec.begin(),vec.end());
  vec.resize( it - vec.begin() );
}

void SiStripDetVOffBuilder::setLastSiStripDetVOff( SiStripDetVOff * lastPayload, cond::Time_t lastTimeStamp ) {
  lastStoredCondObj.first = lastPayload;
  lastStoredCondObj.second = lastTimeStamp;
}

cond::Time_t SiStripDetVOffBuilder::findMostRecentTimeStamp( std::vector<coral::TimeStamp> coralDate ) {
  cond::Time_t latestDate = getCondTime(coralDate[0]);
  
  if( debug_ ) {
    std::cout << "latestDate: condTime = " 
	      << (latestDate>>32) 
	      << " - " 
	      << (latestDate&0xFFFFFFFF) 
      //<< " coralTime= " << coralDate[0] 
	      << std::endl;
  }

  for (unsigned int i = 1; i < coralDate.size(); i++) {
    cond::Time_t testDate = getCondTime(coralDate[i]);
    if (testDate > latestDate) {
      latestDate = testDate;
    }
  }
  return latestDate;
}

void SiStripDetVOffBuilder::reduce( std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator & it,
				    std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator & initialIt,
				    std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > & resultVec,
				    const bool last )
{
  int first = 0;
  // Check if it is the first
  if( distance(resultVec.begin(), initialIt) == 0 ) {
    first = 1;
  }

  if( debug_ && ( it->first->getLVoffCounts() - initialIt->first->getLVoffCounts() == 0 ) && ( it->first->getHVoffCounts() - initialIt->first->getHVoffCounts() == 0 ) ) {
    cout << "Same number of LV and HV at start and end of sequence: LV off = " << it->first->getLVoffCounts() << " HV off = " << it->first->getHVoffCounts() << endl;
  }

  // if it was going off
  if( ( it->first->getLVoffCounts() - initialIt->first->getLVoffCounts() > 0 ) || ( it->first->getHVoffCounts() - initialIt->first->getHVoffCounts() > 0 ) ) {
    // Set the time of the current (last) iov as the time of the initial iov of the sequence
    // replace the first iov with the last one
    (it+last)->second = (initialIt)->second;
    discardIOVs(it, initialIt, resultVec, last, 0);
    if( debug_ ) cout << "going off" << endl;
  }
  // if it was going on
  else if( ( it->first->getLVoffCounts() - initialIt->first->getLVoffCounts() <= 0 ) || ( it->first->getHVoffCounts() - initialIt->first->getHVoffCounts() <= 0 ) ) {
    // replace the last minus one iov with the first one
    discardIOVs(it, initialIt, resultVec, last, first);
    if( debug_ ) cout << "going on" << endl;
  }
}

void SiStripDetVOffBuilder::discardIOVs( std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator & it,
                                         std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator & initialIt,
                                         std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > & resultVec,
                                         const bool last, const unsigned int first )
{
  if( debug_ ) {
    cout << "first = " << first << endl;
    cout << "initial->first = " << initialIt->first << ", second  = " << initialIt->second << endl;
    cout << "last = " << last << endl;
  }
  if( last == true ) {
    resultVec.erase(initialIt+first, it+1);
    // Minus 2 because it will be incremented at the end of the loop becoming end()-1.
    it = resultVec.end()-2;
  }
  else {
    it = resultVec.erase(initialIt+first, it);
  }
}

void SiStripDetVOffBuilder::reduction(const uint32_t deltaTmin, const uint32_t maxIOVlength)
{
  int count = 0;
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator initialIt;

  int resultVecSize = modulesOff.size();
  int resultsIndex = 0;

  if( resultVecSize > 1 ) {
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator it = modulesOff.begin();
    for( ; it != modulesOff.end()-1; ++it, ++resultsIndex ) {
      unsigned long long deltaT = ((it+1)->second - it->second) >> 32;
      unsigned long long deltaTsequence = 0;
      if( count > 1 ) {
	deltaTsequence = ((it+1)->second - initialIt->second) >> 32;
      }
      // Save the initial pair
      if( (deltaT < deltaTmin) && ( (count == 0) || ( deltaTsequence < maxIOVlength ) ) ) {
	// If we are not in a the sequence
	if( count == 0 ) {
	  initialIt = it;
	}
	// Increase the counter in any case.
	++count;
      }
      // We do it only if the sequence is bigger than two cases
      else if( count > 1 ) {
	reduce(it, initialIt, modulesOff);
	// reset all
	count = 0;
      }
      else {
	// reset all
	count = 0;
      }
      // Border case
      if( resultsIndex == resultVecSize-2 && count != 0 ) {
	reduce(it, initialIt, modulesOff, true);
      }
    }
  }
}

void SiStripDetVOffBuilder::statusChange( cond::Time_t & lastTime, TimesAndValues & tStruct )
{
  // Setting tmin to the last value IOV of the database tag
  if( lastTime > 0 ) {
    tmin = getCoralTime(lastTime);
  }
  
  coralInterface->doQuery(whichTable, tmin ,tmax, tStruct.changeDate, tStruct.actualValue, tStruct.dpname);

  // preset the size of the status vector
  tStruct.actualStatus.resize(tStruct.actualValue.size());
  tStruct.actualStatus.clear();
  
  BOOST_FOREACH(float val, tStruct.actualValue) {
    tStruct.actualStatus.push_back(static_cast<int>(val));
  }
}

void SiStripDetVOffBuilder::lastValue(TimesAndValues & tStruct)
{
  coralInterface->doQuery(whichTable, tmin ,tmax, tStruct.changeDate, tStruct.actualValue, tStruct.dpname);
  
  tStruct.latestTime = findMostRecentTimeStamp( tStruct.changeDate );
  
  // preset the size of the status vector
  tStruct.actualStatus.resize(tStruct.actualValue.size());
  
  // retrieve the channel settings from the PVSS DB
  std::vector<coral::TimeStamp> settingDate;
  std::vector<float> settingValue;
  std::vector<std::string> settingDpname;
  std::vector<uint32_t> settingDpid;
  coralInterface->doSettingsQuery(tsetmin,tmax,settingDate,settingValue,settingDpname,settingDpid);
  LogDebug("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: Channel settings retrieved";
  LogDebug("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: Number of PSU channels: " << settingDpname.size();
    
  unsigned int missing = 0;
  std::stringstream ss;
  for (unsigned int j = 0; j < tStruct.dpname.size(); j++) {
    int setting = findSetting(tStruct.dpname[j],tStruct.changeDate[j],settingDpname,settingDate);
    if (setting >= 0) {
      if (tStruct.actualValue[j] > (highVoltageOnThreshold_*(settingValue[setting]))) {tStruct.actualStatus[j] = 1;}
      else {tStruct.actualStatus[j] = 0;}
    } else {
      tStruct.actualStatus[j] = -1;
      missing++;
      ss << "Channel = " << tStruct.dpname[j] << std::endl;
    }
  }
  LogDebug("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: Number of channels with no setting information " << missing;
  LogDebug("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: Number of entries in dpname vector " << tStruct.dpname.size();
}

void SiStripDetVOffBuilder::lastValueFromFile(TimesAndValues & tStruct)
{
  readLastValueFromFile(tStruct.dpid,tStruct.actualValue,tStruct.changeDate);
  tStruct.latestTime = findMostRecentTimeStamp( tStruct.changeDate );
  LogDebug("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: File access complete \n\t Number of values read from file: " << tStruct.dpid.size();
  
  // retrieve the channel settings from the PVSS DB
  std::vector<coral::TimeStamp> settingDate;
  std::vector<float> settingValue;
  std::vector<std::string> settingDpname;
  std::vector<uint32_t> settingDpid;
  
  coralInterface->doSettingsQuery(tsetmin,tmax,settingDate,settingValue,settingDpname,settingDpid);
  LogDebug("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: Channel settings retrieved";
  LogDebug("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::BuildDetVOff]: Number of PSU channels: " << settingDpname.size();
  
  unsigned int missing = 0;
  std::stringstream ss;
  // need to get the PSU channel names from settings
  tStruct.dpname.clear();
  tStruct.dpname.resize(tStruct. dpid.size());
  for (unsigned int j = 0; j < tStruct.dpid.size(); j++) {
    int setting = findSetting(tStruct.dpid[j],tStruct.changeDate[j],settingDpid,settingDate);
    if (setting >= 0) {
      if (tStruct.actualValue[j] > (highVoltageOnThreshold_*settingValue[setting])) {tStruct.actualStatus[j] = 1;}
      else {
	tStruct.actualStatus[j] = 0;
      }
      tStruct.dpname[j] = settingDpname[setting];
    } else {
      tStruct.actualStatus[j] = -1;
      tStruct.dpname[j] = "UNKNOWN";
      missing++;
      ss << "DP ID = " << tStruct.dpid[j] << std::endl;
    }
  }
  LogDebug("SiStripDetVOffBuilder") << "Number of missing psu channels = " << missing << std::endl;
  LogDebug("SiStripDetVOffBuilder") << "IDs are: = " << ss.str();
}

string SiStripDetVOffBuilder::timeToStream(const cond::Time_t & condTime, const string & comment)
{
  stringstream ss;
  ss << comment << (condTime>> 32) << " - " << (condTime & 0xFFFFFFFF) << std::endl;
  return ss.str();
}

string SiStripDetVOffBuilder::timeToStream(const coral::TimeStamp & coralTime, const string & comment)
{
  stringstream ss;
  ss << "Starting from IOV time in the database : year = " << coralTime.year()
     << ", month = " << coralTime.month()
     << ", day = " << coralTime.day()
     << ", hour = " << coralTime.hour()
     << ", minute = " << coralTime.minute()
     << ", second = " << coralTime.second()
     << ", nanosecond = " << coralTime.nanosecond() << std::endl;
  return ss.str();
}

void SiStripDetVOffBuilder::buildPSUdetIdMap(TimesAndValues & psuStruct, DetIdListTimeAndStatus & detIdStruct)
{
  SiStripPsuDetIdMap map_;
  if( psuDetIdMapFile_ == "" ) {
    map_.BuildMap();
  }
  else {
    map_.BuildMap(psuDetIdMapFile_);
  }
  LogTrace("SiStripDetVOffBuilder") <<"[SiStripDetVOffBuilder::BuildDetVOff] DCU-DET ID map built";
  map_.printMap();

  // use map info to build input for list of objects
  // no need to check for duplicates, as put method for SiStripDetVOff checks for you!
  
  unsigned int ch0bad = 0, ch1bad = 0, ch2bad = 0, ch3bad = 0;
  std::vector<unsigned int> numLvBad, numHvBad;

  for (unsigned int dp = 0; dp < psuStruct.dpname.size(); dp++) {
    if (psuStruct.dpname[dp] != "UNKNOWN") {

      // figure out the channel
      std::string board = psuStruct.dpname[dp];
      std::string::size_type loc = board.size()-10;
      board.erase(0,loc);
      // now store!
      std::vector<uint32_t> ids = map_.getDetID(psuStruct.dpname[dp]);

      if( debug_ ) cout << "dbname["<<dp<<"] = " << psuStruct.dpname[dp] << ", for time = " << timeToStream(psuStruct.changeDate[dp]) << std::endl;

      if (!ids.empty()) {
	// DCU-PSU maps only channel000 and channel000 and channel001 switch on and off together
	// so check only channel000
	//	if (board == "channel000" || board == "channel001") {
	if (board == "channel000") {
	  detIdStruct.detidV.push_back( std::make_pair(ids,psuStruct.changeDate[dp]) );
	  if (psuStruct.actualStatus[dp] != 1) {
	    // 	    if (board == "channel000") {ch0bad++;}
	    // 	    if (board == "channel001") {ch1bad++;}
	    ++ch0bad;
	    ++ch1bad;
	    detIdStruct.StatusGood.push_back(false);
	    numLvBad.insert(numLvBad.end(),ids.begin(),ids.end());
	  }
	  else {
	    detIdStruct.StatusGood.push_back(true);
	  }
	  detIdStruct.isHV.push_back(0);
	  detIdStruct.psuName.push_back( psuStruct.dpname[dp] );
	}
	else if( board == "channel002" || board == "channel003" ) {
	  detIdStruct.detidV.push_back( std::make_pair(ids,psuStruct.changeDate[dp]) );
	  if( debug_ ) cout << "actualStatus = " << psuStruct.actualStatus[dp] << " for psu: " << psuStruct.dpname[dp] << endl;
	  if (psuStruct.actualStatus[dp] != 1) {
	    if (board == "channel002") {ch2bad++;}
	    if (board == "channel003") {ch3bad++;}
	    detIdStruct.StatusGood.push_back(false);
	    numHvBad.insert(numHvBad.end(),ids.begin(),ids.end());
	  }
	  else {
	    detIdStruct.StatusGood.push_back(true);
	  }
	  detIdStruct.isHV.push_back(1);
	  detIdStruct.psuName.push_back( psuStruct.dpname[dp] );
	}
	else {
	  if (board != "channel001") {
	    LogTrace("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::" << __func__ << "] channel name not recognised! " << board;
	  }
	}
      }
    } else {
      detIdStruct.notMatched++;
    }
  }

  removeDuplicates(numLvBad);
  removeDuplicates(numHvBad);

  // useful debugging stuff!
  if( debug_ ) {
    std::cout << "Bad 000 = " << ch0bad << " Bad 001 = " << ch1bad << std::endl;
    std::cout << "Bad 002 = " << ch0bad << " Bad 003 = " << ch1bad << std::endl;
    std::cout << "Number of bad LV detIDs = " << numLvBad.size() << std::endl;
    std::cout << "Number of bad HV detIDs = " << numHvBad.size() << std::endl;
  }

  LogTrace("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::" << __func__ << "]: Number of PSUs retrieved from DB with map information    " << detIdStruct.detidV.size();
  LogTrace("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::" << __func__ << "]: Number of PSUs retrieved from DB with no map information " << detIdStruct.notMatched;
  
  unsigned int dupCount = 0;
  for (unsigned int t = 0; t < numLvBad.size(); t++) {
    std::vector<unsigned int>::iterator iter = std::find(numHvBad.begin(),numHvBad.end(),numLvBad[t]);
    if (iter != numHvBad.end()) {dupCount++;}
  }
  if( debug_ ) std::cout << "Number of channels with LV & HV bad = " << dupCount << std::endl;
}

void SiStripDetVOffBuilder::setPayloadStats(const uint32_t afterV, const uint32_t numAdded, const uint32_t numRemoved)
{
  std::vector<uint32_t> pStats(3,0);
  pStats.push_back(afterV);
  pStats.push_back(numAdded);
  pStats.push_back(numRemoved);
  payloadStats.push_back(pStats);
}

pair<int, int> SiStripDetVOffBuilder::extractDetIdVector( const unsigned int i, SiStripDetVOff * modV, DetIdListTimeAndStatus & detIdStruct )
{
  // set the LV and HV off flags ready for storing
  int lv_off = -1, hv_off = -1;
  if (detIdStruct.isHV[i] == 0) {lv_off = !(detIdStruct.StatusGood[i]);}
  if (detIdStruct.isHV[i] == 1) {
    hv_off = !(detIdStruct.StatusGood[i]);

    // TESTING WITHOUT THE FIX
    // -----------------------

    if( psuDetIdMapFile_ == "" ) {
      // temporary fix to handle the fact that we don't know which HV channel the detIDs are associated to
      if (i > 0) {
	std::string iChannel = detIdStruct.psuName[i].substr( (detIdStruct.psuName[i].size()-3) );
	std::string iPsu = detIdStruct.psuName[i].substr(0, (detIdStruct.psuName[i].size()-3) );
	if (iChannel == "002" || iChannel == "003") {
	  bool lastStatusOfOtherChannel = true;
	  for (unsigned int j = 0; j < i; j++) {
	    std::string jPsu = detIdStruct.psuName[j].substr(0, (detIdStruct.psuName[j].size()-3) );
	    std::string jChannel = detIdStruct.psuName[j].substr( (detIdStruct.psuName[j].size()-3) );
	    if (iPsu == jPsu && iChannel != jChannel && (jChannel == "002" || jChannel == "003")) {
	      if( debug_ ) cout << "psu["<<i<<"] = " << detIdStruct.psuName[i] << " with status = " << detIdStruct.StatusGood[i] << " and psu["<<j<<"] = " << detIdStruct.psuName[j] << " with status " << detIdStruct.StatusGood[j] << endl;
	      lastStatusOfOtherChannel = detIdStruct.StatusGood[j];
	    }
	  }
	  if (detIdStruct.StatusGood[i] != lastStatusOfOtherChannel) {
	    if( debug_ ) cout << "turning off hv" << endl;
	    hv_off = 1;
	  }
	}
      }
    }

    // -----------------------

  }

  return make_pair(hv_off, lv_off);
}
