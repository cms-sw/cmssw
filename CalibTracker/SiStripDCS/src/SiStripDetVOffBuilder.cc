#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"
#include "boost/foreach.hpp"
#include <sys/stat.h>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

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
  deltaTmin_(pset.getParameter<uint32_t>("DeltaTmin")),
  maxIOVlength_(pset.getParameter<uint32_t>("MaxIOVlength")),
  detIdListFile_(pset.getParameter< std::string >("DetIdListFile")),
  excludedDetIdListFile_(pset.getParameter< std::string >("ExcludedDetIdListFile")),
  highVoltageOnThreshold_(pset.getParameter<double>("HighVoltageOnThreshold"))
{ 
  lastStoredCondObj.first = nullptr;
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
     << "     Table to be queried: "  << whichTable               << "\n"
     << "     MapFile: "  << psuDetIdMapFile_                << "\n";
  
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

void SiStripDetVOffBuilder::BuildDetVOffObj(const TrackerTopology* trackerTopo)
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
  //We've been using the STATUSCHANGE query only in last year or so... LASTVALUE may have untested issues...
  //In either case the idea is that the results of the query are saved into the timesAndValues struct
  //ready to be anaylized (i.e. translated into detIDs, HV/LV statuses)

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

  //Initialize the stuct that will be used to keep the detID-translated information 
  DetIdListTimeAndStatus dStruct;

  // build PSU - det ID map
  //The following method actually "builds" 4 maps: LVMap, HVMap, HVUnmapped_Map, HVCrosstalking_Map.
  //It also takes the timesAndValues from the query above and using the maps, it processes the information
  //populating the DetIDListTimeAndStatus struct that will hold the information by detid.
  buildPSUdetIdMap(timesAndValues, dStruct);


  // initialize variables
  modulesOff.clear();
  cond::Time_t saveIovTime = 0;
  

  // - If there is already an object stored in the database
  // -- store it in the modulesOff vector
  // -- set the saveIovTime as that
  // -- set the payload stats to empty
  // Successivamente:
  // - loop su tutti gli elementi del detidV, che stato letto dal pvss (questi elementi sono pair<vettore di detid, time>)
  // -- setta il tempo dell'IOV:
  // --- LASTVALUE -> iovtime settato a latestTime
  // --- altrimenti iovtime = tempo associato al detId vector del loop


  // check if there is already an object stored in the DB
  // This happens only if you are using STATUSCHANGE
  if (lastStoredCondObj.first != nullptr && lastStoredCondObj.second > 0) {
    modulesOff.push_back( lastStoredCondObj );
    saveIovTime = lastStoredCondObj.second;
    setPayloadStats(0, 0, 0);
  }


  //Master loop over all the results of the query stored in the dStruct (that contains vectors with vectors of detids, statuses, isHV flags, etc and in particular a vector of timestamps for which the info is valid... basically it is a loop over the timestamps (i.e. IOVs).
  for (unsigned int i = 0; i < dStruct.detidV.size(); i++) {
 
    //     std::vector<uint32_t> detids = dStruct.detidV[i].first;
    //     removeDuplicates(detids);
    std::vector<uint32_t> * detids = &(dStruct.detidV[i].first);

    // set the condition time for the transfer
    cond::Time_t iovtime = 0;

    if (whichTable == "LASTVALUE") {iovtime = timesAndValues.latestTime;}

    else {iovtime = getCondTime((dStruct.detidV[i]).second);}

    // decide how to initialize modV
    SiStripDetVOff *modV = nullptr;

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
        const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >& detInfos  = reader.getAllData();

	//FIXME:
	//Following code is actually broken (well not until the cfg has "" for excludedDetIDListFile parameter!
	//Fix it if felt necessary (remember that it assumes that whatever detids are excluded should NOT be in the regular map
	//breaking our current situation with a fully mapped (LV-wise) tracker...
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
      modV = (modulesOff.back()).first; // modify previous object (TEST THIS if possible! it's fundamental in handling changes at the edges of O2O executions and also in case of PVSS DB buffering!
    }


    
    // extract the detID vector before modifying for stats calculation
    std::vector<uint32_t> beforeV;
    modV->getDetIds(beforeV);

    //CHECKTHIS
    //The following method call is potentially problematic: 
    //passing modV as argument while extracting information about dStruct,
    //modV is not currently used in the method!
    std::pair<int, int> hvlv = extractDetIdVector(i, modV, dStruct);//Returns a pair like this HV OFF->1,-1 HV ON->0,-1 LV OFF->-1,1 LV ON->-1,0
    //Basically a LV OFF of -1  means that the information (IOV) in question is from an HV channel turning on or off and viceversa for an HVOFF of -1.
    //This could be confusing when reading the debug output!
 
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
      SiStripDetVOff * testV = nullptr;
      if (!modulesOff.empty()) {testV = modulesOff.back().first;}
      if (modulesOff.empty() ||  !(*modV == *testV) ) {
        modulesOff.push_back( std::make_pair(modV,iovtime) );
        // save the time of the object
        saveIovTime = iovtime;
        // save stats
        setPayloadStats(afterV.size(), numAdded, numRemoved);
      } else {
        // modV will not be used anymore, DELETE it to avoid memory leak!
        delete modV;
      }
    } else {
      (payloadStats.back())[0] = afterV.size();
      (payloadStats.back())[1] = numAdded;
      (payloadStats.back())[2] = numRemoved;
    }
  }


  // compare the first element and the last from previous transfer
  if (lastStoredCondObj.first != nullptr && lastStoredCondObj.second > 0) {
    if ( *(lastStoredCondObj.first) == *(modulesOff[0].first) ) {
      if ( modulesOff.size() == 1 ){
        // if no HV/LV transition was found in this period: update the last IOV to be tmax
        modulesOff[0].second = getCondTime(tmax);
      }else{
        // HV/LV transitions found: remove the first one (which came from previous transfer)
        modulesOff.erase(modulesOff.begin());
        payloadStats.erase(payloadStats.begin());
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

int SiStripDetVOffBuilder::findSetting(uint32_t id, const coral::TimeStamp& changeDate, const std::vector<uint32_t>& settingID, const std::vector<coral::TimeStamp>& settingDate) {
  int setting = -1;
  // find out how many channel entries there are
  std::vector<int> locations;
  for (unsigned int i = 0; i < settingID.size(); i++) { if (settingID[i] == id) {locations.push_back((int)i);} }

  // simple cases
  if (locations.empty()) {setting = -1;}
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

int SiStripDetVOffBuilder::findSetting(std::string dpname, const coral::TimeStamp& changeDate, const std::vector<std::string>& settingDpname, const std::vector<coral::TimeStamp>& settingDate) {
  int setting = -1;
  // find out how many channel entries there are
  std::vector<int> locations;
  for (unsigned int i = 0; i < settingDpname.size(); i++) { if (settingDpname[i] == dpname) {locations.push_back((int)i);} }
  
  // simple cases
  if (locations.empty()) {setting = -1;}
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

cond::Time_t SiStripDetVOffBuilder::getCondTime(const coral::TimeStamp& coralTime) {

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

cond::Time_t SiStripDetVOffBuilder::findMostRecentTimeStamp( const std::vector<coral::TimeStamp>& coralDate ) {
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
  //const bool last is set to false by default in the header file...
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
    //Naughty use of const bool last... by default it is false (=0), and for the case of the last timestamp in the query results it is set to true(=1) in the call 
    (it+last)->second = (initialIt)->second;
    discardIOVs(it, initialIt, resultVec, last, 0);
    if( debug_ ) cout << "Reducing IOV sequence (going off)" << endl;
  }
  // if it was going on
  else if( ( it->first->getLVoffCounts() - initialIt->first->getLVoffCounts() <= 0 ) || ( it->first->getHVoffCounts() - initialIt->first->getHVoffCounts() <= 0 ) ) {
    // replace the last minus one iov with the first one
    discardIOVs(it, initialIt, resultVec, last, first);
    if( debug_ ) cout << "Reducing IOV sequence (going on)" << endl;
  }
}

void SiStripDetVOffBuilder::discardIOVs( std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator & it,
                                         std::vector< std::pair<SiStripDetVOff*,cond::Time_t> >::iterator & initialIt,
                                         std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > & resultVec,
                                         const bool last, const unsigned int first )
{
  if( debug_ ) {
    cout << "first (1->means the sequence started at the first timestamp in the query results, 0-> that it did not)= " << first << endl;
    cout << "initial->first (initial SiStripDetVOff object of the IOV sequence)= " << initialIt->first << ", second (initial timestamp of the IOV sequence) = " << initialIt->second << endl;
    cout << "last (0->means that the sequence is not ending with the last item in the query results, 1-> that it DOES!)= " << last << endl;
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

//This is the method that (called by GetModulesOff, declared in the header file) executes the reduction by massaging modulesOff
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
  //UNIT TEST DEBUG to bypass the query wait time!!!
  //coral::TimeStamp testtime=getCoralTime(lastTime);
  //tStruct.changeDate.push_back(testtime);
  //tStruct.actualValue.push_back(1.);
  //tStruct.dpname.push_back("cms_trk_dcs_03:CAEN/CMS_TRACKER_SY1527_3/branchController00/easyCrate3/easyBoard17/channel002");

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
bool SiStripDetVOffBuilder::FileExists(string FileName) {
  //Helper method to check if local files exist (needed to handle HVUnmapped, HVCrosstalking modules)
  struct stat FileInfo;
  bool Existence;
  int Stat;
  //Try to get file attributes
  Stat=stat(FileName.c_str(),&FileInfo);
  if (Stat==0) {
    Existence=true;
  }
  else {
    Existence=false;
  }
  return Existence;
}

void SiStripDetVOffBuilder::buildPSUdetIdMap(TimesAndValues & psuStruct, DetIdListTimeAndStatus & detIdStruct)
//This function builds a PSU to DetID map one way or the other. Then it processes the psuStruct that contains
//the results of the CAEN status query to the Online DB, filling the detIdStruct with the detIDs corresponding
//to the PSU (channel in some cases, PSU only in others) reported in the CAEN status query to the Online DB.
//It may make sense to split this method eventually.
{
  SiStripPsuDetIdMap map_;
  if( psuDetIdMapFile_ == "" ) {
    std::cout<<"PLEASE provide the name of a valid PSUDetIDMapFile in the cfg: currently still necessary to have a file, soon will access the info straight from the DB!"<<endl;
    //map_.BuildMap();//This method is not currently used (it would try to build a map based on a query to SiStripConfigDB, and the info there is STALE!)
  }
  else {
    map_.BuildMap(psuDetIdMapFile_,debug_); //This is the method used to build the map.
  }
  LogTrace("SiStripDetVOffBuilder") <<"[SiStripDetVOffBuilder::BuildDetVOff] PSU(Channel)-detID map(s) built";
  //Following method to be replaced by printMaps... to print all 4 maps built!
  map_.printMap(); //This method prints to the info.log file, notice that it is overwritten by each individual O2O job running in the same dir.

  // use map info to build input for list of objects
  // no need to check for duplicates, as put method for SiStripDetVOff checks for you!

  //Debug variables
  unsigned int ch0bad = 0, ch1bad = 0, ch2bad = 0, ch3bad = 0;
  std::vector<unsigned int> numLvBad, numHvBad;

  //Create 2 extra maps that we'll use to keep track of unmapped and crosstalking detids when turning ON and OFF HV:
  //-unmapped need to be both turned OFF when any HV goes OFF and to be turned ON when both are ON
  //-crosstaling need to be both turned ON when any HV goes ON and to be turned OFF ONLY when BOTH are OFF.
  std::map<std::string,bool> UnmappedState, CrosstalkingState;
  //Get the HVUnmapped map from the map, so that we can set know which PSU are unmapped:
  std::map<std::string,std::vector<uint32_t> > UnmappedPSUs=map_.getHVUnmappedMap();
  //Check here if there is a file already, otherwise initialize to OFF all channels in these PSU!
  if (FileExists("HVUnmappedChannelState.dat")) {
    std::cout<<"File HVUnmappedChannelState.dat exists!"<<std::endl;
    std::ifstream ifs("HVUnmappedChannelState.dat");
    string line;
    while( getline( ifs, line ) ) {
      if( line != "" ) {
	// split the line and insert in the map
	stringstream ss(line);
	string PSUChannel;
	bool HVStatus;
	ss >> PSUChannel;
	ss >> HVStatus;
	//Extract the PSU from the PSUChannel (since the HVUnmapped_Map uses PSU as key
	std::string PSU=PSUChannel.substr(0,PSUChannel.size()-10);
	//Look for the PSU in the unmapped map!
	std::map<std::string,std::vector<uint32_t> >::iterator iter=UnmappedPSUs.find(PSU);
	if (iter!=UnmappedPSUs.end()) {
	  UnmappedState[PSUChannel]=HVStatus;
	}
	else {
	  std::cout<<"WARNING!!! There are channels in the local file with the channel status for HVUnmapped channels, that ARE NOT CONSIDERED AS UNMAPPED in the current map!"<<std::endl;
	}
      }
    }//End of the while loop reading and initializing UnmappedState map from file
    //Extra check:
    //Should check if there any HVUnmapped channels in the map that are not listed in the local file!
    bool MissingChannels=false;
    for (std::map<std::string, vector<uint32_t> >::iterator it=UnmappedPSUs.begin(); it!=UnmappedPSUs.end(); it++) {
      std::string chan002=it->first+"channel002";
      std::string chan003=it->first+"channel003";
      std::map<std::string,bool>::iterator iter=UnmappedState.find(chan002);
      if (iter==UnmappedState.end()) {
	std::cout<<"ERROR! The local file with the channel status for HVUnmapped channels IS MISSING one of the following unmapped channel voltage status information:"<<std::endl;
	std::cout<<chan002<<std::endl;
	MissingChannels=true;
      }
      iter=UnmappedState.find(chan003);
      if (iter==UnmappedState.end()) {
	std::cout<<"ERROR! The local file with the channel status for HVUnmapped channels IS MISSING one of the following unmapped channel voltage status information:"<<std::endl;
	std::cout<<chan003<<std::endl;
	MissingChannels=true;
      }
    }
    //Now if any channel WAS missing, exit!
    if (MissingChannels) {
      std::cout<<"!!!!\n"<<"Exiting now... please check the local HVUnmappedChannelState.dat and the mapfile you provided ("<<psuDetIdMapFile_<<")"<<std::endl;
      exit(1);
    }
  }
  else { //If the file HVUnmappedChannelState.dat does not exist, initialize the map to all OFF. 
    //(see below for creating the file at the end of the execution with the latest state of unmapped channels. 
    for (std::map<std::string, vector<uint32_t> >::iterator it=UnmappedPSUs.begin(); it!=UnmappedPSUs.end(); it++) {
      std::string chan002=it->first+"channel002";
      std::string chan003=it->first+"channel003";
      UnmappedState[chan002]=false;
      UnmappedState[chan003]=false;
    }
  }
  //Get the HVCrosstalking map from the map, so that we can set know which PSU are crosstalking:
  std::map<std::string,std::vector<uint32_t> > CrosstalkingPSUs=map_.getHVCrosstalkingMap();
  //Check here if there is a file already, otherwise initialize to OFF all channels in these PSU!
  if (FileExists("HVCrosstalkingChannelState.dat")) {
    std::cout<<"File HVCrosstalkingChannelState.dat exists!"<<std::endl;
    std::ifstream ifs("HVCrosstalkingChannelState.dat");
    string line;
    while( getline( ifs, line ) ) {
      if( line != "" ) {
	// split the line and insert in the map
	stringstream ss(line);
	string PSUChannel;
	bool HVStatus;
	ss >> PSUChannel;
	ss >> HVStatus;
	//Extract the PSU from the PSUChannel (since the HVCrosstalking_Map uses PSU as key
	std::string PSU=PSUChannel.substr(0,PSUChannel.size()-10);
	//Look for the PSU in the unmapped map!
	std::map<std::string,std::vector<uint32_t> >::iterator iter=CrosstalkingPSUs.find(PSU);
	if (iter!=CrosstalkingPSUs.end()) {
	  CrosstalkingState[PSUChannel]=HVStatus;
	}
	else {
	  std::cout<<"WARNING!!! There are channels in the local file with the channel status for HVUnmapped channels, that ARE NOT CONSIDERED AS UNMAPPED in the current map!"<<std::endl;
	}
      }
    }//End of the while loop reading and initializing CrosstalkingState map from file
    //Extra check:
    //Should check if there any HVCrosstalking channels in the map that are not listed in the local file!
    bool MissingChannels=false;
    for (std::map<std::string, vector<uint32_t> >::iterator it=CrosstalkingPSUs.begin(); it!=CrosstalkingPSUs.end(); it++) {
      std::string chan002=it->first+"channel002";
      std::string chan003=it->first+"channel003";
      std::map<std::string,bool>::iterator iter=CrosstalkingState.find(chan002);
      if (iter==CrosstalkingState.end()) {
	std::cout<<"ERROR! The local file with the channel status for HVCrosstalking channels IS MISSING one of the following unmapped channel voltage status information:"<<std::endl;
	std::cout<<chan002<<std::endl;
	MissingChannels=true;
      }
      iter=CrosstalkingState.find(chan003);
      if (iter==CrosstalkingState.end()) {
	std::cout<<"ERROR! The local file with the channel status for HVCrosstalking channels IS MISSING one of the following unmapped channel voltage status information:"<<std::endl;
	std::cout<<chan003<<std::endl;
	MissingChannels=true;
      }
    }
    //Now if any channel WAS missing, exit!
    if (MissingChannels) {
      std::cout<<"!!!!\n"<<"Exiting now... please check the local HVCrosstalkingChannelState.dat and the mapfile you provided ("<<psuDetIdMapFile_<<")"<<std::endl;
      exit(1);
    }
  }
  else { //If the file HVCrosstalkingChannelState.dat does not exist, initialize the map to all OFF. 
    //(see below for creating the file at the end of the execution with the latest state of unmapped channels. 
    for (std::map<std::string, vector<uint32_t> >::iterator it=CrosstalkingPSUs.begin(); it!=CrosstalkingPSUs.end(); it++) {
      std::string chan002=it->first+"channel002";
      std::string chan003=it->first+"channel003";
      CrosstalkingState[chan002]=false;
      CrosstalkingState[chan003]=false;
    }
  }
  
  if (debug_) {
    //print out the UnmappedState map:
    std::cout<<"Printing the UnmappedChannelState initial map:"<<std::endl;
    std::cout<<"PSUChannel\t\tHVON?(true or false)"<<std::endl;
    for (std::map<std::string,bool>::iterator it=UnmappedState.begin(); it!=UnmappedState.end(); it++) {
      std::cout<<it->first<<"\t\t"<<it->second<<std::endl;
    }
    //print out the CrosstalkingState map:
    std::cout<<"Printing the CrosstalkingChannelState initial map:"<<std::endl;
    std::cout<<"PSUChannel\t\tHVON?(true or false)"<<std::endl;
    for (std::map<std::string,bool>::iterator it=CrosstalkingState.begin(); it!=CrosstalkingState.end(); it++) {
      std::cout<<it->first<<"\t\t"<<it->second<<std::endl;
    }
  }
  
  //Loop over the psuStruct (DB query results), lopping over the PSUChannels
  //This will probably change int he future when we will change the query itself 
  //to report directly the detIDs associated with a channel
  //Probably we will report in the query results the detID, the changeDate 
  //and whether the channel is HV mapped, HV unmapped, HV crosstalking using a flag...
  for (unsigned int dp = 0; dp < psuStruct.dpname.size(); dp++) {
    //FIX ME:
    //Check if the following if condition can EVER be true!
    std::string PSUChannel=psuStruct.dpname[dp];
    if (PSUChannel != "UNKNOWN") {

      // figure out the channel and the PSU individually
      std::string Channel = PSUChannel.substr(PSUChannel.size()-10); //Channel is the channel, i.e. channel000, channel001 etc
      std::string PSU = PSUChannel.substr(0,PSUChannel.size()-10);
      
      // Get the detIDs corresponding to the given PSU channel using the getDetID function of SiStripPsuDetIdMap.cc
      //NOTA BENE
      //Need to make sure the information is treated consistently here:
      //The map by convention has 
      //detID-> channel002 or channel003 IF the channel is HV mapped,
      //detID->channel000 if it is not HV mapped
      //We want to differentiate the behavior depending on the status reported for the channel for channels that are unmapped!
      //1-if the channel is turning OFF (!=1) then we want to report all detIDs for that channel and all detIDs that are listed as channel000 for that PSU.
      //2-if the channel is turning ON (==1) then we want to turn on all detIDs for that channel but turn on all detIDs listed as channel000 for that PSU ONLY IF BOTH channel002 and channel003 are BOTH ON!
      //Need to handle the case of coupled Power supplies (that only turn off when both are turned off).

      //Fixed SiStripPSUdetidMap.cc to make sure now getDetID gets the correct list of detIDs:
      //-for channels 000/001 all the detIDs connected to the PSU
      //-for channels 002/003 HV1/HV2 modules only (exclusively) 
      //UPDATE for HV channels: 
      //actually fixed it to report also detIDs listed as 
      //channel000 on the same supply of channel002 or channel003 
      //and the crosstalking ones (channel999) too..

      //Get the detIDs associated with the DPNAME (i.e. PSUChannel) reported by the query
      //Declare the vector to be passed as reference parameters to the getDetID method
      //std::vector<uint32_t> ids,unmapped_ids,crosstalking_ids;
      std::vector<uint32_t> ids;
      //map_.getDetID(PSUChannel, debug_, ids, unmapped_ids, crosstalking_ids);
      //Actually the method above is a bit of an overkill, we could already use the individual methods:
      //getLvDetID
      //getHvDetID

      //Declaring the two vector needed for the HV case in this scope.
      std::vector<uint32_t> unmapped_ids,crosstalking_ids;
      bool LVCase;
      //LV CASE
      if (Channel=="channel000" || Channel=="channel001") {
	LVCase=true;
	ids=map_.getLvDetID(PSU); //Since in the LV case only 1 list of detids is returned (unmapped and crosstalking are irrelevant for LV) return the vector directly
      }
      //HV CASE
      else { //if (Channel=="channel002" || Channel=="channel003") {
	LVCase=false;
	map_.getHvDetID(PSUChannel,ids,unmapped_ids,crosstalking_ids); //In the HV case since 3 vectors are filled, use reference parameters
      }

      if ( debug_ ) {
	cout <<"dpname["<<dp<<"] = "<<PSUChannel<<", for time = "<<timeToStream(psuStruct.changeDate[dp])<<endl;
	if (!ids.empty()) {
	  if (Channel=="channel000" || Channel=="channel001") {
	    cout << "Corresponding to LV (PSU-)matching detids: "<<endl;
	    for (unsigned int i_detid=0;i_detid<ids.size();i_detid++) {
	      cout<< ids[i_detid] << std::endl;
	    }
	  }
	  else {
	    cout << "Corresponding to straight HV matching detids: "<<endl;
	    for (unsigned int i_detid=0;i_detid<ids.size();i_detid++) {
	      cout<< ids[i_detid] << std::endl;
	    }
	  }
	}
	//The unmapped_ids and crosstalking_ids are only filled for HV channels!
	if (!unmapped_ids.empty()) {
	  cout << "Corresponding to HV unmapped (PSU-)matching detids: "<<endl;
	  for (unsigned int i_detid=0;i_detid<unmapped_ids.size();i_detid++) {
	    cout<< unmapped_ids[i_detid] << std::endl;
	  }
	}
	if (!crosstalking_ids.empty()) {
	  cout << "Corresponding to HV crosstalking (PSU-)matching detids: "<<endl;
	  for (unsigned int i_detid=0;i_detid<crosstalking_ids.size();i_detid++) {
	    cout<< crosstalking_ids[i_detid] << std::endl;
	  }
	}
      }
	      
      //NOW implement the new logic using the detids, unmapped_detids, crosstalking_detids!

      //First check whether the channel we're looking at is turning OFF or turning ON!
      
      //TURN OFF case:
      if (psuStruct.actualStatus[dp] != 1) {
	//Behavior is different for LV vs HV channels:
	//LV case:
	if (LVCase) {
	  //Turn OFF all: 
	  //-positively matching 
	  //-unmapped matching
	  //-crosstalking
	  //for the LV case all the detids are automatically reported in the ids vector
	  //unmapped and crosstalking are only differentiated (relevant) for HV.
	  if (!ids.empty()) {
	    //debug variables increment
	    ch0bad++;
	    ch1bad++;
	    
	    //Create a pair with the relevant detIDs (vector) and its timestamp
	    //And put it in the detidV vector of the detIdStruct that will contain all the 
	    //results
	    detIdStruct.detidV.push_back( std::make_pair(ids,psuStruct.changeDate[dp]) );
	    
	    //Set the status to OFF
	    detIdStruct.StatusGood.push_back(false);
	    
	    //debug variable population
	    numLvBad.insert(numLvBad.end(),ids.begin(),ids.end());

	    //Set the flag for LV/HV:
	    detIdStruct.isHV.push_back(0); //LV

	    //Set the PSUChannel (I guess for debug purposes?)
	    detIdStruct.psuName.push_back( PSUChannel );
	  }
	}
	//HV case:
	else { //if (!LVCase) {
	  //Debug variables increment:
	  if (!ids.empty() || !unmapped_ids.empty() || !crosstalking_ids.empty()) {
	    if (Channel=="channel002") {
	      ch2bad++;
	    }
	    else if (Channel=="channel003") {
	      ch3bad++;
	    }
	  }
	  //First sum the ids (positively matching detids) and the unmapped_ids (since both should be TURNED OFF):
	  std::vector<uint32_t> OFFids;
	  OFFids.insert(OFFids.end(),ids.begin(),ids.end()); //Add the ids (if any!)
	  OFFids.insert(OFFids.end(),unmapped_ids.begin(),unmapped_ids.end()); //Add the unmapped_ids (if any!)
	  //Now for the cross-talking ids this is a bit more complicated!
	  if (!crosstalking_ids.empty()) {//This already means that the PSUChannel is one of the crosstalking ones (even if only a few modules in that PSU are showing crosstalking behavior both its channels have to be considered crosstalking of course!
	    //Set the channel OFF in the CrosstalkingState map!
	    CrosstalkingState[PSUChannel]=false; //Turn OFF the channel in the state map!
	    
	    //Need to check if both channels (HV1==channel002 or HV2==channel003) are OFF!
	    if (!CrosstalkingState[PSUChannel.substr(0,PSUChannel.size()-1)+"2"] && !CrosstalkingState[PSUChannel.substr(0,PSUChannel.size()-1)+"3"]) { //if HV1 & HV2 both OFF (false)
		OFFids.insert(OFFids.end(),crosstalking_ids.begin(),crosstalking_ids.end()); //Add the crosstalking_ids (if any!) since both HV1 and HV2 are OFF!
		if (debug_) {
		  std::cout<<"Adding the unmapped detids corresponding to (HV1/2 cross-talking) PSU "<<PSUChannel.substr(0,PSUChannel.size()-10)<<" to the list of detids turning OFF"<<std::endl;
		}
	    }
	  }
	  //Handle the crosstalking channel by setting it to OFF in the CrosstalkingState map!
	  if (!unmapped_ids.empty()) {//This already means that the PSUChannel is one of the unmapped ones (even if only a few modules in that PSU are unmapped both its channels have to be considered crosstalking of course!
	    UnmappedState[PSUChannel]=false; //Turn OFF the channel in the state map!
	  }
	  if (!OFFids.empty()) {
	    //Create a pair with the relevant detIDs (vector) and its timestamp
	    //And put it in the detidV vector of the detIdStruct that will contain all the 
	    //results
	    
	    //Going OFF HV:
	    //report not only ids, but also unmapped_ids.
	    //have to handle crosstalking_ids here... (only OFF if they corresponding PSU HV1/HV2 is off already...
	    //then add all three vectors to the pair below...
	    detIdStruct.detidV.push_back( std::make_pair(OFFids,psuStruct.changeDate[dp]) );
	    
	    //Set the status to OFF
	    detIdStruct.StatusGood.push_back(false);
	    
	    //debug variable population
	    numHvBad.insert(numHvBad.end(),ids.begin(),ids.end());

	    //Set the flag for LV/HV:
	    detIdStruct.isHV.push_back(1); //HV

	    //Set the PSUChannel (I guess for debug purposes?)
	    detIdStruct.psuName.push_back( PSUChannel );
	  }
	}
      }
      //TURNING ON CASE
      else {
	//Implement the rest of the logic!
	//Behavior is different for LV vs HV channels:
	//LV case:
	if (LVCase) {
	  //Turn ON all (PSU)matching detids: 
	  //for the LV case all the detids are automatically reported in the ids vector
	  //unmapped and crosstalking are only differentiated (relevant) for HV.
	  if (!ids.empty()) {
	    //Create a pair with the relevant detIDs (vector) and its timestamp
	    //And put it in the detidV vector of the detIdStruct that will contain all the 
	    //results
	    detIdStruct.detidV.push_back( std::make_pair(ids,psuStruct.changeDate[dp]) );
	    
	    //Set the status to ON
	    detIdStruct.StatusGood.push_back(true);
	    
	    //Set the flag for LV/HV:
	    detIdStruct.isHV.push_back(0); //LV

	    //Set the PSUChannel (I guess for debug purposes?)
	    detIdStruct.psuName.push_back( PSUChannel );
	  }
	}
	//HV case:
	else { //if (!LVCase) {
	  //First sum the ids (positively matching detids) and the crosstalking_ids (since all ids on a crosstalking PSU should be TURNED ON when at least one HV channel is ON):
	  std::vector<uint32_t> ONids;
	  ONids.insert(ONids.end(),ids.begin(),ids.end()); //Add the ids (if any!)
	  ONids.insert(ONids.end(),crosstalking_ids.begin(),crosstalking_ids.end()); //Add the crosstalking_ids (if any!)
	  //Now for the unmapped ids this is a bit more complicated!
	  if (!unmapped_ids.empty()) {//This already means that the PSUChannel is one of the unmapped ones (even if only a few modules in that PSU are unmapped both its channels have to be considered unmapped of course!
	    //Set the HV1 channel on in the UnmappedState map!
	    UnmappedState[PSUChannel]=true; //Turn ON the channel in the state map!

	    //Need to check if BOTH channels (HV1==channel002 or HV2==channel003) are ON!
	    if (UnmappedState[PSUChannel.substr(0,PSUChannel.size()-1)+"2"] && UnmappedState[PSUChannel.substr(0,PSUChannel.size()-1)+"3"]) { //if HV1 & HV2 are both ON (true)
	      ONids.insert(ONids.end(),unmapped_ids.begin(),unmapped_ids.end()); //Add the unmapped_ids (if any!) since both HV1 and HV2 are ON!
	      if (debug_) {
		  std::cout<<"Adding the detids corresponding to HV-unmapped PSU "<<PSUChannel.substr(0,PSUChannel.size()-10)<<" to the list of detids turning ON"<<std::endl;
		}
	    }
	  }
	  //Handle the crosstalking channel by setting it to OFF in the CrosstalkingState map!
	  if (!crosstalking_ids.empty()) {//This already means that the PSUChannel is one of the crosstalking ones (even if only a few modules in that PSU are showing crosstalking behavior both its channels have to be considered crosstalking of course!
	    CrosstalkingState[PSUChannel]=true; //Turn ON the channel in the state map!
	  }
	  if (!ONids.empty()) {
	    //Create a pair with the relevant detIDs (vector) and its timestamp
	    //And put it in the detidV vector of the detIdStruct that will contain all the 
	    //results
	    
	    //Going OFF HV:
	    //report not only ids, but also unmapped_ids.
	    //have to handle crosstalking_ids here... (only OFF if they corresponding PSU HV1/HV2 is off already...
	    //then add all three vectors to the pair below...
	    detIdStruct.detidV.push_back( std::make_pair(ONids,psuStruct.changeDate[dp]) );
	    
	    //Set the status to ON
	    detIdStruct.StatusGood.push_back(true);
	    
	    //Set the flag for LV/HV:
	    detIdStruct.isHV.push_back(1); //HV

	    //Set the PSUChannel (I guess for debug purposes?)
	    detIdStruct.psuName.push_back( PSUChannel );
	  }
	}
      }
    }//End of if dpname not "UNKNOWN" 
    else {
      //if (debug) {
      //std::cout<<"PSU Channel name WAS NOT RECOGNIZED"<<std::endl;
      //}
      detIdStruct.notMatched++;
    }
  }//End of the loop over all PSUChannels reported by the DB query.
  //At this point we need to (over)write the 2 files that will keep the HVUnmapped and HVCrosstalking channels status:
  std::ofstream ofsUnmapped("HVUnmappedChannelState.dat");
  for (std::map<std::string,bool>::iterator it=UnmappedState.begin(); it!=UnmappedState.end(); it++) {
    ofsUnmapped<<it->first<<"\t"<<it->second<<std::endl;
  }
  std::ofstream ofsCrosstalking("HVCrosstalkingChannelState.dat");
  for (std::map<std::string,bool>::iterator it=CrosstalkingState.begin(); it!=CrosstalkingState.end(); it++) {
    ofsCrosstalking<<it->first<<"\t"<<it->second<<std::endl;
  }

  removeDuplicates(numLvBad);
  removeDuplicates(numHvBad);


  // useful debugging stuff!
  if( debug_ ) {
    std::cout << "Number of channels that turned OFF in this O2O interval"<<std::endl;
    std::cout << "Channel000 = " << ch0bad << " Channel001 = " << ch1bad << std::endl;
    std::cout << "Channel002 = " << ch2bad << " Channel003 = " << ch3bad << std::endl;
    std::cout << "Number of LV detIDs that turned OFF in this O2O interval = " << numLvBad.size() << std::endl;
    std::cout << "Number of HV detIDs that turned OFF in this O2O interval = " << numHvBad.size() << std::endl;
  }

  LogTrace("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::" << __func__ << "]: Number of PSUs retrieved from DB with map information    " << detIdStruct.detidV.size();
  LogTrace("SiStripDetVOffBuilder") << "[SiStripDetVOffBuilder::" << __func__ << "]: Number of PSUs retrieved from DB with no map information " << detIdStruct.notMatched;
  
  unsigned int dupCount = 0;
  for (unsigned int t = 0; t < numLvBad.size(); t++) {
    std::vector<unsigned int>::iterator iter = std::find(numHvBad.begin(),numHvBad.end(),numLvBad[t]);
    if (iter != numHvBad.end()) {dupCount++;}
  }
  if( debug_ ) std::cout << "Number of channels for which LV & HV turned OFF in this O2O interval = " << dupCount << std::endl;
  
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
