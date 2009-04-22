#include "CalibTracker/SiStripDCS/test/plugins/dpLocationMap.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;
using namespace sistrip;

dpLocationMap::dpLocationMap(const edm::ParameterSet& pset) :
  onlineDbConnectionString(pset.getUntrackedParameter<std::string>("onlineDB","")),
  authenticationPath(pset.getUntrackedParameter<std::string>("authPath","../data")),
  tDefault(7,0)
{ 
  // set up vectors based on pset parameters (tDefault purely for initialization)
  tmin_par = pset.getUntrackedParameter< std::vector<int> >("Tmin",tDefault);
  tmax_par = pset.getUntrackedParameter< std::vector<int> >("Tmax",tDefault);
  
  // initialize the coral timestamps
  if (tmin_par != tDefault && tmax_par != tDefault) {
    coral::TimeStamp mincpy(tmin_par[0],tmin_par[1],tmin_par[2],tmin_par[3],tmin_par[4],tmin_par[5],tmin_par[6]);
    tmin = mincpy;
    coral::TimeStamp maxcpy(tmax_par[0],tmax_par[1],tmax_par[2],tmax_par[3],tmax_par[4],tmax_par[5],tmax_par[6]);
    tmax = maxcpy;
  } else {
    LogTrace("dpLocationMap") << "[dpLocationMap::" << __func__ << "] time interval not set properly ... Returning ...";
  }
  
}

dpLocationMap::~dpLocationMap() {}

void dpLocationMap::beginRun( const edm::Run& run, const edm::EventSetup& setup ) {
  // build map
  SiStripPsuDetIdMap map_;
  map_.BuildMap();
  LogTrace("dpLocationMap") <<"[dpLocationMap::" << __func__ << "] DCU-DET ID map built";
  
  // Open PVSS Cond DB access
  SiStripCoralIface * iFace_ = new SiStripCoralIface(onlineDbConnectionString,authenticationPath);
  std::vector<std::string> Dpname;
  std::vector<uint32_t> Dpid;
  iFace_->doNameQuery(Dpname,Dpid);

  std::vector< std::pair<std::string,std::string> > NameLocationMap;
  std::vector< std::pair<uint32_t,std::string> > idLocationMap;
  for (unsigned int i = 0; i < Dpname.size(); i++) {
    std::string location_ = map_.getDetectorLocation(Dpname[i]);
    if (location_ != "UNKNOWN") {
      idLocationMap.push_back( std::make_pair(Dpid[i],location_) );
      NameLocationMap.push_back( std::make_pair(Dpname[i],location_) );
    }
  }
  
  std::sort(idLocationMap.begin(),idLocationMap.end());
  std::vector< std::pair<uint32_t,std::string> >::iterator it = std::unique(idLocationMap.begin(),idLocationMap.end());
  idLocationMap.resize( it - idLocationMap.begin() );
  
  /*
  std::cout << "Size of vector = " << idLocationMap.size() << std::endl;
  for (unsigned int j = 0; j < idLocationMap.size(); j++) {
    std::cout << std::setw(6) << idLocationMap[j].first << " " << std::setw(20) << idLocationMap[j].second << std::endl;
  }
  */

  std::cout << "Size of vector = " << NameLocationMap.size() << std::endl;
  for (unsigned int j = 0; j < NameLocationMap.size(); j++) {
    std::cout << NameLocationMap[j].first << " " << NameLocationMap[j].second << std::endl;
  }
}

