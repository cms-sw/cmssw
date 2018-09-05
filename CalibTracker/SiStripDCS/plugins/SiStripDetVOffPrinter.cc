#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include "CondCore/CondDB/interface/ConnectionPool.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TLegend.h>
#include <TGraph.h>
#include <TH1.h>


class SiStripDetVOffPrinter : public edm::one::EDAnalyzer<> {
public:

  explicit SiStripDetVOffPrinter(const edm::ParameterSet& iConfig );
  ~SiStripDetVOffPrinter() override;
  void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void endJob() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:

  cond::persistency::ConnectionPool m_connectionPool;
  std::string m_condDb;
  std::string m_tagName;

  // Manually specify the start/end time. Format: "2002-01-20 23:59:59.000".
  std::string m_startTime;
  std::string m_endTime;
  // Specify output text file name. Leave empty if do not want to dump HV/LV counts in a file.
  std::string m_output;
  edm::Service<SiStripDetInfoFileReader> detidReader;

  //          IOV                 DETIDs
  std::map<cond::Time_t, std::set< uint32_t > > iovMap_HVOff;
  std::map<cond::Time_t, std::set< uint32_t > > iovMap_LVOff;
  //          DETIDs              IOV
  std::map<uint32_t, std::vector< cond::Time_t > > detidMap;
};

SiStripDetVOffPrinter::SiStripDetVOffPrinter(const edm::ParameterSet& iConfig):
  m_connectionPool(),
  m_condDb( iConfig.getParameter< std::string >("conditionDatabase") ),
  m_tagName( iConfig.getParameter< std::string >("tagName") ),
  m_startTime( iConfig.getParameter< std::string >("startTime") ),
  m_endTime( iConfig.getParameter< std::string >("endTime") ),
  m_output( iConfig.getParameter< std::string >("output") )
{
  m_connectionPool.setParameters( iConfig.getParameter<edm::ParameterSet>("DBParameters")  );
  m_connectionPool.configure();
}

SiStripDetVOffPrinter::~SiStripDetVOffPrinter() {
}

void SiStripDetVOffPrinter::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  
  // get total number of modules
  //auto num_modules = detidReader->getAllDetIds().size();

  // use start and end time from config file
  boost::posix_time::ptime p_start, p_end;
  p_start = boost::posix_time::time_from_string(m_startTime);
  p_end   = boost::posix_time::time_from_string(m_endTime);
  cond::Time_t startIov = cond::time::from_boost(p_start);
  cond::Time_t endIov   = cond::time::from_boost(p_end);
  if (startIov > endIov)
    throw cms::Exception("endTime must be greater than startTime!");
  edm::LogInfo("SiStripDetVOffPrinter") << "[SiStripDetVOffPrinter::" << __func__ << "] "
					<< "Set start time " << startIov << " (" << boost::posix_time::to_simple_string(p_start) << ")"
					<< "\n ... Set end time " << endIov << " (" << boost::posix_time::to_simple_string(p_end) << ")" ;

  // open db session
  edm::LogInfo("SiStripDetVOffPrinter") << "[SiStripDetVOffPrinter::" << __func__ << "] "
					<< "Query the condition database " << m_condDb;
  cond::persistency::Session condDbSession = m_connectionPool.createSession( m_condDb );
  condDbSession.transaction().start( true );

  std::stringstream ss;
  // list of times with new IOVs within the time range
  std::vector< cond::Time_t > vTime;
  
  // query the database
  edm::LogInfo("SiStripDetVOffPrinter") << "[SiStripDetVOffPrinter::" << __func__ << "] "
					<< "Reading IOVs from tag " << m_tagName;
  cond::persistency::IOVProxy iovProxy = condDbSession.readIov(m_tagName, true); // load all?
  auto iiov = iovProxy.find(startIov);
  auto eiov = iovProxy.find(endIov);
  int niov = 0;
  while (iiov != iovProxy.end() && (*iiov).since <= (*eiov).since){
    // convert cond::Time_t to seconds since epoch
    if ((*iiov).since<startIov){
      vTime.push_back(startIov);
    } else {
      vTime.push_back((*iiov).since);
    }
    auto payload = condDbSession.fetchPayload<SiStripDetVOff>( (*iiov).payloadId );
    // print IOVs summary
    ss  << boost::posix_time::to_simple_string(cond::time::to_boost((*iiov).since))
	<< " (" << (*iiov).since <<  ")"
	<< ", # HV Off=" << std::setw(6) << payload->getHVoffCounts()
	<< ", # LV Off=" << std::setw(6) << payload->getLVoffCounts() << std::endl;

    // list of detids with HV/LV Off
    std::vector<uint32_t> detIds;
    payload->getDetIds(detIds);
    std::set<uint32_t> detIds_HVOff;
    std::set<uint32_t> detIds_LVOff;
    std::vector<uint32_t>::const_iterator it = detIds.begin();
    for( ; it!=detIds.end(); ++it ) {
      if(payload->IsModuleHVOff(*it) ) detIds_HVOff.insert(*it);
      if(payload->IsModuleLVOff(*it) ) detIds_LVOff.insert(*it);
      
      if(detidMap.find(*it)==detidMap.end()) {
	std::vector< cond::Time_t > vec;
	detidMap[*it] = vec;		  
      }
      
      // for each module concerned by the IOV, add the time in an history vector
      detidMap[*it].push_back(vTime.back());
    }
    
    // fill list of channels Off at a given time
    iovMap_HVOff[vTime.back()] = detIds_HVOff;
    iovMap_LVOff[vTime.back()] = detIds_LVOff;
    
    /*std::vector<uint32_t>::const_iterator it = detIds.begin();
      for( ; it!=detIds.end(); ++it ) {
      std::cout << *it << std::endl;
      }*/
    
    ++iiov;
    ++niov;
  }
  vTime.push_back(endIov); // used to compute last IOV duration

  edm::LogInfo("SiStripDetVOffPrinter") << "[SiStripDetVOffPrinter::" << __func__ << "] "
					<< "Read " << niov << " IOVs from tag " << m_tagName << " corresponding to the specified time interval.\n" << ss.str();


  // Create a map of IOVs time_duration
  std::map< cond::Time_t, boost::posix_time::time_duration > mIOVsDuration;
  std::vector< cond::Time_t >::const_iterator itTime = ++vTime.begin();
  std::vector< cond::Time_t >::const_iterator itPreviousTime = vTime.begin();
  //std::vector< cond::Time_t >::const_iterator itLastTime = --vTime.end();
  for( ; itTime!=vTime.end(); ++itTime ) {
    mIOVsDuration[ *itPreviousTime ] = cond::time::to_boost(*itTime) - cond::time::to_boost(*itPreviousTime);
    itPreviousTime = itTime;
  }
  boost::posix_time::time_duration time_period = cond::time::to_boost(*(--vTime.end())) - cond::time::to_boost(*(vTime.begin()));
  
  // debug
  /*for( itTime=vTime.begin(); itTime!=itLastTime; ++itTime ) {
	std::cout<<boost::posix_time::to_simple_string( cond::time::to_boost(*itTime) ) << "    " <<boost::posix_time::to_simple_string(mIOVsDuration[ *itTime ])<<std::endl;
  }*/

  // Print summary per module	
  edm::LogInfo("SiStripDetVOffPrinter") << "[SiStripDetVOffPrinter::" << __func__ << "] "
					<< detidMap.size() << " modules were Off at some point during the time interval";
  ss.str("");
  
  // Loop over detIds
  std::map<uint32_t, std::vector< cond::Time_t > >::const_iterator itMap = detidMap.begin();
  for( ; itMap!=detidMap.end(); ++itMap ) {
    std::vector< cond::Time_t > vecTime = itMap->second;
    
    boost::posix_time::time_duration cumul_time_HVOff(0,0,0,0);
    boost::posix_time::time_duration cumul_time_LVOff(0,0,0,0);
    // Loop over IOVs
    std::vector< cond::Time_t >::const_iterator itTime = vecTime.begin();
    for( ; itTime!=vecTime.end(); ++itTime ) {
      if(iovMap_HVOff[*itTime].find(itMap->first) != iovMap_HVOff[*itTime].end()) cumul_time_HVOff+=mIOVsDuration[*itTime];
      if(iovMap_LVOff[*itTime].find(itMap->first) != iovMap_LVOff[*itTime].end()) cumul_time_LVOff+=mIOVsDuration[*itTime];
    }
    ss<<"detId "<< itMap->first <<" #IOVs: "<<vecTime.size()
      <<"  HVOff: "<<cumul_time_HVOff<<" "<<cumul_time_HVOff.total_milliseconds()*100.0/time_period.total_milliseconds()<<"% "
      <<"  LVOff: "<<cumul_time_LVOff<<" "<<cumul_time_LVOff.total_milliseconds()*100.0/time_period.total_milliseconds()<<"%"<<std::endl;
    
  }

  condDbSession.transaction().commit();
 
  if (!m_output.empty()) {
    std::ofstream fout;
    fout.open(m_output);
    fout << ss.str();
    fout.close();
  }  
}

void
SiStripDetVOffPrinter::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<std::string>("conditionDatabase","frontier://FrontierProd/CMS_CONDITIONS");
  desc.add<std::string>("tagName","SiStripDetVOff_1hourDelay_v1_Validation");
  desc.add<std::string>("startTime","2002-01-20 23:59:59.000");
  desc.add<std::string>("endTime","2002-01-20 23:59:59.000");
  desc.add<std::string>("output","PerModuleSummary.txt");
  desc.add<std::string>("connect","");

  edm::ParameterSetDescription descDBParameters;
  descDBParameters.addUntracked<std::string>("authenticationPath","");  
  descDBParameters.addUntracked<int>("authenticationSystem",0);  
  descDBParameters.addUntracked<std::string>("security","");  
  descDBParameters.addUntracked<int>("messageLevel",0);  

  desc.add<edm::ParameterSetDescription>("DBParameters",descDBParameters);
  descriptions.add("siStripDetVOffPrinter", desc);
}


void SiStripDetVOffPrinter::endJob() {
}

DEFINE_FWK_MODULE(SiStripDetVOffPrinter);

