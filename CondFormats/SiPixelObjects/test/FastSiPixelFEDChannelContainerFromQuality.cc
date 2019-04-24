//
//
//
//

#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <TROOT.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TLegend.h>
#include <TGraph.h>
#include <TH1.h>

namespace SiPixelFEDChannelUtils {
  std::pair<unsigned int,unsigned int> unpack(cond::Time_t since){
    auto kLowMask = 0XFFFFFFFF;
    auto run  = (since >> 32);
    auto lumi = (since & kLowMask);
    return std::make_pair(run,lumi);
  }
}

class FastSiPixelFEDChannelContainerFromQuality : public edm::one::EDAnalyzer<> {
public:

  explicit FastSiPixelFEDChannelContainerFromQuality(const edm::ParameterSet& iConfig );
  ~FastSiPixelFEDChannelContainerFromQuality() override;
  void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void endJob() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  SiPixelFEDChannelContainer::SiPixelFEDChannelCollection createFromSiPixelQuality(const SiPixelQuality & theQuality, const SiPixelFedCablingMap& theFedCabling,const SiPixelFedCablingTree& theCablingTree);

private:

  cond::persistency::ConnectionPool m_connectionPool;
  const std::string m_condDb;
  const std::string m_QualityTagName;
  const std::string m_CablingTagName; 
  const std::string m_record;

  // Specify output text file name. Leave empty if do not want to dump in a file
  const std::string m_output;

  // Manually specify the start/end time.
  unsigned long long m_startTime;
  unsigned long long m_endTime;    

  const bool printdebug_;
  const bool isMC_;
  const bool removeEmptyPayloads_;

  SiPixelFEDChannelContainer* myQualities;

  inline unsigned int closest_from_above(std::vector<unsigned int> const& vec, unsigned int value) {
    auto const it = std::lower_bound(vec.begin(), vec.end(), value);
    if (it == vec.end() && vec.size()>1) { return -1; }
    return *it;
  }

  inline unsigned int closest_from_below(std::vector<unsigned int> const& vec, unsigned int value) {
    auto const it = std::upper_bound(vec.begin(), vec.end(), value);
    if (it == vec.end() && vec.size()>1) { return -1; }
    return *it;
  }


};

FastSiPixelFEDChannelContainerFromQuality::FastSiPixelFEDChannelContainerFromQuality(const edm::ParameterSet& iConfig):
  m_connectionPool(),
  m_condDb   ( iConfig.getParameter< std::string >("conditionDatabase") ),
  m_QualityTagName  ( iConfig.getParameter< std::string >("qualityTagName") ),
  m_CablingTagName ( iConfig.getParameter< std::string >("cablingMapTagName") ),
  m_record   ( iConfig.getParameter<std::string>("record")),
  m_output   ( iConfig.getParameter< std::string >("output") ),
  m_startTime( iConfig.getParameter< unsigned long long >("startIOV") ),
  m_endTime  ( iConfig.getParameter< unsigned long long >("endIOV") ),
  printdebug_( iConfig.getUntrackedParameter<bool>("printDebug",false)),
  isMC_      ( iConfig.getUntrackedParameter<bool>("isMC",true)),
  removeEmptyPayloads_( iConfig.getUntrackedParameter<bool>("removeEmptyPayloads",false))
{
  m_connectionPool.setParameters( iConfig.getParameter<edm::ParameterSet>("DBParameters")  );
  m_connectionPool.configure();

  //now do what ever initialization is needed
  myQualities = new SiPixelFEDChannelContainer();

}

FastSiPixelFEDChannelContainerFromQuality::~FastSiPixelFEDChannelContainerFromQuality() {
  delete myQualities;
}

void FastSiPixelFEDChannelContainerFromQuality::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  

  cond::Time_t startIov = m_startTime; 
  cond::Time_t endIov   = m_endTime;   
  if (startIov > endIov)
    throw cms::Exception("endTime must be greater than startTime!");
  edm::LogInfo("FastSiPixelFEDChannelContainerFromQuality") << "[FastSiPixelFEDChannelContainerFromQuality::" << __func__ << "] "
  				     << "Set start time " << startIov 
  				     << "\n ... Set end time " << endIov;

  // open db session
  edm::LogInfo("FastSiPixelFEDChannelContainerFromQuality") << "[FastSiPixelFEDChannelContainerFromQuality::" << __func__ << "] "
				     << "Query the condition database " << m_condDb;
  cond::persistency::Session condDbSession = m_connectionPool.createSession( m_condDb );
  condDbSession.transaction().start( true );

  std::stringstream ss;
  // list of times with new IOVs within the time range
  //std::vector< cond::Time_t > vTime;
  
  // query the database
  edm::LogInfo("FastSiPixelFEDChannelContainerFromQuality") << "[FastSiPixelFEDChannelContainerFromQuality::" << __func__ << "] "
				     << "Reading IOVs from tag " << m_QualityTagName;

  // cond::persistency::IOVProxy iovProxy = condDbSession.readIov(m_QualityTagName, true); // load all?
  // auto iiov = iovProxy.find(startIov);
  // auto eiov = iovProxy.find(endIov);
  // int niov = 0;
  // while (iiov != iovProxy.end() && (*iiov).since <= (*eiov).since){
  //   // convert cond::Time_t to seconds since epoch
  //   if ((*iiov).since<startIov){
  //     vTime.push_back(startIov);
  //   } else {
  //     vTime.push_back((*iiov).since);
  //   }
  //   auto payload = condDbSession.fetchPayload<SiPixelQuality>( (*iiov).payloadId );
  //   auto runLS = SiPixelFEDChannelUtils::unpack((*iiov).since);
  //   // print IOVs summary
  //   ss  << runLS.first << ","<< runLS.second << " ("<< (*iiov).since <<")" << " [hash: " << (*iiov).payloadId << "] \n";
    
  //   cond::persistency::IOVProxy iovProxyMap = condDbSession.readIov(m_CablingTagName, true); // load all?  
  //   auto iiovMap = iovProxyMap.find(runLS.first);
  //   if( iiovMap == iovProxyMap.end() ){
  //     std::cout <<"Could not find iov with since="<< runLS.first <<" in tag "<< m_CablingTagName <<std::endl;
  //   }

  //   auto map_payload = condDbSession.fetchPayload<SiPixelFedCablingMap>( (*iiovMap).payloadId );
  //   ss  << "cabling IOV: " << runLS.first << " ("<< (*iiovMap).since <<")" << " [hash: " << (*iiovMap).payloadId << "] \n" << std::endl;
    
  //   std::string scenario = std::to_string(runLS.first)+"_"+std::to_string(runLS.second);
  
  //   edm::LogInfo("SiPixelFEDChannelContainerFromQualityConverter")<<"Found IOV:" << runLS.first  <<"("<<  runLS.second <<")"<<std::endl;
    
  //   auto theSiPixelFEDChannelCollection = this->createFromSiPixelQuality(*payload,*map_payload);

  //   if(removeEmptyPayloads_ && theSiPixelFEDChannelCollection.empty()) return;

  //   myQualities->setScenario(scenario,theSiPixelFEDChannelCollection);
    
  //   ++iiov;
  //   ++niov;
  // }
 
  // load all the IOVs of the cabling map once and for all outside the loop
  cond::persistency::IOVProxy iovProxyMap = condDbSession.readIov(m_CablingTagName, true);   

  int niov = 0;
  std::vector<std::tuple<cond::Time_t,cond::Hash> > m_iovs;
  condDbSession.getIovRange(m_QualityTagName,startIov,endIov, m_iovs);

  std::vector<std::tuple<cond::Time_t,cond::Hash> > m_cabling_iovs;
  condDbSession.getIovRange(m_CablingTagName,startIov,endIov, m_cabling_iovs);

  // create here the unpacked list of IOVs (run numbers)
  std::vector<unsigned int> listOfIOVs;
  std::vector<unsigned int> listOfCablingIOVs;
  std::transform(m_iovs.begin(), m_iovs.end(), std::back_inserter(listOfIOVs),
		 [](std::tuple<cond::Time_t,cond::Hash> myIOV) -> unsigned int { return SiPixelFEDChannelUtils::unpack(std::get<0>(myIOV)).first; });

  std::transform(m_cabling_iovs.begin(), m_cabling_iovs.end(), std::back_inserter(listOfCablingIOVs),
		 [](std::tuple<cond::Time_t,cond::Hash> myIOV) -> unsigned int { return std::get<0>(myIOV); });

  std::cout<<" listOfIOVs.size(): " << listOfIOVs.size() << " listOfCablingIOVs.size(): " << listOfCablingIOVs.size() << std::endl; 
  
  if(listOfCablingIOVs.size()>1){
    if( closest_from_below(listOfCablingIOVs,listOfIOVs.front()) != closest_from_above(listOfCablingIOVs,listOfIOVs.back()) ){
      throw cms::Exception("")<< " The Pixel FED Cabling map does not cover all the requested SiPixelQuality IOVs in the same interval of validity \n";
    }
  } else {
    if(listOfIOVs.front() < listOfCablingIOVs.front() ){
      throw cms::Exception("")<< " The Pixel FED Cabling map does not cover all the requested IOVs \n";
    }
  }

  std::cout<< " listOfIOVs.front() " << listOfIOVs.front() << " listOfIOVs.back() " << listOfIOVs.back() << std::endl;

  for(const auto& cb : listOfCablingIOVs ){
    std::cout<< " " << cb;
  }
  std::cout << std::endl;

  std::cout<< " closest_from_above(listOfCablingIOVs,listOfIOVs.front()): " << closest_from_above(listOfCablingIOVs,listOfIOVs.front()) << std::endl;
  std::cout<< " closest_from_below(listOfCablingIOVs,listOfIOVs.back()): " << closest_from_below(listOfCablingIOVs,listOfIOVs.back()) << std::endl;

  auto it = std::find(listOfCablingIOVs.begin(),listOfCablingIOVs.end(),closest_from_below(listOfCablingIOVs,listOfIOVs.front()));
  int index = std::distance(listOfCablingIOVs.begin(),it);  

  auto theCablingMapPayload = condDbSession.fetchPayload<SiPixelFedCablingMap>(std::get<1>(m_cabling_iovs.at(index)));
  auto theCablingTree = (*theCablingMapPayload).cablingTree();

  printf("Progressing Bar                               :0%%       20%%       40%%       60%%       80%%       100%%\n");
  printf("Translating into SiPixelFEDChannelCollection  :");
  int step = m_iovs.size()/50;

  for (const auto & myIOV : m_iovs){
    
    if(niov%step==0){printf(".");fflush(stdout);}
    auto payload = condDbSession.fetchPayload<SiPixelQuality>(std::get<1>(myIOV));
    auto runLS = SiPixelFEDChannelUtils::unpack(std::get<0>(myIOV));
    // print IOVs summary
    ss  << runLS.first << ","<< runLS.second << " ("<< std::get<0>(myIOV)  <<")" << " [hash: " << std::get<1>(myIOV)  << "] \n";
    
    auto iiovMap = iovProxyMap.find(runLS.first);
    if( iiovMap == iovProxyMap.end() ){
      std::cout <<"Could not find iov with since="<< runLS.first <<" in tag "<< m_CablingTagName <<std::endl;
    }

    //auto map_payload = condDbSession.fetchPayload<SiPixelFedCablingMap>( (*iiovMap).payloadId );
    //ss  << "cabling IOV: " << runLS.first << " ("<< (*iiovMap).since <<")" << " [hash: " << (*iiovMap).payloadId << "] \n" << std::endl;
    
    std::string scenario = std::to_string(runLS.first)+"_"+std::to_string(runLS.second);
  
    edm::LogInfo("SiPixelFEDChannelContainerFromQualityConverter")<<"Found IOV:" << runLS.first  <<"("<<  runLS.second <<")"<<std::endl;
    
    //auto theSiPixelFEDChannelCollection = this->createFromSiPixelQuality(*payload,*map_payload);
    auto theSiPixelFEDChannelCollection = this->createFromSiPixelQuality(*payload,*theCablingMapPayload,*theCablingTree);

    if(removeEmptyPayloads_ && theSiPixelFEDChannelCollection.empty()) return;

    myQualities->setScenario(scenario,theSiPixelFEDChannelCollection);
    
    //SiPixelFEDChannelContainer::SiPixelFEDChannelCollection theBadFEDChannels;
    //myQualities->setScenario(scenario,theBadFEDChannels);

    ++niov;
  }

  edm::LogInfo("FastSiPixelFEDChannelContainerFromQuality") << "[FastSiPixelFEDChannelContainerFromQuality::" << __func__ << "] "
				     << "Read " << niov << " IOVs from tag " << m_QualityTagName << " corresponding to the specified time interval.\n\n" << ss.str();

  if(printdebug_){
    edm::LogInfo("FastSiPixelFEDChannelContainerFromQuality") << "[FastSiPixelFEDChannelContainerFromQuality::" << __func__ << "] "
							      << ss.str();
  }

  condDbSession.transaction().commit();
  
  if (!m_output.empty()) {
    std::ofstream fout;
    fout.open(m_output);
    fout << ss.str();
    fout.close();
  }  
}

// ------------ method called once each job just before starting event loop  ------------
SiPixelFEDChannelContainer::SiPixelFEDChannelCollection
FastSiPixelFEDChannelContainerFromQuality::createFromSiPixelQuality(const SiPixelQuality & theQuality, const SiPixelFedCablingMap& theFedCabling,const SiPixelFedCablingTree& theCablingTree)
{
  auto fedid_ = theFedCabling.det2fedMap();

  SiPixelFEDChannelContainer::SiPixelFEDChannelCollection theBadChannelCollection;

  auto theDisabledModules = theQuality.getBadComponentList();
  for (const auto &mod : theDisabledModules){
    //mod.DetID, mod.errorType,mod.BadRocs
    
    int coded_badRocs = mod.BadRocs;
    std::vector<PixelFEDChannel> disabledChannelsDetSet;
    std::vector<sipixelobjects::CablingPathToDetUnit> path = theFedCabling.pathToDetUnit(mod.DetID);
    unsigned int nrocs_inLink(0);
    if (path.size() != 0){
      const sipixelobjects::PixelFEDCabling * aFed = theCablingTree.fed(path.at(0).fed);
      const sipixelobjects::PixelFEDLink    * link = aFed->link(path.at(0).link);
      nrocs_inLink = link->numberOfROCs();
    }	
    
    std::bitset<16> bad_rocs(coded_badRocs);	  
    unsigned int n_ch = bad_rocs.size()/nrocs_inLink;
	  
    for (unsigned int i_roc = 0; i_roc < n_ch; ++i_roc){
      
      unsigned int first_idx = nrocs_inLink*i_roc;
      unsigned int sec_idx   = nrocs_inLink*(i_roc+1)-1;
      unsigned int mask      = pow(2,nrocs_inLink)-1;
      unsigned int n_setbits = (coded_badRocs >> (i_roc*nrocs_inLink)) & mask;

      if (n_setbits==0){
	continue;
      }

      if(n_setbits != mask){
	if(printdebug_){      
	  edm::LogWarning("FastSiPixelFEDChannelContainerFromQuality")  << "Mismatch! DetId: "<< mod.DetID << " " << n_setbits <<  " " <<  mask << std::endl;
	}
	continue;
      }
	    
      if(printdebug_){      
	edm::LogVerbatim("FastSiPixelFEDChannelContainerFromQuality") << "passed" << std::endl;
      }

      unsigned int link_id = 99999;
      unsigned int fed_id = 99999;
	    
      for (auto const& p: path){
	const sipixelobjects::PixelFEDCabling * aFed = theCablingTree.fed(p.fed);
	const sipixelobjects::PixelFEDLink * link = aFed->link(p.link);
	const sipixelobjects::PixelROC * roc = link->roc(p.roc);
	unsigned int first_roc = roc->idInDetUnit();
	
	if (first_roc == first_idx){	
	  link_id = p.link;
	  fed_id = p.fed;
	  break; 
	}
      }

      if(printdebug_){
	edm::LogVerbatim("FastSiPixelFEDChannelContainerFromQuality") << " " << fed_id << " " << link_id << " " << first_idx << "  " << sec_idx << std::endl;
      }	 

      PixelFEDChannel ch = {fed_id, link_id, first_idx, sec_idx};
      disabledChannelsDetSet.push_back(ch);
      
      if(printdebug_){
	edm::LogVerbatim("FastSiPixelFEDChannelContainerFromQuality") << i_roc << " " << coded_badRocs <<  " " << first_idx << " " << sec_idx << std::endl;
	edm::LogVerbatim("FastSiPixelFEDChannelContainerFromQuality") << "=======================================" << std::endl;
      }
    }
    
    if (!disabledChannelsDetSet.empty()) {
      theBadChannelCollection[mod.DetID] = disabledChannelsDetSet;
    }
  }
  return theBadChannelCollection;
}


void
FastSiPixelFEDChannelContainerFromQuality::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

  edm::ParameterSetDescription desc;
  desc.setComment("Writes payloads of type SiPixelFEDChannelContainer");
  desc.addUntracked<bool>("printDebug",false);
  desc.addUntracked<bool>("removeEmptyPayloads",false);
  desc.add<std::string>("record","SiPixelStatusScenariosRcd");
  desc.add<std::string>("conditionDatabase","frontier://FrontierPrep/CMS_CONDITIONS");
  desc.add<std::string>("qualityTagName","SiPixelQualityOffline_2017_threshold1percent_stuckTBM");
  desc.add<std::string>("cablingMapTagName","SiPixelFedCablingMap_phase1_v7");
  desc.add<unsigned long long>("startIOV",1310841198608821);
  desc.add<unsigned long long>("endIOV",1312696624480350);
  desc.add<std::string>("output","summary.txt");
  desc.add<std::string>("connect","");

  edm::ParameterSetDescription descDBParameters;
  descDBParameters.addUntracked<std::string>("authenticationPath","");  
  descDBParameters.addUntracked<int>("authenticationSystem",0);  
  descDBParameters.addUntracked<std::string>("security","");  
  descDBParameters.addUntracked<int>("messageLevel",0);  

  desc.add<edm::ParameterSetDescription>("DBParameters",descDBParameters);
  descriptions.add("FastSiPixelFEDChannelContainerFromQuality", desc);
}


void 
FastSiPixelFEDChannelContainerFromQuality::endJob() {

  //  edm::LogInfo("FastSiPixelFEDChannelContainerFromQuality")<<"Analyzed "<<IOVcount_<<" IOVs"<<std::endl;
  edm::LogInfo("FastSiPixelFEDChannelContainerFromQuality")<<"Size of SiPixelFEDChannelContainer object "<< myQualities->size() <<std::endl<<std::endl;
  
  if(printdebug_){
    edm::LogInfo("FastSiPixelFEDChannelContainerFromQuality")<<"Content of SiPixelFEDChannelContainer "<<std::endl;
    
    // use built-in method in the CondFormat
     myQualities->printAll();
  }

  // Form the data here
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() ){
    cond::Time_t valid_time = poolDbService->currentTime(); 
    // this writes the payload to begin in current run defined in cfg
    if(!isMC_){
      poolDbService->writeOne(myQualities,valid_time, m_record);
    } else {
      // for MC IOV since=1
      poolDbService->writeOne(myQualities,1, m_record);
    }
  } 
}

DEFINE_FWK_MODULE(FastSiPixelFEDChannelContainerFromQuality);
