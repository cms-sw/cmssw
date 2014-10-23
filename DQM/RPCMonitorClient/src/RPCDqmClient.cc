// Package:    RPCDqmClient
// Original Author:  Anna Cimmino

#include "DQM/RPCMonitorClient/interface/RPCDqmClient.h"
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"
//include client headers
#include  "DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h"
#include "DQM/RPCMonitorClient/interface/RPCMultiplicityTest.h"
#include "DQM/RPCMonitorClient/interface/RPCClusterSizeTest.h"
#include "DQM/RPCMonitorClient/interface/RPCOccupancyTest.h"
#include "DQM/RPCMonitorClient/interface/RPCNoisyStripTest.h"
//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
//Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <FWCore/Framework/interface/ESHandle.h>

RPCDqmClient::RPCDqmClient(const edm::ParameterSet& parameters_){

  edm::LogVerbatim ("rpcdqmclient") << "[RPCDqmClient]: Constructor";

  //  parameters_ = iConfig;

  offlineDQM_ = parameters_.getUntrackedParameter<bool> ("OfflineDQM",true); 
  useRollInfo_=  parameters_.getUntrackedParameter<bool>("UseRollInfo", false);
  //check enabling
  enableDQMClients_ = parameters_.getUntrackedParameter<bool> ("EnableRPCDqmClient",true); 
  minimumEvents_= parameters_.getUntrackedParameter<int>("MinimumRPCEvents", 10000);

  std::string subsystemFolder = parameters_.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  std::string recHitTypeFolder= parameters_.getUntrackedParameter<std::string>("RecHitTypeFolder", "AllHits");
  std::string summaryFolder = parameters_.getUntrackedParameter<std::string>("SummaryFolder", "SummaryHistograms");
  
  prefixDir_ =   subsystemFolder+ "/"+ recHitTypeFolder;
  globalFolder_ = subsystemFolder + "/"+ recHitTypeFolder + "/"+ summaryFolder;

  //get prescale factor
  prescaleGlobalFactor_ = parameters_.getUntrackedParameter<int>("DiagnosticGlobalPrescale", 5);


  //make default client list  
  clientList_.push_back("RPCMultiplicityTest");
  clientList_.push_back("RPCDeadChannelTest");
  clientList_.push_back("RPCClusterSizeTest");
  clientList_= parameters_.getUntrackedParameter<std::vector<std::string> >("RPCDqmClientList",clientList_);


  //get all the possible RPC DQM clients 
  this->makeClientMap(parameters_);

  lumiCounter_ = prescaleGlobalFactor_;
  init_ = false;

}

RPCDqmClient::~RPCDqmClient(){}

void RPCDqmClient::beginJob(){

  if (!enableDQMClients_) {return;}                 ;
  edm::LogVerbatim ("rpcdqmclient") << "[RPCDqmClient]: Begin Job";

    //Do whatever the begin jobs of all client modules do
  for(std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
    (*it)->beginJob( globalFolder_ );
  }
}



void RPCDqmClient::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, edm::LuminosityBlock const  & lumiSeg, edm::EventSetup const& c){
 
  if (!enableDQMClients_ ) {return;}
  edm::LogVerbatim ("rpcdqmclient") << "[RPCDqmClient]: End DQM LB";
  
  
  if(!init_) {  //At the end of the first LumiBlock...
    // ...get chamber based histograms 
    this->getMonitorElements(igetter, c);

    //...book summary histograms
    for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
      (*it)->myBooker( ibooker);
    }

    lumiCounter_ = prescaleGlobalFactor_  - 1;
    init_= true;
  }

  lumiCounter_++;
  
  if (offlineDQM_) {return;} // If in offlineDQM mode, do nothing. Client operation will be done only at endJob
  
  //Check if there's enough statistics
  float   rpcevents = minimumEvents_;
  if(RPCEvents_) {rpcevents = RPCEvents_->getBinContent(1);}
  if( rpcevents < minimumEvents_) {return;}
  //Do not perform client oparations every lumi block  
  if (lumiCounter_%prescaleGlobalFactor_ != 0) {return;}

  edm::LogVerbatim ("rpcdqmclient") <<"[RPCDqmClient]: Client operations";
  for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
    (*it)->clientOperation();
  }

}



void  RPCDqmClient::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter){

  if (!enableDQMClients_ || !init_) {return;}

  edm::LogVerbatim ("rpcdqmclient") << "[RPCDqmClient]: End DQM Job";
  
  float   rpcevents = minimumEvents_;
  if(RPCEvents_) {rpcevents = RPCEvents_ ->getBinContent(1);}
  if(rpcevents < minimumEvents_) {return;}

  edm::LogVerbatim ("rpcdqmclient") <<"[RPCDqmClient]: Client operations";
  for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
    (*it)->clientOperation();
  }
  
}



void  RPCDqmClient::getMonitorElements(DQMStore::IGetter & igetter, const edm::EventSetup& c){
 
  std::vector<MonitorElement *>  myMeVect;
  std::vector<RPCDetId>   myDetIds;
   
  edm::ESHandle<RPCGeometry> rpcGeo;
  c.get<MuonGeometryRecord>().get(rpcGeo);
  
  //dbe_->setCurrentFolder(prefixDir_);
  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
  MonitorElement * myMe = NULL;
  std::string rollName= "";

  //loop on all geometry and get all histos
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< const RPCChamber* >( *it ) != 0 ){
       
       const RPCChamber* ch = dynamic_cast< const RPCChamber* >( *it ); 
       std::vector< const RPCRoll*> roles = (ch->rolls());
       
       //Loop on rolls in given chamber
       for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	 
	 RPCDetId detId = (*r)->id();
	 
	 //Get name
	 RPCGeomServ RPCname(detId);	   
	 rollName= "";
	 if(useRollInfo_) {
	   rollName =  RPCname.name();
	 }else{
	   rollName =   RPCname.chambername();
	 }

	 //loop on clients
	 for( unsigned int cl = 0; cl<clientModules_.size(); cl++ ){
	   
	   myMe = NULL;
	   myMe = igetter.get(prefixDir_ +"/"+ folderStr->folderStructure(detId)+"/"+clientHisto_[cl]+ "_"+rollName); 
	   
	   if (!myMe)continue;
	   //	   dbe_->tag(myMe, clientTag_[cl]);
	   myMeVect.push_back(myMe);
	   myDetIds.push_back(detId);
	   
	 }//end loop on clients
       }//end loop on roll in given chamber
    }
  }//end loop on all geometry and get all histos  
  
  
  RPCEvents_ = igetter.get(prefixDir_ +"/RPCEvents");  
  unsigned int cl = 0;
  for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
    (*it)->getMonitorElements(myMeVect, myDetIds, clientHisto_[cl]);
    cl++;
  }

  delete folderStr;
}
 




void RPCDqmClient::makeClientMap(const edm::ParameterSet& parameters_) {

  for(unsigned int i = 0; i<clientList_.size(); i++){
    
    if( clientList_[i] == "RPCMultiplicityTest" ) {
      clientHisto_.push_back("Multiplicity");
      // clientTag_.push_back(rpcdqm::MULTIPLICITY);
      clientModules_.push_back( new RPCMultiplicityTest(parameters_));
    } else if ( clientList_[i] == "RPCDeadChannelTest" ){
      clientHisto_.push_back("Occupancy");
      clientModules_.push_back( new RPCDeadChannelTest(parameters_));
      // clientTag_.push_back(rpcdqm::OCCUPANCY);
    } else if ( clientList_[i] == "RPCClusterSizeTest" ){
      clientHisto_.push_back("ClusterSize");
      clientModules_.push_back( new RPCClusterSizeTest(parameters_));
      // clientTag_.push_back(rpcdqm::CLUSTERSIZE);
    } else if ( clientList_[i] == "RPCOccupancyTest" ){
      clientHisto_.push_back("Occupancy");
      clientModules_.push_back( new RPCOccupancyTest(parameters_));
      // clientTag_.push_back(rpcdqm::OCCUPANCY);
    } else if ( clientList_[i] == "RPCNoisyStripTest" ){
      clientHisto_.push_back("Occupancy");
      clientModules_.push_back( new RPCNoisyStripTest(parameters_));
      //clientTag_.push_back(rpcdqm::OCCUPANCY);
    }
  }

  return;

}
