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

RPCDqmClient::RPCDqmClient(const edm::ParameterSet& iConfig){

  edm::LogVerbatim ("rpcdqmclient") << "[RPCDqmClient]: Constructor";

  parameters_ = iConfig;

  offlineDQM_ = parameters_.getUntrackedParameter<bool> ("OfflineDQM",true); 
  useRollInfo_=  parameters_.getUntrackedParameter<bool>("UseRollInfo", false);
  //check enabling
  enableDQMClients_ =parameters_.getUntrackedParameter<bool> ("EnableRPCDqmClient",true); 
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
  this->makeClientMap();
}

RPCDqmClient::~RPCDqmClient(){dbe_ = 0;}

void RPCDqmClient::beginJob(){

  edm::LogVerbatim ("rpcdqmclient") << "[RPCDqmClient]: Begin Job";
  if (!enableDQMClients_) return;                 ;

  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  

  //Do whatever the begin jobs of all client modules do
  for(std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ )
    (*it)->beginJob(dbe_, globalFolder_);
  
}


void  RPCDqmClient::beginRun(const edm::Run& r, const edm::EventSetup& c){

  if (!enableDQMClients_) return;

  
  for ( std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
    (*it)->beginRun(r,c);
  }

  if(!offlineDQM_) this->getMonitorElements(r, c);
  
  lumiCounter_ = prescaleGlobalFactor_;
  init_ = false;
}



void  RPCDqmClient::endRun(const edm::Run& r, const edm::EventSetup& c){
  edm::LogVerbatim ("rpcdqmclient") << "[RPCDqmClient]: End Run";

  if (!enableDQMClients_) return;

  if(offlineDQM_) this->getMonitorElements(r, c);

  float   rpcevents = minimumEvents_;
  if(RPCEvents_) rpcevents = RPCEvents_ ->getBinContent(1);
  
  if(rpcevents < minimumEvents_) return;
  
  for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
    (*it)->clientOperation(c);
    (*it)->endRun(r,c);
  }
}


void  RPCDqmClient::getMonitorElements(const edm::Run& r, const edm::EventSetup& c){
 
  std::vector<MonitorElement *>  myMeVect;
  std::vector<RPCDetId>   myDetIds;
   
  edm::ESHandle<RPCGeometry> rpcGeo;
  c.get<MuonGeometryRecord>().get(rpcGeo);
  
  //dbe_->setCurrentFolder(prefixDir_);
  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
  MonitorElement * myMe = NULL;
  std::string rollName= "";
  //  std::set<int> disk_set, ring_set;

  //loop on all geometry and get all histos
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
       
       RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
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
	   myMe = dbe_->get(prefixDir_ +"/"+ folderStr->folderStructure(detId)+"/"+clientHisto_[cl]+ "_"+rollName); 
	   
	   if (!myMe)continue;
	   
	   dbe_->tag(myMe, clientTag_[cl]);
	   myMeVect.push_back(myMe);
	   myDetIds.push_back(detId);
	   
	 }//end loop on clients
	 
       }//end loop on roll in given chamber
    }
  }//end loop on all geometry and get all histos  
  
  
  RPCEvents_ = dbe_->get(prefixDir_ +"/RPCEvents");  

  for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
    (*it)->getMonitorElements(myMeVect, myDetIds);
  }

  delete folderStr;
 
}
 


void RPCDqmClient::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) {
  if (!enableDQMClients_) return;

  for ( std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ )
    (*it)->beginLuminosityBlock(lumiSeg,context);
}

void RPCDqmClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

 if (!enableDQMClients_) return;

 for ( std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ )
    (*it)->analyze( iEvent,iSetup);
}


void RPCDqmClient::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c){
 
  if (!enableDQMClients_ ) return;

  if(offlineDQM_) return;

  edm::LogVerbatim ("rpcdqmclient") <<"[RPCDqmClient]: End of LS ";

  for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ )
    (*it)->endLuminosityBlock( lumiSeg, c);
  
  float   rpcevents = minimumEvents_;
  if(RPCEvents_) rpcevents = RPCEvents_ ->getBinContent(1);
  
  if( rpcevents < minimumEvents_) return;

  if( !init_ ){

    for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
      (*it)->clientOperation(c);
    }
    init_ = true;
    return;
  }

  lumiCounter_++;

  if (lumiCounter_%prescaleGlobalFactor_ != 0) return;


  for (std::vector<RPCClient*>::iterator it = clientModules_.begin(); it!=clientModules_.end(); it++ ){
    (*it)->clientOperation(c);
  }

}



void RPCDqmClient::endJob() {
  if (!enableDQMClients_) return;
  
  for ( std::vector<RPCClient*>::iterator it= clientModules_.begin(); it!=clientModules_.end(); it++ )
    (*it)->endJob();
}


void RPCDqmClient::makeClientMap() {

  for(unsigned int i = 0; i<clientList_.size(); i++){
    
    if( clientList_[i] == "RPCMultiplicityTest" ) {
      clientHisto_.push_back("Multiplicity");
      clientTag_.push_back(rpcdqm::MULTIPLICITY);
      clientModules_.push_back( new RPCMultiplicityTest(parameters_));
    } else if ( clientList_[i] == "RPCDeadChannelTest" ){
      clientHisto_.push_back("Occupancy");
      clientModules_.push_back( new RPCDeadChannelTest(parameters_));
      clientTag_.push_back(rpcdqm::OCCUPANCY);
    } else if ( clientList_[i] == "RPCClusterSizeTest" ){
      clientHisto_.push_back("ClusterSize");
      clientModules_.push_back( new RPCClusterSizeTest(parameters_));
      clientTag_.push_back(rpcdqm::CLUSTERSIZE);
    } else if ( clientList_[i] == "RPCOccupancyTest" ){
      clientHisto_.push_back("Occupancy");
      clientModules_.push_back( new RPCOccupancyTest(parameters_));
      clientTag_.push_back(rpcdqm::OCCUPANCY);
    } else if ( clientList_[i] == "RPCNoisyStripTest" ){
      clientHisto_.push_back("Occupancy");
      clientModules_.push_back( new RPCNoisyStripTest(parameters_));
      clientTag_.push_back(rpcdqm::OCCUPANCY);
    }
  }

  return;

}
