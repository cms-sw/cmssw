/** \file
 *
 *  implementation of RPCMonitorDigi class
 *
 *  $Date: 2006/02/10 10:42:57 $
 *  $Revision: 1.4 $
 *
 * \author Ilaria Segoni
 */

#include <map>
#include <string>

#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>

///Data Format
#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>

///Digi Cluster
#include <DQM/RPCMonitorDigi/interface/RPCClusterHandle.h>


///Log messages
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


RPCMonitorDigi::RPCMonitorDigi( const edm::ParameterSet& pset ):counter(0){

  nameInLog = pset.getUntrackedParameter<std::string>("moduleLogName", "RPC_DQM");
  nameInLog = pset.getUntrackedParameter<std::string>("moduleLogName", "RPC_DQM");

  /// get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  dbe->showDirStructure();


}


RPCMonitorDigi::~RPCMonitorDigi(){
}

void RPCMonitorDigi::endJob(void)
{
  dbe->save("test.root");  
}


void RPCMonitorDigi::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup ){
 counter++;
 edm::LogInfo (nameInLog) <<"Beginning analyzing event = " << counter;

 edm::Handle<RPCDigiCollection> rpcdigis;
 iEvent.getByLabel("rpcunpacker", rpcdigis);

 RPCDigiCollection::DigiRangeIterator collectionItr;
 for(collectionItr=rpcdigis->begin(); collectionItr!=rpcdigis->end(); ++collectionItr){

 RPCDetId detId=(*collectionItr ).first;
 
 uint32_t id=detId();
 
  /// ME's name components common to current RPDDetId  
 char detUnitLabel[128];
 char layerLabel[128];
 sprintf(detUnitLabel ,"%d",detId());
 sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());

 char meId [128];
 char meTitle [128];

 edm::LogInfo (nameInLog) <<"For DetId = "<<id<<" components: "<<(*collectionItr ).first;
 
 std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
 if (meItr == meCollection.end() || (meCollection.size()==0)) {
 	meCollection[id]=bookDetUnitME(detId);
 }
 std::map<std::string, MonitorElement*> meMap=meCollection[id];
 	
 int roll = detId.roll();
 RPCClusterHandle clusterFinder(nameInLog);
 clusterFinder.reset();
 
 int numberOfDigi= 0;
 edm::LogInfo (nameInLog) <<"For roll = "<< roll;
	
	RPCDigiCollection::const_iterator digiItr; 
	for (digiItr = ((*collectionItr ).second).first;
		digiItr!=((*collectionItr).second).second; ++digiItr){
		
		int strip= (*digiItr).strip();
		int bx=(*digiItr).bx();
		//(*digiItr).print();
		clusterFinder.addStrip(strip);
	        ++numberOfDigi;

		sprintf(meId,"Oppupancy_%s",detUnitLabel);
		sprintf(meTitle,"Occupancy_for_%s",layerLabel);
		meMap[meId]->Fill(strip);

		sprintf(meId,"BXN_%s",detUnitLabel);
		sprintf(meTitle,"BXN_for_%s",layerLabel);
		meMap[meId]->Fill(bx);
	
	}/// loop on Digi

	sprintf(meId,"NumberOfDigi_%s",detUnitLabel);
	sprintf(meTitle,"NumberOfDigi_or_%s",layerLabel);
	meMap[meId]->Fill(numberOfDigi);

	/// CLUSTERS
 	std::vector<int> clusterMultiplicities = clusterFinder.findClustersFromStrip();
        
	//edm::LogInfo (nameInLog) <<"Number Of Clusters :"<<clusterMultiplicities.size();
	sprintf(meId,"NumberOfClusters_%s",detUnitLabel);
	sprintf(meTitle,"NumberOfClusters_or_%s",layerLabel);
	meMap[meId]->Fill(clusterMultiplicities.size());

	for(std::vector<int>::iterator mult = clusterMultiplicities.begin(); 
	       mult != clusterMultiplicities.end(); ++mult ){
        	edm::LogInfo (nameInLog) <<"Cluster size:"<<*mult;
			sprintf(meId,"ClusterSize_%s",detUnitLabel);
			sprintf(meTitle,"BXN_for_%s",layerLabel);
			if(*mult<=10) meMap[meId]->Fill(*mult);
			if(*mult>10) meMap[meId]->Fill(11);
	}
	
	//edm::LogInfo (nameInLog) <<"Number Of Digi :"<<numberOfDigi;
 



	dbe->showDirStructure();

 
 }/// loop on RPC Det Unit

      
  
 usleep(1000000);


}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCMonitorDigi)

 
 

