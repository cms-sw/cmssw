/** \file
 *
 *  implementation of RPCMonitorDigi class
 *
 *  $Date: 2006/06/17 09:09:52 $
 *  $Revision: 1.5 $
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

///RPCRecHits
#include <DataFormats/RPCRecHit/interface/RPCRecHitCollection.h>
#include <Geometry/Surface/interface/LocalError.h>
#include <Geometry/Vector/interface/LocalPoint.h>

///Log messages
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>


RPCMonitorDigi::RPCMonitorDigi( const edm::ParameterSet& pset ):counter(0){

  nameInLog = pset.getUntrackedParameter<std::string>("moduleLogName", "RPC_DQM");

  saveRootFile  = pset.getUntrackedParameter<bool>("DQMSaveRootFileDigi", false); 
  saveRootFileEventsInterval  = pset.getUntrackedParameter<int>("EventsIntervalForRootFileDigi", 10000); 
  RootFileName  = pset.getUntrackedParameter<std::string>("RootFileNameDigi", "RPCMonitorModuleDigi.root"); 
  
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

 char detUnitLabel[128];
 char layerLabel[128];
 char meId [128];

/// DIGI     

 edm::Handle<RPCDigiCollection> rpcdigis;
 iEvent.getByLabel("rpcunpacker", rpcdigis);

 RPCDigiCollection::DigiRangeIterator collectionItr;
 for(collectionItr=rpcdigis->begin(); collectionItr!=rpcdigis->end(); ++collectionItr){

 RPCDetId detId=(*collectionItr ).first; 
 uint32_t id=detId(); 


 sprintf(detUnitLabel ,"%d",detId());
 sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());
 
 std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
 if (meItr == meCollection.end() || (meCollection.size()==0)) {
 	meCollection[id]=bookDetUnitME(detId);
 }
 std::map<std::string, MonitorElement*> meMap=meCollection[id];
 	
 int roll = detId.roll();
 
 int numberOfDigi= 0;
	
	RPCDigiCollection::const_iterator digiItr; 
	for (digiItr = ((*collectionItr ).second).first;
		digiItr!=((*collectionItr).second).second; ++digiItr){
		
		int strip= (*digiItr).strip();
		int bx=(*digiItr).bx();
		//(*digiItr).print();
	        ++numberOfDigi;

		sprintf(meId,"Occupancy_%s",detUnitLabel);
		meMap[meId]->Fill(strip);

		sprintf(meId,"BXN_%s",detUnitLabel);
		meMap[meId]->Fill(bx);
	
	}/// loop on Digi

	sprintf(meId,"NumberOfDigi_%s",detUnitLabel);
	meMap[meId]->Fill(numberOfDigi);

        

 }/// loop on RPC Det Unit

 /// RPCRecHits/Clusters

//	sprintf(meId,"NumberOfClusters_%s",detUnitLabel);
//	meMap[meId]->Fill(clusterMultiplicities.size());

 edm::Handle<RPCRecHitCollection> rpcHits;
 iEvent.getByType(rpcHits);
 
 RPCRecHitCollection::const_iterator it;
 int numberOfHits=0;
     for (it = rpcHits->begin(); it != rpcHits->end(); it++)
     {
 
	RPCDetId detIdRecHits=it->rpcId();
	uint32_t idRecHits=detIdRecHits(); 
	LocalError error=it->localPositionError();//plot of errors/roll => should be gaussian	
	LocalPoint point=it->localPosition();	  //plot of coordinates/roll =>should be flat
	int mult=it->clusterSize();		  //cluster size plot => should be within 3-4	
	int firstStrip=it->firstClusterStrip();    //plot first Strip => should be flat
	float xposition=point.x();
	float yposition=point.y();
	
	
	
	std::cout<<"Position "<<xposition<<" "<<yposition <<std::endl;
	
	sprintf(detUnitLabel ,"%d",idRecHits);
	sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detIdRecHits.layer(),detIdRecHits.subsector(),detIdRecHits.roll());
 
	std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItrRecHits = meCollection.find(idRecHits);
	if (meItrRecHits == meCollection.end() || (meCollection.size()==0)) {
		meCollection[idRecHits]=bookDetUnitME(detIdRecHits);
	}
	std::map<std::string, MonitorElement*> meMapRecHits=meCollection[idRecHits];
 	
     
	sprintf(meId,"ClusterSize_%s",detUnitLabel);
			if(mult<=10) meMapRecHits[meId]->Fill(mult);
			if(mult>10)  meMapRecHits[meId]->Fill(11);
			
	numberOfHits++;
     }
	
	//sprintf(meId,"NumberOfClusters_%s",detUnitLabel);
	//meMap[meId]->Fill(clusterMultiplicities.size());
  

  if((!(counter%saveRootFileEventsInterval))&&(saveRootFile) ) {
    dbe->save(RootFileName);
  }
  
  
  //dbe->showDirStructure();
  //usleep(10000000);

}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCMonitorDigi)

 
 

