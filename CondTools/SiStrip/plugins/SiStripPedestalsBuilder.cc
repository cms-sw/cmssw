#include "CondTools/SiStrip/plugins/SiStripPedestalsBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <iostream>
#include <fstream>

SiStripPedestalsBuilder::SiStripPedestalsBuilder( const edm::ParameterSet& iConfig ):
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}


void SiStripPedestalsBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup){

  unsigned int run=evt.id().run();

  edm::LogInfo("SiStripPedestalsBuilder") << "... creating dummy SiStripPedestals Data for Run " << run << "\n " << std::endl;

  SiStripPedestals* obj = new SiStripPedestals();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();

  int count=-1;
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    count++;
    //Generate Pedestal for det detid
    std::vector<char> theSiStripVector;   
    for(int strip=0; strip<128*it->second.nApvs;++strip){
	
      float MeanPed   = 100;
      float RmsPed    = 5;
	
      float ped = RandGauss::shoot(MeanPed,RmsPed);
      float lTh = RandFlat::shoot(1.) * 12.6;
      float hTh = RandFlat::shoot(1.) * 12.6;
      if (hTh < lTh){
	float tmp = hTh;
	hTh = lTh;
	lTh = tmp;
      }
	  
      if ( count<printdebug_)
	edm::LogInfo("SiStripPedestalsBuilder") << "detid "  << it->first << " \t"
						<< " strip " << strip << " \t"
						<< ped       << " \t" 
						<< lTh       << " \t" 
						<< hTh       << " \t" 
						<< std::endl; 	    
      obj->setData(ped,lTh,hTh,theSiStripVector);
    }
      
    SiStripPedestals::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(it->first,range) )
      edm::LogError("SiStripPedestalsBuilder")<<"[SiStripPedestalsBuilder::analyze] detid already exists"<<std::endl;
  }
  

  //End now write sistrippedestals data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if( mydbservice.isAvailable() ){
    if ( mydbservice->isNewTagRequest("SiStripPedestalsRcd") ){
      mydbservice->createNewIOV<SiStripPedestals>(obj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripPedestalsRcd");
    } else {
      //mydbservice->createNewIOV<SiStripPedestals>(obj,mydbservice->currentTime(),"SiStripPedestalsRcd");      
      mydbservice->appendSinceTime<SiStripPedestals>(obj,mydbservice->currentTime(),"SiStripPedestalsRcd");      
    }
  }else{
    edm::LogError("SiStripPedestalsBuilder")<<"Service is unavailable"<<std::endl;
  }
}
     
