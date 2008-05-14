#include "CondTools/SiStrip/plugins/SiStripThresholdBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <iostream>
#include <fstream>

SiStripThresholdBuilder::SiStripThresholdBuilder( const edm::ParameterSet& iConfig ):
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",3)){}


void SiStripThresholdBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup){


  unsigned int run=evt.id().run();

  edm::LogInfo("SiStripThresholdBuilder") << "... creating dummy SiStripThreshold Data for Run " << run << "\n " << std::endl;

  SiStripThreshold* obj = new SiStripThreshold();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();

  int count=-1;
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    count++;
    //Generate Pedestal for det detid
    SiStripThreshold::Container theSiStripVector;   
    uint16_t strip=0;
    while(strip<128*it->second.nApvs){
      
      float lTh = (RandFlat::shoot(1.) * 64)/5;
      float hTh = (RandFlat::shoot(1.) * 64)/5;
      if (hTh < lTh){
	float tmp = hTh;
	hTh = lTh;
	lTh = tmp;
      }
      
      obj->setData(strip,lTh,hTh,theSiStripVector);
      if (count<(int)printdebug_)
	edm::LogInfo("SiStripThresholdBuilder") <<"detid: "  << it->first << " \t"
						<< "firstStrip: " << strip << " \t" << theSiStripVector.back().getFirstStrip() << " \t"
						<< "lTh: " << lTh       << " \t" << theSiStripVector.back().getLth() << " \t"
						<< "hTh: " << hTh       << " \t" << theSiStripVector.back().getHth() << " \t"
						<< "FirstStrip_and_Hth: " << theSiStripVector.back().FirstStrip_and_Hth << " \t"
						<< std::endl; 	    
      obj->setData(strip+1,lTh,hTh,theSiStripVector);
      strip=(uint16_t) (RandFlat::shoot(strip+2,128*it->second.nApvs));
    }      
    if ( ! obj->put(it->first,theSiStripVector) )
      edm::LogError("SiStripThresholdBuilder")<<"[SiStripThresholdBuilder::analyze] detid already exists"<<std::endl;
  }
  

  //End now write sistrippedestals data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  if( mydbservice.isAvailable() ){
    if ( mydbservice->isNewTagRequest("SiStripThresholdRcd") ){
      mydbservice->createNewIOV<SiStripThreshold>(obj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripThresholdRcd");
    } else {
      mydbservice->appendSinceTime<SiStripThreshold>(obj,mydbservice->currentTime(),"SiStripThresholdRcd");      
    }
  }else{
    edm::LogError("SiStripThresholdBuilder")<<"Service is unavailable"<<std::endl;
  }
}
     
