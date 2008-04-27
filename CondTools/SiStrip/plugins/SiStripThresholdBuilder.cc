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
  unsigned short NconsecutiveValueTh=1;
  unsigned int encodeValueTh;
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    count++;
    //Generate Pedestal for det detid
    std::vector<unsigned int> theSiStripVector;   
    for(int strip=0; strip<128*it->second.nApvs;++strip){
	
      float lTh = (RandFlat::shoot(1.) * 64)/5;
      float hTh = (RandFlat::shoot(1.) * 64)/5;
      if (hTh < lTh){
	float tmp = hTh;
	hTh = lTh;
	lTh = tmp;
      }
	  
	encodeValueTh=obj->encode(strip,NconsecutiveValueTh,lTh,hTh);
      if (count<printdebug_)
	edm::LogInfo("SiStripThresholdBuilder") <<"detid: "  << it->first << " \t"
						<< "firstStrip: " << strip << " \t"
						<< "NumConsecutiveStrip: " << NconsecutiveValueTh << " \t"
						<< "lTh: " << lTh       << " \t" 
						<< "hTh: " << hTh       << " \t" 
						<< "packed integer: " << std::hex << encodeValueTh << std::dec
						<< std::endl; 	    
	  theSiStripVector.push_back(encodeValueTh);
    }
      
    SiStripThreshold::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(it->first,range) )
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
     
