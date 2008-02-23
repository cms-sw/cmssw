#include "CondTools/SiStrip/plugins/SiStripNoisesBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <iostream>
#include <fstream>

SiStripNoisesBuilder::SiStripNoisesBuilder( const edm::ParameterSet& iConfig ):
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}


void SiStripNoisesBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup){

  unsigned int run=evt.id().run();

  edm::LogInfo("SiStripNoisesBuilder") << "... creating dummy SiStripNoises Data for Run " << run << "\n " << std::endl;

  SiStripNoises* obj = new SiStripNoises();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();

  int count=-1;
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    count++;
    //Generate Noise for det detid
    std::vector<short> theSiStripVector;
    for(int strip=0; strip<128*it->second.nApvs; ++strip){

      float MeanNoise = 5;
      float RmsNoise  = 1;
      double badStripProb = .5;

      float noise =  RandGauss::shoot(MeanNoise,RmsNoise);
      bool disable = (RandFlat::shoot(1.) < badStripProb ? true:false);
	
      if (count<printdebug_)
	edm::LogInfo("SiStripNoisesBuilder") << "detid " << it->first << " \t"
					     << " strip " << strip << " \t"
					     << noise     << " \t" 
					     << disable   << " \t" 
					     << std::endl; 	    
      obj->setData(noise,disable,theSiStripVector);
    }    
      
    SiStripNoises::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(it->first,range) )
      edm::LogError("SiStripNoisesBuilder")<<"[SiStripNoisesBuilder::analyze] detid already exists"<<std::endl;
  }


  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if( mydbservice.isAvailable() ){
    if ( mydbservice->isNewTagRequest("SiStripNoisesRcd") ){
      mydbservice->createNewIOV<SiStripNoises>(obj,mydbservice->endOfTime(),"SiStripNoisesRcd");
    } else {  
      mydbservice->createNewIOV<SiStripNoises>(obj,mydbservice->currentTime(),"SiStripNoisesRcd");      
    }
  }else{
    edm::LogError("SiStripNoisesBuilder")<<"Service is unavailable"<<std::endl;
  }
}
     
