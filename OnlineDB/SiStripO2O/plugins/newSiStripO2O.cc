#include "OnlineDB/SiStripO2O/plugins/newSiStripO2O.h"
#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

newSiStripO2O::newSiStripO2O( const edm::ParameterSet& pset ):
  UsingDb_(pset.getUntrackedParameter<bool>("UsingDb",true))
{}

newSiStripO2O::~newSiStripO2O(){
  edm::LogInfo("newSiStripO2O") << "[newSiStripO2O::~newSiStripO2O]"
			     << " Destructing object...";
}
   
void newSiStripO2O::analyze(const edm::Event& evt, const edm::EventSetup& iSetup){
  
  unsigned int run=evt.id().run();
  std::cout << "RunNb " << run << std::endl;
  
  SiStripCondObjBuilderFromDb condObjBuilder;
  
  condObjBuilder.buildCondObj();


  SiStripFedCabling *cabling_cpy = condObjBuilder.getFedCabling();

  SiStripDetCabling* det_cabling=new SiStripDetCabling(*cabling_cpy);

  SiStripPedestals *ped_cpy = condObjBuilder.getPedestals();
  SiStripPedestals *ped = condObjBuilder.getPedestals();

  SiStripNoises *noise_cpy = condObjBuilder.getNoises();
  SiStripNoises *noise = condObjBuilder.getNoises();


  vector<uint32_t> det_ids;
  det_cabling->addActiveDetectorsRawIds(det_ids);
  if ( det_ids.empty() ) {
    edm::LogWarning("SiStripO2O")
      << "detids vetor empty";
  }  
  edm::LogInfo("SiStripO2O") << " Cabling Found " << det_ids.size() << " active DetIds";
  if (edm::isDebugEnabled()){
    // Iterate through active DetIds
    vector<uint32_t>::const_iterator det_id = det_ids.begin();
    for ( ; det_id != det_ids.end(); det_id++ ) {
      LogTrace("SiStripO2O") << " mySiStripO2O detid " << *det_id << std::endl;
    }    
  }

  det_ids.clear();
  SiStripDetCabling det_cabling_cpy( *cabling_cpy );
  det_cabling_cpy.addActiveDetectorsRawIds(det_ids);
  if ( det_ids.empty() ) {
    edm::LogWarning("SiStripO2O")
      << "detids vetor empty";
  }  
  edm::LogInfo("SiStripO2O") << " Cabling_cpy Found " << det_ids.size() << " active DetIds";
  if (edm::isDebugEnabled()){
    // Iterate through active DetIds
    vector<uint32_t>::const_iterator det_id = det_ids.begin();
    for ( ; det_id != det_ids.end(); det_id++ ) {
      LogTrace("SiStripO2O") << " cabling_cpy detid " << *det_id << std::endl;
    }    
  }

  //COPY NOISE
  

  std::vector<uint32_t> ndetid;
  noise->getDetIds(ndetid);
  edm::LogInfo("SiStripO2O") << " Noise Found " << ndetid.size() << " DetIds";
  for (size_t id=0;id<ndetid.size();id++){
    SiStripNoises::Range range=noise->getRange(ndetid[id]);

    if (edm::isDebugEnabled()){
      int strip=0;
      LogTrace("SiStripO2O")  << "NOISE detid " << ndetid[id] << " \t"
			      << " strip " << strip << " \t"
			      << noise->getNoise(strip,range)     << " \t" 
	//<< noise->getDisable(strip,range)   << " \t" 
			      << std::endl; 	    
    } 
  }

  //COPY PED
  std::vector<uint32_t> pdetid;
  ped->getDetIds(pdetid);
  edm::LogInfo("SiStripO2O") << " Peds Found " << pdetid.size() << " DetIds";
  for (size_t id=0;id<pdetid.size();id++){
    SiStripPedestals::Range range=ped->getRange(pdetid[id]);
    if (edm::isDebugEnabled()){
      int strip=0;
      LogTrace("SiStripO2O")  << "PED detid " << pdetid[id] << " \t"
			      << " strip " << strip << " \t"
			      << ped->getPed   (strip,range)   << " \t" 
			      << std::endl; 	    
    } 
  }  

  //End now write data in DB
  if(UsingDb_){
    edm::LogInfo("SiStripO2O") << "calling PoolDBOutputService" << std::endl;
    edm::Service<cond::service::PoolDBOutputService> mydbservice;

    if( mydbservice.isAvailable() ){

      if( mydbservice->isNewTagRequest("SiStripPedestalsRcd") ){
	edm::LogInfo("SiStripO2O") << "new tag requested for SiStripPedestalsRcd" << std::endl;
	mydbservice->createNewIOV<SiStripPedestals>(ped_cpy,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripPedestalsRcd");      
      } else {
	edm::LogInfo("SiStripO2O") << "append to existing tag for SiStripPedestalsRcd" << std::endl;
	mydbservice->appendSinceTime<SiStripPedestals>(ped_cpy,mydbservice->currentTime(),"SiStripPedestalsRcd");
      }
      
      if( mydbservice->isNewTagRequest("SiStripNoisesRcd") ){
	edm::LogInfo("SiStripO2O") << "new tag requested for SiStripNoisesRcd" << std::endl;
	mydbservice->createNewIOV<SiStripNoises>(noise_cpy,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripNoisesRcd");      
      } else {
	edm::LogInfo("SiStripO2O") << "append to existing tag for SiStripNoisesRcd" << std::endl;
	mydbservice->appendSinceTime<SiStripNoises>(noise_cpy,mydbservice->currentTime(),"SiStripNoisesRcd");      
      }
     
      if( mydbservice->isNewTagRequest("SiStripFedCablingRcd") ){
	edm::LogInfo("SiStripO2O") << "new tag requested for SiStripFedCablingRcd" << std::endl;
	mydbservice->createNewIOV<SiStripFedCabling>(cabling_cpy,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripFedCablingRcd");      
      } else {
	edm::LogInfo("SiStripO2O") << "append to existing tag for SiStripFedCablingRcd" << std::endl;
	mydbservice->appendSinceTime<SiStripFedCabling>(cabling_cpy,mydbservice->currentTime(),"SiStripFedCablingRcd"); 
      }

      if( mydbservice->isNewTagRequest("SiStripThresholdRcd") ){
	edm::LogInfo("SiStripO2O") << "new tag requested for SiStripThresholdRcd" << std::endl;
	mydbservice->createNewIOV<SiStripThreshold>(condObjBuilder.getThreshold(),mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripThresholdRcd");      
      } else {
	edm::LogInfo("SiStripO2O") << "append to existing tag for SiStripThresholdRcd" << std::endl;
	mydbservice->appendSinceTime<SiStripThreshold>(condObjBuilder.getThreshold(),mydbservice->currentTime(),"SiStripThresholdRcd"); 
      }

      SiStripQuality qobj=SiStripQuality(*(condObjBuilder.getQuality()));
      SiStripBadStrip* obj= new SiStripBadStrip(qobj); 
      if( mydbservice->isNewTagRequest("SiStripBadStripRcd") ){
	edm::LogInfo("SiStripO2O") << "new tag requested for SiStripBadStripRcd" << std::endl;
	mydbservice->createNewIOV<SiStripBadStrip>(obj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripBadStripRcd");      
      } else {
	edm::LogInfo("SiStripO2O") << "append to existing tag for SiStripBadStripRcd" << std::endl;
	mydbservice->appendSinceTime<SiStripBadStrip>(obj,mydbservice->currentTime(),"SiStripBadStripRcd"); 
      }

      //edm::LogInfo("SiStripO2O")  << " finished to upload data " << std::endl;    
    }else{
      edm::LogError("newSiStripO2O")<<"Service is unavailable"<<std::endl;
    }    
  }
}

