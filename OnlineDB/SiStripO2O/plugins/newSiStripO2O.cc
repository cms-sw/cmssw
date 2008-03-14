#include "OnlineDB/SiStripO2O/plugins/newSiStripO2O.h"

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

  //getting fed cabling from onlinedb through ES
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get( cabling );
  SiStripFedCabling *cabling_cpy = new SiStripFedCabling(*cabling);

  edm::ESHandle<SiStripDetCabling> det_cabling;
  iSetup.get<SiStripDetCablingRcd>().get( det_cabling );

  edm::ESHandle<SiStripPedestals> ped;
  iSetup.get<SiStripPedestalsRcd>().get(ped);
  SiStripPedestals *ped_cpy = new SiStripPedestals();

  edm::ESHandle<SiStripNoises> noise;
  iSetup.get<SiStripNoisesRcd>().get(noise);
  SiStripNoises *noise_cpy = new SiStripNoises();


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
    noise_cpy->put(ndetid[id],range);      

    if (edm::isDebugEnabled()){
      int strip=0;
      LogTrace("SiStripO2O")  << "NOISE detid " << ndetid[id] << " \t"
			      << " strip " << strip << " \t"
			      << noise->getNoise(strip,range)     << " \t" 
			      << noise->getDisable(strip,range)   << " \t" 
			      << std::endl; 	    
    } 
  }

  //COPY PED
  std::vector<uint32_t> pdetid;
  ped->getDetIds(pdetid);
  edm::LogInfo("SiStripO2O") << " Peds Found " << pdetid.size() << " DetIds";
  for (size_t id=0;id<pdetid.size();id++){
    SiStripPedestals::Range range=ped->getRange(pdetid[id]);
    ped_cpy->put(pdetid[id],range);

    if (edm::isDebugEnabled()){
      int strip=0;
      LogTrace("SiStripO2O")  << "PED detid " << pdetid[id] << " \t"
			      << " strip " << strip << " \t"
			      << ped->getPed   (strip,range)   << " \t" 
			      << ped->getLowTh (strip,range)   << " \t" 
			      << ped->getHighTh(strip,range)   << " \t" 
			      << std::endl; 	    
    } 
  }  
  
  //End now write data in DB
  if(UsingDb_){
    edm::LogInfo("SiStripO2O") << "calling PoolDBOutputService" << std::endl;
    edm::Service<cond::service::PoolDBOutputService> mydbservice;

    if( mydbservice.isAvailable() ){
      try{

	if( mydbservice->isNewTagRequest("SiStripPedestalsRcd") ){
	  edm::LogInfo("SiStripO2O") << "new tag requested for SiStripPedestalsRcd" << std::endl;
	  mydbservice->createNewIOV<SiStripPedestals>(ped_cpy,mydbservice->endOfTime(),"SiStripPedestalsRcd");      
	} else {
	  edm::LogInfo("SiStripO2O") << "append to existing tag for SiStripPedestalsRcd" << std::endl;
	  mydbservice->appendSinceTime<SiStripPedestals>(ped_cpy,mydbservice->currentTime(),"SiStripPedestalsRcd");
	}
      
	if( mydbservice->isNewTagRequest("SiStripNoisesRcd") ){
	  edm::LogInfo("SiStripO2O") << "new tag requested for SiStripNoisesRcd" << std::endl;
	  mydbservice->createNewIOV<SiStripNoises>(noise_cpy,mydbservice->endOfTime(),"SiStripNoisesRcd");      
	} else {
	  edm::LogInfo("SiStripO2O") << "append to existing tag for SiStripNoisesRcd" << std::endl;
	  mydbservice->appendSinceTime<SiStripNoises>(noise_cpy,mydbservice->currentTime(),"SiStripNoisesRcd");      
	}
     
	if( mydbservice->isNewTagRequest("SiStripFedCablingRcd") ){
	  edm::LogInfo("SiStripO2O") << "new tag requested for SiStripFedCablingRcd" << std::endl;
	  mydbservice->createNewIOV<SiStripFedCabling>(cabling_cpy,mydbservice->endOfTime(),"SiStripFedCablingRcd"); 
	} else {
	  edm::LogInfo("SiStripO2O") << "append to existing tag for SiStripFedCablingRcd" << std::endl;
	  mydbservice->appendSinceTime<SiStripFedCabling>(cabling_cpy,mydbservice->currentTime(),"SiStripFedCablingRcd"); 
	}

	//edm::LogInfo("SiStripO2O")  << " finished to upload data " << std::endl;    

      }catch(const cond::Exception& er){
	edm::LogError("newSiStripO2O")<<er.what()<<std::endl;
      }catch(const std::exception& er){
	edm::LogError("newSiStripO2O")<<"caught std::exception "<<er.what()<<std::endl;
      }catch(...){
	edm::LogError("newSiStripO2O")<<"Funny error"<<std::endl;
      }
    }else{
      edm::LogError("newSiStripO2O")<<"Service is unavailable"<<std::endl;
    }    
  }
}

