#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibTracker/SiStripESProducers/plugins/DBWriter/SiStripFedCablingManipulator.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include <fstream>
#include <iostream>

SiStripFedCablingManipulator::SiStripFedCablingManipulator(const edm::ParameterSet& iConfig):iConfig_(iConfig){
  edm::LogInfo("SiStripFedCablingManipulator") << "SiStripFedCablingManipulator constructor "<< std::endl;
}


SiStripFedCablingManipulator::~SiStripFedCablingManipulator(){
 edm::LogInfo("SiStripFedCablingManipulator") << "SiStripFedCablingManipulator::~SiStripFedCablingManipulator()" << std::endl;
}

void SiStripFedCablingManipulator::endRun(const edm::Run & run, const edm::EventSetup & es){
  edm::ESHandle<SiStripFedCabling> esobj;
  es.get<SiStripFedCablingRcd>().get( esobj ); 


  SiStripFedCabling *obj=0;
  manipulate(esobj.product(),obj);

  cond::Time_t Time_ = 0;
  
  //And now write  data in DB
  edm::Service<cond::service::PoolDBOutputService> dbservice;
  if( dbservice.isAvailable() ){

    if(obj==NULL){
      edm::LogError("SiStripFedCablingManipulator")<<"null pointer obj. nothing will be written "<<std::endl;
      return;
    }

    std::string openIovAt=iConfig_.getUntrackedParameter<std::string>("OpenIovAt","beginOfTime");
    if(openIovAt=="beginOfTime")
      Time_=dbservice->beginOfTime();
    else if (openIovAt=="currentTime")
      dbservice->currentTime();
    else
      Time_=iConfig_.getUntrackedParameter<uint32_t>("OpenIovAtTime",1);
    
    //if first time tag is populated
    if( dbservice->isNewTagRequest("SiStripFedCablingRcd")){
      edm::LogInfo("SiStripFedCablingManipulator") << "first request for storing objects with Record "<< "SiStripFedCablingRcd" << " at time " << Time_ << std::endl;
      dbservice->createNewIOV<SiStripFedCabling>(obj, Time_ ,dbservice->endOfTime(), "SiStripFedCablingRcd");      
    } else {
      edm::LogInfo("SiStripFedCablingManipulator") << "appending a new object to existing tag " <<"SiStripFedCablingRcd" <<" in since mode " << std::endl;
      dbservice->appendSinceTime<SiStripFedCabling>(obj, Time_, "SiStripFedCablingRcd"); 
    }    
  } else{
    edm::LogError("SiStripFedCablingManipulator")<<"Service is unavailable"<<std::endl;
  }
}

void SiStripFedCablingManipulator::manipulate(const SiStripFedCabling* iobj,SiStripFedCabling*& oobj){
  std::string fp=iConfig_.getParameter<std::string>("file");
   

  std::ifstream inputFile_; 
  inputFile_.open(fp.c_str());
  
  std::map<uint32_t, std::pair<uint32_t, uint32_t> > dcuDetIdMap;
  uint32_t dcuid, Olddetid, Newdetid; 
  
  // if(fp.c_str()==""){
  if(fp.empty()){
    edm::LogInfo("SiStripFedCablingManipulator") << "::manipulate : since no file is specified, the copy of the input cabling will be applied"<< std::endl;
    oobj= new SiStripFedCabling(*iobj);

  } else if (!inputFile_.is_open()){
    edm::LogError("SiStripFedCablingManipulator") << "::manipulate - ERROR in opening file  " << fp << std::endl;
    throw cms::Exception("CorruptedData")  << "::manipulate - ERROR in opening file  " << fp << std::endl;
  }else{
    
    for(;;) {
      inputFile_ >> dcuid >> Olddetid >> Newdetid;

      if (!(inputFile_.eof() || inputFile_.fail())){
	
	if(dcuDetIdMap.find(dcuid)==dcuDetIdMap.end()){
	  
	  edm::LogInfo("SiStripFedCablingManipulator") << dcuid << " " << Olddetid << " " << Newdetid << std::endl;
	  
	  dcuDetIdMap[dcuid]=std::pair<uint32_t, uint32_t>(Olddetid,Newdetid);
	}else{
	  edm::LogError("SiStripFedCablingManipulator") << "::manipulate - ERROR duplicated dcuid " << dcuid <<std::endl;     
	  throw cms::Exception("CorruptedData") << "SiStripFedCablingManipulator::manipulate - ERROR duplicated dcuid " << dcuid ;
	  break;
	}
      }else if (inputFile_.eof()){
	edm::LogInfo("SiStripFedCablingManipulator")<< "::manipulate - END of file reached"<<std::endl;
	break;
      }else if (inputFile_.fail()) {
	edm::LogError("SiStripFedCablingManipulator")<<"::manipulate - ERROR while reading file"<<std::endl;     
	break;
      }
    }
    inputFile_.close();
    std::map<uint32_t, std::pair<uint32_t, uint32_t> >::const_iterator it=dcuDetIdMap.begin();
    for (;it!=dcuDetIdMap.end();++it)
      edm::LogInfo("SiStripFedCablingManipulator")<< "::manipulate - Map "<< it->first << " " << it->second.first << " " << it->second.second;
   

    std::vector<FedChannelConnection> conns;
  
    const std::vector<uint16_t>& feds=iobj->feds();
    std::vector<uint16_t>::const_iterator ifeds=feds.begin();  
    for(;ifeds!=feds.end();ifeds++){
      const std::vector<FedChannelConnection>& conns_per_fed =iobj->connections( *ifeds );
      std::vector<FedChannelConnection>::const_iterator iconn=conns_per_fed.begin();
      for(;iconn!=conns_per_fed.end();++iconn){
	std::map<uint32_t, std::pair<uint32_t, uint32_t> >::const_iterator it=dcuDetIdMap.find(iconn->dcuId());
	if(it!=dcuDetIdMap.end() && it->second.first==iconn->detId()){
	  edm::LogInfo("SiStripFedCablingManipulator")<< "::manipulate - fedid "<< *ifeds << " dcuid " <<  iconn->dcuId() << " oldDet " << iconn->detId() << " newDetID " << it->second.second ;
	  conns.push_back(FedChannelConnection( 
					       iconn->fecCrate(),
					       iconn->fecSlot(),
					       iconn->fecRing(),
					       iconn->ccuAddr(),
					       iconn->ccuChan(),
					       iconn->i2cAddr(0),
					       iconn->i2cAddr(1),
					       iconn->dcuId(),
					       it->second.second,  //<------ New detid
					       iconn->nApvPairs(),
					       iconn->fedId(),
					       iconn->fedCh(),
					       iconn->fiberLength(),
					       iconn->dcu(),
					       iconn->pll(),
					       iconn->mux(),
					       iconn->lld()
					       )
			  );
	}else{
	  conns.push_back(FedChannelConnection( 
					       iconn->fecCrate(),
					       iconn->fecSlot(),
					       iconn->fecRing(),
					       iconn->ccuAddr(),
					       iconn->ccuChan(),
					       iconn->i2cAddr(0),
					       iconn->i2cAddr(1),
					       iconn->dcuId(),
					       iconn->detId(),
					       iconn->nApvPairs(),
					       iconn->fedId(),
					       iconn->fedCh(),
					       iconn->fiberLength(),
					       iconn->dcu(),
					       iconn->pll(),
					       iconn->mux(),
					       iconn->lld()
					       )
			  );
	}
      }
    }
    
    oobj = new SiStripFedCabling( conns );

  }  
}
