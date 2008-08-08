#include "DQM/SiStripHistoricInfoClient/interface/HistoricOfflineClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include "TNamed.h"

//---- default constructor / destructor
//-----------------------------------------------------------------------------------------------
HistoricOfflineClient::HistoricOfflineClient(const edm::ParameterSet& iConfig):iConfig_(iConfig) { 
//-----------------------------------------------------------------------------------------------
  dqmStore_ = edm::Service<DQMStore>().operator->(); dqmStore_->setVerbose(0); 
}


//-----------------------------------------------------------------------------------------------
HistoricOfflineClient::~HistoricOfflineClient() {}
//-----------------------------------------------------------------------------------------------



//---- called each event
//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
//-----------------------------------------------------------------------------------------------

  if(firstEventInRun){
    firstEventInRun=false;
    
    //for (unsigned int i=0; i<vSummary.size(); i++) vSummary.at(i)->setTimeValue(iEvent.time().value());    
  }
  ++nevents;
}



//---- called each BOR
//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){
//-----------------------------------------------------------------------------------------------
  std::cout<<"HistoricOfflineClient::beginRun() nevents = "<<nevents<<std::endl;  
  vSummary.clear();
  firstEventInRun=true;
  
}


//---- called each EOR
//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::endRun(const edm::Run& run , const edm::EventSetup& iSetup){
//-----------------------------------------------------------------------------------------------

  firstEventInRun=false;
  
  
  //*RETRIVE TAGGED MEs*//
  retrievePointersToModuleMEs(iSetup);
  
   
  //*LOOP OVER THE LIST OF SUMMARY OBJECTS TO BE INSERT IN DB*//
  std::vector<std::string> userDBContent;


  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters List = iConfig_.getParameter<Parameters>("list");
  Parameters::iterator itList = List.begin();
  Parameters::iterator itListEnd = List.end();
  for(; itList != itListEnd; ++itList ) {

    vSummary.push_back(new SiStripSummary());
    vSummary.back()->setRunNr(run.run());

    std::string RecordName = itList->getUntrackedParameter<std::string>("RecordName");
    vSummary.back()->setTag(RecordName);
    
    //* DISCOVER SET OF HISTOGRAMS & QUANTITIES TO BE UPLOADED*//
    std::vector<std::string> userDBContent;
    Parameters histoList = itList->getParameter<Parameters>("histoList");
    Parameters::iterator ithistoList = histoList.begin();
    Parameters::iterator ithistoListEnd = histoList.end();
    
    for(; ithistoList != ithistoListEnd; ++ithistoList ) {
       std::string histoName = ithistoList->getUntrackedParameter<std::string>("Name");
       std::vector<std::string> Quantities = ithistoList->getUntrackedParameter<std::vector<std::string> >("quantitiesToExtract"); 
       for (size_t i=0;i<Quantities.size();++i){
	 userDBContent.push_back(histoName+std::string("@")+Quantities[i]);
       }
    }
    vSummary.back()->setUserDBContent(userDBContent);
    
    std::cout << "QUANTITIES TO BE INSERTED IN DB : " << std::endl;

    std::vector<std::string> userDBContentA = vSummary.back()->getUserDBContent();
    for (size_t i=0;i<userDBContentA.size();++i){
      std::cout << userDBContentA[i]<< std::endl;
    }
    
    //*FILL SUMMARY*//
    std::cout << "FILL OBJECT " << std::endl;
    for(; ithistoList != ithistoListEnd; ++ithistoList ) {
      std::string histoName = ithistoList->getUntrackedParameter<std::string>("Name");
      std::vector<std::string> Quantities = ithistoList->getUntrackedParameter<std::vector<std::string> >("quantitiesToExtract"); 
      fillSummaryObjects( vSummary.back(), histoName, Quantities);
      
    }
  }
  
    
  //* PRINT STUFF*//
  std::cout<<"HistoricOfflineClient::endRun() nevents = "<<nevents<<std::endl;
  for (unsigned int i=0; i<vSummary.size(); i++) vSummary.at(i)->print();
  
  
  //*WRITE TO DB*//
  writeToDB();
}


//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::beginJob(const edm::EventSetup&) {
//-----------------------------------------------------------------------------------------------
  nevents = 0;
}


//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::endJob() {
//-----------------------------------------------------------------------------------------------
  if ( iConfig_.getUntrackedParameter<bool>("writeHisto", true) ){
    std::string outputfile = iConfig_.getUntrackedParameter<std::string>("outputFile", "historicOffline.root");
    std::cout<<"HistoricOfflineClient::endJob() outputFile = "<<outputfile<<std::endl;
    dqmStore_->save(outputfile);
  }
}


//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::retrievePointersToModuleMEs(const edm::EventSetup& iSetup) {
//-----------------------------------------------------------------------------------------------

  // take from eventSetup the SiStripDetCabling object
  
  std::cout<<"HistoricOfflineClient::retrievePointersToModuleMEs() "<<std::endl;
  
  edm::ESHandle<SiStripDetCabling> tkmechstruct;
  iSetup.get<SiStripDetCablingRcd>().get(tkmechstruct);
  // get list of active detectors from SiStripDetCabling - this will change and be taken from a SiStripDetControl object
  std::vector<uint32_t> activeDets;
  activeDets.clear(); // just in case
  tkmechstruct->addActiveDetectorsRawIds(activeDets);
  /// get all MonitorElements tagged as <tag>
  ClientPointersToModuleMEs.clear();
  for(std::vector<uint32_t>::const_iterator idet = activeDets.begin(); idet != activeDets.end(); ++idet){
    std::vector<MonitorElement *> local_mes =  dqmStore_->get(*idet); // get tagged MEs
    ClientPointersToModuleMEs.insert(std::make_pair(*idet, local_mes));
  }
  std::cout<<"HistoricOfflineClient::retrievePointersToModuleMEs() ClientPointersToModuleMEs.size()="<<ClientPointersToModuleMEs.size()<<std::endl;
}


//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::fillSummaryObjects(SiStripSummary* summary,std::string& histoName, std::vector<std::string>& Quantities){
//-----------------------------------------------------------------------------------------------

  std::cout<<"HistoricOfflineClient::fillSummaryObjects() called. ClientPointersToModuleMEs.size()="<<ClientPointersToModuleMEs.size()<< std::endl;
  for(std::map<uint32_t , std::vector<MonitorElement *> >::const_iterator imapmes = ClientPointersToModuleMEs.begin(); imapmes != ClientPointersToModuleMEs.end(); imapmes++){
     //uint32_t local_detid = imapmes->first;
     std::vector<MonitorElement*> locvec = imapmes->second;
     
     for(std::vector<MonitorElement*>::const_iterator iterMes = locvec.begin(); iterMes != locvec.end() ; iterMes++){
        std::string me_name = (*iterMes)->getName();
        if (me_name.find(histoName) == 0) { 
	
	SiStripSummary::InputVector values;
        std::vector<std::string> userDBContent;
      
        std::cout << "\n-----------------------------\n Found compatible ME " << me_name << " " << histoName << std::endl;
      
        for(size_t i=0;i<Quantities.size();++i){
	
	 userDBContent.push_back(histoName+std::string("@")+Quantities[i]);
	 if(Quantities[i] =="mean") 
	  values.push_back( (*iterMes)->getMean());
	 else if(Quantities[i]  =="rms")  
	  values.push_back( (*iterMes)->getRMS());
	 else if(Quantities[i]  =="entries")  
	  values.push_back( (*iterMes)->getEntries());
       }
      
       summary->put(returnDetComponent(me_name),values,userDBContent);
		
      }
    }     
  }
}

//----------------------------------------------------------------------------------------------- 
uint32_t HistoricOfflineClient::returnDetComponent(std::string histoName)
//-----------------------------------------------------------------------------------------------
{
  if(histoName.find("__det__")!= std::string::npos)
    return atoi(histoName.substr(histoName.find("__det__")+7,9).c_str());
  else if(histoName.find("TIB")!= std::string::npos)
    {
      if (histoName.find("layer")!= std::string::npos) 
	return 10+atoi(histoName.substr(histoName.find("layer__")+7,1).c_str());
      else 
	return sistripsummary::TIB;
    }
  else if(histoName.find("TOB")!= std::string::npos)
    {
      if (histoName.find("layer")!= std::string::npos) 
	return 13+atoi(histoName.substr(histoName.find("layer__")+7,1).c_str());
      else    
	return sistripsummary::TOB;
    }
  else if(histoName.find("TID")!= std::string::npos)
    {  
      if (histoName.find("layer")!= std::string::npos) 
	return 20+atoi(histoName.substr(histoName.find("layer__")+7,1).c_str());
      else    
	return sistripsummary::TID;
    } 
  else if(histoName.find("TEC")!= std::string::npos)
    {  
      if (histoName.find("layer")!= std::string::npos) 
	return 22+atoi(histoName.substr(histoName.find("layer__")+7,1).c_str());
      else    
	return sistripsummary::TEC;
    } 
  
}


//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::writeToDB() const 
//-----------------------------------------------------------------------------------------------
{
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  if( mydbservice.isAvailable() ){
    
    std::vector<SiStripSummary *>::const_iterator iter=vSummary.begin();
    std::vector<SiStripSummary *>::const_iterator iterEnd=vSummary.end();

    for(;iter!=iterEnd;++iter){
      if ( mydbservice->isNewTagRequest( (*iter)->getTag() ) ){   
	edm::LogVerbatim("SiStripSummary") << "createNewIOV " << mydbservice->beginOfTime() <<" " <<mydbservice->currentTime() <<std::endl;
	mydbservice->createNewIOV<SiStripSummary>((SiStripSummary*) *iter,mydbservice->beginOfTime(),mydbservice->endOfTime(),(*iter)->getTag());
      } else {	 
	edm::LogVerbatim("SiStripSummary") << "appendSinceTime " << mydbservice->beginOfTime() <<" " <<mydbservice->currentTime() <<std::endl;
	mydbservice->appendSinceTime<SiStripSummary>((SiStripSummary*) *iter,mydbservice->currentTime(),(*iter)->getTag());      
      }
    }
  }else{
    edm::LogError("writeToDB")<<"Service is unavailable"<<std::endl;
  }
  
}

