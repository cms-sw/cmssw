#include "DQM/SiStripHistoricInfoClient/interface/ReadFromFile.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripPerformanceSummaryRcd.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include <cctype>
#include <time.h>
#include <boost/cstdint.hpp>
//-----------------------------------------------------------------------------------------------
ReadFromFile::ReadFromFile(const edm::ParameterSet& iConfig):iConfig_(iConfig){}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
ReadFromFile::~ReadFromFile() { vSummary.clear();}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
void ReadFromFile::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){} 
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
void ReadFromFile::endRun(const edm::Run& run , const edm::EventSetup& iSetup)
//-----------------------------------------------------------------------------------------------
{

  dqmStore_ = edm::Service<DQMStore>().operator->(); 
  dqmStore_->setVerbose(0); 


  //*GET Parameters*//
  ROOTFILE_DIR = iConfig_.getUntrackedParameter<std::string>("ROOTFILE_DIR","");
  FILE_NAME = iConfig_.getUntrackedParameter<std::string>("FILE_NAME","");
  ME_DIR = iConfig_.getUntrackedParameter<std::string>("ME_DIR","DQMData");


  //* OPEN DQM FILE*//
  openRequestedFile();
  
  
  //*LOOP OVER THE LIST OF SUMMARY OBJECTS TO INSERT IN DB*//
  std::vector<std::string> userDBContent;

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters List = iConfig_.getParameter<Parameters>("list");
  Parameters::iterator itList = List.begin();
  Parameters::iterator itListEnd = List.end();
  for(; itList != itListEnd; ++itList ) {

    vSummary.push_back(new SiStripSummary());
    vSummary.back()->setRunNr(getRunNumber());

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
    std::cout << "\n STARTING TO FILL OBJECT " << std::endl;
    ithistoList = histoList.begin();
    for(; ithistoList != ithistoListEnd; ++ithistoList ) {
      std::string histoName = ithistoList->getUntrackedParameter<std::string>("Name");
      std::vector<std::string> Quantities = ithistoList->getUntrackedParameter<std::vector<std::string> >("quantitiesToExtract"); 
      scanTreeAndFillSummary(ME_DIR, vSummary.back(), histoName, Quantities);
    }
  }
  
  //*WRITE TO DB*//
  writeToDB();
  
}


//-----------------------------------------------------------------------------------------------
void ReadFromFile::openRequestedFile() 
//-----------------------------------------------------------------------------------------------
{ 
  std::string fpath = ROOTFILE_DIR + FILE_NAME;
  dqmStore_->open(fpath.c_str(), false); 
}


//-----------------------------------------------------------------------------------------------
void ReadFromFile::scanTreeAndFillSummary(std::string top_dir, SiStripSummary* summary,std::string& histoName, std::vector<std::string>& Quantities) 
//-----------------------------------------------------------------------------------------------
//
// -- Scan full root file and fill module numbers and histograms
//
//-----------------------------------------------------------------------------------------------
{

  const std::vector<MonitorElement*>& MEs = dqmStore_->getAllContents(top_dir);
  std::vector<MonitorElement*>::const_iterator iterMes = MEs.begin(); 
  std::vector<MonitorElement*>::const_iterator iterMesEnd = MEs.end(); 

  for (; iterMes!=iterMesEnd; ++iterMes){
    std::string me_name = (*iterMes)->getName();
    if (me_name.find(histoName) == 0){ 

      SiStripSummary::InputVector values;
      std::vector<std::string> userDBContent;
      
      std::cout << "\n-----------------------------\nFound compatible ME " << me_name << " " << histoName << std::endl;
      
      for(size_t i=0;i<Quantities.size();++i){
	
	userDBContent.push_back(histoName+std::string("@")+Quantities[i]);
	if(Quantities[i] =="mean") 
	  values.push_back( (*iterMes)->getMean());
	else if(Quantities[i]  =="rms")  
	  values.push_back( (*iterMes)->getRMS());
	 else if(Quantities[i]  =="entries")  
	   values.push_back( (*iterMes)->getEntries());
      }
      
      for(size_t i=0;i<values.size();++i)
      std::cout << "Quantity " << userDBContent[i] << " value " << values[i] << std::endl;
      std::cout << std::endl;
      
      summary->put(returnDetComponent(me_name),values,userDBContent);
    }
  }
}   
 

//-----------------------------------------------------------------------------------------------
void ReadFromFile::writeToDB() const 
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
 
//----------------------------------------------------------------------------------------------- 
uint32_t ReadFromFile::returnDetComponent(std::string histoName)
//-----------------------------------------------------------------------------------------------
{
  if(histoName.find("__det__")!= std::string::npos)
    return atoi(histoName.substr(histoName.find("__det__")+7,9).c_str());
  else if(histoName.find("TIB")!= std::string::npos)
    {
      if (histoName.find("layer")!= std::string::npos) 
	return 10+atoi(histoName.substr(histoName.find("layer__")+7,1).c_str());
      else 
	return SiStripSummary::TIB;
    }
  else if(histoName.find("TOB")!= std::string::npos)
    {
      if (histoName.find("layer")!= std::string::npos) 
	return 13+atoi(histoName.substr(histoName.find("layer__")+7,1).c_str());
      else    
	return SiStripSummary::TOB;
    }
  else if(histoName.find("TID")!= std::string::npos)
    {  
      if (histoName.find("layer")!= std::string::npos) 
	return 20+atoi(histoName.substr(histoName.find("layer__")+7,1).c_str());
      else    
	return SiStripSummary::TID;
    } 
  else if(histoName.find("TEC")!= std::string::npos)
    {  
      if (histoName.find("layer")!= std::string::npos) 
	return 22+atoi(histoName.substr(histoName.find("layer__")+7,1).c_str());
      else    
	return SiStripSummary::TEC;
    } 
  
}

//-----------------------------------------------------------------------------------------------
uint32_t ReadFromFile::getRunNumber() const 
//-----------------------------------------------------------------------------------------------
{
  //Get RunNumber
  int pos1 = ME_DIR.find("Run ");
  int pos2 = ME_DIR.find("/SiStrip");
  
  std::string runStr = ME_DIR.substr(pos1+4,pos2-pos1-4);
  return atoi(runStr.c_str());
}
