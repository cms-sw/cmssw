#include "DQMServices/Diagnostic/interface/DQMHistoryServiceBase.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Diagnostic/interface/HDQMfitUtilities.h"
#include <string>
#include <sstream>
#include <cctype>
#include <time.h>
#include <boost/cstdint.hpp>
#

DQMHistoryServiceBase::DQMHistoryServiceBase(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
 iConfig_(iConfig)
{
  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::DQMHistoryServiceBase]";
}


DQMHistoryServiceBase::~DQMHistoryServiceBase() { 
  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::~DQMHistoryServiceBase]";
}

void DQMHistoryServiceBase::initialize(){
  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::initialize]";
  fitME = new HDQMfitUtilities();
}

bool DQMHistoryServiceBase::checkForCompatibility(std::string ss){
  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::checkForCompatibility]";
  if(ss=="")
    return true;

  uint32_t previousRun=atoi(ss.substr(ss.find("Run ")+4).c_str());
  
  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::checkForCompatibility] extracted string " << previousRun ;
  return previousRun<getRunNumber();
}

void DQMHistoryServiceBase::createSummary(){
    
  //LOOP OVER THE LIST OF SUMMARY OBJECTS TO INSERT IN DB

  obj_=new HDQMSummary();

  obj_->setRunNr(getRunNumber());

  // DISCOVER SET OF HISTOGRAMS & QUANTITIES TO BE UPLOADED
  std::vector<std::string> userDBContent;
  typedef std::vector<edm::ParameterSet> VParameters;
  VParameters histoList = iConfig_.getParameter<VParameters>("histoList");
  VParameters::iterator ithistoList = histoList.begin();
  VParameters::iterator ithistoListEnd = histoList.end();
  
  for(; ithistoList != ithistoListEnd; ++ithistoList ) {    
    std::string keyName = ithistoList->getUntrackedParameter<std::string>("keyName");
    std::vector<std::string> Quantities = ithistoList->getUntrackedParameter<std::vector<std::string> >("quantitiesToExtract"); 
    for (size_t i=0;i<Quantities.size();++i){
      
      if  ( Quantities[i] == "landau" )
	setDBLabelsForLandau(keyName, userDBContent);
      else if  ( Quantities[i] == "gauss" )
	setDBLabelsForGauss(keyName, userDBContent);
      else if  ( Quantities[i] == "stat" )
	setDBLabelsForStat(keyName, userDBContent);
      else 
	setDBLabelsForUser(keyName, userDBContent, Quantities[i]);
    }
  }
  obj_->setUserDBContent(userDBContent);
  
  std::stringstream ss;
  ss << "[DQMHistoryServiceBase::scanTreeAndFillSummary] QUANTITIES TO BE INSERTED IN DB :" << std::endl;  
  std::vector<std::string> userDBContentA = obj_->getUserDBContent();
  for (size_t i=0;i<userDBContentA.size();++i){
    ss << userDBContentA[i]<< std::endl;
  }
  edm::LogInfo("HDQMSummary") << ss.str();

  // OPEN DQM FILE
  openRequestedFile();
  const std::vector<MonitorElement*>& MEs = dqmStore_->getAllContents(iConfig_.getUntrackedParameter<std::string>("ME_DIR","DQMData"));

  // FILL SUMMARY
  edm::LogInfo("HDQMSummary") << "\nSTARTING TO FILL OBJECT " << std::endl;
  ithistoList = histoList.begin();
  for(; ithistoList != ithistoListEnd; ++ithistoList ) {
    std::string keyName = ithistoList->getUntrackedParameter<std::string>("keyName");
    std::vector<std::string> Quantities = ithistoList->getUntrackedParameter<std::vector<std::string> >("quantitiesToExtract"); 
    scanTreeAndFillSummary(MEs, obj_, keyName, Quantities);
  }
  
}

void DQMHistoryServiceBase::openRequestedFile() { 

  dqmStore_ = edm::Service<DQMStore>().operator->(); 

  if( iConfig_.getParameter<bool>("accessDQMFile") ){
    
    std::string fileName = iConfig_.getUntrackedParameter<std::string>("FILE_NAME","");
    
    edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::openRequestedFile] Accessing root File" << fileName;

    dqmStore_->open(fileName, false); 
  } else {
    edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::openRequestedFile] Accessing dqmStore stream in Online Operation";
  }
}


void DQMHistoryServiceBase::scanTreeAndFillSummary(const std::vector<MonitorElement*>& MEs,HDQMSummary* summary,std::string& keyName, std::vector<std::string>& Quantities){
  //
  // -- Scan full root file and fill module numbers and histograms
  //
  //-----------------------------------------------------------------------------------------------

  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::scanTreeAndFillSummary] keyName " << keyName;

  std::vector<MonitorElement*>::const_iterator iterMes = MEs.begin(); 
  std::vector<MonitorElement*>::const_iterator iterMesEnd = MEs.end(); 
  std::stringstream ss;
  for (; iterMes!=iterMesEnd; ++iterMes){
    std::string me_name = (*iterMes)->getName();  
    if (me_name.find(keyName) == 0){ 

      HDQMSummary::InputVector values;
      std::vector<std::string> userDBContent;
      
      ss << "\nFound compatible ME " << me_name << " for key " << keyName << std::endl;
      
      for(size_t i=0;i<Quantities.size();++i){
	

	if(Quantities[i]  == "landau"){
	  setDBLabelsForLandau(keyName, userDBContent);
	  setDBValuesForLandau(iterMes,values);
	}
	else if(Quantities[i]  == "gauss"){
	  setDBLabelsForGauss(keyName, userDBContent);
	  setDBValuesForGauss(iterMes,values);
	}
	else if(Quantities[i]  == "stat"){
	  setDBLabelsForStat(keyName, userDBContent);
  	  setDBValuesForStat(iterMes,values);
	}
	else{
	  setDBLabelsForUser(keyName, userDBContent,Quantities[i]);
	  setDBValuesForUser(iterMes,values,Quantities[i]);
	}
      }  
      

      uint32_t detid=returnDetComponent(*iterMes);

      ss << "detid " << detid << " \n";
      for(size_t i=0;i<values.size();++i)
	ss << "Quantity " << userDBContent[i] << " value " << values[i] << std::endl;
      
      summary->put(detid,values,userDBContent);

    }
  }
  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::scanTreeAndFillSummary] " << ss.str();
}   
 

bool DQMHistoryServiceBase::setDBLabelsForLandau(std::string& keyName, std::vector<std::string>& userDBContent){ 
  userDBContent.push_back(keyName+std::string("@")+std::string("landauPeak"));
  userDBContent.push_back(keyName+std::string("@")+std::string("landauPeakErr"));
  userDBContent.push_back(keyName+std::string("@")+std::string("landauSFWHM"));
  userDBContent.push_back(keyName+std::string("@")+std::string("landauChi2NDF"));
  return true;
}

bool DQMHistoryServiceBase::setDBLabelsForGauss(std::string& keyName, std::vector<std::string>& userDBContent){ 
  userDBContent.push_back(keyName+std::string("@")+std::string("gaussMean"));
  userDBContent.push_back(keyName+std::string("@")+std::string("gaussSigma"));
  userDBContent.push_back(keyName+std::string("@")+std::string("gaussChi2NDF"));
  return true;
}	
bool DQMHistoryServiceBase::setDBLabelsForStat(std::string& keyName, std::vector<std::string>& userDBContent){ 
  userDBContent.push_back(keyName+std::string("@")+std::string("entries"));
  userDBContent.push_back(keyName+std::string("@")+std::string("mean"));
  userDBContent.push_back(keyName+std::string("@")+std::string("rms"));
  return true;
}

bool DQMHistoryServiceBase::setDBValuesForLandau(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values){
  fitME->doLanGaussFit(*iterMes);
  values.push_back( fitME->getLanGaussPar("mpv")    ); 
  values.push_back( fitME->getLanGaussParErr("mpv") ); 
  values.push_back( fitME->getLanGaussConv("fwhm")  );
  if (fitME->getFitnDof()!=0 ) values.push_back( fitME->getFitChi()/fitME->getFitnDof() );
  else                         values.push_back(-99.);
  return true;
}

bool DQMHistoryServiceBase::setDBValuesForGauss(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values){  
  fitME->doGaussFit(*iterMes);
  values.push_back( fitME->getGaussPar("mean")  );
  values.push_back( fitME->getGaussPar("sigma") );
  if (fitME->getFitnDof()!=0 ) values.push_back( fitME->getFitChi()/fitME->getFitnDof() );
  else                         values.push_back(-99.);
  return true;
}

bool DQMHistoryServiceBase::setDBValuesForStat(std::vector<MonitorElement*>::const_iterator iterMes, HDQMSummary::InputVector& values){  
  values.push_back( (*iterMes)->getEntries());
  values.push_back( (*iterMes)->getMean());
  values.push_back( (*iterMes)->getRMS());
  return true;
}

uint32_t DQMHistoryServiceBase::getRunNumber() const {
  edm::LogInfo("DQMHistoryServiceBase") <<  "[DQMHistoryServiceBase::getRunNumber] " << iConfig_.getParameter<uint32_t>("RunNb");
  return iConfig_.getParameter<uint32_t>("RunNb");
}

