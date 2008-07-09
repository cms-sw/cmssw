#include "DQM/SiStripMonitorSummary/interface/SiStripApvGainsDQM.h"


#include "DQMServices/Core/interface/MonitorElement.h"

// -----
SiStripApvGainsDQM::SiStripApvGainsDQM(const edm::EventSetup & eSetup,
                                       edm::ParameterSet const& hPSet,
                                       edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup,hPSet, fPSet){

  eSetup_.get<SiStripApvGainRcd>().get(gainHandle_);
}
// -----

// -----
SiStripApvGainsDQM::~SiStripApvGainsDQM(){}
// -----


// -----
void SiStripApvGainsDQM::fillModMEs(){
 
  std::vector<uint32_t> DetIds;
  gainHandle_->getDetIds(DetIds);

  selectedDetIds = selectModules(DetIds);

  ModMEs CondObj_ME;

  for(std::vector<uint32_t>::const_iterator detIter_ =selectedDetIds.begin();
                                            detIter_!=selectedDetIds.end();++detIter_){
    fillMEsForDet(CondObj_ME,*detIter_);
  }  
}  

  
// -----
void SiStripApvGainsDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_){
  
  std::vector<uint32_t> DetIds;
  gainHandle_->getDetIds(DetIds);

  SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
  
  int nApv =  reader->getNumberOfApvsAndStripLength(selDetId_).first;
    
  getModMEs(selModME_,selDetId_);
 
  for( int iapv=0;iapv<nApv;++iapv){
    try{
      if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
        selModME_.ProfileDistr->Fill(iapv+1,gainHandle_->getApvGain(iapv,gainRange));
      }
      if( CondObj_fillId_ =="onlyCumul" || CondObj_fillId_ =="ProfileAndCumul"){
        selModME_.CumulDistr  ->Fill(gainHandle_->getApvGain(iapv,gainRange));
      }
    } 
    catch(cms::Exception& e){
      edm::LogError("SiStripApvGainsDQM")          
	 << "[SiStripApvGainsDQM::fillMEsForDet] cms::Exception accessing gainHandle_->getApvGain(iapv,gainRange) for apv "  
	 << iapv 
	 << " and detid " 
	 << selDetId_  
	 << " :  " 
	 << e.what() ;
    }
  }
}

// -----
void SiStripApvGainsDQM::fillSummaryMEs(){

  std::vector<uint32_t> DetIds;
  gainHandle_->getDetIds(DetIds);
  
  selectedDetIds = selectModules(DetIds);
  
  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
                                            detIter_!= selectedDetIds.end();detIter_++){
    fillMEsForLayer(SummaryMEsMap_, *detIter_);
    
  } 
}  

// -----
void SiStripApvGainsDQM::fillMEsForLayer( std::map<uint32_t, ModMEs> selMEsMap_, uint32_t selDetId_){
    
  int subdetectorId_ = ((selDetId_>>25)&0x7);
  
  if( subdetectorId_<3 ||subdetectorId_>6 ){ 
    edm::LogError("SiStripApvGainsDQM")
       << "[SiStripApvGainsDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
       << subdetectorId_ << " no folder set!" 
       << std::endl;
    return;
  }
  // ----
         
  std::map<uint32_t, ModMEs>::iterator selMEsMapIter_  = selMEsMap_.find(getLayerNameAndId(selDetId_).second);
  ModMEs selME_;
  selME_ =selMEsMapIter_->second;
  getSummaryMEs(selME_,selDetId_);
  
  SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
  int nApv =  reader->getNumberOfApvsAndStripLength(selDetId_).first;

  SiStripHistoId hidmanager;

  
  if( CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyProfile"){
    // --> profile summary    
    std::string hSummaryOfProfile_description;
    hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");
      
    std::string hSummaryOfProfile_name; 
    hSummaryOfProfile_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, 
                                                         "layer", 
						         getLayerNameAndId(selDetId_).first, 
						         "") ;
 	
    for( int iapv=0;iapv<nApv;++iapv){
      
      try{ 
       selME_.SummaryOfProfileDistr->Fill(iapv+1,gainHandle_->getApvGain(iapv,gainRange));
      } 
      catch(cms::Exception& e){
        edm::LogError("SiStripApvGainsDQM")          
	   << "[SiStripApvGainsDQM::fillMEsForLayer] cms::Exception accessing gainHandle_->getApvGain(istrip,gainRange) for strip "  
	   << iapv
	   << " and detid " 
	   << selDetId_  
	   << " :  " 
	   << e.what() ;
      }
    }// istrip
  }
    
  if( CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyCumul" ){ 
  
    // --> cumul summary    
    std::string hSummaryOfCumul_description;
    hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("SummaryOfCumul_description");
      
    std::string hSummaryOfCumul_name; 
    hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, 
                                                       "layer", 
						       getLayerNameAndId(selDetId_).first, 
						       "") ;
 
	
    for( int iapv=0;iapv<nApv;++iapv){
      try{ 
	selME_.SummaryOfCumulDistr->Fill(gainHandle_->getApvGain(iapv,gainRange));
      } 
      catch(cms::Exception& e){
        edm::LogError("SiStripApvGainsDQM")          
	   << "[SiStripApvGainsDQM::fillMEsForLayer] cms::Exception accessing gainHandle_->getApvGain(iapv,gainRange) for apv "  
	   << iapv 
	   << "and detid " 
	   << selDetId_  
	   << " :  " 
	   << e.what() ;
      }
    }// iapv
  }
}  
// -----
 




  
