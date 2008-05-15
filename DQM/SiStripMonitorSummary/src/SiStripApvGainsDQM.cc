#include "DQM/SiStripMonitorSummary/interface/SiStripApvGainsDQM.h"


#include "DQMServices/Core/interface/MonitorElement.h"

// -----
SiStripApvGainsDQM::SiStripApvGainsDQM(const edm::EventSetup & eSetup,
                                       edm::ParameterSet const& hPSet,
                                       edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup,hPSet, fPSet){

}
// -----

// -----
SiStripApvGainsDQM::~SiStripApvGainsDQM(){}
// -----


// -----
void SiStripApvGainsDQM::fillModMEs(){

  edm::ESHandle<SiStripApvGain> gainHandle_;
  eSetup_.get<SiStripApvGainRcd>().get(gainHandle_);
  
  std::vector<uint32_t> DetIds;
  gainHandle_->getDetIds(DetIds);

  std::vector<uint32_t> selectedDetIds;
  selectedDetIds = selectModules(DetIds);

  ModMEs CondObj_ME;

  for(std::vector<uint32_t>::const_iterator detIter_ =selectedDetIds.begin();
                                            detIter_!=selectedDetIds.end();++detIter_){
    fillMEsForDet(CondObj_ME,*detIter_);
  }  
}  

  
// -----
void SiStripApvGainsDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_){
  
  edm::ESHandle<SiStripApvGain> gainHandle_;
  eSetup_.get<SiStripApvGainRcd>().get(gainHandle_);
  
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
void SiStripApvGainsDQM::fillSummaryMEs(){ }
  
