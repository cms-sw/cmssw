#include "DQM/SiStripMonitorSummary/interface/SiStripNoisesDQM.h"


#include "DQMServices/Core/interface/MonitorElement.h"

// -----
SiStripNoisesDQM::SiStripNoisesDQM(const edm::EventSetup & eSetup,
                                   edm::ParameterSet const& hPSet,
                                   edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){}
// -----

// -----
SiStripNoisesDQM::~SiStripNoisesDQM(){}
// -----


// -----
void SiStripNoisesDQM::fillModMEs(){

  edm::ESHandle<SiStripNoises> noiseHandle_;
  eSetup_.get<SiStripNoisesRcd>().get(noiseHandle_);
  
  std::vector<uint32_t> DetIds;
  noiseHandle_->getDetIds(DetIds);

  std::vector<uint32_t> selectedDetIds;
  selectedDetIds = selectModules(DetIds);

  ModMEs CondObj_ME;
 
  for(std::vector<uint32_t>::const_iterator detIter_=selectedDetIds.begin();
                                           detIter_!=selectedDetIds.end();++detIter_){
    fillMEsForDet(CondObj_ME,*detIter_);
  }
}    



// -----
void SiStripNoisesDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_){
  
  edm::ESHandle<SiStripNoises> noiseHandle_;
  eSetup_.get<SiStripNoisesRcd>().get(noiseHandle_);
  
  std::vector<uint32_t> DetIds;
  noiseHandle_->getDetIds(DetIds);

  SiStripNoises::Range noiseRange = noiseHandle_->getRange(selDetId_);
  
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;
    
  getModMEs(selModME_,selDetId_);

  for( int istrip=0;istrip<nStrip;++istrip){
    try{
      if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
        selModME_.ProfileDistr->Fill(istrip+1,noiseHandle_->getNoise(istrip,noiseRange));
      } 	
      if( CondObj_fillId_ =="onlyCumul" || CondObj_fillId_ =="ProfileAndCumul"){
        selModME_.CumulDistr  ->Fill(noiseHandle_->getNoise(istrip,noiseRange));
      }
    } 
    catch(cms::Exception& e){
      edm::LogError("SiStripNoisesDQM")          
	 << "[SiStripNoisesDQM::fillMEsForDet] cms::Exception accessing noiseHandle_->getNoise(istrip,noiseRange) for strip "  
	 << istrip 
	 << " and detid " 
	 << selDetId_  
	 << " :  " 
	 << e.what() ;
    }
  }
}
  
// -----
void SiStripNoisesDQM::fillSummaryMEs(){

  edm::ESHandle<SiStripNoises> noiseHandle_;
  eSetup_.get<SiStripNoisesRcd>().get(noiseHandle_);
  
  std::vector<uint32_t> DetIds;
  noiseHandle_->getDetIds(DetIds);
  
  std::vector<uint32_t> selectedDetIds;
  selectedDetIds = selectModules(DetIds);
  
  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
                                            detIter_!= selectedDetIds.end();detIter_++){
    fillMEsForLayer(SummaryMEsMap_, *detIter_);

  } 
}    
// -----


// -----
void SiStripNoisesDQM::fillMEsForLayer( std::map<uint32_t, ModMEs> selMEsMap_, uint32_t selDetId_){

   
  edm::ESHandle<SiStripNoises> noiseHandle_;
  eSetup_.get<SiStripNoisesRcd>().get(noiseHandle_);
    
  // ----
  int subdetectorId_ = ((selDetId_>>25)&0x7);
  
  if( subdetectorId_<3 ||subdetectorId_>6 ){ 
    edm::LogError("SiStripNoisesDQM")
       << "[SiStripNoisesDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
       << subdetectorId_ << " no folder set!" 
       << std::endl;
    return;
  }
  // ----
         
  std::map<uint32_t, ModMEs>::iterator selMEsMapIter_  = selMEsMap_.find(getLayerNameAndId(selDetId_).second);
  ModMEs selME_;
  selME_ =selMEsMapIter_->second;
  getSummaryMEs(selME_,selDetId_);
  
  SiStripNoises::Range noiseRange = noiseHandle_->getRange(selDetId_);
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;

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
 
	
    for( int istrip=0;istrip<nStrip;++istrip){
    
      try{ 
       selME_.SummaryOfProfileDistr->Fill(istrip+1,noiseHandle_->getNoise(istrip,noiseRange));
      } 
      catch(cms::Exception& e){
        edm::LogError("SiStripNoisesDQM")          
	   << "[SiStripNoisesDQM::fillMEsForLayer] cms::Exception accessing noiseHandle_->getNoise(istrip,noiseRange) for strip "  
	   << istrip 
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
 
	
    for( int istrip=0;istrip<nStrip;++istrip){
    
      try{ 
       selME_.SummaryOfCumulDistr->Fill(noiseHandle_->getNoise(istrip,noiseRange));
      } 
      catch(cms::Exception& e){
        edm::LogError("SiStripNoisesDQM")          
	   << "[SiStripNoisesDQM::fillMEsForLayer] cms::Exception accessing noiseHandle_->getNoise(istrip,noiseRange) for strip "  
	   << istrip 
	   << "and detid " 
	   << selDetId_  
	   << " :  " 
	   << e.what() ;
      }
    }// istrip
  
  }
}  
// -----
 


