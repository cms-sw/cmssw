#include "DQM/SiStripMonitorSummary/interface/SiStripQualityDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// -----
SiStripQualityDQM::SiStripQualityDQM(const edm::EventSetup & eSetup,
                                         edm::ParameterSet const& hPSet,
                                         edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){
  qualityLabel_ = fPSet.getParameter<std::string>("StripQualityLabel");
}
// -----



// -----
SiStripQualityDQM::~SiStripQualityDQM(){}
// -----


// -----
void SiStripQualityDQM::getActiveDetIds(const edm::EventSetup & eSetup){
  getConditionObject(eSetup);
  qualityHandle_->getDetIds(activeDetIds);
  selectModules(activeDetIds);  
}
// -----


// -----
void SiStripQualityDQM::fillModMEs(const std::vector<uint32_t> & selectedDetIds){
   
  ModMEs CondObj_ME;
  
  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
                                            detIter_!= selectedDetIds.end();detIter_++){
    fillMEsForDet(CondObj_ME,*detIter_);
      
  }
}    
// -----




// -----
void SiStripQualityDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_){
    
  getModMEs(selModME_,selDetId_);
  
  SiStripQuality::Range qualityRange = qualityHandle_->getRange(selDetId_);
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;
  
  for( int istrip=0;istrip<nStrip;++istrip){
    try{      
         selModME_.ProfileDistr->Fill(istrip+1,qualityHandle_->IsStripBad(qualityRange,istrip)?1.:0.);
    } 
    catch(cms::Exception& e){
      edm::LogError("SiStripQualityDQM")          
	<< "[SiStripQualityDQM::fillMEsForDet] cms::Exception accessing qualityHandle_->IsStripBad(qualityRange,istrip)?1.:0.) for strip "  
	<< istrip 
	<< " and detid " 
	<< selDetId_  
	<< " :  " 
	<< e.what() ;
    }
  }// istrip
  
}    
// -----


// -----
void SiStripQualityDQM::fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds){
                                                   
  bool fillNext = true; 
  
  for(unsigned int i=0;i<selectedDetIds.size();i++){					    
					    
    int subDetId_ = ((selectedDetIds[i]>>25)&0x7);

    if( subDetId_<3 ||subDetId_>6 ){ 
      edm::LogError("SiStripBaseCondObjDQM")
         << "[SiStripBaseCondObjDQM::bookSummaryProfileMEs] WRONG INPUT : no such subdetector type : "
         << subDetId_ << " and detId " << selectedDetIds[i] << " therefore no filling!" 
         << std::endl;
    }
    else {
    
     if( fillNext) { fillMEsForLayer(SummaryMEsMap_, selectedDetIds[i]);} 
     
     if( getLayerNameAndId(selectedDetIds[i+1])==getLayerNameAndId(selectedDetIds[i])){ fillNext=false;}
     else { fillNext=true;}
     
    } 
    
  }
  // -----

}    
// -----



// -----
void SiStripQualityDQM::fillMEsForLayer( std::map<uint32_t, ModMEs> selMEsMap_, uint32_t selDetId_){  
  
  float numberOfBadStrips=0;
  
  SiStripHistoId hidmanager;
      
  std::string hSummaryOfProfile_description;
  hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");
      
  std::string hSummary_name; 
  
  // ----
  int subDetId_ = ((selDetId_>>25)&0x7);
  
  if( subDetId_<3 || subDetId_>6 ){ 
    edm::LogError("SiStripQualityDQM")
       << "[SiStripQualityDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
       << subDetId_ << " no folder set!" 
       << std::endl;
    return;
  }
  // ----

  hSummary_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, 
                                              "layer", 
					      getLayerNameAndId(selDetId_).first, 
					      "") ;
        
  std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = selMEsMap_.find(getLayerNameAndId(selDetId_).second);
    
  ModMEs selME_;
  selME_ =selMEsMapIter_->second;

  getSummaryMEs(selME_,selDetId_ );
  
  // -----   					   
  uint32_t selSubDetId_ =  ((selDetId_>>25)&0x7);
  SiStripSubStructure substructure_;
  
  std::vector<uint32_t> sameLayerDetIds_;
  sameLayerDetIds_.clear();
  
  if(selSubDetId_==3){  //  TIB
    substructure_.getTIBDetectors(activeDetIds, sameLayerDetIds_, TIBDetId(selDetId_).layerNumber(),0,0,0);  
  }
  else if(selSubDetId_==4){  // TID
    substructure_.getTIDDetectors(activeDetIds, sameLayerDetIds_, 0,0,0,0);
  }
  else if(selSubDetId_==5){  // TOB
    substructure_.getTOBDetectors(activeDetIds, sameLayerDetIds_, TOBDetId(selDetId_).layerNumber(),0,0);
  }
  else if(selSubDetId_==6){  // TEC
    substructure_.getTECDetectors(activeDetIds, sameLayerDetIds_, 0,0,0,0,0,0);
  }
     
  //sort(sameLayerDetIds_.begin(),sameLayerDetIds_.end()); 
        
  // -----
  
  for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
    
    SiStripQuality::Range qualityRange = qualityHandle_->getRange(sameLayerDetIds_[i]);
    int nStrip =  reader->getNumberOfApvsAndStripLength(sameLayerDetIds_[i]).first*128;
    
    numberOfBadStrips=0;
    
    for( int istrip=0;istrip<nStrip;++istrip){
      if(qualityHandle_->IsStripBad(qualityRange,istrip)) { numberOfBadStrips++;}
    }
    
    try{ 
      selME_.SummaryOfProfileDistr->Fill(i+1,100*float(numberOfBadStrips)/nStrip);
    }
    catch(cms::Exception& e){
      edm::LogError("SiStripQualityDQM")
	<< "[SiStripQualityDQM::fillMEsForLayer] cms::Exception filling fraction of bad strips for detId "
	<< sameLayerDetIds_[i]
	<< " :  "
	<< e.what() ;
    } 
  } 
  
  std::string hSummaryOfCumul_description;
  hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("SummaryOfCumul_description");
  
  std::string hSummaryOfCumul_name; 
  
  if( subDetId_<3 || subDetId_>6 ){ 
    edm::LogError("SiStripQualityDQM")
      << "[SiStripQualityDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
      << subDetId_ << " no folder set!" 
      << std::endl;
    return;
  }
  
  hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, 
						     "layer", 
						     getLayerNameAndId(selDetId_).first, 
						     "") ;
  
  
  for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
    
    SiStripQuality::Range qualityRange = qualityHandle_->getRange(sameLayerDetIds_[i]);
    int nStrip =  reader->getNumberOfApvsAndStripLength(sameLayerDetIds_[i]).first*128;
    
    numberOfBadStrips=0;
    
    for( int istrip=0;istrip<nStrip;++istrip){
      if(qualityHandle_->IsStripBad(qualityRange,istrip)) { numberOfBadStrips++;}
    }
    
    try{ 
      selME_.SummaryDistr->Fill(100*float(numberOfBadStrips)/nStrip);
    }
    catch(cms::Exception& e){
      edm::LogError("SiStripQualityDQM")
	<< "[SiStripQualityDQM::fillMEsForLayer] cms::Exception filling fraction of bad strips for detId "
	<< sameLayerDetIds_[i]
	<< " :  "
	<< e.what() ;
    } 
  } 
  
  
}  
// -----
 



