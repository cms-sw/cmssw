#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"



// -----
SiStripBaseCondObjDQM::SiStripBaseCondObjDQM(const edm::EventSetup & eSetup,
                                             edm::ParameterSet const& hPSet,
                                             edm::ParameterSet const& fPSet ):
  eSetup_(eSetup),
  hPSet_(hPSet),
  fPSet_(fPSet),
  dqmStore_(edm::Service<DQMStore>().operator->()),
  m_cacheID_(0){

  reader = new SiStripDetInfoFileReader(edm::FileInPath(std::string("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat") ).fullPath());
  
  Mod_On_   = fPSet_.getParameter<bool>("Mod_On");
  SummaryOnLayerLevel_On_ = fPSet_.getParameter<bool>("SummaryOnLayerLevel_On");
   
  CondObj_fillId_    = hPSet_.getParameter<std::string>("CondObj_fillId");
  CondObj_name_      = hPSet_.getParameter<std::string>("CondObj_name");

}
// -----



// -----
void SiStripBaseCondObjDQM::analysis(const edm::EventSetup & eSetup_){
 
  unsigned long long cacheID_=  getCache(eSetup_);
  
  if (m_cacheID_ == cacheID_) return;
  
  m_cacheID_ = cacheID_;

  if(Mod_On_ )  { fillModMEs(); }
  if(SummaryOnLayerLevel_On_ ){ fillSummaryMEs();}

}
// -----



// -----
std::vector<uint32_t> SiStripBaseCondObjDQM::getCabledModules() {     
 
  std::vector<uint32_t> cabledDetIds_;  
  
  eSetup_.get<SiStripDetCablingRcd>().get(detCablingHandle_);
    
  detCablingHandle_->addActiveDetectorsRawIds(cabledDetIds_);

  return cabledDetIds_;  

}
// -----



// -----
std::vector<uint32_t> SiStripBaseCondObjDQM::selectModules(std::vector<uint32_t> detIds_){

  std::vector<uint32_t> selDetIds_;
  
  ModulesToBeExcluded_     = fPSet_.getParameter< std::vector<unsigned int> >("ModulesToBeExcluded");
  ModulesToBeIncluded_     = fPSet_.getParameter< std::vector<unsigned int> >("ModulesToBeIncluded");
  SubDetectorsToBeExcluded_= fPSet_.getParameter< std::vector<std::string> >("SubDetectorsToBeExcluded");
  
 
  if(fPSet_.getParameter<bool>("excludeModules") 
     && ModulesToBeExcluded_.size()==0 
     && ModulesToBeIncluded_.size()==0 ){
    edm::LogWarning("SiStripBaseCondObjDQM") 
       << "[SiStripBaseCondObjDQM::selectModules] PLEASE CHECK : no modules to be exclude/included in your cfg"
       << std::endl; 
    return selDetIds_;
  }
 

  // --> detIds to start with

  if( fPSet_.getParameter<bool>("excludeModules")){
      
    if( ModulesToBeIncluded_.size()>0 ){
    
      for( std::vector<uint32_t>::iterator modIter_  = ModulesToBeIncluded_.begin(); 
                                           modIter_ != ModulesToBeIncluded_.end(); modIter_++){
	  selDetIds_.push_back(*modIter_);				 
      } 
    }
    else { selDetIds_ = detIds_;}
  }
  else { selDetIds_ = detIds_;}

  

  // -----
  // *** exclude modules ***
  
  if( fPSet_.getParameter<bool>("excludeModules") ){
    

    for( std::vector<uint32_t>::const_iterator modIter_  = ModulesToBeExcluded_.begin(); 
                                               modIter_ != ModulesToBeExcluded_.end(); modIter_++){
      
      std::sort(selDetIds_.begin(),selDetIds_.end());
      std::vector<uint32_t>::iterator detIter_=std::lower_bound(selDetIds_.begin(),selDetIds_.end(),*modIter_);
      selDetIds_.erase(detIter_);
      detIter_--;
     
    }
  
  }
  // *** exclude modules ***
  // -----
   

  // -----
  // *** restrict to a particular subdetector ***
  
  if( *(SubDetectorsToBeExcluded_.begin()) !="none" ){
    
    std::string currSubDetector_;
    
    for( std::vector<std::string>::const_iterator modIter_  = SubDetectorsToBeExcluded_.begin(); 
                                                  modIter_ != SubDetectorsToBeExcluded_.end(); modIter_++){
      
      for(std::vector<uint32_t>::iterator detIter_ = selDetIds_.begin();
                                          detIter_!= selDetIds_.end();detIter_++){
	
        currSubDetector_ = folder_organizer.GetSubDetAndLayer(*(detIter_)).first;
					  
        if( currSubDetector_==*modIter_){  
	  selDetIds_.erase(detIter_);
	  detIter_--;
        } 

      }

    } 

  } 
  // *** restrict to a particular subdetector ***
  // -----

  
  // -----
  // *** fill only one Module per layer ***
  
  if(fPSet_.getParameter<std::string>("ModulesToBeFilled") == "onlyOneModulePerLayer"){
    
    unsigned int refLayer_;
    unsigned int nextLayer_;
    std::string refSubDetector_;
    std::string nextSubDetector_;    
     
    for(std::vector<uint32_t>::iterator detIter_ = selDetIds_.begin();
                                        detIter_!= selDetIds_.end();detIter_++){
      
      refSubDetector_  = folder_organizer.GetSubDetAndLayer(*detIter_).first;
      nextSubDetector_ = folder_organizer.GetSubDetAndLayer(*detIter_+1).first;
      
      if(nextSubDetector_==refSubDetector_ && detIter_!=(selDetIds_.end()-1) ){ // we want to have plot only one module per layer for each subdetector
        
        refLayer_= folder_organizer.GetSubDetAndLayer(*detIter_).second;
        nextLayer_= folder_organizer.GetSubDetAndLayer(*(detIter_+1)).second;
 
        if( nextLayer_== refLayer_ ){ // layer already stored, so omit it
          selDetIds_.erase(detIter_+1);
	  detIter_--;
        }

      }

    }
    
  }
  // *** fill only one Module per layer ***
  // -----

  
  return selDetIds_;

   
} //selectModules
// -----



// -----
void SiStripBaseCondObjDQM::getModMEs(ModMEs& CondObj_ME, const uint32_t& detId_){
  
  std::map< uint32_t, ModMEs >::const_iterator ModMEsMap_iter = ModMEsMap_.find(detId_);

  if (ModMEsMap_iter != ModMEsMap_.end()){ 
  
    CondObj_ME=ModMEsMap_iter->second;
  
  
    if( ( CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyProfile") && CondObj_ME.ProfileDistr ) { 
      CondObj_ME.ProfileDistr ->Reset();    
    }
       
    if( (CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyCumul" ) &&  CondObj_ME.CumulDistr ){
      CondObj_ME.CumulDistr ->Reset();
    }
    return; 

  }
  
  // --> profile defined for all CondData
  if ( (CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyProfile") ) {
    bookProfileMEs(CondObj_ME,detId_);
  }  
  
  // --> cumul currently only defined for noise and apvgain
  if( (CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyCumul" )
      &&(CondObj_name_ == "noise" || CondObj_name_ == "apvgain")          ) bookCumulMEs(CondObj_ME,detId_);
  
 
  ModMEsMap_.insert( std::make_pair(detId_,CondObj_ME) );
  
}
// ---- 


// -----
void SiStripBaseCondObjDQM::getSummaryMEs(ModMEs& CondObj_ME, const uint32_t& detId_){

  std::map<uint32_t, ModMEs>::const_iterator SummaryMEsMap_iter = SummaryMEsMap_.find(getLayerNameAndId(detId_).second);
   
  if (SummaryMEsMap_iter != SummaryMEsMap_.end()){ 
    
    CondObj_ME=SummaryMEsMap_iter->second;
  
    if( ( CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyProfile") && CondObj_ME.SummaryOfProfileDistr ) { 
      CondObj_ME.SummaryOfProfileDistr ->Reset();    
    }
       
    if( (CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyCumul" ) &&  CondObj_ME.SummaryOfCumulDistr ){
      CondObj_ME.SummaryOfCumulDistr ->Reset();
    }
    
    return; 

  }

  // --> currently only profile summary defined for pedestal, noise and LA
  if( (CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyProfile" ) 
    && (CondObj_name_ == "pedestal" || CondObj_name_ == "noise" || CondObj_name_ == "apvgain" || CondObj_name_ == "lorentzangle")   ) {
    
    if (CondObj_ME.SummaryOfProfileDistr) { bookSummaryProfileMEs(CondObj_ME,detId_);}
  
  }
    
  // --> currently only cumul summary for noise and LA
  if( (CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyCumul") 
      &&( CondObj_name_ == "noise"|| CondObj_name_ == "apvgain" || CondObj_name_ == "lorentzangle")  ) {
              
      if (CondObj_ME.SummaryOfCumulDistr) { bookSummaryCumulMEs(CondObj_ME,detId_); } 
      
  }                         

  SummaryMEsMap_.insert( std::make_pair(getLayerNameAndId(detId_).second,CondObj_ME) );
}
// ---- 


// -----
void SiStripBaseCondObjDQM::bookProfileMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_){
     
  int    hProfile_NchX  = 0;
  double hProfile_LowX  = 0;
  double hProfile_HighX = 0;
  
  std::string hProfile_description;
  hProfile_description   = hPSet_.getParameter<std::string>("Profile_description");
      
  std::string hProfile_xTitle, hProfile_yTitle;
  hProfile_xTitle          = hPSet_.getParameter<std::string>("Profile_xTitle");
  hProfile_yTitle          = hPSet_.getParameter<std::string>("Profile_yTitle");
            
  if( CondObj_name_!= "apvgain" ){
    
    int nStrip      = reader->getNumberOfApvsAndStripLength(detId_).first*128;
	
    hProfile_NchX           = nStrip;
    hProfile_LowX           = 0.5;
    hProfile_HighX          = nStrip+0.5;
  }
  else {
	
    int nApv      = reader->getNumberOfApvsAndStripLength(detId_).first;
	
    hProfile_NchX           = nApv;
    hProfile_LowX           = 0.5;
    hProfile_HighX          = nApv+0.5;
  }
      
  folder_organizer.setDetectorFolder(detId_); 
      
  std::string hProfile_Name; 
  hProfile_Name = hidmanager.createHistoId(hProfile_description, "det", detId_); ;
      
  std::string hProfile;
  hProfile = hProfile_Name ;
    
  CondObj_ME.ProfileDistr = dqmStore_->book1D(hProfile_Name, hProfile, hProfile_NchX, hProfile_LowX, hProfile_HighX);
  CondObj_ME.ProfileDistr->setAxisTitle(hProfile_xTitle,1);
  CondObj_ME.ProfileDistr->setAxisTitle(hProfile_yTitle,2);
  dqmStore_->tag(CondObj_ME.ProfileDistr, detId_);
  
}
// -----


      
// -----
void SiStripBaseCondObjDQM::bookCumulMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_){

  int    hCumul_NchX    = 0;
  double hCumul_LowX    = 0;
  double hCumul_HighX   = 0;
     
  std::string hCumul_description;
  hCumul_description   = hPSet_.getParameter<std::string>("Cumul_description");
      
  std::string hCumul_xTitle, hCumul_yTitle;
  hCumul_xTitle        = hPSet_.getParameter<std::string>("Cumul_xTitle");
  hCumul_yTitle        = hPSet_.getParameter<std::string>("Cumul_yTitle");
      
  hCumul_NchX          = hPSet_.getParameter<int>("Cumul_NchX");
  hCumul_LowX          = hPSet_.getParameter<double>("Cumul_LowX");
  hCumul_HighX         = hPSet_.getParameter<double>("Cumul_HighX");
      
  folder_organizer.setDetectorFolder(detId_); 
      
  std::string hCumul_name; 
  hCumul_name   = hidmanager.createHistoId(hCumul_description  , "det", detId_); ;
      
  std::string hCumul_title;
  hCumul_title   = hCumul_name ;
      
  CondObj_ME.CumulDistr = dqmStore_->book1D(hCumul_name, 
                                            hCumul_title, 
					    hCumul_NchX, 
					    hCumul_LowX, 
					    hCumul_HighX);
  CondObj_ME.CumulDistr->setAxisTitle(hCumul_xTitle,1);
  CondObj_ME.CumulDistr->setAxisTitle(hCumul_yTitle,2);
  dqmStore_->tag(CondObj_ME.CumulDistr, detId_);
      
} 
// ---- 



// -----
void SiStripBaseCondObjDQM::bookSummaryProfileMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_){
  
  std::vector<uint32_t> sameLayerDetIds_;

  int    hSummaryOfProfile_NchX    = 0;
  double hSummaryOfProfile_LowX    = 0;
  double hSummaryOfProfile_HighX   = 0;
     
  std::string hSummaryOfProfile_description;
  hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");
      
  std::string hSummaryOfProfile_xTitle, hSummaryOfProfile_yTitle;
  hSummaryOfProfile_xTitle        = hPSet_.getParameter<std::string>("SummaryOfProfile_xTitle");
  hSummaryOfProfile_yTitle        = hPSet_.getParameter<std::string>("SummaryOfProfile_yTitle");
  
  int nStrip, nApv;    
  int layerId_= getLayerNameAndId(detId_).second;
     
  if( CondObj_name_ == "pedestal" || CondObj_name_ == "noise" || CondObj_name_ == "quality"){ // plot in strip number
    
    if( (layerId_ > 610 && layerId_ < 620) || // TID & TEC have 768 strips at maximum
        (layerId_ > 620 && layerId_ < 630) ||
        (layerId_ > 410 && layerId_ < 414) ||
        (layerId_ > 420 && layerId_ < 424) ){ nStrip =768;} 
    else { nStrip      = reader->getNumberOfApvsAndStripLength(detId_).first*128;}
    
    hSummaryOfProfile_NchX           = nStrip;
    hSummaryOfProfile_LowX           = 0.5;
    hSummaryOfProfile_HighX          = nStrip+0.5;
  
  }  
  else if( CondObj_name_ == "lorentzangle"){ // plot in detId-number

    // -----
    // get detIds belonging to same layer to fill X-axis with detId-number
  					   
    uint32_t subDetId_ =  ((detId_>>25)&0x7);
    SiStripSubStructure substructure_;
  
    if(subDetId_==3){  //  TIB
      substructure_.getTIBDetectors(selectedDetIds, sameLayerDetIds_, TIBDetId(detId_).layerNumber(),0,0,0);  
    }
    else if(subDetId_==4){  // TID
      substructure_.getTIDDetectors(selectedDetIds, sameLayerDetIds_, 0,0,0,0);
    }
    else if(subDetId_==5){  // TOB
      substructure_.getTOBDetectors(selectedDetIds, sameLayerDetIds_, TOBDetId(detId_).layerNumber(),0,0);
    }
    else if(subDetId_==6){  // TEC
      substructure_.getTECDetectors(selectedDetIds, sameLayerDetIds_, 0,0,0,0,0,0);
    }

    hSummaryOfProfile_NchX           = sameLayerDetIds_.size(); 
    hSummaryOfProfile_LowX           = 0.5;
    hSummaryOfProfile_HighX          = sameLayerDetIds_.size()+0.5;
 
  } 
  else if( CondObj_name_ == "apvgain"){
 
    if( (layerId_ > 610 && layerId_ < 620) || // TID & TEC have 6 apvs at maximum
        (layerId_ > 620 && layerId_ < 630) ||
        (layerId_ > 410 && layerId_ < 414) ||
        (layerId_ > 420 && layerId_ < 424) ){ nApv =6;} 
    else { nApv     = reader->getNumberOfApvsAndStripLength(detId_).first;}
    
    hSummaryOfProfile_NchX           = nApv;
    hSummaryOfProfile_LowX           = 0.5;
    hSummaryOfProfile_HighX          = nApv+0.5;
 
  }
  else {
    edm::LogWarning("SiStripBaseCondObjDQM") 
       << "[SiStripBaseCondObjDQM::bookSummaryProfileMEs] PLEASE CHECK : x-axis label in your cfg"
       << std::endl; 
  }
  
  uint32_t layer_=0;
      
  layer_ = folder_organizer.GetSubDetAndLayer(detId_).second;
      
  folder_organizer.setLayerFolder(detId_,layer_); 
      
  std::string hSummaryOfProfile_name; 
  
  // ---
  int subdetectorId_ = ((detId_>>25)&0x7);
  
 
  if( subdetectorId_<3 ||subdetectorId_>6 ){ 
    edm::LogError("SiStripBaseCondObjDQM")
       << "[SiStripBaseCondObjDQM::bookSummaryProfileMEs] WRONG INPUT : no such subdetector type : "
       << subdetectorId_ << " no folder set!" 
       << std::endl;
    return;
  }
  // ---
  
  hSummaryOfProfile_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, 
                                                       "layer" , 
						        getLayerNameAndId(detId_).first,
							"") ;
      
  std::string hSummaryOfProfile_title;
  hSummaryOfProfile_title   = hSummaryOfProfile_name ;
           
  CondObj_ME.SummaryOfProfileDistr = dqmStore_->book1D(hSummaryOfProfile_name, 
                                                       hSummaryOfProfile_title, 
						       hSummaryOfProfile_NchX, 
						       hSummaryOfProfile_LowX, 
						       hSummaryOfProfile_HighX);
  CondObj_ME.SummaryOfProfileDistr->setAxisTitle(hSummaryOfProfile_xTitle,1);
  CondObj_ME.SummaryOfProfileDistr->setAxisTitle(hSummaryOfProfile_yTitle,2);
 
  // -----
  // in order to get the right detId-number labelled in right bin of x-axis
  
  if( CondObj_name_ == "lorentzangle"){
    
    unsigned int iBin=0;
    
    for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
    
      iBin++;
      char sameLayerDetIds_Name[1024];
      sprintf(sameLayerDetIds_Name,"%u",sameLayerDetIds_[i]);
      CondObj_ME.SummaryOfProfileDistr->setBinLabel(iBin, sameLayerDetIds_Name);
    
    }
  } 
  // -----
      
  dqmStore_->tag(CondObj_ME.SummaryOfProfileDistr, layer_);
      
} 
// ---- 



// -----
void SiStripBaseCondObjDQM::bookSummaryCumulMEs(SiStripBaseCondObjDQM::ModMEs& CondObj_ME, const uint32_t& detId_){
    
  int    hSummaryOfCumul_NchX    = 0;
  double hSummaryOfCumul_LowX    = 0;
  double hSummaryOfCumul_HighX   = 0;
	
  std::string hSummaryOfCumul_description;
  hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("SummaryOfCumul_description");
	
  std::string hSummaryOfCumul_xTitle, hSummaryOfCumul_yTitle;
  hSummaryOfCumul_xTitle        = hPSet_.getParameter<std::string>("SummaryOfCumul_xTitle");
  hSummaryOfCumul_yTitle        = hPSet_.getParameter<std::string>("SummaryOfCumul_yTitle");
	
  hSummaryOfCumul_NchX          = hPSet_.getParameter<int>("SummaryOfCumul_NchX");
  hSummaryOfCumul_LowX          = hPSet_.getParameter<double>("SummaryOfCumul_LowX");
  hSummaryOfCumul_HighX         = hPSet_.getParameter<double>("SummaryOfCumul_HighX");
	
  uint32_t layer_=0;
	
  layer_ = folder_organizer.GetSubDetAndLayer(detId_).second;
	
  folder_organizer.setLayerFolder(detId_,layer_); 
	
  std::string hSummaryOfCumul_name; 
  
  // ---
  int subdetectorId_ = ((detId_>>25)&0x7);
  
  if( subdetectorId_<3 || subdetectorId_>6 ){ 
    edm::LogError("SiStripBaseCondObjDQM")
       << "[SiStripBaseCondObjDQM::bookSummaryCumulMEs] WRONG INPUT : no such subdetector type : "
       << subdetectorId_ << " no folder set!" 
       << std::endl;
    return;
  }
  // ---
  
  hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, 
                                                     "layer" ,
						     getLayerNameAndId(detId_).first, 
						     "") ;
	
  std::string hSummaryOfCumul_title;
  hSummaryOfCumul_title   = hSummaryOfCumul_name ;
	
  CondObj_ME.SummaryOfCumulDistr = dqmStore_->book1D(hSummaryOfCumul_name, 
                                                     hSummaryOfCumul_title, 
						     hSummaryOfCumul_NchX, 
						     hSummaryOfCumul_LowX, 
						     hSummaryOfCumul_HighX);

  CondObj_ME.SummaryOfCumulDistr->setAxisTitle(hSummaryOfCumul_xTitle,1);
  CondObj_ME.SummaryOfCumulDistr->setAxisTitle(hSummaryOfCumul_yTitle,2);
	
  dqmStore_->tag(CondObj_ME.SummaryOfCumulDistr, layer_);
	
}
// -----





// -----
std::pair<std::string,uint32_t> SiStripBaseCondObjDQM::getLayerNameAndId(const uint32_t& detId_){

  int subdetectorId_ = ((detId_>>25)&0x7);
  int layerId_=0;
  std::string layerName_;
  
  char tempLayerName_;
  char tempLayerNumber_[20];
  
  if(      subdetectorId_==3 ){ //TIB
    
    for( unsigned int i=1; i < 5 ;i++){
      
      sprintf(tempLayerNumber_,"%u",i);
      
      if(TIBDetId(detId_).layer()==i){ 
        sprintf( &tempLayerName_, "%s%s","TIB__layer__", tempLayerNumber_); 
	layerId_ = 300 + i;
      }
      layerName_ = &tempLayerName_;
      
    }
    
  }
  else if( subdetectorId_==4 ){ //TIDD
    
    if(TIDDetId(detId_).side()==1){ // TIDD side 1
      
      for( unsigned int i=1; i < 4 ;i++){
	
	sprintf(tempLayerNumber_,"%u",i);
	
	if(TIDDetId(detId_).wheel()==i){ 
	  sprintf( &tempLayerName_, "%s%s","TID__side__1__wheel__", tempLayerNumber_); 
	  layerId_ = 410 + i;
	}
	layerName_ = &tempLayerName_;
	
      }
      
      
    }
    else if(TIDDetId(detId_).side()==2){// TIDD side 2
      
      for( unsigned int i=1; i < 4 ;i++){
	
	sprintf(tempLayerNumber_,"%u",i);
	
	if(TIDDetId(detId_).wheel()==i){ 
	  sprintf( &tempLayerName_, "%s%s","TID__side__2__wheel__", tempLayerNumber_); 
	  layerId_ = 420 + i;
	}
	layerName_ = &tempLayerName_;
	
      }
      
    }
  }
  else if( subdetectorId_==5 ){ // TOB
    
    for( unsigned int i=1; i < 7 ;i++){
      
      sprintf(tempLayerNumber_,"%u",i);
      
      if(TOBDetId(detId_).layer()==i){ 
        sprintf( &tempLayerName_, "%s%s","TOB__layer__", tempLayerNumber_); 
	layerId_ = 500 + i;
      }
      layerName_ = &tempLayerName_;
      
    }
    
    
  }
  else if( subdetectorId_==6 ){ // TEC
    
    
    if(TECDetId(detId_).side()==1){ // TEC side 1
      
      for( unsigned int i=1; i < 10 ;i++){
	
	sprintf(tempLayerNumber_,"%u",i);
	
	if(TECDetId(detId_).wheel()==i){ 
	  sprintf( &tempLayerName_, "%s%s","TEC__side__1__wheel__", tempLayerNumber_); 
	  layerId_ = 610 + i;
	}
	layerName_ = &tempLayerName_;
	
      }
    }
    else if(TECDetId(detId_).side()==2){ // TEC side 2
      
      for( unsigned int i=1; i < 10 ;i++){
	
	sprintf(tempLayerNumber_,"%u",i);
	
	if(TECDetId(detId_).wheel()==i){ 
	  sprintf( &tempLayerName_, "%s%s","TEC__side__2__wheel__", tempLayerNumber_); 
	  layerId_ = 620 + i;
	}
	layerName_ = &tempLayerName_;
	
      }
     }
  }
  
  return std::make_pair(layerName_,layerId_);
}
