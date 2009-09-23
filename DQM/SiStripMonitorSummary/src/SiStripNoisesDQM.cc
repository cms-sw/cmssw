#include "DQM/SiStripMonitorSummary/interface/SiStripNoisesDQM.h"


#include "DQMServices/Core/interface/MonitorElement.h"

// -----
SiStripNoisesDQM::SiStripNoisesDQM(const edm::EventSetup & eSetup,
                                   edm::ParameterSet const& hPSet,
                                   edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){  
  gainRenormalisation_ = hPSet_.getParameter<bool>("GainRenormalisation");
  if( gainRenormalisation_){ eSetup.get<SiStripApvGainRcd>().get(gainHandle_);}


  // Build the Histo_TkMap:
  if(HistoMaps_On_ ) Tk_HM_ = new TkHistoMap("SiStrip/Histo_Map","MeanNoise_TkMap",0.);

}
// -----

// -----
SiStripNoisesDQM::~SiStripNoisesDQM(){}
// -----


// -----
void SiStripNoisesDQM::getActiveDetIds(const edm::EventSetup & eSetup){
  
  getConditionObject(eSetup);
  noiseHandle_->getDetIds(activeDetIds);
  selectModules(activeDetIds);

}
// -----


// -----
void SiStripNoisesDQM::fillModMEs(const std::vector<uint32_t> & selectedDetIds){

  ModMEs CondObj_ME;
 
  for(std::vector<uint32_t>::const_iterator detIter_=selectedDetIds.begin();
                                           detIter_!=selectedDetIds.end();++detIter_){
    fillMEsForDet(CondObj_ME,*detIter_);
  }
}    



// -----
void SiStripNoisesDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_){
  
  std::vector<uint32_t> DetIds;
  noiseHandle_->getDetIds(DetIds);

  SiStripNoises::Range noiseRange = noiseHandle_->getRange(selDetId_);
  
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;
    
  getModMEs(selModME_,selDetId_);
  
  float gainFactor;

  for( int istrip=0;istrip<nStrip;++istrip){
    try{
      if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
	if( gainRenormalisation_){
          SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
	  gainFactor= gainHandle_ ->getStripGain(istrip,gainRange);
          selModME_.ProfileDistr->Fill(istrip+1,noiseHandle_->getNoise(istrip,noiseRange)/gainFactor);
	}
	else{
          selModME_.ProfileDistr->Fill(istrip+1,noiseHandle_->getNoise(istrip,noiseRange));
	}
      } 	
      if( CondObj_fillId_ =="onlyCumul" || CondObj_fillId_ =="ProfileAndCumul"){
	if( gainRenormalisation_){
          SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
 	  gainFactor= gainHandle_ ->getStripGain(istrip,gainRange);
          selModME_.CumulDistr->Fill(noiseHandle_->getNoise(istrip,noiseRange)/gainFactor);
 	}
	else {
          selModME_.CumulDistr  ->Fill(noiseHandle_->getNoise(istrip,noiseRange));
        }
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
void SiStripNoisesDQM::fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds){
  
  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
                                            detIter_!= selectedDetIds.end();detIter_++){
    fillMEsForLayer(SummaryMEsMap_, *detIter_);

  } 
}    
// -----


// -----
void SiStripNoisesDQM::fillMEsForLayer( std::map<uint32_t, ModMEs> selMEsMap_, uint32_t selDetId_){
  
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
  float gainFactor=1;
  
  if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")){
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
	if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){	
	  if(gainRenormalisation_ ){
	    SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
	    gainFactor= gainHandle_ ->getStripGain(istrip,gainRange);
	    selME_.SummaryOfProfileDistr->Fill(istrip+1,noiseHandle_->getNoise(istrip,noiseRange)/gainFactor);
	  }
	  else{
	    selME_.SummaryOfProfileDistr->Fill(istrip+1,noiseHandle_->getNoise(istrip,noiseRange));
	  }
	}
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
  }//if fill
  
  if(hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")){
    // --> cumul summary    
    std::string hSummary_description;
    hSummary_description  = hPSet_.getParameter<std::string>("Summary_description");
    
    std::string hSummary_name; 
    hSummary_name = hidmanager.createHistoLayer(hSummary_description, 
						"layer", 
						getLayerNameAndId(selDetId_).first, 
						"") ;
    gainFactor=1;
    
    float meanNoise=0;
    
    for( int istrip=0;istrip<nStrip;++istrip){
      
      if( gainRenormalisation_){           
	SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
	gainFactor= gainHandle_ ->getStripGain(istrip,gainRange);
	try{
	  meanNoise = meanNoise +noiseHandle_->getNoise(istrip,noiseRange)/gainFactor;
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
      }	
      else {  
	try{
	  meanNoise = meanNoise +noiseHandle_->getNoise(istrip,noiseRange)/gainFactor;
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
      } 
    }//istrip
    meanNoise = meanNoise/nStrip;
  // get detIds belonging to same layer to fill X-axis with detId-number
  
    std::vector<uint32_t> sameLayerDetIds_;
    
    sameLayerDetIds_.clear();
    
    sameLayerDetIds_=GetSameLayerDetId(activeDetIds,selDetId_);
    
    
    unsigned int iBin=0;
    for(unsigned int i=0;i<sameLayerDetIds_.size();i++){
      if(sameLayerDetIds_[i]==selDetId_){iBin=i+1;}
    }  
    selME_.SummaryDistr->Fill(iBin,meanNoise);

    // Fill the Histo_TkMap with the mean Noise:
        if(HistoMaps_On_ ) Tk_HM_->fill(selDetId_, meanNoise);
    
    
  }//if fill

  /// Cumulative distr. for Noise:
  if(hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel")){
    std::string hSummaryOfCumul_description;
    hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("Cumul_description");
    
    std::string hSummaryOfCumul_name; 
    hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getStringNameAndId(selDetId_).first, "") ;
    
    for( int istrip=0;istrip<nStrip;++istrip){
      try{ 
	if( CondObj_fillId_ =="onlyCumul" || CondObj_fillId_ =="ProfileAndCumul"){
	  if(gainRenormalisation_){           
	    SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
	    gainFactor= gainHandle_ ->getStripGain(istrip,gainRange);
	    selME_.SummaryOfCumulDistr->Fill(noiseHandle_->getNoise(istrip,noiseRange)/gainFactor);
	  }
	  else{
	    selME_.SummaryOfCumulDistr->Fill(noiseHandle_->getNoise(istrip,noiseRange));
	  }
	}
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
    }//istrip
  }//if fill
  // -----
}  
// -----



