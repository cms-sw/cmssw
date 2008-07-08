#include "DQM/SiStripMonitorSummary/interface/SiStripLorentzAngleDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// -----
SiStripLorentzAngleDQM::SiStripLorentzAngleDQM(const edm::EventSetup & eSetup,
                                                   edm::ParameterSet const& hPSet,
                                                   edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){
}
// -----



// -----
SiStripLorentzAngleDQM::~SiStripLorentzAngleDQM(){}
// -----


// -----
void SiStripLorentzAngleDQM::getActiveDetIds(const edm::EventSetup & eSetup){
  
  getConditionObject(eSetup);
  std::map<uint32_t,float>::const_iterator LAMapIter_;
  std::map<uint32_t,float> LAMap_ = lorentzangleHandle_->getLorentzAngles(); 
  
  std::vector<uint32_t> cabledModules_ = getCabledModules(); 
  
  for(std::vector<uint32_t>::const_iterator cablingIter_ = cabledModules_.begin();
                                           cablingIter_!= cabledModules_.end(); cablingIter_++){
    
    unsigned int cablingId;
    cablingId=*cablingIter_;
    LAMapIter_=LAMap_.find(cablingId);    
    
    if (LAMapIter_!=LAMap_.end() && (*LAMapIter_).first==cablingId) { 
      activeDetIds.push_back(*cablingIter_);
    }
  }

  selectModules(activeDetIds);

}
// -----


// -----
void SiStripLorentzAngleDQM::fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds){
   
   
  // -----
  // LA on layer-level : fill at once all detIds belonging to same layer when encountering first detID in the layer 

  bool fillNext = true; 
    for(unsigned int i=0;i<selectedDetIds.size();i++){					    
      int subDetId_ = ((selectedDetIds[i]>>25)&0x7);
      if( subDetId_<3 ||subDetId_>6 ){ 
	edm::LogError("SiStripLorentzAngle")
         << "[SiStripLorentzAngle::fillSummaryMEs] WRONG INPUT : no such subdetector type : "
         << subDetId_ << " and detId " << selectedDetIds[i] << " therefore no filling!" 
         << std::endl;
      }    
      else if (SummaryOnLayerLevel_On_) {    
	if( fillNext) { fillMEsForLayer(SummaryMEsMap_, selectedDetIds[i]);} 
	if( getLayerNameAndId(selectedDetIds[i+1])==getLayerNameAndId(selectedDetIds[i])){ fillNext=false;}
	else { fillNext=true;}
      } 
      else if (SummaryOnStringLevel_On_) {
	if( fillNext) { fillMEsForLayer(SummaryMEsMap_, selectedDetIds[i]);} 
	if( getStringNameAndId(selectedDetIds[i+1])==getStringNameAndId(selectedDetIds[i])){ fillNext=false;}
	else { fillNext=true;}
      } 
    }
}    
// -----



// -----
void SiStripLorentzAngleDQM::fillMEsForLayer( std::map<uint32_t, ModMEs> selMEsMap_, uint32_t selDetId_){  

  SiStripHistoId hidmanager;
      
  std::string hSummaryOfProfile_description;
  hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");
      
  std::string hSummary_name; 

  int subDetId_ = ((selDetId_>>25)&0x7);
  
  if( subDetId_<3 || subDetId_>6 ){ 
    edm::LogError("SiStripLorentzAngleDQM")
      << "[SiStripLorentzAngleDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
      << subDetId_ << " no folder set!" 
      << std::endl;
    return;
  }

  uint32_t selSubDetId_ =  ((selDetId_>>25)&0x7);
  SiStripSubStructure substructure_;
  
  std::vector<uint32_t> sameLayerDetIds_;
  sameLayerDetIds_.clear();

  
  if (SummaryOnStringLevel_On_) {  //FILLING FOR STRING LEVEL

    hSummary_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getStringNameAndId(selDetId_).first, "") ;
    std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = selMEsMap_.find(getStringNameAndId(selDetId_).second);
    
    ModMEs selME_;
    selME_ =selMEsMapIter_->second;

    getSummaryMEs(selME_,selDetId_ );
  
    // -----   					   
    sameLayerDetIds_.clear();
   
    if(selSubDetId_==3){  //  TIB
      if(TIBDetId(selDetId_).isInternalString()){
	substructure_.getTIBDetectors(activeDetIds, sameLayerDetIds_, TIBDetId(selDetId_).layerNumber(),0,1,TIBDetId(selDetId_).stringNumber());
      }
      if(TIBDetId(selDetId_).isExternalString()){
	substructure_.getTIBDetectors(activeDetIds, sameLayerDetIds_, TIBDetId(selDetId_).layerNumber(),0,2,TIBDetId(selDetId_).stringNumber());
      } 
    }
    else if(selSubDetId_==4){  // TID
      substructure_.getTIDDetectors(activeDetIds, sameLayerDetIds_, 0,0,0,0);
    }
    else if(selSubDetId_==5){  // TOB
      substructure_.getTOBDetectors(activeDetIds, sameLayerDetIds_, TOBDetId(selDetId_).layerNumber(),0,TOBDetId(selDetId_).rodNumber());
    }
    else if(selSubDetId_==6){  // TEC
      substructure_.getTECDetectors(activeDetIds, sameLayerDetIds_, 0,0,0,0,0,0);
    }
 
    // -----

    for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
      try{ 
	selME_.SummaryOfProfileDistr->Fill(i+1,lorentzangleHandle_->getLorentzAngle(sameLayerDetIds_[i]));
      }
      catch(cms::Exception& e){
	edm::LogError("SiStripLorentzAngleDQM")
	  << "[SiStripLorentzAngleDQM::fillMEsForLayer] cms::Exception accessing lorentzangleHandle_->getLorentzAngle() for detId "
	  << selDetId_
	  << " :  "
	  << e.what() ;
      } 
    } 
    
    std::string hSummaryOfCumul_description;
    hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("SummaryOfCumul_description");
    
    std::string hSummaryOfCumul_name; 
    
    if( subDetId_<3 || subDetId_>6 ){ 
      edm::LogError("SiStripLorentzAngleDQM")
	<< "[SiStripLorentzAngleDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
	<< subDetId_ << " no folder set!" 
	<< std::endl;
      return;
    }
    
    hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getStringNameAndId(selDetId_).first, "") ;
    
    for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
      try{ 
	selME_.SummaryOfCumulDistr->Fill(lorentzangleHandle_->getLorentzAngle(sameLayerDetIds_[i]));
      }
      catch(cms::Exception& e){
	edm::LogError("SiStripLorentzAngleDQM")
	  << "[SiStripLorentzAngleDQM::fillMEsForLayer] cms::Exception accessing lorentzangleHandle_->getLorentzAngle() for detId "
	  << selDetId_
	  << " :  "
	  << e.what() ;
      } 
    } 
  } //FILLING FOR STRING LEVEL
  
  
  else { //FILLING FOR LAYER LEVEL
    
    hSummary_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getLayerNameAndId(selDetId_).first, "") ;    
    std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = selMEsMap_.find(getLayerNameAndId(selDetId_).second);
    
    ModMEs selME_;
    selME_ =selMEsMapIter_->second;
    
    getSummaryMEs(selME_,selDetId_ );
    
    // -----   					   
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
      substructure_.getTECDetectors(activeDetIds, sameLayerDetIds_, 0,0,0,0,0);
    }
     

    for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
      try{ 
	selME_.SummaryOfProfileDistr->Fill(i+1,lorentzangleHandle_->getLorentzAngle(sameLayerDetIds_[i]));
      }
      catch(cms::Exception& e){
	edm::LogError("SiStripLorentzAngleDQM")
	  << "[SiStripLorentzAngleDQM::fillMEsForLayer] cms::Exception accessing lorentzangleHandle_->getLorentzAngle() for detId "
	  << selDetId_
	  << " :  "
	  << e.what() ;
      } 
    } 
    
    std::string hSummaryOfCumul_description;
    hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("SummaryOfCumul_description");
    
    std::string hSummaryOfCumul_name; 
    
    if( subDetId_<3 || subDetId_>6 ){ 
      edm::LogError("SiStripLorentzAngleDQM")
	<< "[SiStripLorentzAngleDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
	<< subDetId_ << " no folder set!" 
	<< std::endl;
      return;
    }
    
    hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getLayerNameAndId(selDetId_).first, "") ;
    
    for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
      try{ 
	selME_.SummaryOfCumulDistr->Fill(lorentzangleHandle_->getLorentzAngle(sameLayerDetIds_[i]));
      }
      catch(cms::Exception& e){
	edm::LogError("SiStripLorentzAngleDQM")
	  << "[SiStripLorentzAngleDQM::fillMEsForLayer] cms::Exception accessing lorentzangleHandle_->getLorentzAngle() for detId "
	  << selDetId_
	  << " :  "
	  << e.what() ;
      } 
    } 
  } //FILLING FOR LAYER LEVEL
  
}  
// -----
 
