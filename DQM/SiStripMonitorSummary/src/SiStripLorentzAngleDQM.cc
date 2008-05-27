#include "DQM/SiStripMonitorSummary/interface/SiStripLorentzAngleDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// -----
SiStripLorentzAngleDQM::SiStripLorentzAngleDQM(const edm::EventSetup & eSetup,
                                         edm::ParameterSet const& hPSet,
                                         edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){
					 
  eSetup_.get<SiStripLorentzAngleRcd>().get(lorentzangleHandle_);
}
// -----



// -----
SiStripLorentzAngleDQM::~SiStripLorentzAngleDQM(){}
// -----


// -----
void SiStripLorentzAngleDQM::fillSummaryMEs(){
   
  std::vector<uint32_t> DetIds;
  
  std::map<uint32_t,float>::const_iterator LAMapIter_;
  std::map<uint32_t,float> LAMap_ = lorentzangleHandle_->getLorentzAngles(); 
  
  std::vector<uint32_t> cabledModules_ = getCabledModules(); 
  
  for(std::vector<uint32_t>::const_iterator cablingIter_ = cabledModules_.begin();
                                           cablingIter_!= cabledModules_.end(); cablingIter_++){
    
    unsigned int cablingId;
    cablingId=*cablingIter_;
    LAMapIter_=LAMap_.find(cablingId);    
    
    if (LAMapIter_!=LAMap_.end() && (*LAMapIter_).first==cablingId) { 
      DetIds.push_back(*cablingIter_);
    }
  }

  selectedDetIds = selectModules(DetIds);  

  
  // -----
  // LA on layer-level : fill at once all detIds belonging to same layer when encountering first detID in the layer 
                                                
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
void SiStripLorentzAngleDQM::fillMEsForLayer( std::map<uint32_t, ModMEs> selMEsMap_, uint32_t selDetId_){  
  
  SiStripHistoId hidmanager;
      
  std::string hSummaryOfProfile_description;
  hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");
      
  std::string hSummary_name; 
  
  // ----
  int subDetId_ = ((selDetId_>>25)&0x7);
  
  if( subDetId_<3 || subDetId_>6 ){ 
    edm::LogError("SiStripLorentzAngleDQM")
       << "[SiStripLorentzAngleDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
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
  
  if(selSubDetId_==3){  //  TIB
    substructure_.getTIBDetectors(selectedDetIds, sameLayerDetIds_, TIBDetId(selDetId_).layerNumber(),0,0,0);  
  }
  else if(selSubDetId_==4){  // TID
    substructure_.getTIDDetectors(selectedDetIds, sameLayerDetIds_, 0,0,0,0);
  }
  else if(selSubDetId_==5){  // TOB
    substructure_.getTOBDetectors(selectedDetIds, sameLayerDetIds_, TOBDetId(selDetId_).layerNumber(),0,0);
  }
  else if(selSubDetId_==6){  // TEC
    substructure_.getTECDetectors(selectedDetIds, sameLayerDetIds_, 0,0,0,0,0,0);
  }
     
  //sort(sameLayerDetIds_.begin(),sameLayerDetIds_.end()); 
        
  // -----

  if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
  
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
  } 
 
  if( CondObj_fillId_ =="ProfileAndCumul" || CondObj_fillId_ =="onlyCumul"){

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

    hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, 
                                                       "layer", 
						       getLayerNameAndId(selDetId_).first, 
						       "") ;

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
  }


}  
// -----
 
