#include "DQM/SiStripMonitorSummary/interface/SiStripThresholdDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TCanvas.h"

// -----
SiStripThresholdDQM::SiStripThresholdDQM(const edm::EventSetup & eSetup,
                                         edm::ParameterSet const& hPSet,
                                         edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){
  WhichThreshold=hPSet.getParameter<std::string>("WhichThreshold");

  // Build the Histo_TkMap:
  if(HistoMaps_On_ ){
    if(WhichThreshold=="Low") Tk_HM_L = new TkHistoMap("SiStrip/Histo_Map","LowThresh_TkMap",0.);
    if(WhichThreshold=="High") Tk_HM_H = new TkHistoMap("SiStrip/Histo_Map","HighThresh_TkMap",0.);
  }
}

// -----



// -----
SiStripThresholdDQM::~SiStripThresholdDQM(){}
// -----


// -----
void SiStripThresholdDQM::getActiveDetIds(const edm::EventSetup & eSetup){
  
  getConditionObject(eSetup);
  thresholdHandle_->getDetIds(activeDetIds);
}
// -----

//=====================================================================================


// -----
void SiStripThresholdDQM::fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){
   
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  ModMEs CondObj_ME;
  
    
  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
      detIter_!= selectedDetIds.end();detIter_++){
      
    fillMEsForDet(CondObj_ME,*detIter_,tTopo);
      
  }
}    
// -----



//======================================================================================
// -----



void SiStripThresholdDQM::fillMEsForDet(const ModMEs& _selModME_, uint32_t selDetId_, const TrackerTopology* tTopo){
  ModMEs selModME_ = _selModME_;
  std::vector<uint32_t> DetIds;
  thresholdHandle_->getDetIds(DetIds);
    
 
  SiStripThreshold::Range ThresholdRange = thresholdHandle_->getRange(selDetId_);
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;


    getModMEs(selModME_,selDetId_,tTopo);


    
  for( int istrip=0;istrip<nStrip;++istrip){
      if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
	if(WhichThreshold=="Low")
	  selModME_.ProfileDistr->Fill(istrip+1,thresholdHandle_->getData(istrip,ThresholdRange).getLth());
	if(WhichThreshold=="High")
	  selModME_.ProfileDistr->Fill(istrip+1,thresholdHandle_->getData(istrip,ThresholdRange).getHth());
      }
  }// istrip
  
    
} 

   
// -----



//=======================================================================================
// -----
void SiStripThresholdDQM::fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){
   
   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHandle;
   es.get<IdealGeometryRecord>().get(tTopoHandle);
   const TrackerTopology* const tTopo = tTopoHandle.product();

   for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
       detIter_!= selectedDetIds.end();detIter_++){
     fillMEsForLayer(/*SummaryMEsMap_,*/ *detIter_,tTopo);

   }
  
}    
// -----


//=======================================================================================
// -----
void SiStripThresholdDQM::fillMEsForLayer( /*std::map<uint32_t, ModMEs> selMEsMap_,*/ uint32_t selDetId_, const TrackerTopology* tTopo){

  // ----
  int subdetectorId_ = ((selDetId_>>25)&0x7);
    
  if( subdetectorId_<3 || subdetectorId_>6 ){ 
    edm::LogError("SiStripThresholdDQM")
      << "[SiStripThresholdDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
      << subdetectorId_ << " no folder set!" 
      << std::endl;
    return;
  }
  // ----
     
   std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = SummaryMEsMap_.find(getLayerNameAndId(selDetId_,tTopo).second);
   ModMEs selME_;
   if ( selMEsMapIter_ != SummaryMEsMap_.end())
     selME_ =selMEsMapIter_->second;

   getSummaryMEs(selME_,selDetId_,tTopo);

    
   SiStripThreshold::Range ThresholdRange = thresholdHandle_->getRange(selDetId_);
  
   int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;
  
     SiStripHistoId hidmanager;
  
  
  if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")){
    // --> profile summary    
  
    std::string hSummaryOfProfile_description;
    hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");
  
    std::string hSummaryOfProfile_name; 
  
    hSummaryOfProfile_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getLayerNameAndId(selDetId_,tTopo).first, "") ;

  
    
    for( int istrip=0;istrip<nStrip;++istrip){
    
	if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
	  if(WhichThreshold=="Low"){
	    selME_.SummaryOfProfileDistr->Fill(istrip+1,thresholdHandle_->getData(istrip,ThresholdRange).getLth());
	    if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) fillTkMap(selDetId_, thresholdHandle_->getData(istrip,ThresholdRange).getLth());
	  }
	  if(WhichThreshold=="High"){
	    selME_.SummaryOfProfileDistr->Fill(istrip+1,thresholdHandle_->getData(istrip,ThresholdRange).getHth());
	    if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) fillTkMap(selDetId_, thresholdHandle_->getData(istrip,ThresholdRange).getHth());
	  }
	}
    }// istrip	



  }//if Fill

 

  if(hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")){
    
    // --> summary  
    
    std::string hSummary_description;
    hSummary_description  = hPSet_.getParameter<std::string>("Summary_description");
  
    std::string hSummary_name; 
    hSummary_name = hidmanager.createHistoLayer(hSummary_description, 
						"layer", 
						getLayerNameAndId(selDetId_,tTopo).first, 
						"") ;


    float meanLowThreshold=0;
    float meanHighThreshold=0;
  
    for( int istrip=0;istrip<nStrip;++istrip){
    
	meanLowThreshold = meanLowThreshold + thresholdHandle_->getData(istrip,ThresholdRange).getLth();
	meanHighThreshold = meanHighThreshold + thresholdHandle_->getData(istrip,ThresholdRange).getHth();
    }//istrip
  
  
    meanLowThreshold = meanLowThreshold/nStrip;
    meanHighThreshold = meanHighThreshold/nStrip;
  
  
    // -----
    // get detIds belonging to same layer to fill X-axis with detId-number
  
    std::vector<uint32_t> sameLayerDetIds_=GetSameLayerDetId(activeDetIds,selDetId_,tTopo);
  
    unsigned int iBin=0;
    for(unsigned int i=0;i<sameLayerDetIds_.size();i++){
      if(sameLayerDetIds_[i]==selDetId_){iBin=i+1;}
    }   
  
    if(WhichThreshold=="Low"){
      selME_.SummaryDistr->Fill(iBin,meanLowThreshold);
      if(HistoMaps_On_ ) Tk_HM_L->fill(selDetId_, meanLowThreshold);
    }
    if(WhichThreshold=="High"){
      selME_.SummaryDistr->Fill(iBin,meanHighThreshold);
      if(HistoMaps_On_ )Tk_HM_H->fill(selDetId_, meanHighThreshold);
    }
  
  }//if Fill ...
  


}  
// -----

