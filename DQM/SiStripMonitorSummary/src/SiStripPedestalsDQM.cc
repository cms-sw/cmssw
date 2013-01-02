#include "DQM/SiStripMonitorSummary/interface/SiStripPedestalsDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TCanvas.h"

// -----
SiStripPedestalsDQM::SiStripPedestalsDQM(const edm::EventSetup & eSetup,
                                         edm::ParameterSet const& hPSet,
                                         edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){

  // Build the Histo_TkMap:
  if(HistoMaps_On_ ) Tk_HM_ = new TkHistoMap("SiStrip/Histo_Map","MeanPed_TkMap",0.);
}
// -----



// -----
SiStripPedestalsDQM::~SiStripPedestalsDQM(){}
// -----


// -----
void SiStripPedestalsDQM::getActiveDetIds(const edm::EventSetup & eSetup){
  
  getConditionObject(eSetup);
  pedestalHandle_->getDetIds(activeDetIds);

}
// -----


// -----
void SiStripPedestalsDQM::fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){
   
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);

  ModMEs CondObj_ME;
  
  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
      detIter_!= selectedDetIds.end();detIter_++){
      
    fillMEsForDet(CondObj_ME,*detIter_,tTopo);
      
  }
}    
// -----




// -----
void SiStripPedestalsDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){
  
  getModMEs(selModME_,selDetId_,tTopo);
  
  SiStripPedestals::Range pedRange = pedestalHandle_->getRange(selDetId_);
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;
  
  for( int istrip=0;istrip<nStrip;++istrip){
      if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
        selModME_.ProfileDistr->Fill(istrip+1,pedestalHandle_->getPed(istrip,pedRange));
      }
  }// istrip
  
}    
// -----



// -----
void SiStripPedestalsDQM::fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){
   
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);

  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
      detIter_!= selectedDetIds.end();detIter_++){
    fillMEsForLayer(/*SummaryMEsMap_,*/ *detIter_,tTopo);
  }

  for (std::map<uint32_t, ModMEs>::iterator iter=SummaryMEsMap_.begin(); iter!=SummaryMEsMap_.end(); iter++){

    ModMEs selME;
    selME = iter->second;

    if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel") && fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")){

      if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){

	TCanvas c1("c1");
	selME.SummaryOfProfileDistr->getTProfile()->Draw();
	std::string name (selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
	name+=".png";
	c1.Print(name.c_str());
      }
    }
    if(hPSet_.getParameter<bool>("FillSummaryAtLayerLevel") && fPSet_.getParameter<bool>("OutputSummaryAtLayerLevelAsImage")){

      TCanvas c1("c1");
      selME.SummaryDistr->getTH1()->Draw();
      std::string name (selME.SummaryDistr->getTH1()->GetTitle());
      name+=".png";
      c1.Print(name.c_str());
    }

  }

}           
  
// -----



// -----
void SiStripPedestalsDQM::fillMEsForLayer( /*std::map<uint32_t, ModMEs> selMEsMap_,*/ uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){

  // ----
  int subdetectorId_ = ((selDetId_>>25)&0x7);
    
  if( subdetectorId_<3 || subdetectorId_>6 ){ 
    edm::LogError("SiStripPedestalsDQM")
      << "[SiStripPedestalsDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
      << subdetectorId_ << " no folder set!" 
      << std::endl;
    return;
  }
  // ----

//     // Cumulative distribution with average Ped value on a layer (not needed):  
     
  std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = SummaryMEsMap_.find(getLayerNameAndId(selDetId_,tTopo).second);
  ModMEs selME_;
  if ( selMEsMapIter_ != SummaryMEsMap_.end())
    selME_ =selMEsMapIter_->second;
  getSummaryMEs(selME_,selDetId_,tTopo);
    
  SiStripPedestals::Range pedRange = pedestalHandle_->getRange(selDetId_);
  
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;
  
  SiStripHistoId hidmanager;

   if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")){
  
     // --> profile summary    
  
     std::string hSummaryOfProfile_description;
     hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");
  
     std::string hSummaryOfProfile_name; 
  
  
    hSummaryOfProfile_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, 
							 "layer", 
							 getLayerNameAndId(selDetId_,tTopo).first, 
							 "") ;
 
    for( int istrip=0;istrip<nStrip;++istrip){
    
	if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
	  selME_.SummaryOfProfileDistr->Fill(istrip+1,pedestalHandle_->getPed(istrip,pedRange));
	}

      //fill the TkMap
    if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")){
      fillTkMap(selDetId_, pedestalHandle_->getPed(istrip,pedRange));
    }

    }// istrip	
   }//if Fill ...

  if(hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")){

    // -->  summary  
  
    std::string hSummary_description;
    hSummary_description  = hPSet_.getParameter<std::string>("Summary_description");
  
    std::string hSummary_name; 
    hSummary_name = hidmanager.createHistoLayer(hSummary_description, 
						"layer", 
						getLayerNameAndId(selDetId_,tTopo).first, 
						"") ;
    float meanPedestal=0;
  
    for( int istrip=0;istrip<nStrip;++istrip){
    
	meanPedestal = meanPedestal + pedestalHandle_->getPed(istrip,pedRange);
    
    }//istrip
  
    meanPedestal = meanPedestal/nStrip;


    // -----
    // get detIds belonging to same layer to fill X-axis with detId-number
  
    std::vector<uint32_t> sameLayerDetIds_;
  
    sameLayerDetIds_.clear();

    sameLayerDetIds_=GetSameLayerDetId(activeDetIds,selDetId_,tTopo);
  
  
    unsigned int iBin=0;
    for(unsigned int i=0;i<sameLayerDetIds_.size();i++){
      if(sameLayerDetIds_[i]==selDetId_){iBin=i+1;}
    }   
  
    selME_.SummaryDistr->Fill(iBin,meanPedestal);

    // Fill the Histo_TkMap with the mean Pedestal:
    if(HistoMaps_On_ ) Tk_HM_->fill(selDetId_, meanPedestal);


  }//if Fill ...
  
  
  
} 
// -----

