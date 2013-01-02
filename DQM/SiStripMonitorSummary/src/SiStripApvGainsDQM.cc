#include "DQM/SiStripMonitorSummary/interface/SiStripApvGainsDQM.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "TCanvas.h"

// -----
SiStripApvGainsDQM::SiStripApvGainsDQM(const edm::EventSetup & eSetup,
                                       edm::ParameterSet const& hPSet,
                                       edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup,hPSet, fPSet){

  // Build the Histo_TkMap:
  if(HistoMaps_On_ ) Tk_HM_ = new TkHistoMap("SiStrip/Histo_Map","MeanApvGain_TkMap",0.);

}
// -----

// -----
SiStripApvGainsDQM::~SiStripApvGainsDQM(){}
// -----


// -----
void SiStripApvGainsDQM::getActiveDetIds(const edm::EventSetup & eSetup){

  getConditionObject(eSetup);
  gainHandle_->getDetIds(activeDetIds);

}
// -----




// -----
void SiStripApvGainsDQM::fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){

  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);

  ModMEs CondObj_ME;

  for(std::vector<uint32_t>::const_iterator detIter_ =selectedDetIds.begin();
      detIter_!=selectedDetIds.end();++detIter_){
    fillMEsForDet(CondObj_ME,*detIter_,tTopo);
  }  
}  

  
// -----
void SiStripApvGainsDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){
  
  std::vector<uint32_t> DetIds;
  gainHandle_->getDetIds(DetIds);

  SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
  
  int nApv =  reader->getNumberOfApvsAndStripLength(selDetId_).first;
    
  getModMEs(selModME_,selDetId_, tTopo);
 
  for( int iapv=0;iapv<nApv;++iapv){
      if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
        selModME_.ProfileDistr->Fill(iapv+1,gainHandle_->getApvGain(iapv,gainRange));
      }
      if( CondObj_fillId_ =="onlyCumul" || CondObj_fillId_ =="ProfileAndCumul"){
        selModME_.CumulDistr  ->Fill(gainHandle_->getApvGain(iapv,gainRange));
      }

    // Fill the TkMap
    if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")){
      fillTkMap(selDetId_, gainHandle_->getApvGain(iapv,gainRange));
    }

  }
}

// -----
void SiStripApvGainsDQM::fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){
  
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);

  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
      detIter_!= selectedDetIds.end();detIter_++){
    fillMEsForLayer(/*SummaryMEsMap_,*/ *detIter_, tTopo);
  }

  for (std::map<uint32_t, ModMEs>::iterator iter=SummaryMEsMap_.begin(); iter!=SummaryMEsMap_.end(); iter++){

    ModMEs selME;
    selME = iter->second;

    if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel") && fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")){

      TCanvas c1("c1");
      selME.SummaryOfProfileDistr->getTProfile()->Draw();
      std::string name (selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
      name+=".png";
      c1.Print(name.c_str());
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
void SiStripApvGainsDQM::fillMEsForLayer( /*std::map<uint32_t, ModMEs> selMEsMap_, */uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){
    
  int subdetectorId_ = ((selDetId_>>25)&0x7);
  
  if( subdetectorId_<3 ||subdetectorId_>6 ){ 
    edm::LogError("SiStripApvGainsDQM")
      << "[SiStripApvGainsDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
      << subdetectorId_ << " no folder set!" 
      << std::endl;
    return;
  }
  // ----
         
  std::map<uint32_t, ModMEs>::iterator selMEsMapIter_  = SummaryMEsMap_.find(getLayerNameAndId(selDetId_,tTopo).second);
  ModMEs selME_;
  if ( selMEsMapIter_ != SummaryMEsMap_.end())
  selME_ =selMEsMapIter_->second;
  getSummaryMEs(selME_,selDetId_,tTopo);
  
  SiStripApvGain::Range gainRange = gainHandle_->getRange(selDetId_);
  int nApv =  reader->getNumberOfApvsAndStripLength(selDetId_).first;
  
  float meanApvGain=0;

  SiStripHistoId hidmanager;

  if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")){

    // --> profile summary    
    std::string hSummaryOfProfile_description;
    hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");
  
    std::string hSummaryOfProfile_name; 
    hSummaryOfProfile_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, 
							 "layer", 
							 getLayerNameAndId(selDetId_,tTopo).first, "") ;
  
    for( int iapv=0;iapv<nApv;++iapv){
    
	meanApvGain = meanApvGain +gainHandle_ ->getApvGain(iapv,gainRange);
	selME_.SummaryOfProfileDistr->Fill(iapv+1,gainHandle_->getApvGain(iapv,gainRange));

    // Fill the TkMap
    if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")){
      fillTkMap(selDetId_, gainHandle_->getApvGain(iapv,gainRange));
    }

    }// iapv

    meanApvGain  = meanApvGain/nApv;

    // Fill the TkHistoMap with meanApvgain:
    if(HistoMaps_On_ ) Tk_HM_->setBinContent(selDetId_, meanApvGain);

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


    // get detIds belonging to same layer to fill X-axis with detId-number
  
    std::vector<uint32_t> sameLayerDetIds_;
  
    sameLayerDetIds_.clear();

    sameLayerDetIds_=GetSameLayerDetId(activeDetIds,selDetId_,tTopo);
  
    unsigned int iBin=0;
    for(unsigned int i=0;i<sameLayerDetIds_.size();i++){
      if(sameLayerDetIds_[i]==selDetId_){iBin=i+1;}
    }

    for( int iapv=0;iapv<nApv;++iapv){
	meanApvGain = meanApvGain +gainHandle_ ->getApvGain(iapv,gainRange);
	selME_.SummaryDistr->Fill(iBin,gainHandle_->getApvGain(iapv,gainRange));
    }//iapv
    meanApvGain  = meanApvGain/nApv;

    // Fill the TkHistoMap with meanApvgain:
    //    if(HistoMaps_On_ ) Tk_HM_->setBinContent(selDetId_, meanApvGain);

  }//if Fill ...
}  
// -----





  
