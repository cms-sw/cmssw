#include "DQM/SiStripMonitorSummary/interface/SiStripBackPlaneCorrectionDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TCanvas.h"

// -----
SiStripBackPlaneCorrectionDQM::SiStripBackPlaneCorrectionDQM(const edm::EventSetup & eSetup,
					       edm::ParameterSet const& hPSet,
					       edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){

  // Build the Histo_TkMap:
  if(HistoMaps_On_ ) Tk_HM_ = new TkHistoMap("SiStrip/Histo_Map","BP_TkMap",0.);

}
// -----



// -----
SiStripBackPlaneCorrectionDQM::~SiStripBackPlaneCorrectionDQM(){}
// -----


// -----
void SiStripBackPlaneCorrectionDQM::getActiveDetIds(const edm::EventSetup & eSetup){
  
  getConditionObject(eSetup);

  std::map<uint32_t,float>::const_iterator BPMapIter_;
  std::map<uint32_t,float> BPMap_ = bpcorrectionHandle_->getBackPlaneCorrections();

  for( BPMapIter_ = BPMap_.begin();
       BPMapIter_!= BPMap_.end(); BPMapIter_++){

    activeDetIds.push_back((*BPMapIter_).first);
  }

}
// -----


// -----
void SiStripBackPlaneCorrectionDQM::fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // -----
  // BP on layer-level : fill at once all detIds belonging to same layer when encountering first detID in the layer 

  bool fillNext = true; 
  for(unsigned int i=0;i<selectedDetIds.size();i++){					    
    int subDetId_ = ((selectedDetIds[i]>>25)&0x7);
    if( subDetId_<3 ||subDetId_>6 ){ 
      edm::LogError("SiStripBackPlaneCorrection")
	<< "[SiStripBackPlaneCorrection::fillSummaryMEs] WRONG INPUT : no such subdetector type : "
	<< subDetId_ << " and detId " << selectedDetIds[i] << " therefore no filling!" 
	<< std::endl;
    }    
    else if (SummaryOnLayerLevel_On_) {    
      if( fillNext) { fillMEsForLayer(/*SummaryMEsMap_,*/ selectedDetIds[i],tTopo);} 
      if( getLayerNameAndId(selectedDetIds[i+1],tTopo)==getLayerNameAndId(selectedDetIds[i],tTopo)){ fillNext=false;}
      else { fillNext=true;}
    } 
    else if (SummaryOnStringLevel_On_) {
      if( fillNext) { fillMEsForLayer(/*SummaryMEsMap_,*/ selectedDetIds[i],tTopo);} 
      if( getStringNameAndId(selectedDetIds[i+1],tTopo)==getStringNameAndId(selectedDetIds[i],tTopo)){ fillNext=false;}
      else { fillNext=true;}
    } 
  }

  for (std::map<uint32_t, ModMEs>::iterator iter=SummaryMEsMap_.begin(); iter!=SummaryMEsMap_.end(); iter++){

    ModMEs selME;
    selME = iter->second;

    if(SummaryOnStringLevel_On_){

      if (fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")){

	TCanvas c1("c1");
	selME.SummaryOfProfileDistr->getTProfile()->Draw();
	std::string name (selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
	name+=".png";
	c1.Print(name.c_str());
      }

      if (fPSet_.getParameter<bool>("OutputCumulativeSummaryAtLayerLevelAsImage")){

	TCanvas c2("c2");
	selME.SummaryOfCumulDistr->getTH1()->Draw();
	std::string name2 (selME.SummaryOfCumulDistr->getTH1()->GetTitle());
	name2+=".png";
	c2.Print(name2.c_str());
      }

    }
    else{
      if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel") && fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")){

	TCanvas c1("c1");
	selME.SummaryOfProfileDistr->getTProfile()->Draw();
	std::string name (selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
	name+=".png";
	c1.Print(name.c_str());
      }

      if(hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel") && fPSet_.getParameter<bool>("OutputCumulativeSummaryAtLayerLevelAsImage")){

	TCanvas c1("c1");
	selME.SummaryOfCumulDistr->getTH1()->Draw();
	std::string name (selME.SummaryOfCumulDistr->getTH1()->GetTitle());
	name+=".png";
	c1.Print(name.c_str());
      }
    }

  }

}
// -----



// -----
void SiStripBackPlaneCorrectionDQM::fillMEsForLayer( /*std::map<uint32_t, ModMEs> selMEsMap_,*/ uint32_t selDetId_, const TrackerTopology* tTopo){

  SiStripHistoId hidmanager;

      
  std::string hSummaryOfProfile_description;
  hSummaryOfProfile_description  = hPSet_.getParameter<std::string>("SummaryOfProfile_description");

  std::string hSummary_name; 

  int subDetId_ = ((selDetId_>>25)&0x7);
  
  if( subDetId_<3 || subDetId_>6 ){ 
    edm::LogError("SiStripBackPlaneCorrectionDQM")
      << "[SiStripBackPlaneCorrectionDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
      << subDetId_ << " no folder set!" 
      << std::endl;
    return;
  }

  uint32_t selSubDetId_ =  ((selDetId_>>25)&0x7);
  SiStripSubStructure substructure_;
  
  std::vector<uint32_t> sameLayerDetIds_;
  sameLayerDetIds_.clear();

  if (SummaryOnStringLevel_On_) {  //FILLING FOR STRING LEVEL

    hSummary_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getStringNameAndId(selDetId_,tTopo).first, "") ;
    std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = SummaryMEsMap_.find(getStringNameAndId(selDetId_,tTopo).second);
    
    ModMEs selME_;
    if ( selMEsMapIter_ != SummaryMEsMap_.end())
      selME_ =selMEsMapIter_->second;

    getSummaryMEs(selME_,selDetId_,tTopo);
  
    // -----   					   
    sameLayerDetIds_.clear();
   
    if(selSubDetId_==3){  //  TIB
      if(tTopo->tibIsInternalString(selDetId_)){
	substructure_.getTIBDetectors(activeDetIds, sameLayerDetIds_, tTopo->tibLayer(selDetId_),0,1,tTopo->tibString(selDetId_));
      }
      if(tTopo->tibIsExternalString(selDetId_)){
	substructure_.getTIBDetectors(activeDetIds, sameLayerDetIds_, tTopo->tibLayer(selDetId_),0,2,tTopo->tibString(selDetId_));
      } 
    }
    else if(selSubDetId_==4){  // TID
      substructure_.getTIDDetectors(activeDetIds, sameLayerDetIds_, 0,0,0,0);
    }
    else if(selSubDetId_==5){  // TOB
      substructure_.getTOBDetectors(activeDetIds, sameLayerDetIds_, tTopo->tobLayer(selDetId_),0,tTopo->tobRod(selDetId_));
    }
    else if(selSubDetId_==6){  // TEC
      substructure_.getTECDetectors(activeDetIds, sameLayerDetIds_, 0,0,0,0,0,0);
    }
 
    // -----

    for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
	selME_.SummaryOfProfileDistr->Fill(i+1,bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i]));

	// Fill the Histo_TkMap+TkMap with the BP:
        if(HistoMaps_On_ ) Tk_HM_->fill(sameLayerDetIds_[i], bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i]));

	  std::cout<<sameLayerDetIds_[i]<<"\t"<<bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i])<<std::endl;

	if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")){
	  fillTkMap(sameLayerDetIds_[i], bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i]));
	}
	
    } 
    
    std::string hSummaryOfCumul_description;
    hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("SummaryOfCumul_description");
    
    std::string hSummaryOfCumul_name; 
    
    if( subDetId_<3 || subDetId_>6 ){ 
      edm::LogError("SiStripBackPlaneCorrectionDQM")
	<< "[SiStripBackPlaneCorrectionDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
	<< subDetId_ << " no folder set!" 
	<< std::endl;
      return;
    }
    
    hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getStringNameAndId(selDetId_,tTopo).first, "") ;
    
    for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
	selME_.SummaryOfCumulDistr->Fill(bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i]));
    } 
  } //FILLING FOR STRING LEVEL
  
  
  else { //FILLING FOR LAYER LEVEL

    std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = SummaryMEsMap_.find(getLayerNameAndId(selDetId_,tTopo).second);
    
    ModMEs selME_;
    if ( selMEsMapIter_ != SummaryMEsMap_.end())
      selME_ =selMEsMapIter_->second;
    
    getSummaryMEs(selME_,selDetId_,tTopo);
    
    if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")){

      hSummary_name = hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getLayerNameAndId(selDetId_,tTopo).first, "") ;    
    
      // -----   					   
      sameLayerDetIds_.clear();

      sameLayerDetIds_=GetSameLayerDetId(activeDetIds,selDetId_,tTopo);     

      for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
	  selME_.SummaryOfProfileDistr->Fill(i+1,bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i]));

	  // Fill the Histo_TkMap with BP:
	  if(HistoMaps_On_ ) Tk_HM_->fill(sameLayerDetIds_[i], bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i]));

	if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")){
	  fillTkMap(sameLayerDetIds_[i], bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i]));

	}

      } 
    }//if Fill ...

    if(hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel")){

      std::string hSummaryOfCumul_description;
      hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("SummaryOfCumul_description");
    
      std::string hSummaryOfCumul_name; 
    
      if( subDetId_<3 || subDetId_>6 ){ 
	edm::LogError("SiStripBackPlaneCorrectionDQM")
	  << "[SiStripBackPlaneCorrectionDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
	  << subDetId_ << " no folder set!" 
	  << std::endl;
	return;
      }
    
      hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getLayerNameAndId(selDetId_,tTopo).first, "") ;
    
      for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
	  selME_.SummaryOfCumulDistr->Fill(bpcorrectionHandle_->getBackPlaneCorrection(sameLayerDetIds_[i]));
      }
    }//if Fill ... 
  } //FILLING FOR LAYER LEVEL
  
}  
// -----
 
