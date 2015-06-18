#include "DQM/SiStripMonitorSummary/interface/SiStripNoisesDQM.h"


#include "DQMServices/Core/interface/MonitorElement.h"

#include "TCanvas.h"

// -----
SiStripNoisesDQM::SiStripNoisesDQM(const edm::EventSetup & eSetup,
                                   edm::ParameterSet const& hPSet,
                                   edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){  
  gainRenormalisation_ = hPSet_.getParameter<bool>("GainRenormalisation");
  simGainRenormalisation_ = hPSet_.getParameter<bool>("SimGainRenormalisation");
  if( gainRenormalisation_ && !simGainRenormalisation_){ eSetup.get<SiStripApvGainRcd>().get(gainHandle_);}
  if( simGainRenormalisation_){ eSetup.get<SiStripApvGainSimRcd>().get(gainHandle_);}


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

}

// -----
void SiStripNoisesDQM::fillMEsForDet(const ModMEs& _selModME_, uint32_t selDetId_, const TrackerTopology* tTopo){
  ModMEs selModME_ = _selModME_;
  std::vector<uint32_t> DetIds;
  noiseHandle_->getDetIds(DetIds);

  SiStripNoises::Range noiseRange = noiseHandle_->getRange(selDetId_);
  
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;

  getModMEs(selModME_,selDetId_, tTopo);
  
  float gainFactor;
  float stripnoise;

  SiStripApvGain::Range gainRange;
  if( gainRenormalisation_ ||  simGainRenormalisation_ ){
    gainRange = gainHandle_->getRange(selDetId_);
  }

  for( int istrip=0;istrip<nStrip;++istrip){
    if( gainRenormalisation_ ||  simGainRenormalisation_ )
      gainFactor= gainHandle_ ->getStripGain(istrip,gainRange) ? gainHandle_ ->getStripGain(istrip,gainRange) : 1.;
    else
      gainFactor=1;

      stripnoise=noiseHandle_->getNoise(istrip,noiseRange)/gainFactor;
    if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
      selModME_.ProfileDistr->Fill(istrip+1,stripnoise);
    }
    if( CondObj_fillId_ =="onlyCumul" || CondObj_fillId_ =="ProfileAndCumul"){
      selModME_.CumulDistr->Fill(stripnoise);
    } 
  } //istrip

}
  

// -----
//FIXME too long. factorize this method. 
//FIXME the number of lines of code in the derived classes should be reduced ONLY at what cannot be done in the base class because of the specific implementation
//FIXME of the derived class. Moreover, several loops on the same quantities should be avoided...

void SiStripNoisesDQM::fillMEsForLayer( /*std::map<uint32_t, ModMEs> selMEsMap_,*/ uint32_t selDetId_, const TrackerTopology* tTopo){

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
  
  std::map<uint32_t, ModMEs>::iterator selMEsMapIter_  = SummaryMEsMap_.find(getLayerNameAndId(selDetId_,tTopo).second);
  ModMEs selME_;
  if ( selMEsMapIter_ != SummaryMEsMap_.end())
    selME_ =selMEsMapIter_->second;
  getSummaryMEs(selME_,selDetId_,tTopo);
  
  SiStripNoises::Range noiseRange = noiseHandle_->getRange(selDetId_);
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;

  float stripnoise=-1.;
  float meanNoise=0;
  int Nbadstrips=0;

  SiStripApvGain::Range gainRange;
  if(gainRenormalisation_ ||  simGainRenormalisation_ ){
    gainRange = gainHandle_->getRange(selDetId_);
  }
  float gainFactor=1;
  
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
  }
  if(hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel")){
    std::string hSummaryOfCumul_description;
    hSummaryOfCumul_description  = hPSet_.getParameter<std::string>("Cumul_description");
    
    std::string hSummaryOfCumul_name; 
    hSummaryOfCumul_name = hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getStringNameAndId(selDetId_,tTopo).first, "") ;    
  }   
  if(hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")){
    // --> cumul summary    
    std::string hSummary_description;
    hSummary_description  = hPSet_.getParameter<std::string>("Summary_description");
    
    std::string hSummary_name; 
    hSummary_name = hidmanager.createHistoLayer(hSummary_description, 
						"layer", 
						getLayerNameAndId(selDetId_,tTopo).first, 
						"") ;
  }


  for( int istrip=0;istrip<nStrip;++istrip){

    if(gainRenormalisation_  || simGainRenormalisation_ ){
      gainFactor= gainHandle_ ->getStripGain(istrip,gainRange) ? gainHandle_ ->getStripGain(istrip,gainRange) : 1.;
    } else{
      gainFactor=1.;
    }
    
    stripnoise=noiseHandle_->getNoise(istrip,noiseRange)/gainFactor;
    meanNoise+=stripnoise;
    if(hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")){
      if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){	
	selME_.SummaryOfProfileDistr->Fill(istrip+1,stripnoise);
      }
    }

    if(hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel")){
      if( CondObj_fillId_ =="onlyCumul" || CondObj_fillId_ =="ProfileAndCumul"){
	selME_.SummaryOfCumulDistr->Fill(stripnoise);
      }
    }

    // Fill the TkMap
    if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")){
      fillTkMap(selDetId_, stripnoise);   
    }
  
  } //istrip


  if(hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")){
    
    meanNoise = meanNoise/(nStrip-Nbadstrips);
    // get detIds belonging to same layer to fill X-axis with detId-number
    
    std::vector<uint32_t> sameLayerDetIds_;
    sameLayerDetIds_=GetSameLayerDetId(activeDetIds,selDetId_,tTopo);
    
    std::vector<uint32_t>::const_iterator ibound=lower_bound(sameLayerDetIds_.begin(),sameLayerDetIds_.end(),selDetId_);
    if(ibound!=sameLayerDetIds_.end() && *ibound==selDetId_)
      selME_.SummaryDistr->Fill(ibound-sameLayerDetIds_.begin()+1,meanNoise);
    

    // Fill the Histo_TkMap with the mean Noise:
    if(HistoMaps_On_ ){Tk_HM_->fill(selDetId_, meanNoise); }


    //Check the axis range for tkmap, and in case redefine;
    int intNoise = int(meanNoise);
    if(intNoise+1 > (int)tkMapScaler.size()){
      tkMapScaler.resize(intNoise+1,0);
    }
    tkMapScaler[intNoise]++;
      
  } 

}

