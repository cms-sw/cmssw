#include "DQM/SiStripMonitorSummary/interface/SiStripQualityDQM.h"


#include "DQMServices/Core/interface/MonitorElement.h"

// -----
SiStripQualityDQM::SiStripQualityDQM(const edm::EventSetup & eSetup,
                                               edm::ParameterSet const& hPSet,
                                               edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){
  qualityLabel_ = fPSet.getParameter<std::string>("StripQualityLabel");
  eSetup_.get<SiStripQualityRcd>().get(qualityLabel_, qualityHandle_);
  }

// -----

// -----
SiStripQualityDQM::~SiStripQualityDQM(){}
// -----


// -----
void SiStripQualityDQM::fillModMEs(){
 				         
  std::vector<uint32_t> DetIds;
  qualityHandle_->getDetIds(DetIds);
  
  selectedDetIds = selectModules(DetIds);
  
 
  ModMEs CondObj_ME;
  
  for(std::vector<uint32_t>::const_iterator detIter_ =selectedDetIds.begin();
                                            detIter_!=selectedDetIds.end();++detIter_){
   fillMEsForDet(CondObj_ME,*detIter_);
  } 
} 

   
// -----
void SiStripQualityDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_){
				       
  SiStripBadStrip::RegistryIterator rbegin = qualityHandle_->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator rend   = qualityHandle_->getRegistryVectorEnd();
  
  uint32_t detid;
    
  if (rbegin==rend) return;

  for (SiStripBadStrip::RegistryIterator rp=rbegin; rp != rend; ++rp) {
  
    detid = rp->detid;
    
    if (detid != selDetId_) { continue;}
    
    getModMEs(selModME_,selDetId_);

    SiStripBadStrip::Range range = SiStripBadStrip::Range(qualityHandle_->getDataVectorBegin()+rp->ibegin , 
                                                          qualityHandle_->getDataVectorBegin()+rp->iend );
      
    SiStripBadStrip::ContainerIterator it=range.first;
      
    for(;it!=range.second;++it){
      unsigned int value=(*it);
      short str_start = qualityHandle_->decode(value).firstStrip;
      short str_end   = str_start + qualityHandle_->decode(value).range;
     
      if ( qualityHandle_->decode(value).flag ==0){ // currently always the case for bad strips
	for ( short isr = str_start; isr < str_end + 1; isr++) { 
	 if( CondObj_fillId_ =="onlyProfile" || CondObj_fillId_ =="ProfileAndCumul"){
           if (isr <= (selModME_.ProfileDistr->getNbinsX()-1)) selModME_.ProfileDistr->Fill(isr+1, 1.0);
	 }  
        }
      }
      
      
    } // it
  } // rp
  
}

void SiStripQualityDQM::fillSummaryMEs(){}
  
