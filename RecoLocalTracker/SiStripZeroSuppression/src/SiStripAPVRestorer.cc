#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripAPVRestorer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <iostream>
#include <algorithm>


SiStripAPVRestorer::SiStripAPVRestorer(const edm::ParameterSet& conf):
  quality_cache_id(-1), noise_cache_id(-1), pedestal_cache_id(-1),
  ForceNoRestore_(conf.getParameter<bool>("ForceNoRestore")),
  SelfSelectRestoreAlgo_(conf.getParameter<bool>("SelfSelectRestoreAlgo")),
  InspectAlgo_(conf.getParameter<std::string>("APVInspectMode")),
  RestoreAlgo_(conf.getParameter<std::string>("APVRestoreMode")),
  useRealMeanCM_(conf.getParameter<bool>("useRealMeanCM")),
  fraction_(conf.getParameter<double>("Fraction")),
  deviation_(conf.getParameter<uint32_t>("Deviation")),
  restoreThreshold_(conf.getParameter<double>("restoreThreshold")),
  DeltaCMThreshold_(conf.getParameter<uint32_t>("DeltaCMThreshold")),
  nSigmaNoiseDerTh_(conf.getParameter<uint32_t>("nSigmaNoiseDerTh")),
  consecThreshold_(conf.getParameter<uint32_t>("consecThreshold")),
  hitStripThreshold_(conf.getParameter<uint32_t>("hitStripThreshold")),
  nSmooth_(conf.getParameter<uint32_t>("nSmooth")),
  minStripsToFit_(conf.getParameter<uint32_t>("minStripsToFit")),
  distortionThreshold_(conf.getParameter<uint32_t>("distortionThreshold")),
  CutToAvoidSignal_(conf.getParameter<double>("CutToAvoidSignal")),
  nSaturatedStrip_(conf.getParameter<uint32_t>("nSaturatedStrip")),
  ApplyBaselineCleaner_(conf.getParameter<bool>("ApplyBaselineCleaner")),
  slopeX_(conf.getParameter<int32_t>("slopeX")),
  slopeY_(conf.getParameter<int32_t>("slopeY")),
  CleaningSequence_(conf.getParameter<uint32_t>("CleaningSequence")),
  ApplyBaselineRejection_(conf.getParameter<bool>("ApplyBaselineRejection")),
  MeanCM_(conf.getParameter<int32_t>("MeanCM")),
  filteredBaselineMax_(conf.getParameter<double>("filteredBaselineMax")),
  filteredBaselineDerivativeSumSquare_(conf.getParameter<double>("filteredBaselineDerivativeSumSquare"))  

{
  apvFlags_.clear();
  median_.clear();
  SmoothedMaps_.clear();
  BaselineMap_.erase(BaselineMap_.begin(), BaselineMap_.end());
}


void SiStripAPVRestorer::init(const edm::EventSetup& es){
  uint32_t n_cache_id = es.get<SiStripNoisesRcd>().cacheIdentifier();
  uint32_t q_cache_id = es.get<SiStripQualityRcd>().cacheIdentifier();
  uint32_t p_cache_id = es.get<SiStripPedestalsRcd>().cacheIdentifier();
  
  if(n_cache_id != noise_cache_id) {
    es.get<SiStripNoisesRcd>().get( noiseHandle );
    noise_cache_id = n_cache_id;
  } else {
    noise_cache_id = n_cache_id;
  }
  if(q_cache_id != quality_cache_id) {
    es.get<SiStripQualityRcd>().get( qualityHandle );
    quality_cache_id = q_cache_id;
  }else {
    quality_cache_id = q_cache_id;
  }
  
  if(p_cache_id != pedestal_cache_id) {
		es.get<SiStripPedestalsRcd>().get( pedestalHandle );
		pedestal_cache_id = p_cache_id;
  }else {
    pedestal_cache_id = p_cache_id;
  }
  
}

 
int16_t SiStripAPVRestorer::InspectAndRestore( const uint32_t& detId, const uint16_t& firstAPV, std::vector<int16_t>& rawDigisPedSubtracted,  std::vector<int16_t>& processedRawDigi, const std::vector< std::pair<short,float> >& vmedians ){
  int16_t nAPVFlagged = this->inspect(detId, firstAPV, rawDigisPedSubtracted, vmedians);
  this->restore(firstAPV, processedRawDigi);
  return nAPVFlagged;
}


int16_t SiStripAPVRestorer::inspect( const uint32_t& detId, const uint16_t& firstAPV, std::vector<int16_t>& digis, const std::vector< std::pair<short,float> >& vmedians) {
  
  detId_ = detId;
  
  apvFlagsBool_.clear();
  apvFlagsBoolOverride_.clear();
  apvFlagsBoolOverride_.insert(apvFlagsBoolOverride_.begin(), 6, false);
  apvFlags_.clear();
  apvFlags_.insert(apvFlags_.begin(), 6, "");
  median_.clear();
  median_.insert(median_.begin(), 6, -999);
  badAPVs_.clear();
  badAPVs_.insert(badAPVs_.begin(), 6, false);
  SmoothedMaps_.erase(SmoothedMaps_.begin(),  SmoothedMaps_.end());
  BaselineMap_.erase(BaselineMap_.begin(), BaselineMap_.end()); 
    
  for(size_t i=0; i< vmedians.size(); ++i){
         short APV =  vmedians[i].first;
         median_[APV]= vmedians[i].second;
         badAPVs_[APV] = qualityHandle->IsApvBad(detId_, APV);
  }
	
  if(InspectAlgo_=="BaselineFollower") return this->BaselineFollowerInspect(firstAPV, digis); 
  if(InspectAlgo_=="AbnormalBaseline") return this->AbnormalBaselineInspect(firstAPV, digis);
  if(InspectAlgo_=="Null") return this->NullInspect(firstAPV, digis);
  if(InspectAlgo_=="BaselineAndSaturation") return this->BaselineAndSaturationInspect(firstAPV, digis);
  throw cms::Exception("Unregistered Inspect Algorithm") << "SiStripAPVRestorer possibilities: (Null), (AbnormalBaseline),(BaselineFollower)";
  
}


void SiStripAPVRestorer::restore(const uint16_t& firstAPV, std::vector<int16_t>& digis ) {
	
  if(ForceNoRestore_) return;
  
  for( uint16_t APV=firstAPV; APV< digis.size()/128 + firstAPV; ++APV){
    std::string	algoToUse = *( apvFlags_.begin() + APV );
    
    if ( algoToUse != ""){
      if(!SelfSelectRestoreAlgo_) algoToUse = RestoreAlgo_;
 
      if(algoToUse=="Flat"){
	this->FlatRestore(APV, firstAPV, digis);
      }else if(algoToUse=="BaselineFollower"){
	this->BaselineFollowerRestore(APV, firstAPV, median_[APV], digis);
      }else{
	throw cms::Exception("Unregistered Restore Algorithm") << "SiStripAPVRestorer possibilities: (Flat), (BaselineFollower)";
      }
      
      
    }
  }
  
}


//Inspect method implementation ========================================================================================================================================================================
//======================================================================================================================================================================================================
//======================================================================================================================================================================================================

template<typename T>
inline
int16_t SiStripAPVRestorer::BaselineFollowerInspect(const uint16_t& firstAPV, std::vector<T>& digis){
  std::vector<T> singleAPVdigi;  
  int16_t nAPVflagged = 0;
  
  CMMap::iterator itCMMap;
  if(useRealMeanCM_) itCMMap = MeanCMmap_.find(detId_);
  
  for(uint16_t APV=firstAPV ; APV< digis.size()/128 + firstAPV; ++APV){

    DigiMap smoothedmap;
    smoothedmap.erase(smoothedmap.begin(), smoothedmap.end());

    if(!badAPVs_[APV]){
      float MeanAPVCM = MeanCM_;
      if(useRealMeanCM_&&itCMMap!= MeanCMmap_.end()) MeanAPVCM =(itCMMap->second)[APV];
    
      singleAPVdigi.clear(); 
      for(int16_t strip = (APV-firstAPV)*128; strip < (APV-firstAPV+1)*128; ++strip){
        singleAPVdigi.push_back(digis[strip]); 
      }
   
   
      float DeltaCM = median_[APV] - MeanAPVCM; 
      
      //std::cout << "Delta CM: " << DeltaCM << " CM: " << median_[APV] << " detId " << (uint32_t) detId_ << std::endl; 	
      if(DeltaCM < 0 && std::abs(DeltaCM) > DeltaCMThreshold_){
      
        bool isFlat = FlatRegionsFinder(singleAPVdigi,smoothedmap,APV);
        if(!isFlat){
	      apvFlags_[APV]= "BaselineFollower";    //specify any algo to make the restore
	      nAPVflagged++;
        }
      }	
      
    } 
    SmoothedMaps_.insert(SmoothedMaps_.end(), std::pair<uint16_t, DigiMap>(APV, smoothedmap));
   }
  
  return nAPVflagged;
}

//======================================================================================================================================================================================================
template<typename T>
inline
int16_t SiStripAPVRestorer::BaselineAndSaturationInspect(const uint16_t& firstAPV, std::vector<T>& digis){
  std::vector<T> singleAPVdigi;
  singleAPVdigi.clear();
  
  
  int16_t nAPVflagged = 0;
  
  CMMap::iterator itCMMap;
  if(useRealMeanCM_) itCMMap = MeanCMmap_.find(detId_);
  
  for(uint16_t APV=firstAPV ; APV< digis.size()/128 + firstAPV; ++APV){
     apvFlags_.push_back( "" );
    if(!badAPVs_[APV]){
     float MeanAPVCM = MeanCM_;
     if(useRealMeanCM_&&itCMMap!= MeanCMmap_.end()) MeanAPVCM =(itCMMap->second)[APV];
    
     singleAPVdigi.clear();
   
     uint16_t nSatStrip =0;
     for(int16_t strip = (APV-firstAPV)*128; strip < (APV-firstAPV+1)*128; ++strip){
        singleAPVdigi.push_back(digis[strip]); 
        if(digis[strip] >=1023) ++nSatStrip;
      }
         
     float DeltaCM = median_[APV] -MeanAPVCM; 
    
    
     if(DeltaCM < 0 && std::abs(DeltaCM) > DeltaCMThreshold_&&nSatStrip>= nSaturatedStrip_){
       apvFlags_[APV] = RestoreAlgo_;    //specify any algo to make the restore
       nAPVflagged++;
     } 
    }	
  }
  
  return nAPVflagged;
}

//======================================================================================================================================================================================================
template<typename T>
inline
int16_t SiStripAPVRestorer::AbnormalBaselineInspect( const uint16_t& firstAPV, std::vector<T>& digis){

  SiStripQuality::Range detQualityRange = qualityHandle->getRange(detId_);
  
  typename std::vector<T>::iterator fs;
  
  int16_t nAPVflagged=0;
  
  CMMap::iterator itCMMap;
  if(useRealMeanCM_) itCMMap = MeanCMmap_.find(detId_);
  
  
  int devCount = 0, qualityCount = 0, minstrip = 0; 
 for(uint16_t APV=firstAPV ; APV< digis.size()/128 + firstAPV; ++APV){
    apvFlags_.push_back( "" );
    if(!badAPVs_[APV]){
      float MeanAPVCM = MeanCM_;
      if(useRealMeanCM_&&itCMMap!= MeanCMmap_.end()) MeanAPVCM =(itCMMap->second)[APV];
      for (uint16_t istrip=APV*128; istrip<(APV+1)*128; ++istrip){
        fs = digis.begin() + istrip-firstAPV*128;
        if ( !qualityHandle->IsStripBad(detQualityRange,istrip) ){
	       qualityCount++; 
	       if ( std::abs((int) *fs - MeanAPVCM) > (int)deviation_ ){ 
                devCount++;
	            minstrip = std::min((int) *fs, minstrip);
           }
         }
      }
    
      if( devCount > fraction_ * qualityCount ) {
        apvFlags_[APV] = RestoreAlgo_;      //specify any algo to make the restore
        nAPVflagged++;
      } 
    } 
  }
  
  return nAPVflagged;
  
}


//======================================================================================================================================================================================================
template<typename T>
inline
int16_t SiStripAPVRestorer::NullInspect(const uint16_t& firstAPV, std::vector<T>& digis){

  SiStripQuality::Range detQualityRange = qualityHandle->getRange(detId_);

  typename std::vector<T>::iterator fs;

  int16_t nAPVflagged = 0;

  for(uint16_t APV=firstAPV ; APV< digis.size()/128 + firstAPV; ++APV){
   apvFlags_.push_back( "" );
   if(!badAPVs_[APV]){ 
     int zeroCount = 0, qualityCount = 0; 
     for (uint16_t istrip=APV*128; istrip<(APV+1)*128; ++istrip){
       fs = digis.begin() + istrip-firstAPV*128;
       if ( !qualityHandle->IsStripBad(detQualityRange,istrip) ){
        qualityCount++; 
        if ( (int) *fs < 1 ) zeroCount++;
       }
      }
    
      if( zeroCount > restoreThreshold_ * qualityCount ) {
        apvFlags_[APV] = RestoreAlgo_;     //specify any algo to make the restore
        nAPVflagged++;
      } 
   } 
   }
 
  return nAPVflagged;

}



//Restore method implementation ========================================================================================================================================================================
//======================================================================================================================================================================================================
//======================================================================================================================================================================================================

inline
void SiStripAPVRestorer::BaselineFollowerRestore(const uint16_t& APVn, const uint16_t& firstAPV, const float& median, std::vector<int16_t>& digis){
  //typename std::vector<T>::iterator firstStrip(digis.begin() + APVn*128), lastStrip(firstStrip + 128), actualStrip;
  
  
  std::vector<int16_t> baseline;
  baseline.clear();
  baseline.insert(baseline.begin(),128, 0);
  	
	 
 
    
  //============================= Find Flat Regions & Interpolating the baseline & subtracting the baseline  =================	
  
  if(SmoothedMaps_.size()){
    std::map<uint16_t, DigiMap >::iterator itSmootedMap = SmoothedMaps_.find(APVn);	
    this->BaselineFollower(itSmootedMap->second, baseline, median);
  } else {
    //median=0;
    DigiMap  smoothedpoints;
    std::vector<int16_t> singleAPVdigi;
    singleAPVdigi.clear(); 
    for(int16_t strip = (APVn-firstAPV)*128; strip < (APVn-firstAPV+1)*128; ++strip) singleAPVdigi.push_back(digis[strip]); 
    this->FlatRegionsFinder(singleAPVdigi,smoothedpoints, APVn);
    this->BaselineFollower(smoothedpoints, baseline, median);		
    
  }	

  if(ApplyBaselineRejection_){
    if(CheckBaseline(baseline)) apvFlagsBoolOverride_[APVn] = true;
  }
  
  //============================= subtracting the baseline =============================================
  
  for(int16_t itStrip= 0 ; itStrip< 128; ++itStrip){
    digis[(APVn-firstAPV)*128+itStrip] -= baseline[itStrip] - median;
  }
  
		
  //============================= storing baseline to the map =============================================	
  BaselineMap_.insert(BaselineMap_.end(),  std::pair< uint16_t, std::vector < int16_t> >(APVn, baseline));
  
}


//======================================================================================================================================================================================================
inline
void SiStripAPVRestorer::FlatRestore(const uint16_t& APVn, const uint16_t& firstAPV, std::vector<int16_t>& digis ){
 
  std::vector<int16_t> baseline;
  baseline.clear();
  baseline.insert(baseline.begin(),128, 150);
  baseline[0]=0; baseline[127]=0;
  BaselineMap_.insert(BaselineMap_.end(),  std::pair< uint16_t, std::vector < int16_t> >(APVn, baseline));  
  
  for(int16_t itStrip= 0 ; itStrip< 128; ++itStrip){
    digis[(APVn-firstAPV)*128+itStrip] = baseline[itStrip];
  }
 
  
}



//Baseline calculation implementation ==================================================================================================================================================================
//======================================================================================================================================================================================================
//======================================================================================================================================================================================================

bool inline SiStripAPVRestorer::FlatRegionsFinder(const std::vector<int16_t>& adcs, DigiMap& smoothedpoints, const uint16_t& APVn){
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detId_);
  
  DigiMap consecpoints;
  DigiMapIter itConsecpoints, itSmoothedpoints;
  consecpoints.erase(consecpoints.begin(), consecpoints.end());
  smoothedpoints.erase(smoothedpoints.begin(), smoothedpoints.end());
  
   
  //============================= Height above local minimum ===============================                    
  std::vector<float> adcsLocalMinSubtracted;
  adcsLocalMinSubtracted.clear();
  adcsLocalMinSubtracted.insert(adcsLocalMinSubtracted.begin(), 128,0);
  for(uint32_t istrip=0; istrip<128; ++istrip) {
    float localmin = 999.9;		
    for(uint16_t jstrip=std::max(0,(int)(istrip-nSmooth_/2)); jstrip<std::min(128,(int)(istrip+nSmooth_/2)); ++jstrip) {
      float nextvalue = adcs[jstrip];
      if(nextvalue < localmin) localmin=nextvalue;			
    }
    adcsLocalMinSubtracted[istrip] = adcs[istrip] - localmin;
  }
  
  
  //============================= Find regions with stable slopes ========================
  std::vector<uint16_t> nConsStrip;
  nConsStrip.clear();
  
  //Creating maps with all the neighborhood strip and putting in a nCosntStip vector how many we have
  uint16_t consecStrips=0;
  for(uint32_t istrip=0; istrip<128; ++istrip) {    
    int16_t adc = adcs[istrip]; 
 
   //if( adcsLocalMinSubtracted[istrip] < nSigmaNoiseDerTh_ * (float)noiseHandle->getNoise(istrip+APVn*128,detNoiseRange) && (adc - median) < hitStripThreshold_){
   if( adcsLocalMinSubtracted[istrip] < nSigmaNoiseDerTh_ * (float)noiseHandle->getNoiseFast(istrip+APVn*128,detNoiseRange)){
      consecpoints.insert(consecpoints.end(), std::pair<uint16_t, int16_t >(istrip, adc));
      ++consecStrips;
    }else if (consecStrips >0){
      nConsStrip.push_back(consecStrips);
      consecStrips = 0;
    }    
  }     		

  //to cope with the last flat region of the APV
  if(consecStrips >0) nConsStrip.push_back(consecStrips);

  //removing from the map the fist and last points in wide flat regions and erasing from the map too small regions
  itConsecpoints = consecpoints.begin();
  float MinSmoothValue=20000., MaxSmoothValue=0.;
  for(std::vector<uint16_t>::iterator itnConsStrip = nConsStrip.begin(); itnConsStrip < nConsStrip.end(); ++itnConsStrip){
    
    consecStrips = *itnConsStrip;
    if(consecStrips >=consecThreshold_){
      ++itConsecpoints;  //skipping first point
      uint16_t nFirstStrip = itConsecpoints->first;
      uint16_t nLastStrip;
      float smoothValue = 0.0;
      float stripCount =1;
      for(uint16_t n =0; n < consecStrips-2; ++n){
		smoothValue += itConsecpoints->second;
		if(stripCount == consecThreshold_){
		  smoothValue /= (float)stripCount;
	  	  nLastStrip = nFirstStrip + stripCount -1;				                    
	  	  smoothedpoints.insert(smoothedpoints.end(), std::pair<uint16_t, int16_t >(nFirstStrip, smoothValue));
		  smoothedpoints.insert(smoothedpoints.end(), std::pair<uint16_t, int16_t >(nLastStrip, smoothValue));
		  if(smoothValue > MaxSmoothValue) MaxSmoothValue = smoothValue;
		  if(smoothValue < MinSmoothValue) MinSmoothValue = smoothValue;
		  nFirstStrip = nLastStrip+1;
		  smoothValue=0;
		  stripCount=0;
		}
		++stripCount;
		++itConsecpoints;
     }
     ++itConsecpoints;  //and putting the pointer to the new seies of point 
      
     if(stripCount>1) {
     //if(smoothValue>0){
		--stripCount;
		smoothValue /= (float)(stripCount);
		nLastStrip = nFirstStrip + stripCount -1;
		smoothedpoints.insert(smoothedpoints.end(), std::pair<uint16_t, int16_t >(nFirstStrip, smoothValue));
		smoothedpoints.insert(smoothedpoints.end(), std::pair<uint16_t, int16_t >(nLastStrip, smoothValue));
		if(smoothValue > MaxSmoothValue) MaxSmoothValue = smoothValue;
		if(smoothValue < MinSmoothValue) MinSmoothValue = smoothValue;
     }
   } else{
      for(int n =0; n< consecStrips ; ++n) ++itConsecpoints;
   }
  }
  
  	
  if( (MaxSmoothValue-MinSmoothValue) > distortionThreshold_){
 	if(ApplyBaselineCleaner_) this->BaselineCleaner(adcs, smoothedpoints, APVn);
	return false;
  }
  return true;
}


void inline SiStripAPVRestorer::BaselineCleaner(const std::vector<int16_t>& adcs, DigiMap& smoothedpoints, const uint16_t& APVn){
  
  if(CleaningSequence_==0) {  //default sequence used up to now
    this->Cleaner_HighSlopeChecker(smoothedpoints);
    this->Cleaner_LocalMinimumAdder(adcs, smoothedpoints, APVn);
  }else if(CleaningSequence_==1){     
    this->Cleaner_LocalMinimumAdder(adcs, smoothedpoints, APVn);
    this->Cleaner_HighSlopeChecker(smoothedpoints);
    this->Cleaner_MonotonyChecker(smoothedpoints);
  }else if(CleaningSequence_==2){
    this->Cleaner_HighSlopeChecker(smoothedpoints);
  }else if(CleaningSequence_==3){
    this->Cleaner_LocalMinimumAdder(adcs, smoothedpoints, APVn);
    this->Cleaner_HighSlopeChecker(smoothedpoints);    
  }else{
    this->Cleaner_HighSlopeChecker(smoothedpoints);
    this->Cleaner_LocalMinimumAdder(adcs, smoothedpoints, APVn);
  }     
    
}


void inline SiStripAPVRestorer::Cleaner_MonotonyChecker(DigiMap& smoothedpoints){
//Removing points without monotony
	//--------------------------------------------------------------------------------------------------
         if(smoothedpoints.size() < 3) return;         
         DigiMapIter itSmoothedpoints, itSmoothedpointsNext, itSmoothedpointsNextNext, itSmoothedpointsBegin, itSmoothedpointsEnd;
	 
	itSmoothedpoints=smoothedpoints.begin();
       	while (smoothedpoints.size() > 3 && itSmoothedpoints != --(--(smoothedpoints.end()))) { //while we are not at the last point
	        // get info about current and next points
		itSmoothedpointsNext = itSmoothedpoints;
                ++itSmoothedpointsNext;
		itSmoothedpointsNextNext = itSmoothedpointsNext;
                ++itSmoothedpointsNextNext;
                float adc1 = itSmoothedpoints->second;
		float adc2 = itSmoothedpointsNext->second;
	  	float adc3 = itSmoothedpointsNextNext->second;
                
       	        if((adc2-adc1) > hitStripThreshold_ && (adc2-adc3) > hitStripThreshold_){
		  smoothedpoints.erase(itSmoothedpointsNext);
		}else {
		  ++itSmoothedpoints;
		}
		        
	}
}

void inline SiStripAPVRestorer::Cleaner_LocalMinimumAdder(const std::vector<int16_t>& adcs, DigiMap& smoothedpoints, const uint16_t& APVn){
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detId_);
  //inserting extra point is case of local minimum
	//--------------------------------------------------------------------------------------------------
	// these should be reset now for the point-insertion that follows
	
        DigiMapIter itSmoothedpoints, itSmoothedpointsNext, itSmoothedpointsBegin, itSmoothedpointsEnd; 
	if(smoothedpoints.size() >= 2){
 
    	itSmoothedpointsBegin = smoothedpoints.begin();
    	itSmoothedpointsEnd = --(smoothedpoints.end());
		for(itSmoothedpoints = itSmoothedpointsBegin; itSmoothedpoints != itSmoothedpointsEnd; ++itSmoothedpoints){  
    		itSmoothedpointsNext = itSmoothedpoints;
	       	++itSmoothedpointsNext;
      		float strip1 = itSmoothedpoints->first;
      		float strip2 = itSmoothedpointsNext->first;
      		float adc1 = itSmoothedpoints->second;
      		float adc2 = itSmoothedpointsNext->second;
	  	float m = (adc2 -adc1)/(strip2 -strip1);
    
		//2,4
        	if((strip2 - strip1) >slopeX_ && abs(adc1 -adc2) >slopeY_){
		       	float itStrip = 1;
        		float strip = itStrip + strip1;
 			while(strip < strip2){
				
			float adc = adcs[strip];
                	if( adc < (adc1 + m * itStrip - 2 * (float)noiseHandle->getNoiseFast(strip+APVn*128,detNoiseRange))){
						//std::cout << "applying correction strip: " << strip + APVn*128 << " adc " << adc << " detId: " << detId_ << std::endl;
						smoothedpoints.insert(itSmoothedpointsNext, std::pair<uint16_t, int16_t >(strip,adc));
						++itSmoothedpoints;
						++itSmoothedpointsNext;
						itSmoothedpointsEnd = --(smoothedpoints.end());
					} 
					++itStrip;
					++strip;
				}
			

	    	}
		}
	}
	
	
    itSmoothedpointsBegin = smoothedpoints.begin();
    itSmoothedpointsEnd = --(smoothedpoints.end());
    uint16_t firstStripFlat = itSmoothedpointsBegin->first;
    uint16_t lastStripFlat = itSmoothedpointsEnd->first;
    int16_t firstStripFlatADC= itSmoothedpointsBegin->second;
    int16_t lastStripFlatADC= itSmoothedpointsEnd->second;
    	
    itSmoothedpoints = itSmoothedpointsBegin;
    if(firstStripFlat >3){
		float strip = 0;
       	while(strip < firstStripFlat){
			float adc = adcs[strip];
            if( adc < ( firstStripFlatADC - 2 * (float)noiseHandle->getNoiseFast(strip+APVn*128,detNoiseRange))){
					smoothedpoints.insert(itSmoothedpoints, std::pair<uint16_t, int16_t >(strip,adc));
					++itSmoothedpoints;
			} 
			++strip;
		}
	}
	
	itSmoothedpoints = itSmoothedpointsEnd;
	if(lastStripFlat <125){
		float strip = lastStripFlat+1;
       	while(strip < 128){
			float adc = adcs[strip];
            if( adc < ( lastStripFlatADC - 2 * (float)noiseHandle->getNoiseFast(strip+APVn*128,detNoiseRange))){
            	smoothedpoints.insert(smoothedpoints.end(), std::pair<uint16_t, int16_t >(strip,adc));
			} 
			++strip;
		}
	}
}


void inline SiStripAPVRestorer::Cleaner_HighSlopeChecker(DigiMap& smoothedpoints){
  //Removing points in the slope is too high
	//--------------------------------------------------------------------------------------------------
  
        if(smoothedpoints.size() < 4) return;
	DigiMapIter itSmoothedpoints, itSmoothedpointsNext, itSmoothedpointsBegin, itSmoothedpointsEnd; 
	itSmoothedpoints=smoothedpoints.begin();
	while (smoothedpoints.size() >2 && itSmoothedpoints != --(smoothedpoints.end()) ) { //while we are not at the last point
	  //if(smoothedpoints.size() <2) break;
		// get info about current and next points
		itSmoothedpointsNext = itSmoothedpoints;
		++itSmoothedpointsNext;
		float strip1 = itSmoothedpoints->first;
		float strip2 = itSmoothedpointsNext->first;
		float adc1 = itSmoothedpoints->second;
		float adc2 = itSmoothedpointsNext->second;
	  	float m = (adc2 -adc1)/(strip2 -strip1);
       	
		if (m>2) { // in case of large positive slope, remove next point and try again from same current point
			smoothedpoints.erase(itSmoothedpointsNext);
		} else if (m<-2) { // in case of large negative slope, remove current point and either...
			// move to next point if we have reached the beginning (post-increment to avoid invalidating pointer during erase) or...
			if(itSmoothedpoints==smoothedpoints.begin()) smoothedpoints.erase(itSmoothedpoints++); 
			// try again from the previous point if we have not reached the beginning
			else smoothedpoints.erase(itSmoothedpoints--); 
		} else { // in case of a flat enough slope, continue on to the next point
			itSmoothedpoints++;
		}
		
	}
}

void inline SiStripAPVRestorer::BaselineFollower(DigiMap& smoothedpoints, std::vector<int16_t>& baseline, const float& median){
  
  baseline.clear();
  DigiMapIter itSmoothedpoints;
  
  
  //if not enough points
  if(smoothedpoints.size() < minStripsToFit_){
     baseline.insert(baseline.begin(),128, median);
  } else {
     baseline.insert(baseline.begin(),128, 0);  
    
    DigiMapIter itSmoothedpointsBegin, itSmoothedpointsEnd;
    itSmoothedpointsBegin = smoothedpoints.begin();
    itSmoothedpointsEnd = --(smoothedpoints.end());
    
				
    uint16_t firstStripFlat = itSmoothedpointsBegin->first;
    uint16_t lastStripFlat = itSmoothedpointsEnd->first;
    int16_t firstStripFlatADC= itSmoothedpointsBegin->second;
    int16_t lastStripFlatADC= itSmoothedpointsEnd->second;
   
    //adding here the costant line at the extremities 
    baseline.erase(baseline.begin(), baseline.begin()+firstStripFlat);
    baseline.insert(baseline.begin(), firstStripFlat, firstStripFlatADC);
    
    baseline.erase(baseline.begin()+lastStripFlat, baseline.end());
    baseline.insert(baseline.end(), 128 - lastStripFlat, lastStripFlatADC);
    
    
    //IMPORTANT: the itSmoothedpointsEnd should be at least smaller than smoothedpoints.end() -1
    for(itSmoothedpoints = itSmoothedpointsBegin; itSmoothedpoints != itSmoothedpointsEnd; ++itSmoothedpoints){  
      DigiMapIter itSmoothedpointsNext = itSmoothedpoints;
      ++itSmoothedpointsNext;
      float strip1 = itSmoothedpoints->first;
      float strip2 = itSmoothedpointsNext->first;
      float adc1 = itSmoothedpoints->second;
      float adc2 = itSmoothedpointsNext->second;
     
      baseline[strip1] = adc1;
      baseline[strip2] = adc2;
      float m = (adc2 -adc1)/(strip2 -strip1);
      uint16_t itStrip = strip1 +1;
      float stripadc = adc1 + m; 
      while(itStrip < strip2){
		baseline[itStrip] = stripadc;
		++itStrip;
		stripadc+=m;
      }
      
    }
    
  }
}


bool SiStripAPVRestorer::CheckBaseline(const std::vector<int16_t> &baseline) const
{
	// The Savitzky-Golay (S-G) filter of any left length nL, right
	// length nR, and order m, and with an optional opt equals the
	// derivative order (0 for the magnitude, 1 for the first
	// derivative, etc.) can be calculated using the following
	// Mathematica code:
	//
	// SavitzkyGolay[m_?IntegerQ, {nL_?IntegerQ, nR_?IntegerQ},
	//   opt___] := Module[
	//   {a, d},
	//   d = If[opt === Null, 0, If[IntegerQ[opt], opt, 0]]; 
	//   a = Table[
	//     If[i == 0 && j == 0, 1, i^j], {i, -nL, nR}, {j, 0, 
	//      m}]; (Inverse[Transpose[a].a].Transpose[a])[[d + 1]]];
	//
	// The following coefficients can be then calculated by:
	//
	// N[Join[Table[SavitzkyGolay[2, {k, 16}], {k, 0, 16}], 
	//   Table[SavitzkyGolay[2, {16, k}], {k, 15, 0, -1}]]]

	// nLR = max(nL, nR)
	static const size_t savitzky_golay_n_l_r = 16;
	static const float savitzky_golay_coefficient
		[2 * savitzky_golay_n_l_r + 1][2 * savitzky_golay_n_l_r + 1] = {
		{ 0.422085, 0.325077, 0.23839, 0.162023, 0.0959752, 0.0402477,
		  -0.00515996, -0.0402477, -0.0650155, -0.0794634, -0.0835913,
		  -0.0773994, -0.0608875, -0.0340557, 0.00309598, 0.0505676,
		  0.108359 },
		{ 0.315789, 0.254902, 0.19969, 0.150155, 0.106295, 0.0681115,
		  0.0356037, 0.00877193, -0.0123839, -0.0278638, -0.0376677,
		  -0.0417957, -0.0402477, -0.0330237, -0.0201238, -0.00154799,
		  0.0227038, 0.0526316 },
		{ 0.234586, 0.198496, 0.165207, 0.134719, 0.107032, 0.0821465,
		  0.0600619, 0.0407784, 0.024296, 0.0106148, -0.000265369,
		  -0.00834439, -0.0136223, -0.0160991, -0.0157747, -0.0126493,
		  -0.00672269, 0.00200501, 0.0135338 },
		{ 0.172078, 0.153076, 0.135099, 0.118148, 0.102221, 0.0873206,
		  0.073445, 0.0605947, 0.0487697, 0.0379699, 0.0281955,
		  0.0194463, 0.0117225, 0.00502392, -0.000649351, -0.00529733,
		  -0.00892003, -0.0115174, -0.0130895, -0.0136364 },
		{ 0.123659, 0.116431, 0.109144, 0.101798, 0.0943921, 0.0869268,
		  0.0794021, 0.0718179, 0.0641743, 0.0564712, 0.0487087,
		  0.0408868, 0.0330054, 0.0250646, 0.0170644, 0.00900473,
		  0.000885613, -0.00729294, -0.0155309, -0.0238283,
		  -0.0321852 },
		{ 0.0859684, 0.0868154, 0.0869565, 0.0863919, 0.0851214,
		  0.0831451, 0.080463, 0.0770751, 0.0729814, 0.0681818,
		  0.0626765, 0.0564653, 0.0495483, 0.0419255, 0.0335968,
		  0.0245624, 0.0148221, 0.00437606, -0.00677583, -0.0186335,
		  -0.0311971, -0.0444664 },
		{ 0.0565217, 0.0628458, 0.0680971, 0.0722756, 0.0753811,
		  0.0774139, 0.0783738, 0.0782609, 0.0770751, 0.0748165,
		  0.071485, 0.0670807, 0.0616036, 0.0550536, 0.0474308,
		  0.0387352, 0.0289667, 0.0181254, 0.00621118, -0.00677583,
		  -0.0208357, -0.0359684, -0.0521739 },
		{ 0.0334615, 0.0434281, 0.0521329, 0.0595759, 0.0657571,
		  0.0706765, 0.0743341, 0.07673, 0.0778641, 0.0777364,
		  0.0763469, 0.0736957, 0.0697826, 0.0646078, 0.0581712,
		  0.0504728, 0.0415126, 0.0312907, 0.0198069, 0.00706142,
		  -0.00694588, -0.022215, -0.0387458, -0.0565385 },
		{ 0.0153846, 0.0276923, 0.0386622, 0.0482943, 0.0565886,
		  0.0635452, 0.0691639, 0.0734448, 0.076388, 0.0779933,
		  0.0782609, 0.0771906, 0.0747826, 0.0710368, 0.0659532,
		  0.0595318, 0.0517726, 0.0426756, 0.0322408, 0.0204682,
		  0.00735786, -0.0070903, -0.0228763, -0.04, -0.0584615 },
		{ 0.001221, 0.0149451, 0.027326, 0.0383639, 0.0480586,
		  0.0564103, 0.0634188, 0.0690842, 0.0734066, 0.0763858,
		  0.078022, 0.078315, 0.077265, 0.0748718, 0.0711355,
		  0.0660562, 0.0596337, 0.0518681, 0.0427595, 0.0323077,
		  0.0205128, 0.00737485, -0.00710623, -0.0229304, -0.0400977,
		  -0.0586081 },
		{ -0.00985222, 0.00463138, 0.0178098, 0.029683, 0.0402509,
		  0.0495137, 0.0574713, 0.0641236, 0.0694708, 0.0735127,
		  0.0762494, 0.0776809, 0.0778073, 0.0766284, 0.0741442,
		  0.0703549, 0.0652604, 0.0588607, 0.0511557, 0.0421456,
		  0.0318302, 0.0202097, 0.0072839, -0.00694708, -0.0224833,
		  -0.0393247, -0.0574713 },
		{ -0.0184729, -0.00369458, 0.00984169, 0.0221359, 0.0331881,
		  0.0429982, 0.0515662, 0.0588923, 0.0649762, 0.0698181,
		  0.073418, 0.0757758, 0.0768915, 0.0767652, 0.0753968,
		  0.0727864, 0.0689339, 0.0638394, 0.0575028, 0.0499242,
		  0.0411035, 0.0310408, 0.019736, 0.00718917, -0.00659972,
		  -0.0216307, -0.0379037, -0.0554187 },
		{ -0.025139, -0.0103925, 0.00318873, 0.0156046, 0.0268552,
		  0.0369405, 0.0458605, 0.0536151, 0.0602045, 0.0656285,
		  0.0698872, 0.0729806, 0.0749086, 0.0756714, 0.0752688,
		  0.0737009, 0.0709677, 0.0670692, 0.0620054, 0.0557763,
		  0.0483818, 0.039822, 0.0300969, 0.0192065, 0.0071508,
		  -0.00607024, -0.0204566, -0.0360083, -0.0527253 },
		{ -0.0302419, -0.0157536, -0.00234785, 0.00997537, 0.021216,
		  0.0313741, 0.0404497, 0.0484427, 0.0553532, 0.0611811,
		  0.0659264, 0.0695892, 0.0721695, 0.0736672, 0.0740823,
		  0.0734149, 0.0716649, 0.0688324, 0.0649174, 0.0599198,
		  0.0538396, 0.0466769, 0.0384316, 0.0291038, 0.0186934,
		  0.00720046, -0.00537502, -0.0190331, -0.0337736,
		  -0.0495968 },
		{ -0.0340909, -0.0200147, -0.006937, 0.00514208, 0.0162226,
		  0.0263045, 0.0353878, 0.0434725, 0.0505587, 0.0566463,
		  0.0617353, 0.0658257, 0.0689175, 0.0710107, 0.0721054,
		  0.0722014, 0.0712989, 0.0693978, 0.0664981, 0.0625999,
		  0.057703, 0.0518076, 0.0449135, 0.0370209, 0.0281297,
		  0.01824, 0.0073516, -0.00453534, -0.0174209, -0.031305,
		  -0.0461877 },
		{ -0.0369318, -0.0233688, -0.0107221, 0.00100806, 0.0118218,
		  0.0217192, 0.0307001, 0.0387647, 0.0459128, 0.0521444,
		  0.0574597, 0.0618585, 0.0653409, 0.0679069, 0.0695565,
		  0.0702896, 0.0701063, 0.0690066, 0.0669905, 0.0640579,
		  0.0602089, 0.0554435, 0.0497617, 0.0431635, 0.0356488,
		  0.0272177, 0.0178702, 0.0076063, -0.00357405, -0.0156708,
		  -0.028684, -0.0426136 },
		{ -0.038961, -0.025974, -0.0138249, -0.00251362, 0.00795978,
		  0.0175953, 0.026393, 0.0343527, 0.0414747, 0.0477587,
		  0.0532049, 0.0578132, 0.0615836, 0.0645161, 0.0666108,
		  0.0678676, 0.0682866, 0.0678676, 0.0666108, 0.0645161,
		  0.0615836, 0.0578132, 0.0532049, 0.0477587, 0.0414747,
		  0.0343527, 0.026393, 0.0175953, 0.00795978, -0.00251362,
		  -0.0138249, -0.025974, -0.038961 },
		{ -0.0426136, -0.028684, -0.0156708, -0.00357405, 0.0076063,
		  0.0178702, 0.0272177, 0.0356488, 0.0431635, 0.0497617,
		  0.0554435, 0.0602089, 0.0640579, 0.0669905, 0.0690066,
		  0.0701063, 0.0702896, 0.0695565, 0.0679069, 0.0653409,
		  0.0618585, 0.0574597, 0.0521444, 0.0459128, 0.0387647,
		  0.0307001, 0.0217192, 0.0118218, 0.00100806, -0.0107221,
		  -0.0233688, -0.0369318 },
		{ -0.0461877, -0.031305, -0.0174209, -0.00453534, 0.0073516,
		  0.01824, 0.0281297, 0.0370209, 0.0449135, 0.0518076,
		  0.057703, 0.0625999, 0.0664981, 0.0693978, 0.0712989,
		  0.0722014, 0.0721054, 0.0710107, 0.0689175, 0.0658257,
		  0.0617353, 0.0566463, 0.0505587, 0.0434725, 0.0353878,
		  0.0263045, 0.0162226, 0.00514208, -0.006937, -0.0200147,
		  -0.0340909 },
		{ -0.0495968, -0.0337736, -0.0190331, -0.00537502, 0.00720046,
		  0.0186934, 0.0291038, 0.0384316, 0.0466769, 0.0538396,
		  0.0599198, 0.0649174, 0.0688324, 0.0716649, 0.0734149,
		  0.0740823, 0.0736672, 0.0721695, 0.0695892, 0.0659264,
		  0.0611811, 0.0553532, 0.0484427, 0.0404497, 0.0313741,
		  0.021216, 0.00997537, -0.00234785, -0.0157536, -0.0302419 },
		{ -0.0527253, -0.0360083, -0.0204566, -0.00607024, 0.0071508,
		  0.0192065, 0.0300969, 0.039822, 0.0483818, 0.0557763,
		  0.0620054, 0.0670692, 0.0709677, 0.0737009, 0.0752688,
		  0.0756714, 0.0749086, 0.0729806, 0.0698872, 0.0656285,
		  0.0602045, 0.0536151, 0.0458605, 0.0369405, 0.0268552,
		  0.0156046, 0.00318873, -0.0103925, -0.025139 },
		{ -0.0554187, -0.0379037, -0.0216307, -0.00659972, 0.00718917,
		  0.019736, 0.0310408, 0.0411035, 0.0499242, 0.0575028,
		  0.0638394, 0.0689339, 0.0727864, 0.0753968, 0.0767652,
		  0.0768915, 0.0757758, 0.073418, 0.0698181, 0.0649762,
		  0.0588923, 0.0515662, 0.0429982, 0.0331881, 0.0221359,
		  0.00984169, -0.00369458, -0.0184729 },
		{ -0.0574713, -0.0393247, -0.0224833, -0.00694708, 0.0072839,
		  0.0202097, 0.0318302, 0.0421456, 0.0511557, 0.0588607,
		  0.0652604, 0.0703549, 0.0741442, 0.0766284, 0.0778073,
		  0.0776809, 0.0762494, 0.0735127, 0.0694708, 0.0641236,
		  0.0574713, 0.0495137, 0.0402509, 0.029683, 0.0178098,
		  0.00463138, -0.00985222 },
		{ -0.0586081, -0.0400977, -0.0229304, -0.00710623, 0.00737485,
		  0.0205128, 0.0323077, 0.0427595, 0.0518681, 0.0596337,
		  0.0660562, 0.0711355, 0.0748718, 0.077265, 0.078315,
		  0.078022, 0.0763858, 0.0734066, 0.0690842, 0.0634188,
		  0.0564103, 0.0480586, 0.0383639, 0.027326, 0.0149451,
		  0.001221 },
		{ -0.0584615, -0.04, -0.0228763, -0.0070903, 0.00735786,
		  0.0204682, 0.0322408, 0.0426756, 0.0517726, 0.0595318,
		  0.0659532, 0.0710368, 0.0747826, 0.0771906, 0.0782609,
		  0.0779933, 0.076388, 0.0734448, 0.0691639, 0.0635452,
		  0.0565886, 0.0482943, 0.0386622, 0.0276923, 0.0153846 },
		{ -0.0565385, -0.0387458, -0.022215, -0.00694588, 0.00706142,
		  0.0198069, 0.0312907, 0.0415126, 0.0504728, 0.0581712,
		  0.0646078, 0.0697826, 0.0736957, 0.0763469, 0.0777364,
		  0.0778641, 0.07673, 0.0743341, 0.0706765, 0.0657571,
		  0.0595759, 0.0521329, 0.0434281, 0.0334615 },
		{ -0.0521739, -0.0359684, -0.0208357, -0.00677583, 0.00621118,
		  0.0181254, 0.0289667, 0.0387352, 0.0474308, 0.0550536,
		  0.0616036, 0.0670807, 0.071485, 0.0748165, 0.0770751,
		  0.0782609, 0.0783738, 0.0774139, 0.0753811, 0.0722756,
		  0.0680971, 0.0628458, 0.0565217 },
		{ -0.0444664, -0.0311971, -0.0186335, -0.00677583, 0.00437606,
		  0.0148221, 0.0245624, 0.0335968, 0.0419255, 0.0495483,
		  0.0564653, 0.0626765, 0.0681818, 0.0729814, 0.0770751,
		  0.080463, 0.0831451, 0.0851214, 0.0863919, 0.0869565,
		  0.0868154, 0.0859684 },
		{ -0.0321852, -0.0238283, -0.0155309, -0.00729294, 0.000885613,
		  0.00900473, 0.0170644, 0.0250646, 0.0330054, 0.0408868,
		  0.0487087, 0.0564712, 0.0641743, 0.0718179, 0.0794021,
		  0.0869268, 0.0943921, 0.101798, 0.109144, 0.116431,
		  0.123659 },
		{ -0.0136364, -0.0130895, -0.0115174, -0.00892003, -0.00529733,
		  -0.000649351, 0.00502392, 0.0117225, 0.0194463, 0.0281955,
		  0.0379699, 0.0487697, 0.0605947, 0.073445, 0.0873206,
		  0.102221, 0.118148, 0.135099, 0.153076, 0.172078 },
		{ 0.0135338, 0.00200501, -0.00672269, -0.0126493, -0.0157747,
		  -0.0160991, -0.0136223, -0.00834439, -0.000265369, 0.0106148,
		  0.024296, 0.0407784, 0.0600619, 0.0821465, 0.107032,
		  0.134719, 0.165207, 0.198496, 0.234586 },
		{ 0.0526316, 0.0227038, -0.00154799, -0.0201238, -0.0330237,
		  -0.0402477, -0.0417957, -0.0376677, -0.0278638, -0.0123839,
		  0.00877193, 0.0356037, 0.0681115, 0.106295, 0.150155,
		  0.19969, 0.254902, 0.315789 },
		{ 0.108359, 0.0505676, 0.00309598, -0.0340557, -0.0608875,
		  -0.0773994, -0.0835913, -0.0794634, -0.0650155, -0.0402477,
		  -0.00515996, 0.0402477, 0.0959752, 0.162023, 0.23839,
		  0.325077, 0.422085 }
	};

	float filtered_baseline[128];
	float filtered_baseline_derivative[127];

	// Zero filtered_baseline
	memset(filtered_baseline, 0, 128 * sizeof(float));
	// Filter the left edge using (nL, nR) = (0, 16) .. (15, 16) S-G
	// filters
	for (size_t i = 0; i < savitzky_golay_n_l_r; i++) {
		for (size_t j = 0; j < savitzky_golay_n_l_r + 1 + i; j++) {
			filtered_baseline[i] +=
				savitzky_golay_coefficient[i][j] * baseline[j];
		}
	}
	// Filter the middle section using the (nL, nR) = (16, 16) S-G
	// filter, while taking advantage of the symmetry to save 16
	// multiplications.
	for (size_t i = savitzky_golay_n_l_r;
		 i < 128 - savitzky_golay_n_l_r; i++) {
		filtered_baseline[i] =
			savitzky_golay_coefficient
			[savitzky_golay_n_l_r][savitzky_golay_n_l_r] * baseline[i];
		for (size_t j = 0; j < savitzky_golay_n_l_r; j++) {
			filtered_baseline[i] +=
				savitzky_golay_coefficient[savitzky_golay_n_l_r][j] *
				(baseline[i + j - savitzky_golay_n_l_r] +
				 baseline[i - j + savitzky_golay_n_l_r]);
		}
#if 0
		// Test that the indexing above is correct
		float test = 0;
		for (size_t j = 0; j < 2 * savitzky_golay_n_l_r + 1; j++) {
			test +=
				savitzky_golay_coefficient[savitzky_golay_n_l_r][j] *
				baseline[i + j - savitzky_golay_n_l_r];
		}
		// test == filtered_baseline[i] should hold now
#endif
	}
	// Filter the right edge using (nL, nR) = (16, 15) .. (16, 0) S-G
	// filters
	for (size_t i = 128 - savitzky_golay_n_l_r; i < 128; i++) {
		for (size_t j = 0; j < 128 - i + savitzky_golay_n_l_r; j++) {
			filtered_baseline[i] +=
				savitzky_golay_coefficient
				[2 * savitzky_golay_n_l_r + i + 1 - 128][j] *
				baseline[i + j - savitzky_golay_n_l_r];
		}
	}
	// In lieu of a spearate S-G derivative filter, the finite
	// difference is used here (since the output is sufficiently
	// smooth).
	for (size_t i = 0; i < 127; i++) {
		filtered_baseline_derivative[i] =
			filtered_baseline[i + 1] - filtered_baseline[i];
	}

	// Calculate the maximum deviation between filtered and unfiltered
	// baseline, plus the sum square of the derivative.

	double filtered_baseline_max = 0;
	double filtered_baseline_derivative_sum_square = 0;

	for (size_t i = 0; i < 128; i++) {
		const double d = filtered_baseline[i] - baseline[i];

		filtered_baseline_max =
			std::max(filtered_baseline_max,
					 static_cast<double>(fabs(d)));
	}
	for (size_t i = 0; i < 127; i++) {
		filtered_baseline_derivative_sum_square +=
			filtered_baseline_derivative[i] *
			filtered_baseline_derivative[i];
	}

#if 0
	std::cerr << __FILE__ << ':' << __LINE__ << ": "
			  << filtered_baseline_max << ' '
			  << filtered_baseline_derivative_sum_square << std::endl;
#endif

	// Apply the cut
	return !(filtered_baseline_max >= filteredBaselineMax_ ||
 			 filtered_baseline_derivative_sum_square >= filteredBaselineDerivativeSumSquare_);
}





//Other methods implementation ==============================================
//==========================================================================

void SiStripAPVRestorer::LoadMeanCMMap(const edm::Event& iEvent){
  if(useRealMeanCM_){  
	edm::Handle< edm::DetSetVector<SiStripRawDigi> > input;
    iEvent.getByLabel("siStripDigis","VirginRaw", input);
   this->CreateCMMapRealPed(*input);
  } else {
    edm::Handle< edm::DetSetVector<SiStripProcessedRawDigi> > inputCM;
    iEvent.getByLabel("MEANAPVCM",inputCM);
    this->CreateCMMapCMstored(*inputCM);
  }
}


void SiStripAPVRestorer::CreateCMMapRealPed(const edm::DetSetVector<SiStripRawDigi>& input){
  
  MeanCMmap_.erase(MeanCMmap_.begin(), MeanCMmap_.end());
  	
 //std::cout<< "===============================================" << std::endl;
 
 for ( edm::DetSetVector<SiStripRawDigi>::const_iterator 
	  rawDigis = input.begin(); rawDigis != input.end(); rawDigis++) {
         SiStripPedestals::Range detPedestalRange = pedestalHandle->getRange(rawDigis->id);
		 std::vector<float> MeanCMDetSet;
		 MeanCMDetSet.clear();
		
		for(uint16_t APV = 0; APV < rawDigis->size()/128; ++APV){
			uint16_t MinPed =0;
			for(uint16_t strip = APV*128; strip< (APV+1)*128; ++strip){
			  uint16_t ped =  (uint16_t)pedestalHandle->getPed(strip,detPedestalRange);
			 
			  if(ped < MinPed) MinPed = ped;
			}
			if(MinPed>128) MinPed=128;
			MeanCMDetSet.push_back(MinPed);
		       
		}
		MeanCMmap_.insert(std::pair<uint32_t, std::vector<float> >(rawDigis->id,MeanCMDetSet));
		
 }
}

void SiStripAPVRestorer::CreateCMMapCMstored(const edm::DetSetVector<SiStripProcessedRawDigi>& Input){

  MeanCMmap_.erase(MeanCMmap_.begin(), MeanCMmap_.end());
  uint32_t detId;
  edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator itInput;
  edm::DetSet<SiStripProcessedRawDigi>::const_iterator itCM;
  std::vector<float> MeanCMNValue;
  
  for(itInput = Input.begin(); itInput != Input.end(); ++itInput){
    detId = itInput->id;
    MeanCMNValue.clear();
    for(itCM = itInput->begin(); itCM != itInput->end(); ++itCM) MeanCMNValue.push_back(itCM->adc()); 			
    MeanCMmap_.insert(std::pair<uint32_t, std::vector<float> >(detId,MeanCMNValue));
  }
}

std::vector<bool>& SiStripAPVRestorer::GetAPVFlags(){
    apvFlagsBool_.clear();
    for(size_t i =0; i < apvFlags_.size(); ++i){
      if(apvFlags_[i] != "" && !apvFlagsBoolOverride_[i]) apvFlagsBool_.push_back(true);
      else apvFlagsBool_.push_back(false);
    }
    return apvFlagsBool_;	
}




