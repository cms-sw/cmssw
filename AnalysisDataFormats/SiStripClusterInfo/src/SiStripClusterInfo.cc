#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"

#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "CommonTools/SiStripZeroSuppression/interface/SiStripMedianCommonModeNoiseSubtraction.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripTT6CommonModeNoiseSubtraction.h"

#include <cmath>

SiStripClusterInfo::SiStripClusterInfo(const SiStripCluster&  cluster, 
                                       const edm::EventSetup&      es,
				       std::string CMNSubtractionMode): es_(es) {
  SiStripClusterInfo(cluster.geographicalId(),cluster,es,CMNSubtractionMode);
}

SiStripClusterInfo::SiStripClusterInfo(const uint32_t  cluster_detId,
                                       const SiStripCluster&  cluster, 
                                       const edm::EventSetup&      es,
				       std::string CMNSubtractionMode):
  es_(es){
  
  cluster_ = & cluster;  
  cluster_detId_ = cluster_detId;  

  es.get<SiStripPedestalsRcd>().get(pedestalsHandle_);
  es.get<SiStripNoisesRcd>().get(noiseHandle_);
  es.get<SiStripGainRcd>().get(gainHandle_);
  
  
  SiStripPedestalsSubtractor_ = new SiStripPedestalsSubtractor();
    
    
    
  //------------------------
  if ( CMNSubtractionMode== "Median") { 
    SiStripCommonModeNoiseSubtractor_ = new SiStripMedianCommonModeNoiseSubtraction();
    validCMNSubtraction_ = true;
  }
  else if ( CMNSubtractionMode== "TT6") { 
    //FIXME : SiStripCommonModeNoiseSubtractor_ = new SiStripTT6CommonModeNoiseSubtraction(conf.getParameter<double>("CutToAvoidSignal"));
    validCMNSubtraction_ = true;
  }
  else {
    edm::LogError("SiStripClusterInfoProducer") << "[SiStripClusterInfoProducer::SiStripClusterInfoProducer] No valid CommonModeNoiseSubtraction Mode selected, possible CMNSubtractionMode: Median or TT6" << std::endl;
    validCMNSubtraction_ = false;
  } 
    
  //------------------------
    
  
  
}


SiStripClusterInfo::~SiStripClusterInfo(){}


float SiStripClusterInfo::getCharge() const {
  
  float charge_=0;
  const std::vector<uint8_t>& amplitudes_ =  cluster_->amplitudes();
  
  for(size_t i=0; i< amplitudes_.size();i++){
    if (amplitudes_[i] > 0){  // redundant as always fullfilled for cluster amplitudes
      charge_+=amplitudes_[i];
    } 
  } 
  
  return charge_ ;
  
}


uint16_t SiStripClusterInfo::getMaxPosition() const {
  return getMaxPosition(cluster_);
}

uint16_t SiStripClusterInfo::getMaxPosition(const SiStripCluster *cluster) const {
  
  uint16_t maxPosition_=0;
  const std::vector<uint8_t>& amplitudes_ =  cluster->amplitudes();  
  float maxCharge_=0;
  
  for(size_t i=0; i< amplitudes_.size();i++){
    if (amplitudes_[i] > 0){ // redundant as always fullfilled for cluster amplitudes
      if (maxCharge_<amplitudes_[i]){ 
        maxCharge_=amplitudes_[i];
        maxPosition_=i;
      }
    } 
  } 
  maxPosition_+=cluster->firstStrip();
  
  return maxPosition_;
  
}


float SiStripClusterInfo::getMaxCharge() const {
  
  const std::vector<uint8_t>& amplitudes_ =  cluster_->amplitudes();  
  float maxCharge_=0;
  
  for(size_t i=0; i< amplitudes_.size();i++){
    if (amplitudes_[i] > 0){
      if (maxCharge_<amplitudes_[i]){ 
        maxCharge_=amplitudes_[i];
      }
    } 
  } 
  return maxCharge_;
}


std::vector<float> SiStripClusterInfo::getRawChargeCLR( const edm::DetSet<SiStripRawDigi>&    ds_SiStripRawDigi,
							      edmNew::DetSet<SiStripCluster>& ds_SiStripCluster,
							std::string                           rawDigiLabel ) {
  std::vector<float> adcCLR(3*0);
  
  // Get First Left and First Right
  uint32_t neighbourStrips = 1;
  std::pair< std::vector<float>,std::vector<float> > rawChargesLR =
    this->getRawDigiAmplitudesLR(neighbourStrips, ds_SiStripRawDigi, ds_SiStripCluster, std::string(rawDigiLabel));
  
  if (edm::isDebugEnabled()){
    std::stringstream sssL;
    std::stringstream sssR;
    // Left
    for(std::vector<float>::const_iterator digi_adc_iter=rawChargesLR.first.begin();
	digi_adc_iter!=rawChargesLR.first.end();digi_adc_iter++)
      sssL << "\n digi adc " << *digi_adc_iter;
    LogDebug("SiStripClusterInfo") << " detid " << cluster_detId_  << "\n Max Left RawDigis \n " << sssL.str();
    // Right
    for(std::vector<float>::const_iterator digi_adc_iter=rawChargesLR.second.begin();
	digi_adc_iter!=rawChargesLR.second.end();digi_adc_iter++)
      sssR << "\n digi adc " << *digi_adc_iter;
    LogDebug("SiStripClusterInfo") << " detid " << cluster_detId_  << "\n Max Right RawDigis \n " << sssR.str();
  }
  
  adcCLR[0] = amplitudeC_;
  adcCLR[1] = rawChargesLR.first[0];
  adcCLR[2] = rawChargesLR.second[0];
  
  return adcCLR;
}


std::pair<float,float> SiStripClusterInfo::getChargeLR() const {
  
  float chargeL_=0;
  float chargeR_=0; 
  uint16_t maxPosition_ = this->getMaxPosition();
  const std::vector<uint8_t>& amplitudes_ =  cluster_->amplitudes();
  
  for(size_t i=0; i<amplitudes_.size();i++){
    if ((cluster_->firstStrip())+i < maxPosition_) chargeL_+=(float)amplitudes_[i]; 	  
    if ((cluster_->firstStrip())+i > maxPosition_) chargeR_+=(float)amplitudes_[i]; 	  
  }
  
  return std::pair<float,float>(chargeL_,chargeR_);
  
}


std::pair<float,float> SiStripClusterInfo::getChargeLRFirstNeighbour() const {
  
  uint16_t maxPosition = this->getMaxPosition();
  uint16_t firstStrip  = this->getFirstStrip();
  const std::vector<uint8_t>& amplitudes =  cluster_->amplitudes();
  
  float chargeL = ( maxPosition > firstStrip                           ? (float)amplitudes[maxPosition-firstStrip-1] : 0. );
  float chargeR = ( maxPosition < firstStrip + (amplitudes.size() - 1) ? (float)amplitudes[maxPosition-firstStrip+1] : 0. );
  
  //  std::cout << this->getDetId() << ": " << chargeL << " " << chargeR << " width " << amplitudes.size() << std::endl;
  
  return std::pair<float,float>(chargeL,chargeR);
  
}


std::vector<float> SiStripClusterInfo::getStripNoises() const {
  
  std::vector<float>   stripNoises_;
  SiStripNoises::Range detNoiseRange = noiseHandle_->getRange(cluster_detId_);  
  const std::vector<uint8_t>& amplitudes_ =  cluster_->amplitudes();
  
  for(size_t i=0; i<amplitudes_.size();i++){
    
    float noise_=noiseHandle_->getNoise(cluster_->firstStrip()+i,detNoiseRange);	
    stripNoises_.push_back(noise_);
    
  } 
  
  return stripNoises_;
  
}


std::vector<float> SiStripClusterInfo::getStripNoisesRescaledByGain() const {
  
  std::vector<float>   stripNoises_;
  SiStripNoises::Range detNoiseRange = noiseHandle_->getRange(cluster_detId_);  
  SiStripApvGain::Range detGainRange = gainHandle_->getRange(cluster_detId_);	
  const std::vector<uint8_t>& amplitudes_ =  cluster_->amplitudes();
  
  for(size_t i=0; i<amplitudes_.size();i++){
    
    float gain_=gainHandle_->getStripGain(cluster_->firstStrip()+i,detGainRange);
    float noise_=noiseHandle_->getNoise(cluster_->firstStrip()+i,detNoiseRange);
    noise_=noise_/gain_;
    
    stripNoises_.push_back(noise_);
    
  } 
  
  return stripNoises_;
  
}


float SiStripClusterInfo::getNoise() {
    
  int numberOfPosAmplitudes_=0;
  float        clusterNoise_=0;  
  float       clusterNoise2_=0;
  
  SiStripNoises::Range detNoiseRange = noiseHandle_->getRange(cluster_detId_);
  
  const std::vector<uint8_t>& amplitudes_ =  cluster_->amplitudes();
  
  for(size_t i=0; i<amplitudes_.size();i++){
    
    float noise_=noiseHandle_->getNoise(cluster_->firstStrip()+i,detNoiseRange);
    
    if (amplitudes_[i]>0){
      clusterNoise2_+=noise_*noise_;
      numberOfPosAmplitudes_++;  
    } 
  }   
  clusterNoise_= sqrt(clusterNoise2_/numberOfPosAmplitudes_);
  
  return clusterNoise_;
  
}


float SiStripClusterInfo::getNoiseRescaledByGain() {
    
  int numberOfPosAmplitudes_=0;
  float       clusterNoise_ =0;  
  float       clusterNoise2_=0;  
  
  SiStripNoises::Range detNoiseRange = noiseHandle_->getRange(cluster_detId_);
  SiStripApvGain::Range detGainRange = gainHandle_->getRange(cluster_detId_);	
  
  const std::vector<uint8_t>& amplitudes_ =  cluster_->amplitudes();
  
  for(size_t i=0; i<amplitudes_.size();i++){
    
    float gain_=gainHandle_->getStripGain(cluster_->firstStrip()+i,detGainRange);
    float noise_=noiseHandle_->getNoise(cluster_->firstStrip()+i,detNoiseRange);
    noise_=noise_/gain_;
    
    if (amplitudes_[i]>0){
      clusterNoise2_+=noise_*noise_;
      numberOfPosAmplitudes_++;  
    } 
  }   
  clusterNoise_= sqrt(clusterNoise2_/numberOfPosAmplitudes_);
  
  return clusterNoise_;
  
}


float SiStripClusterInfo::getNoiseForStripNb(uint16_t istrip) const{
  
  std::vector<float>   stripNoises_;	
  stripNoises_ = this->getStripNoises();
  
  short strip_of_istrip     = (int) ((cluster_->firstStrip()+istrip)/128);  
  short strip_of_firstStrip_ = (int) (cluster_->firstStrip()/128);  
  
  return stripNoises_[strip_of_istrip-strip_of_firstStrip_];
  
}


std::vector<float> SiStripClusterInfo::getApvGains() const {
  
  std::vector<float>   apvGains_;
  SiStripApvGain::Range detGainRange = gainHandle_->getRange(cluster_detId_);	
  
  const std::vector<uint8_t>& amplitudes_ =  cluster_->amplitudes();
  
  for(size_t i=0; i<amplitudes_.size();i++){	
    float gain_=gainHandle_->getStripGain(cluster_->firstStrip()+i,detGainRange);
    if (apvGains_.empty())           apvGains_.push_back(gain_);
    else if (apvGains_.back()!=gain_) apvGains_.push_back(gain_);	  
  } 
  
  return apvGains_;
  
}


float SiStripClusterInfo::getGainForStripNb(uint16_t istrip) const{
  
  std::vector<float>   apvGains_;	
  apvGains_ = this->getApvGains();
  
  short apv_of_istrip = (int) ((cluster_->firstStrip()+istrip)/128);  
  short apv_of_firstStrip_ = (int) (cluster_->firstStrip()/128);  
  
  return apvGains_[apv_of_istrip-apv_of_firstStrip_];
  
}


float SiStripClusterInfo::getSignalOverNoise() {
  
  return (this->getCharge())/(this->getNoise());
  
}


float SiStripClusterInfo::getSignalOverNoiseRescaledByGain() {
  
  return (this->getCharge())/(this->getNoiseRescaledByGain());
  
}


std::pair< std::vector<float>,std::vector<float> > SiStripClusterInfo::getRawDigiAmplitudesLR(      uint32_t&                          neighbourStripNr, 
                                                                                              const edm::DetSetVector<SiStripRawDigi>&    rawDigis_dsv_, 
                                                                                                    edmNew::DetSetVector<SiStripCluster>& clusters_dsv_,
                                                                                                    std::string                            rawDigiLabel ) {
  neighbourStripNr_ = neighbourStripNr;
  
  const edm::DetSet<SiStripRawDigi> rawDigis_ds_ = rawDigis_dsv_[cluster_detId_];
  edmNew::DetSet<SiStripCluster>    clusters_ds_ = clusters_dsv_[cluster_detId_];
 
  if (rawDigis_dsv_.find(cluster_detId_)!=rawDigis_dsv_.end()){
  
    rawdigi_algorithm(rawDigis_ds_,clusters_ds_,rawDigiLabel);
  
    return std::pair< std::vector<float>,std::vector<float> > (amplitudesL_, amplitudesR_);
  }    
  else {
    throw cms::Exception("CorruptedData")
      << "[SiStripClusterInfo::getRawDigiAmplitudesLR] reached already end " << std::endl;
  }
}


std::pair< std::vector<float>,std::vector<float> > SiStripClusterInfo::getRawDigiAmplitudesLR(      uint32_t&                   neighbourStripNr, 
                                                                                              const edm::DetSet<SiStripRawDigi>&    rawDigis_ds_, 
                                                                                                    edmNew::DetSet<SiStripCluster>& clusters_ds_,
                                                                                                    std::string                     rawDigiLabel ) {
  neighbourStripNr_ = neighbourStripNr;
  
  //QUESTION: here no test?
  rawdigi_algorithm(rawDigis_ds_,clusters_ds_,rawDigiLabel);
  
  return std::pair< std::vector<float>,std::vector<float> > (amplitudesL_, amplitudesR_);
}



std::pair< std::vector<float>,std::vector<float> > SiStripClusterInfo::getDigiAmplitudesLR(      uint32_t&                          neighbourStripNr,
                                                                                           const edm::DetSetVector<SiStripDigi>&          digis_dsv_,
                                                                                                 edmNew::DetSetVector<SiStripCluster>& clusters_dsv_ ){
  
  neighbourStripNr_ = neighbourStripNr;
  
  const edm::DetSet<SiStripDigi>    digis_ds_ =   digis_dsv_[cluster_detId_];
  edmNew::DetSet<SiStripCluster> clusters_ds_ = clusters_dsv_[cluster_detId_];

  if (digis_dsv_.find(cluster_detId_)!=digis_dsv_.end()){
  
    digi_algorithm(digis_ds_,clusters_ds_);          
    
    return std::pair< std::vector<float>,std::vector<float> > (amplitudesL_, amplitudesR_);
  }
  else {
    throw cms::Exception("CorruptedData")
      << "[SiStripClusterInfo::getDigiAmplitudesLR] reached already end " << std::endl;
  }
 
}

std::pair< std::vector<float>,std::vector<float> > SiStripClusterInfo::getDigiAmplitudesLR(      uint32_t&                  neighbourStripNr, 
                                                                                           const edm::DetSet<SiStripDigi>&          digis_ds_, 
                                                                                                 edmNew::DetSet<SiStripCluster>& clusters_ds_ ) {
  neighbourStripNr_ = neighbourStripNr;
  
  digi_algorithm(digis_ds_,clusters_ds_);
  
  return std::pair< std::vector<float>,std::vector<float> > (amplitudesL_, amplitudesR_);
}


void SiStripClusterInfo::rawdigi_algorithm(const edm::DetSet<SiStripRawDigi>     rawDigis_ds_,
				                 edmNew::DetSet<SiStripCluster>  clusters_ds_,
		              		         std::string                     rawDigiLabel ){
  
  std::vector<int16_t> vssRd(rawDigis_ds_.size());
  
  if ( rawDigiLabel == "ProcessedRaw"){
    
    for(edm::DetSet<SiStripRawDigi>::const_iterator digis_iter=rawDigis_ds_.data.begin();
                                                   digis_iter!=rawDigis_ds_.data.end();digis_iter++){
      vssRd.push_back(digis_iter->adc());
      
      if (edm::isDebugEnabled()){
	std::stringstream sss;
	int idig=0;
	
	for(std::vector<int16_t>::const_iterator digi_adc_iter=vssRd.begin();
	                                        digi_adc_iter!=vssRd.end();digi_adc_iter++)
	  sss << "\n digi strip " << idig++ << " digi adc " << *digi_adc_iter;
	LogDebug("SiStripClusterInfo") << " detid " << cluster_detId_  << " Pedestal subtracted digis \n" << sss.str();	
      }	    
    }
  } else if ( rawDigiLabel == "VirginRaw" ) {
    
    if (edm::isDebugEnabled()){
      std::stringstream sss;
      int idig=0;
      
      for(edm::DetSet<SiStripRawDigi>::const_iterator digis_iter=rawDigis_ds_.data.begin();
                                                     digis_iter!=rawDigis_ds_.data.end();digis_iter++)
	sss << "\n digi strip " << idig++ << " digi adc " << digis_iter->adc();
      LogDebug("SiStripClusterInfo") << " detid " << cluster_detId_  << "\n RawDigis \n" << sss.str();
    }
    
    //Subtract Pedestals
    SiStripPedestalsSubtractor_->init(es_);
    SiStripPedestalsSubtractor_->subtract(rawDigis_ds_,vssRd);
    
    if (edm::isDebugEnabled()){
      std::stringstream sss;
      int idig=0;
      
      for(std::vector<int16_t>::const_iterator digi_adc_iter=vssRd.begin();
                                              digi_adc_iter!=vssRd.end();digi_adc_iter++)
	sss << "\n digi strip " << idig++ << " digi adc " << *digi_adc_iter;
      LogDebug("SiStripClusterInfo") << " detid " << cluster_detId_  << "\n Pedestal subtracted digis \n " << sss.str();	
    }
    
    //Subtract CMN
    if (validCMNSubtraction_){
      SiStripCommonModeNoiseSubtractor_->init(es_);
      SiStripCommonModeNoiseSubtractor_->subtract(rawDigis_ds_.id,vssRd);
      
      if (edm::isDebugEnabled()){
	std::stringstream sss;
	int idig=0;
	
	for(std::vector<int16_t>::const_iterator digi_adc_iter=vssRd.begin();
	                                        digi_adc_iter!=vssRd.end();digi_adc_iter++)
	  sss << "\n digi strip " << idig++ << " digi adc " << *digi_adc_iter;
	LogDebug("SiStripClusterInfo") << " detid " << cluster_detId_  << "\n CMN subtracted digis \n " << sss.str();	
      }
      
    }else{
      throw cms::Exception("") 
	<< "[" << __PRETTY_FUNCTION__<< "] No valid CommonModeNoiseSubtraction Mode selected, possible CMNSubtractionMode: Median or TT6"
	<< std::endl;
    }
  } else {
    return;
  }  
  findNeigh("raw",clusters_ds_,vssRd,vssRd);	
}


void SiStripClusterInfo::digi_algorithm(const edm::DetSet<SiStripDigi>          digis_ds_,
                                              edmNew::DetSet<SiStripCluster> clusters_ds_ ){
  
  std::vector<int16_t> vstrip;
  std::vector<int16_t> vadc;
  
  //Get list of digis for the current DetId
  vstrip.clear();
  vadc.clear();
  
  for(edm::DetSet<SiStripDigi>::const_iterator digis_iter=digis_ds_.data.begin();
                                               digis_iter!=digis_ds_.data.end();digis_iter++){
    vstrip.push_back(digis_iter->strip());
    vadc.push_back(digis_iter->adc());
  }
  findNeigh("digi",clusters_ds_,vadc,vstrip);
}  



void SiStripClusterInfo::findNeigh(char*                                  mode,
                                   edmNew::DetSet<SiStripCluster> clusters_ds_,
				   std::vector<int16_t>&                  vadc,
				   std::vector<int16_t>&                vstrip ){
  
  // clean the private members
  //  before filling with the neighbours
  amplitudeC_ = 0;
  amplitudesL_.clear();
  amplitudesR_.clear();
  //
  
  // Find Digi adiacent to the clusters of this detid
  int16_t lastStrip_previousCluster=-1;
  int16_t firstStrip_nextCluster=10000;
  
  for (edmNew::DetSet<SiStripCluster>::const_iterator clusters_iter=clusters_ds_.begin(); 
       clusters_iter!=clusters_ds_.end(); clusters_iter++){	
    
    // Avoid overlapping with neighbour clusters
    if (clusters_iter!=clusters_ds_.begin())
      lastStrip_previousCluster=(clusters_iter-1)->firstStrip()+(clusters_iter-1)->amplitudes().size() -1;	  
    if (clusters_iter!=clusters_ds_.end()-1)
      firstStrip_nextCluster=(clusters_iter+1)->firstStrip();
    
    
    // Get Gain Range
    SiStripApvGain::Range detGainRange = gainHandle_->getRange(clusters_ds_.id());
    
    // Left or Right with respect to the Central Strip (Cluster Seed)
    //    int16_t firstStrip=clusters_iter->firstStrip();
    //    int16_t lastStrip=firstStrip + clusters_iter->amplitudes().size() -1;
    int16_t firstStrip=getMaxPosition(clusters_iter);
    int16_t lastStrip=firstStrip;
    //    std::cout << "firstStrip = " << firstStrip << " lastStrip = " << lastStrip << std::endl;
    std::vector<int16_t>::iterator ptr;
    if (mode=="digi"){
      ptr=std::find(vstrip.begin(),vstrip.end(),firstStrip); 
      if (ptr==vstrip.end())
	throw cms::Exception("") << "\n Expected Digi not found in detid " << clusters_ds_.id() << " strip " << firstStrip << std::endl;
    }
    else{ 
      ptr=vstrip.begin()+firstStrip; // For raw mode vstrip==vadc==vector of digis for all strips in the det
    }
    
    // Central Digi from the ptr to the central strip
    int centralPos = ptr-vstrip.begin();
    if (mode=="digi")
      centralPos=*(vstrip.begin()+centralPos);
    //    std::cout << "centralPos = " << centralPos << std::endl;
    float gain=gainHandle_->getStripGain(centralPos,detGainRange);
    amplitudeC_ = ( (*(vadc.begin()+(ptr-vstrip.begin()))) / gain );
    
    // Looking at digis before firstStrip	  
    for (uint16_t istrip=1;istrip<neighbourStripNr_+1;istrip++){
      if (istrip>ptr-vstrip.begin()) //avoid underflow
	{break;}
      if (mode=="digi")
	if (firstStrip-istrip!=*(ptr-istrip)) //avoid not contiguous digis
	  {break;}
      if (firstStrip-istrip==lastStrip_previousCluster) //avoid clusters overlapping 
	{break;}
      int stripPos=ptr-vstrip.begin()-istrip;
      if (mode=="digi")
	stripPos=*(vstrip.begin()+stripPos);
      //      std::cout << "stripPos = " << stripPos << std::endl;
      float gain=gainHandle_->getStripGain(stripPos,detGainRange);
      amplitudesL_.push_back( (*(vadc.begin()+(ptr-vstrip.begin())-istrip)) / gain );
    }
    
    ptr+=lastStrip-firstStrip;
    
    // Looking at digis after LastStrip
    for (uint16_t istrip=1;istrip<neighbourStripNr_+1;istrip++){
      if (istrip>vstrip.end()-ptr-1) //avoid overflow
	{break;}
      if (mode=="digi")
	if (lastStrip+istrip!=*(ptr+istrip)) //avoid not contiguous digis
	  {break;}
      if (lastStrip+istrip==firstStrip_nextCluster) //avoid clusters overlapping 
	{break;}
      int stripPos=ptr-vstrip.begin()+istrip;
      if (mode=="digi")
	stripPos=*(vstrip.begin()+stripPos);
      //      std::cout << "stripPos = " << stripPos << std::endl;
      float gain=gainHandle_->getStripGain(stripPos,detGainRange);
      amplitudesR_.push_back( (*(vadc.begin()+(ptr-vstrip.begin())+istrip)) / gain );
    }   
  }
}





