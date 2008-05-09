#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"
#include "sstream"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void ThreeThresholdStripClusterizer::init(const edm::EventSetup& es) {
  //Get ESObject 
  es.get<SiStripGainRcd>().get(gainHandle_);
  es.get<SiStripNoisesRcd>().get(noiseHandle_);
  es.get<SiStripQualityRcd>().get(qualityLabel_,qualityHandle_);
}

void ThreeThresholdStripClusterizer::clusterizeDetUnit(const edm::DetSet<SiStripDigi>    & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
    clusterizeDetUnit_(digis,output);
}
void ThreeThresholdStripClusterizer::clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
    clusterizeDetUnit_(digis,output);
}

template<typename InputDetSet>
void ThreeThresholdStripClusterizer::clusterizeDetUnit_(const InputDetSet& input,
						        edmNew::DetSetVector<SiStripCluster>::FastFiller& output) {
  
  const uint32_t& detID = input.detId();
  
  if (!qualityHandle_->IsModuleUsable(detID)){
#ifdef DEBUG_SiStripClusterizer_
    LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] detid " << detID << " skipped because flagged NOT Usable in SiStripQuality " << std::endl; 
#endif
    return;
  }
  
  typename InputDetSet::const_iterator begin=input.begin();
  typename InputDetSet::const_iterator end  =input.end();
  
  typename InputDetSet::const_iterator ibeg, iend, ihigh, itest;  
  ibeg = iend = begin;
  cluster_digis_.clear();
  cluster_digis_.reserve(10);

  //output.data.reserve( (end - begin)/3 + 1); // FIXME put back in if needed

  SiStripApvGain::Range detGainRange =  gainHandle_->getRange(detID); 
  SiStripNoises::Range detNoiseRange = noiseHandle_->getRange(detID);
  SiStripQuality::Range detQualityRange = qualityHandle_->getRange(detID);

  //  AboveSeed predicate(seedThresholdInNoiseSigma(),SiStripNoiseService_,detID);
  AboveSeed predicate(seedThresholdInNoiseSigma(), noiseHandle_, detNoiseRange, qualityHandle_, detQualityRange);

#ifdef DEBUG_SiStripClusterizer_
  std::stringstream ss;
#endif

  while ( ibeg != end &&
          (ihigh = std::find_if( ibeg, end, predicate)) != end) {

#ifdef DEBUG_SiStripClusterizer_
    if(edm::isDebugEnabled())
      ss << "\nSeed Channel:\n\t\t detID "<< detID << " digis " << ihigh->strip() 
	 << " adc " << ihigh->adc() << " is " 
	 << " channelNoise " << noiseHandle_->getNoise(ihigh->strip(),detNoiseRange) 
	 <<  " IsBadChannel from SiStripQuality " << qualityHandle_->IsStripBad(detQualityRange,ihigh->strip());
#endif
    
    // The seed strip is ihigh. Scan up and down from it, finding nearby strips above
    // threshold, allowing for some holes. The accepted cluster runs from strip ibeg
    // to iend, and itest is the strip under study, not yet accepted.
    iend = ihigh;
    itest = iend + 1;
    while ( itest != end && (itest->strip() - iend->strip() <= max_holes_ + 1 )) {
      float channelNoise = noiseHandle_->getNoise(itest->strip(),detNoiseRange);
      bool IsBadChannel = qualityHandle_->IsStripBad(detQualityRange,itest->strip());

#ifdef DEBUG_SiStripClusterizer_
      if(edm::isDebugEnabled())
	ss << "\nStrips on the right:\n\t\t detID " << detID << " digis " << itest->strip()  
	   << " adc " << itest->adc() << " " << " channelNoise " << channelNoise 
	   <<  " IsBadChannel from SiStripQuality " << IsBadChannel;
#endif 
      
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
	iend = itest;
      }
      ++itest;
    }
    //if the next digi after iend is an adiacent bad digi then insert into candidate cluster
    itest=iend+1;
    if ( itest != end && (itest->strip() - iend->strip() == 1) && qualityHandle_->IsStripBad(detQualityRange,itest->strip()) ){
#ifdef DEBUG_SiStripClusterizer_
      if(edm::isDebugEnabled())
	ss << "\n\t\tInserted bad strip at the end edge iend->strip()= " << iend->strip() << " itest->strip() = " << itest->strip();
#endif      
      
      iend++;
    }

    ibeg = ihigh;
    itest = ibeg - 1;
    while ( itest >= begin && (ibeg->strip() - itest->strip() <= max_holes_ + 1 )) {
      float channelNoise = noiseHandle_->getNoise(itest->strip(),detNoiseRange);
      bool IsBadChannel = qualityHandle_->IsStripBad(detQualityRange,itest->strip());

#ifdef DEBUG_SiStripClusterizer_
      if(edm::isDebugEnabled())
	ss << "\nStrips on the left:\n\t\t detID " << detID << " digis " << itest->strip()
	   << " adc " << itest->adc() << " " << " channelNoise " << channelNoise 
	   <<  " IsBadChannel from SiStripQuality " << IsBadChannel;
#endif      
      
      
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
        ibeg = itest;
      }
      --itest;
    }
    //if the next digi after ibeg is an adiacent bad digi then insert into candidate cluster
    itest=ibeg-1;
    if ( itest >= begin && (ibeg->strip() - itest->strip() == 1) &&  qualityHandle_->IsStripBad(detQualityRange,itest->strip()) ) {    
#ifdef DEBUG_SiStripClusterizer_
      if(edm::isDebugEnabled())
	ss << "\nInserted bad strip at the begin edge ibeg->strip()= " << ibeg->strip() << " itest->strip() = " << itest->strip();
#endif      
      ibeg--;
    }
    
    float charge = 0;
    float sigmaNoise2=0;
    //int counts=0;
    cluster_digis_.clear();
    for (itest=ibeg; itest<=iend; itest++) {
      float channelNoise = noiseHandle_->getNoise(itest->strip(),detNoiseRange);
      bool IsBadChannel = qualityHandle_->IsStripBad(detQualityRange,itest->strip());

#ifdef DEBUG_SiStripClusterizer_
      if(edm::isDebugEnabled())
	ss << "\nLooking at cluster digis:\n\t\t detID " << detID << " digis " << itest->strip()  
	   << " adc " << itest->adc() << " channelNoise " << channelNoise 
	   <<  " IsBadChannel from SiStripQuality " << IsBadChannel;
#endif
      
 
      //check for consecutive digis
      if (itest!=ibeg && itest->strip()-(itest-1)->strip()!=1){
	//digi *(itest-1) and *itest are not consecutive: create an equivalent number of Digis with zero amp
	for (int j=(itest-1)->strip()+1;j<itest->strip();j++){
	  cluster_digis_.push_back(SiStripDigi(j,0)); //if strip bad or under threshold set StripDigi.adc_=0  
#ifdef DEBUG_SiStripClusterizer_
	  ss << "\n\t\tHole added: detID " << detID << " digis " << j << " adc 0 ";
#endif
	}
      }
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {

	float gainFactor  = gainHandle_->getStripGain(itest->strip(), detGainRange);
        float stripCharge=(static_cast<float>(itest->adc()))/gainFactor;

        charge += stripCharge;
        sigmaNoise2 += channelNoise*channelNoise/(gainFactor*gainFactor);
      
	cluster_digis_.push_back(SiStripDigi(itest->strip(), static_cast<uint8_t>(stripCharge+0.499)));
      } else {
	cluster_digis_.push_back(SiStripDigi(itest->strip(),0)); //if strip bad or under threshold set SiStripDigi.adc_=0
	
#ifdef DEBUG_SiStripClusterizer_
	 if(edm::isDebugEnabled())
	   ss << "\n\t\tBad or under threshold digis: detID " << detID  << " digis " << itest->strip()  
	      << " adc " << itest->adc() << " channelNoise " << channelNoise 
	      <<  " IsBadChannel from SiStripQuality " << IsBadChannel;
#endif
      }
    }
    float sigmaNoise = sqrt(sigmaNoise2);
    
    if (charge >= clusterThresholdInNoiseSigma()*sigmaNoise) {
#ifdef DEBUG_SiStripClusterizer_
       if(edm::isDebugEnabled())
	 ss << "\n\t\tCluster accepted :)";
#endif
      output.push_back( SiStripCluster( detID, SiStripCluster::SiStripDigiRange( cluster_digis_.begin(),
										      cluster_digis_.end())));
    } else {
#ifdef DEBUG_SiStripClusterizer_
      if(edm::isDebugEnabled())
	ss << "\n\t\tCluster rejected :(";
#endif
    }
    ibeg = iend+1;
  }

  
#ifdef DEBUG_SiStripClusterizer_
  LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] \n" << ss.str();
#endif
}


