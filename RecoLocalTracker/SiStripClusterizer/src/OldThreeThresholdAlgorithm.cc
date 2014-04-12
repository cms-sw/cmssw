#include "RecoLocalTracker/SiStripClusterizer/interface/OldThreeThresholdAlgorithm.h"
#include <sstream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define PATCH_FOR_DIGIS_DUPLICATION

void OldThreeThresholdAlgorithm::initialize(const edm::EventSetup& es) {
  //Get ESObject 
  es.get<SiStripGainRcd>().get(gainHandle_);
  es.get<SiStripNoisesRcd>().get(noiseHandle_);
  es.get<SiStripQualityRcd>().get(qualityLabel_,qualityHandle_);
}

void OldThreeThresholdAlgorithm::clusterizeDetUnit(const edm::DetSet<SiStripDigi>    & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
  clusterizeDetUnit_(digis,output);
}
void OldThreeThresholdAlgorithm::clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output) {
  clusterizeDetUnit_(digis,output);
}

template<typename InputDetSet>
void OldThreeThresholdAlgorithm::clusterizeDetUnit_(const InputDetSet& input,
							edmNew::DetSetVector<SiStripCluster>::FastFiller& output) {
  
#ifdef PATCH_FOR_DIGIS_DUPLICATION
  bool printPatchError=false;
  int countErrors=0;
#endif
  const uint32_t& detID = input.detId();
  
  if (!qualityHandle_->IsModuleUsable(detID)){
#ifdef DEBUG_SiStripClusterizer_
    LogDebug("SiStripClusterizer") << "[OldThreeThresholdAlgorithm::clusterizeDetUnit] detid " << detID << " skipped because flagged NOT Usable in SiStripQuality " << std::endl; 
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
#ifdef PATCH_FOR_DIGIS_DUPLICATION
    bool isDigiListBad=false;
    int16_t oldStrip=-1;
#endif

    for (itest=ibeg; itest<=iend; itest++) {
      float channelNoise = noiseHandle_->getNoise(itest->strip(),detNoiseRange);
      bool IsBadChannel = qualityHandle_->IsStripBad(detQualityRange,itest->strip());

#ifdef PATCH_FOR_DIGIS_DUPLICATION
      if(itest->strip()==oldStrip){
	isDigiListBad=true;
	printPatchError=true;
	countErrors++;
	break;
      }
      oldStrip=itest->strip();
#endif


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

	// Begin of Change done by Loic Quertenmont
	// Protection if the Charge/Gain brings the charge to be bigger 
	// than 255 (largest integer containable in uint8)
        // Also, Gain is applied only if the strip is not saturating.
	// Same convention as in SimTracker/SiStripDigitizer/src/SiTrivialDigitalConverter.cc

        float stripCharge=(static_cast<float>(itest->adc()));

	// //dummy DEBUGG
	// float stripCharge;
	// for (uint16_t myadc=0; myadc<=255; myadc++){

	//   stripCharge=(static_cast<float>(myadc));

	//   gainFactor=0.73;
	// //ENDDEBUGG

	//correct for gain only non-truncated channel charge. Throw exception if channel charge exceeding 255 ADC counts is found
 
	if(stripCharge<254){
	  stripCharge /= gainFactor;  
	      
	  if(stripCharge>511.5){stripCharge=255;}
	  else if(stripCharge>253.5){stripCharge=254;}
	}  
	else if(stripCharge>255){
	  throw cms::Exception("LogicError") << "Cluster charge (" << stripCharge << ") out of range. This clustering algorithm should only work with input charges lower than or equal to 255 ADC counts";

	}

	//  //dummy DEBUGG
	//   std::vector<SiStripDigi> b; b.push_back(SiStripDigi(itest->strip(), static_cast<uint8_t>(stripCharge+0.5)));; 
	//   SiStripCluster c( detID, SiStripCluster::SiStripDigiRange( b.begin(),b.end()));
	//   edm::LogInfo("MYTEST") << "myadc="<<myadc<<" stripCharge="<<stripCharge<<" stored charge="<< (unsigned int)(c.amplitudes()[0])<< std::endl;
	//  edm::LogInfo("MYTEST") << "myadc="<< (unsigned int)(static_cast<uint8_t>(255.5));
	// } 
	//   //ENDDEBUGG

	// End of Change done by Loic Quertenmont



        charge += stripCharge;
        sigmaNoise2 += channelNoise*channelNoise/(gainFactor*gainFactor);
      
	cluster_digis_.push_back(SiStripDigi(itest->strip(), static_cast<uint8_t>(stripCharge+0.5)));
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
    
#ifdef PATCH_FOR_DIGIS_DUPLICATION
    if (charge >= clusterThresholdInNoiseSigma()*sigmaNoise && !isDigiListBad) {
#else
      if (charge >= clusterThresholdInNoiseSigma()*sigmaNoise ) {
#endif

#ifdef DEBUG_SiStripClusterizer_
	if(edm::isDebugEnabled())
	  ss << "\n\t\tCluster accepted :)";
#endif
	output.push_back( SiStripCluster(SiStripCluster::SiStripDigiRange( cluster_digis_.begin(),
										   cluster_digis_.end())));
	
      } else {
#ifdef DEBUG_SiStripClusterizer_
	if(edm::isDebugEnabled())
	  ss << "\n\t\tCluster rejected :(";
#endif
      }
      ibeg = iend+1;
    }

#ifdef PATCH_FOR_DIGIS_DUPLICATION
    if(printPatchError)
      edm::LogError("SiStripClusterizer") << "[OldThreeThresholdAlgorithm::clusterizeDetUnit] \n There are errors in " << countErrors << "  clusters due to not unique digis ";
#endif

#ifdef DEBUG_SiStripClusterizer_
    LogDebug("SiStripClusterizer") << "[OldThreeThresholdAlgorithm::clusterizeDetUnit] \n" << ss.str();
#endif
  }


