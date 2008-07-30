#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void ThreeThresholdStripClusterizer::clusterizeDetUnit( 
						       const edm::DetSet<SiStripDigi>& input,
						       edm::DetSet<SiStripCluster>& output, 
						       const edm::EventSetup& es){
  //Get ESObject 
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;

  es.get<SiStripGainRcd>().get(gainHandle);
  es.get<SiStripNoisesRcd>().get(noiseHandle);
  es.get<SiStripQualityRcd>().get(qualityLabel_,qualityHandle);

  bool printPatchError=false;
  int countErrors=0;
  const uint32_t& detID = input.id;
  
  if (!qualityHandle->IsModuleUsable(detID)){
    LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] detid " << detID << " skipped because flagged NOT Usable in SiStripQuality " << std::endl; 
    return;
  }
  
  edm::DetSet<SiStripDigi>::const_iterator begin=input.data.begin();
  edm::DetSet<SiStripDigi>::const_iterator end  =input.data.end();
  
  edm::DetSet<SiStripDigi>::const_iterator ibeg, iend, ihigh, itest;  
  ibeg = iend = begin;
  std::vector<SiStripDigi> cluster_digis;  
  cluster_digis.reserve(10);

  output.data.reserve( (end - begin)/3 + 1);

  SiStripApvGain::Range detGainRange =  gainHandle->getRange(detID); 
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
  SiStripQuality::Range detQualityRange = qualityHandle->getRange(detID);


  //  AboveSeed predicate(seedThresholdInNoiseSigma(),SiStripNoiseService_,detID);
  AboveSeed predicate(seedThresholdInNoiseSigma(), noiseHandle, detNoiseRange, qualityHandle, detQualityRange);

  std::stringstream ss;
  
  while ( ibeg != end &&
          (ihigh = find_if( ibeg, end, predicate)) != end) {

    if(edm::isDebugEnabled())
      ss << "\nSeed Channel:\n\t\t detID "<< detID << " digis " << ihigh->strip() 
	 << " adc " << ihigh->adc() << " is " 
	 << " channelNoise " << noiseHandle->getNoise(ihigh->strip(),detNoiseRange) 
	 <<  " IsBadChannel from SiStripQuality " << qualityHandle->IsStripBad(detQualityRange,ihigh->strip());
    
    // The seed strip is ihigh. Scan up and down from it, finding nearby strips above
    // threshold, allowing for some holes. The accepted cluster runs from strip ibeg
    // to iend, and itest is the strip under study, not yet accepted.
    iend = ihigh;
    itest = iend + 1;

    while ( itest != end && (itest->strip() - iend->strip() <= max_holes_ + 1 )) {
      float channelNoise = noiseHandle->getNoise(itest->strip(),detNoiseRange);
      bool IsBadChannel = qualityHandle->IsStripBad(detQualityRange,itest->strip());

      if(edm::isDebugEnabled())
	ss << "\nStrips on the right:\n\t\t detID " << detID << " digis " << itest->strip()  
	   << " adc " << itest->adc() << " " << " channelNoise " << channelNoise 
	   <<  " IsBadChannel from SiStripQuality " << IsBadChannel;
      
      
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
	iend = itest;
      }
      ++itest;
    }
    //if the next digi after iend is an adiacent bad digi then insert into candidate cluster
    itest=iend+1;
    if ( itest != end && (itest->strip() - iend->strip() == 1) && qualityHandle->IsStripBad(detQualityRange,itest->strip()) ){
      if(edm::isDebugEnabled())
	ss << "\n\t\tInserted bad strip at the end edge iend->strip()= " << iend->strip() << " itest->strip() = " << itest->strip();
      
      iend++;
    }

    ibeg = ihigh;
    itest = ibeg - 1;
    while ( itest >= begin && (ibeg->strip() - itest->strip() <= max_holes_ + 1 )) {
      float channelNoise = noiseHandle->getNoise(itest->strip(),detNoiseRange);
      bool IsBadChannel = qualityHandle->IsStripBad(detQualityRange,itest->strip());

      if(edm::isDebugEnabled())
	ss << "\nStrips on the left:\n\t\t detID " << detID << " digis " << itest->strip()
	   << " adc " << itest->adc() << " " << " channelNoise " << channelNoise 
	   <<  " IsBadChannel from SiStripQuality " << IsBadChannel;
      
      
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
        ibeg = itest;
      }
      --itest;
    }
    //if the next digi after ibeg is an adiacent bad digi then insert into candidate cluster
    itest=ibeg-1;
    if ( itest >= begin && (ibeg->strip() - itest->strip() == 1) &&  qualityHandle->IsStripBad(detQualityRange,itest->strip()) ) {    
      if(edm::isDebugEnabled())
	ss << "\nInserted bad strip at the begin edge ibeg->strip()= " << ibeg->strip() << " itest->strip() = " << itest->strip();
      
      ibeg--;
    }
    
    float charge = 0;
    float sigmaNoise2=0;
    //int counts=0;
    cluster_digis.clear();

    //PATCH for 20X CRUZET 
    bool isDigiListBad=false;
    int16_t oldStrip=-1;
    //PATCH for 20X CRUZET 
    
    for (itest=ibeg; itest<=iend; itest++) {
      float channelNoise = noiseHandle->getNoise(itest->strip(),detNoiseRange);
      bool IsBadChannel = qualityHandle->IsStripBad(detQualityRange,itest->strip());
    
      //PATCH for 20X CRUZET 
      if(itest->strip()==oldStrip){
	isDigiListBad=true;
	printPatchError=true;
	countErrors++;
	break;
      }
      oldStrip=itest->strip();
      //PATCH for 20X CRUZET 
      

      if(edm::isDebugEnabled())
	ss << "\nLooking at cluster digis:\n\t\t detID " << detID << " digis " << itest->strip()  
	   << " adc " << itest->adc() << " channelNoise " << channelNoise 
	   <<  " IsBadChannel from SiStripQuality " << IsBadChannel;
      
 
      //check for consecutive digis
      if (itest!=ibeg && itest->strip()-(itest-1)->strip()!=1){
	//digi *(itest-1) and *itest are not consecutive: create an equivalent number of Digis with zero amp
	for (int j=(itest-1)->strip()+1;j<itest->strip();j++){
	  cluster_digis.push_back(SiStripDigi(j,0)); //if strip bad or under threshold set StripDigi.adc_=0  
	  
	  ss << "\n\t\tHole added: detID " << detID << " digis " << j << " adc 0 ";

	}
      }
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {

	float gainFactor  = gainHandle->getStripGain(itest->strip(), detGainRange);
        float stripCharge=(static_cast<float>(itest->adc()))/gainFactor;

        charge += stripCharge;
        sigmaNoise2 += channelNoise*channelNoise/(gainFactor*gainFactor);
      
	cluster_digis.push_back(SiStripDigi(itest->strip(), static_cast<int>(stripCharge+0.499)));
      } else {
	cluster_digis.push_back(SiStripDigi(itest->strip(),0)); //if strip bad or under threshold set SiStripDigi.adc_=0
	
	 if(edm::isDebugEnabled())
	   ss << "\n\t\tBad or under threshold digis: detID " << detID  << " digis " << itest->strip()  
	      << " adc " << itest->adc() << " channelNoise " << channelNoise 
	      <<  " IsBadChannel from SiStripQuality " << IsBadChannel;
      }
    }
    float sigmaNoise = sqrt(sigmaNoise2);
    
    if (charge >= clusterThresholdInNoiseSigma()*sigmaNoise && !isDigiListBad) {
       if(edm::isDebugEnabled())
	 ss << "\n\t\tCluster accepted :)";
       output.data.push_back( SiStripCluster( detID, SiStripCluster::SiStripDigiRange( cluster_digis.begin(),
										      cluster_digis.end())));
    } else {
      if(edm::isDebugEnabled())
	ss << "\n\t\tCluster rejected :(";
    }
    ibeg = iend+1;
  }
  
  if(printPatchError)
    edm::LogError("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] \n There are errors in " << countErrors << "  clusters due to not unique digis "; 
  
  LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] \n" << ss.str();
}


