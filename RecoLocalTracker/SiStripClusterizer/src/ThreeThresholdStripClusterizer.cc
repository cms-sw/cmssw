#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"

void ThreeThresholdStripClusterizer::clusterizeDetUnit( const edm::DetSet<SiStripDigi>& input,edm::DetSet<SiStripCluster>& output)
{
  const uint32_t& detID = input.id;
  edm::DetSet<SiStripDigi>::const_iterator begin=input.data.begin();
  edm::DetSet<SiStripDigi>::const_iterator end  =input.data.end();
  
  edm::DetSet<SiStripDigi>::const_iterator ibeg, iend, ihigh, itest;  
  ibeg = iend = begin;
  std::vector<SiStripDigi> cluster_digis;  
  cluster_digis.reserve(10);

  output.data.reserve( (end - begin)/3 + 1);

  AboveSeed predicate(seedThresholdInNoiseSigma(),SiStripNoiseService_,detID);

  //FIXME
  //Check if channel is lower than vnoise.size()
//   itest = end - 1;
//   if (vnoise.size()<=itest->channel()) 
//     {
//       std::cout <<  "WARNING for detID " << detID << " there will be a request for noise for channel seed" << itest->channel() << " when this detID has vnoise.size= " <<  vnoise.size() << "\nskip"<< std::endl;
//       return rhits;
//     }
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //  cout << "before while loop..." << endl;

  while ( ibeg != end &&
          (ihigh = find_if( ibeg, end, predicate)) != end) {

    LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] " 
				   << "Seed Channel: detID "<< detID << " digis " << ihigh->strip() 
				   << " adc " << ihigh->adc() << " is " 
				   << " channelNoise " << SiStripNoiseService_->getNoise(detID,ihigh->strip()) 
				   <<  " IsBadChannel  " << SiStripNoiseService_->getDisable(detID,ihigh->strip()) << std::endl;

    // The seed strip is ihigh. Scan up and down from it, finding nearby strips above
    // threshold, allowing for some holes. The accepted cluster runs from strip ibeg
    // to iend, and itest is the strip under study, not yet accepted.
    iend = ihigh;
    itest = iend + 1;
    while ( itest != end && (itest->strip() - iend->strip() <= max_holes_ + 1 )) {
      float channelNoise = SiStripNoiseService_->getNoise(detID,itest->strip());
      bool IsBadChannel = SiStripNoiseService_->getDisable(detID,itest->strip());

      LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] " 
				     << "Strips on the right: detID " << detID << " digis " << itest->strip()  
				     << " adc " << itest->adc() << " " << " channelNoise " << channelNoise <<  " IsBadChannel  " << IsBadChannel << std::endl;
      
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
	iend = itest;
      }
      ++itest;
    }
    //if the next digi after iend is an adiacent bad digi then insert into candidate cluster
    itest=iend+1;
    if ( itest != end && (itest->strip() - iend->strip() == 1) && SiStripNoiseService_->getDisable(detID,itest->strip()) ) {    
      LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] " 
				     << "Inserted bad strip at the end edge iend->strip()= " << iend->strip() << " itest->strip() = " << itest->strip() << std::endl;
      iend++;
    }

    ibeg = ihigh;
    itest = ibeg - 1;
    while ( itest >= begin && (ibeg->strip() - itest->strip() <= max_holes_ + 1 )) {
      float channelNoise = SiStripNoiseService_->getNoise(detID,itest->strip());
      bool IsBadChannel = SiStripNoiseService_->getDisable(detID,itest->strip());

      LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] " 
				     << "Strips on the left : detID " << detID << " digis " << itest->strip()  
				     << " adc " << itest->adc() << " " << " channelNoise " << channelNoise <<  " IsBadChannel  " << IsBadChannel << std::endl;

      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
        ibeg = itest;
      }
      --itest;
    }
    //if the next digi after ibeg is an adiacent bad digi then insert into candidate cluster
    itest=ibeg-1;
    if ( itest >= begin && (ibeg->strip() - itest->strip() == 1) &&  SiStripNoiseService_->getDisable(detID,itest->strip()) ) {    

      LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] " 
				     << "Inserted bad strip at the begin edge ibeg->strip()= " << ibeg->strip() << " itest->strip() = " << itest->strip() << std::endl;

      ibeg--;
    }
    
    int charge = 0;
    float sigmaNoise2=0;
    //int counts=0;
    cluster_digis.clear();
    for (itest=ibeg; itest<=iend; itest++) {
      float channelNoise = SiStripNoiseService_->getNoise(detID,itest->strip());
      bool IsBadChannel = SiStripNoiseService_->getDisable(detID,itest->strip());

      LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] " 
				     << "Looking at cluster digis: detID " << detID << " digis " << itest->strip()  
				     << " adc " << itest->adc() << " channelNoise " << channelNoise << " IsBadChannel  " << IsBadChannel << std::endl;

      //check for consecutive digis
      if (itest!=ibeg && itest->strip()-(itest-1)->strip()!=1){
	//digi *(itest-1) and *itest are not consecutive: create an equivalent number of Digis with zero amp
	for (int j=(itest-1)->strip()+1;j<itest->strip();j++){
	  cluster_digis.push_back(SiStripDigi(j,0)); //if strip bad or under threshold set StripDigi.adc_=0  
	  
	  LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] " 
					 << "Hole added: detID " << detID << " digis " << j << " adc 0 " <<  std::endl;

	}
      }
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {
        charge += itest->adc();
        sigmaNoise2 += channelNoise*channelNoise;
        //counts++;
	cluster_digis.push_back(*itest);
      } else {
	cluster_digis.push_back(SiStripDigi(itest->strip(),0)); //if strip bad or under threshold set SiStripDigi.adc_=0

	LogDebug("SiStripClusterizer") << "[ThreeThresholdStripClusterizer::clusterizeDetUnit] " 
				       << "Bad or under th digis: detID " << detID  << " digis " << itest->strip()  
				       << " adc " << itest->adc() << " channelNoise " << channelNoise << " IsBadChannel  " << IsBadChannel << std::endl;

      }
    }
    //   float sigmaNoise = sqrt(sigmaNoise2/counts);
    float sigmaNoise = sqrt(sigmaNoise2);

    if (charge >= static_cast<int>( clusterThresholdInNoiseSigma()*sigmaNoise)) {
      output.data.push_back( SiStripCluster( detID, SiStripCluster::SiStripDigiRange( cluster_digis.begin(),
										      cluster_digis.end())));
    }
    ibeg = iend+1;
  }
}

