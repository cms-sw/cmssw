#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"


bool ThreeThresholdStripClusterizer::badChannel( int channel, 
						 const std::vector<short>& badChannels) const
{
  const std::vector<short>::size_type linearCutoff = 10;
  if (badChannels.size() < linearCutoff) {
    return (std::find( badChannels.begin(), badChannels.end(), channel) != badChannels.end());
  }
  else return std::binary_search( badChannels.begin(), badChannels.end(), channel);
}



std::vector<SiStripCluster> ThreeThresholdStripClusterizer::clusterizeDetUnit( DigiIterator begin, DigiIterator end,
									       unsigned int detid, const SiStripNoiseVector& vnoise){
  //  std::cout << "I'm in clusterizeDetUnit for detid " << detid << std::endl;
  // const int maxBadChannels_ = 1;

  DigiContainer::const_iterator ibeg, iend, ihigh, itest, i;  
  ibeg = iend = begin;
  DigiContainer my_digis; my_digis.reserve(10);

  std::vector<SiStripCluster> rhits; rhits.reserve( (end - begin)/3 + 1);

  AboveSeed predicate(seedThresholdInNoiseSigma(),vnoise);

  //FIXME
  //Check if channel is lower than vnoise.size()
  itest = end - 1;
  if (vnoise.size()<=itest->channel()) 
    {
      std::cout <<  "WARNING for detid " << detid << " there will be a request for noise for channel seed" << itest->channel() << " when this detid has vnoise.size= " <<  vnoise.size() << "\nskip"<< std::endl;
      return rhits;
    }
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //  cout << "before while loop..." << endl;

  while ( ibeg != end &&
          (ihigh = find_if( ibeg, end, predicate)) != end) {

    std::cout << "Seed Channel: detid "<< detid << " digis " << ihigh->channel() 
	      << " adc " << ihigh->adc() << " is " << " channelNoise " << vnoise[ihigh->channel()].getNoise() <<  " IsBadChannel  " << vnoise[ihigh->channel()].getDisable() << std::endl;
   
    // The seed strip is ihigh. Scan up and down from it, finding nearby strips above
    // threshold, allowing for some holes. The accepted cluster runs from strip ibeg
    // to iend, and itest is the strip under study, not yet accepted.
    iend = ihigh;
    itest = iend + 1;
    while ( itest != end && (itest->strip() - iend->strip() <= max_holes_ + 1 )) {
      float channelNoise = vnoise[itest->channel()].getNoise();
      bool IsBadChannel = vnoise[itest->channel()].getDisable();
      //FIXME
      //for debugging, remove it!!!
      std::cout << "Strips on the right: detid " << detid << " digis " << itest->channel()  
		<< " adc " << itest->adc() << " is " << " channelNoise " << channelNoise <<  " IsBadChannel  " << IsBadChannel << std::endl;
      //////
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
	iend = itest;
      }
      ++itest;
    }
    //if the next digi after iend is an adiacent bad digi then insert into candidate cluster
    itest=iend+1;
    if ( itest != end && (itest->strip() - iend->strip() == 1) && vnoise[itest->channel()].getDisable() ) {    
      std::cout << "Inserted bad strip at the end edge iend->strip()= " << iend->strip() << " itest->strip() = " << itest->strip() << std::endl;
      iend++;
    }

    ibeg = ihigh;
    itest = ibeg - 1;
    while ( itest >= begin && (ibeg->strip() - itest->strip() <= max_holes_ + 1 )) {
      float channelNoise = vnoise[itest->channel()].getNoise();  
      bool IsBadChannel = vnoise[itest->channel()].getDisable();
      //FIXME
      //for debugging, remove it!!!
      std::cout << "Strips on the left : detid " << detid << " digis " << itest->channel()  
		<< " adc " << itest->adc() << " is " << " channelNoise " << channelNoise <<  " IsBadChannel  " << IsBadChannel << std::endl;
      ////////////
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
        ibeg = itest;
      }
      --itest;
    }
    //if the next digi after ibeg is an adiacent bad digi then insert into candidate cluster
    itest=ibeg-1;
    if ( itest >= begin && (ibeg->strip() - itest->strip() == 1) && vnoise[itest->channel()].getDisable() ) {    
      std::cout << "Inserted bad strip at the begin edge ibeg->strip()= " << ibeg->strip() << " itest->strip() = " << itest->strip() << std::endl;
      ibeg--;
    }
 
    int charge = 0;
    float sigmaNoise2=0;
    my_digis.clear();
    for (i=ibeg; i<=iend; i++) {
      float channelNoise = vnoise[i->channel()].getNoise();  
      bool IsBadChannel = vnoise[i->channel()].getDisable();
      std::cout << "Looking at cluster digis: detid " << detid << " digis " << i->channel()  
		<< " adc " << i->adc() << " channelNoise " << channelNoise << " IsBadChannel  " << IsBadChannel << std::endl;
      if (!IsBadChannel && i->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {
        charge += i->adc();
        sigmaNoise2 += channelNoise*channelNoise;
        my_digis.push_back(*i);
      }
    }
    float sigmaNoise = sqrt(sigmaNoise2);

    if (charge >= static_cast<int>( clusterThresholdInNoiseSigma()*sigmaNoise)) {
      rhits.push_back( SiStripCluster( detid, SiStripCluster::StripDigiRange( my_digis.begin(),
									      my_digis.end())));
    }
    ibeg = iend+1;
  }   
  return rhits;
}


//FIXME
//In the future, with blobs, perhaps we will come back at this version
// std::vector<SiStripCluster> 
// ThreeThresholdStripClusterizer::clusterizeDetUnit( DigiIterator begin, DigiIterator end,
// 						   unsigned int detid,
// 						   const std::vector<float>& noiseVec,
// 						   const std::vector<short>& badChannels)
// {
//   // const int maxBadChannels_ = 1;
//   //const int max_holes = 0;

//   DigiContainer::const_iterator ibeg, iend, ihigh, itest, i;  
//   ibeg = iend = begin;
//   DigiContainer my_digis; my_digis.reserve(10);

//   std::vector<SiStripCluster> rhits; rhits.reserve( (end - begin)/3 + 1);

//   //  cout << "before while loop..." << endl;

//   while ( ibeg != end &&
//           (ihigh = find_if( ibeg, end, AboveSeed(seedThresholdInNoiseSigma(),noiseVec))) != end) {

//     //std::cout << ihigh->channel() << std::endl;

//     // The seed strip is ihigh. Scan up and down from it, finding nearby strips above
//     // threshold, allowing for some holes. The accepted cluster runs from strip ibeg
//     // to iend, and itest is the strip under study, not yet accepted.
//     iend = ihigh;
//     itest = iend + 1;
//     while ( itest != end && (itest->strip() - iend->strip() <= max_holes+1)) {
//       float channelNoise = noiseVec.at(itest->channel());  
//       if ( itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
//          iend = itest;
//       }
//       ++itest;
//     }
//     ibeg = ihigh;
//     itest = ibeg - 1;
//     while ( itest >= begin &&
//                (ibeg->strip() - itest->strip() <= max_holes+1)) {
//       float channelNoise = noiseVec.at(itest->channel());   
//       if ( itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
//         ibeg = itest;
//       }
//       --itest;
//     }
 
//     int charge = 0;
//     float sigmaNoise2=0;
//     my_digis.clear();
//     for (i=ibeg; i<=iend; i++) {
//       float channelNoise = noiseVec.at(i->channel());
//       if ( i->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {
// 	// FIXME: should the digi be tested for badChannel before using the adc?
//         charge += i->adc();
//         sigmaNoise2 += channelNoise*channelNoise;
//         my_digis.push_back(*i);
//       }
//     }
//     float sigmaNoise = sqrt(sigmaNoise2);

//     if (charge >= static_cast<int>( clusterThresholdInNoiseSigma()*sigmaNoise)) {
//       rhits.push_back( SiStripCluster( detid, SiStripCluster::StripDigiRange( my_digis.begin(),
// 									      my_digis.end())));
//     }
//     ibeg = iend+1;
//   }   
//   return rhits;
// }

