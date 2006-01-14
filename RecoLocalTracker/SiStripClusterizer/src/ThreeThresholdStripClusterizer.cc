#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"

#include <vector>
#include <algorithm>
#include <cmath>


bool ThreeThresholdStripClusterizer::badChannel( int channel, 
						 const std::vector<short>& badChannels) const
{
  const std::vector<short>::size_type linearCutoff = 10;
  if (badChannels.size() < linearCutoff) {
    return (std::find( badChannels.begin(), badChannels.end(), channel) != badChannels.end());
  }
  else return std::binary_search( badChannels.begin(), badChannels.end(), channel);
}

class AboveSeed {
public:
  AboveSeed( float aseed,  const std::vector<float>& noiseVec) : seed(aseed), noiseVec_(noiseVec) {}

  // FIXME: uses boundary checking with at(), should be replaced with faster operator[]
  // when everything debugged
  bool operator()(const StripDigi& digi) { return digi.adc() >= seed * noiseVec_.at(digi.channel());}
private:
  float seed;
  const std::vector<float>& noiseVec_;
};

class AboveSeed_2 {
public:
  AboveSeed_2( float aseed,  const SiStripPedestalsVector & sistrippedvec) : seed(aseed), sistrippedvec_(sistrippedvec) {}

  // FIXME: uses boundary checking with at(), should be replaced with faster operator[]
  // when everything debugged
  bool operator()(const StripDigi& digi) { return digi.adc() >= seed * sistrippedvec_[digi.channel()].getNoise() ;}
private:
  float seed;
  const SiStripPedestalsVector & sistrippedvec_;
};


std::vector<SiStripCluster> 
ThreeThresholdStripClusterizer::clusterizeDetUnit( DigiIterator begin, DigiIterator end,
		     unsigned int detid,
		     const SiStripPedestalsVector & sistrippedvec)
{
  std::cout << "I'm in new clusterizeDetUnit for detid " << detid << std::endl;
  // const int maxBadChannels_ = 1;
  const int max_holes = 0;

  DigiContainer::const_iterator ibeg, iend, ihigh, itest, i;  
  ibeg = iend = begin;
  DigiContainer my_digis; my_digis.reserve(10);

  std::vector<SiStripCluster> rhits; rhits.reserve( (end - begin)/3 + 1);

  //  cout << "before while loop..." << endl;

  while ( ibeg != end &&
          (ihigh = find_if( ibeg, end, AboveSeed_2(seedThresholdInNoiseSigma(),sistrippedvec))) != end) {

    //std::cout << ihigh->channel() << std::endl;

    // The seed strip is ihigh. Scan up and down from it, finding nearby strips above
    // threshold, allowing for some holes. The accepted cluster runs from strip ibeg
    // to iend, and itest is the strip under study, not yet accepted.
    iend = ihigh;
    itest = iend + 1;
    while ( itest != end && (itest->strip() - iend->strip() <= max_holes+1)) {
      //float channelNoise = noiseVec.at(itest->channel());   
      if (sistrippedvec.size()<=itest->channel()) 
	{
	  std::cout <<  "WARNING for detid " << detid << " there is a request for noise for channel seed" << itest->channel() << " when this detid has sistrippedvec.size= " <<  sistrippedvec.size() << "\nskip"<< std::endl;
	  continue;
	}
      float channelNoise = sistrippedvec[itest->channel()].getNoise();  
      std::cout << "A) channelNoise for detid " << detid << " digis " << itest->channel()  << " adc " << itest->adc() << " is " << channelNoise << std::endl;
      if ( itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
	iend = itest;
      }
      ++itest;
    }
    ibeg = ihigh;
    itest = ibeg - 1;
    while ( itest >= begin &&
	    (ibeg->strip() - itest->strip() <= max_holes+1)) {
      //      float channelNoise = noiseVec.at(itest->channel());   
      
      if (sistrippedvec.size()<=itest->channel()) 
	{
	  std::cout << "WARNING for detid " << detid << " there is a request for noise for channel " << itest->channel() << " when this detid has sistrippedvec.size= " << sistrippedvec.size() << "\nskip"<< std::endl;
	  continue;
	}
      float channelNoise = sistrippedvec[itest->channel()].getNoise();  
      std::cout << "B)channelNoise for detid " << detid << " digis " << itest->channel()  << " adc " << itest->adc() << " is " << channelNoise << std::endl;
      if ( itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
        ibeg = itest;
      }
      --itest;
    }
 
    int charge = 0;
    float sigmaNoise2=0;
    my_digis.clear();
    for (i=ibeg; i<=iend; i++) {
      //float channelNoise = noiseVec.at(i->channel());
      if (sistrippedvec.size()<=i->channel()) 
	{
	  std::cout << "WARNING for detid " << detid << " there is a request for noise for channel " << i->channel() << " when this detid has sistrippedvec.size= " <<  sistrippedvec.size() << "\nskip"<< std::endl;
	  continue;
	}
      float channelNoise = sistrippedvec[i->channel()].getNoise();  
      std::cout << "C) channelNoise for detid " << detid << " digis " << i->channel()  << " adc " << i->adc() << " is " << channelNoise << std::endl;
      if ( i->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {
	// FIXME: should the digi be tested for badChannel before using the adc?
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



std::vector<SiStripCluster> 
ThreeThresholdStripClusterizer::clusterizeDetUnit( DigiIterator begin, DigiIterator end,
						   unsigned int detid,
						   const std::vector<float>& noiseVec,
						   const std::vector<short>& badChannels)
{
  // const int maxBadChannels_ = 1;
  const int max_holes = 0;

  DigiContainer::const_iterator ibeg, iend, ihigh, itest, i;  
  ibeg = iend = begin;
  DigiContainer my_digis; my_digis.reserve(10);

  std::vector<SiStripCluster> rhits; rhits.reserve( (end - begin)/3 + 1);

  //  cout << "before while loop..." << endl;

  while ( ibeg != end &&
          (ihigh = find_if( ibeg, end, AboveSeed(seedThresholdInNoiseSigma(),noiseVec))) != end) {

    //std::cout << ihigh->channel() << std::endl;

    // The seed strip is ihigh. Scan up and down from it, finding nearby strips above
    // threshold, allowing for some holes. The accepted cluster runs from strip ibeg
    // to iend, and itest is the strip under study, not yet accepted.
    iend = ihigh;
    itest = iend + 1;
    while ( itest != end && (itest->strip() - iend->strip() <= max_holes+1)) {
      float channelNoise = noiseVec.at(itest->channel());  
      if ( itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
         iend = itest;
      }
      ++itest;
    }
    ibeg = ihigh;
    itest = ibeg - 1;
    while ( itest >= begin &&
               (ibeg->strip() - itest->strip() <= max_holes+1)) {
      float channelNoise = noiseVec.at(itest->channel());   
      if ( itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
        ibeg = itest;
      }
      --itest;
    }
 
    int charge = 0;
    float sigmaNoise2=0;
    my_digis.clear();
    for (i=ibeg; i<=iend; i++) {
      float channelNoise = noiseVec.at(i->channel());
      if ( i->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {
	// FIXME: should the digi be tested for badChannel before using the adc?
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

