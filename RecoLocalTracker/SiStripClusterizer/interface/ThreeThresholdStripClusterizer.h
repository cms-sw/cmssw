#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H

#include <vector>
class StripDigi;
class SiStripCluster;

#include "CondFormats/SiStripObjects/interface/SiStriNoises.h"

class ThreeThresholdStripClusterizer {
public:

  typedef std::vector<StripDigi>                  DigiContainer;
  typedef DigiContainer::const_iterator           DigiIterator;

  ThreeThresholdStripClusterizer(float strip_thr, float seed_thr,float clust_thr, int max_holes) :
    theChannelThreshold(strip_thr), 
    theSeedThreshold(seed_thr),
    theClusterThreshold(clust_thr),
    max_holes_(max_holes){};  

//FIXME
//In the future, with blobs, perhaps we will come back at this version
/*   std::vector<SiStripCluster>  */
/*     clusterizeDetUnit( DigiIterator begin, DigiIterator end, */
/* 		       unsigned int detid, */
/* 		       const std::vector<float>& noiseVec, */
/* 		       const std::vector<short>& badChannels); */

  std::vector<SiStripCluster> clusterizeDetUnit(DigiIterator begin, DigiIterator end,
						unsigned int detid, const SiStripNoiseVector& vnoise);
  

  float channelThresholdInNoiseSigma() const { return theChannelThreshold;}
  float seedThresholdInNoiseSigma()    const { return theSeedThreshold;}
  float clusterThresholdInNoiseSigma() const { return theClusterThreshold;}

private:

  float theChannelThreshold;
  float theSeedThreshold;
  float theClusterThreshold;
  int max_holes_;

  bool badChannel( int channel, const std::vector<short>& badChannels) const;

};

//FIXME
//In the future, with blobs, perhaps we will come back at this version
/* class AboveSeed { */
/* public: */
/*   AboveSeed( float aseed,  const std::vector<float>& noiseVec) : seed(aseed), noiseVec_(noiseVec) {} */

/*   // FIXME: uses boundary checking with at(), should be replaced with faster operator[] */
/*   // when everything debugged */
/*   bool operator()(const StripDigi& digi) { return digi.adc() >= seed * noiseVec_.at(digi.channel());} */
/* private: */
/*   float seed; */
/*   const std::vector<float>& noiseVec_; */
/* }; */

class AboveSeed {
 public:
  AboveSeed(float aseed,const SiStripNoiseVector& vnoise) : seed(aseed), vnoise_(vnoise) {}

  // FIXME: uses boundary checking with at(), should be replaced with faster operator[]
  // when everything debugged
  bool operator()(const StripDigi& digi) { return digi.adc() >= seed * vnoise_[digi.channel()].getNoise() ;}
private:
  float seed;
  const SiStripNoiseVector& vnoise_;
};

#endif
