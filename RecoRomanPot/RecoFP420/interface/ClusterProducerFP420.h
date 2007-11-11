#ifndef ClusterProducerFP420_h
#define ClusterProducerFP420_h

#include "RecoRomanPot/RecoFP420/interface/ClusterNoiseFP420.h"

#include "DataFormats/FP420Cluster/interface/ClusterFP420.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"

#include <vector>
#include <algorithm>
#include <cmath>


class ClusterProducerFP420 {
public:

  typedef std::vector<HDigiFP420>::const_iterator           HDigiFP420Iter;

  ClusterProducerFP420(float electrode_thr, float seed_thr,float clust_thr, int max_voids) :
    theChannelThreshold(electrode_thr), 
    theSeedThreshold(seed_thr),
    theClusterThreshold(clust_thr),
    max_voids_(max_voids){};  


  std::vector<ClusterFP420> clusterizeDetUnit(HDigiFP420Iter begin, HDigiFP420Iter end,
						unsigned int detid, const ElectrodNoiseVector& vnoise);
  std::vector<ClusterFP420> clusterizeDetUnitPixels(HDigiFP420Iter begin, HDigiFP420Iter end,
						    unsigned int detid, const ElectrodNoiseVector& vnoise, unsigned int zside);
  
  int difNarr(unsigned int zside, HDigiFP420Iter ichannel,
				  HDigiFP420Iter jchannel);
  int difWide(unsigned int zside, HDigiFP420Iter ichannel,
				  HDigiFP420Iter jchannel);

  float channelThresholdInNoiseSigma() const { return theChannelThreshold;}
  float seedThresholdInNoiseSigma()    const { return theSeedThreshold;}
  float clusterThresholdInNoiseSigma() const { return theClusterThreshold;}

private:

  float theChannelThreshold;
  float theSeedThreshold;
  float theClusterThreshold;
  int max_voids_;

  bool badChannel( int channel, const std::vector<short>& badChannels) const;

};

class AboveSeed {
 public:
  AboveSeed(float aseed,const ElectrodNoiseVector& vnoise) : seed(aseed), vnoise_(vnoise) {};

  bool operator()(const HDigiFP420& digi) { return ( !vnoise_[digi.channel()].getDisable() && 
                                               digi.adc() >= seed * vnoise_[digi.channel()].getNoise()) ;}
private:
  float seed;
  const ElectrodNoiseVector& vnoise_;
};

#endif
