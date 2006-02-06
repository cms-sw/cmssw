#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H

#include <vector>
class StripDigi;
class SiStripCluster;

class ThreeThresholdStripClusterizer {
public:

  typedef std::vector<StripDigi>                  DigiContainer;
  typedef DigiContainer::const_iterator           DigiIterator;

  ThreeThresholdStripClusterizer(float strip_thr, float seed_thr,
                                 float clust_thr) :
    theChannelThreshold(strip_thr), theSeedThreshold(seed_thr),
    theClusterThreshold(clust_thr) {}  

  std::vector<SiStripCluster> 
  clusterizeDetUnit( DigiIterator begin, DigiIterator end,
		     unsigned int detid,
		     const std::vector<float>& noiseVec,
		     const std::vector<short>& badChannels);

  float channelThresholdInNoiseSigma() const { return theChannelThreshold;}
  float seedThresholdInNoiseSigma()    const { return theSeedThreshold;}
  float clusterThresholdInNoiseSigma() const { return theClusterThreshold;}

private:

  float theChannelThreshold;
  float theSeedThreshold;
  float theClusterThreshold;

  bool badChannel( int channel, const std::vector<short>& badChannels) const;

};

#endif
