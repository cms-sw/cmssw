#ifndef PRINT_RecoOBJECTS_H
#define PRINT_RecoOBJECTS_H


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "sstream"
#include "boost/foreach.hpp"

typedef edmNew::DetSet<SiStripCluster>::const_iterator ClusIter;
typedef edmNew::DetSetVector<SiStripCluster> ClusterCollection;

class TrackerTopology;

class PrintRecoObjects{

 public:
  PrintRecoObjects(){};
  ~PrintRecoObjects(){};

  void print(std::stringstream& ss, const SiStripCluster& clus);
  void print(std::stringstream& ss, const TrajectorySeed& tjS);
  void print(std::stringstream& ss, const uint32_t& detid, const TrackerTopology *tTopo) const;
  void print(std::stringstream& ss, const reco::Track* track, const math::XYZPoint& vx);
  std::string getString(uint32_t detid, const TrackerTopology *tTopo) const;

};
#endif
