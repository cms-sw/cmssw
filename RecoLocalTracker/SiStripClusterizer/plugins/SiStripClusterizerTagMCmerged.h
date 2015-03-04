#ifndef RecoLocalTracker_SiStripClusterizerTagMCmerged_h
#define RecoLocalTracker_SiStripClusterizerTagMCmerged_h

#include "RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterizer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

//
// Use MC truth to identify merged clusters, i.e., those associated with more than one
// (in-time) SimTrack.
//
// Author:  Bill Ford (wtford)  3 March 2015
//

class SiStripClusterizerTagMCmerged : public SiStripClusterizer {

public:

  explicit SiStripClusterizerTagMCmerged(const edm::ParameterSet& conf);

private:

  edm::ParameterSet confClusterizer_;
  virtual void refiner_iniEvent(edm::Event&, const edm::EventSetup&);
  virtual void refineCluster(const edm::Handle< edm::DetSetVector<SiStripDigi> >& input,
			     std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >& output);

  std::shared_ptr<TrackerHitAssociator> associator_;

};

#endif
