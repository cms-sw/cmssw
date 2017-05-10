#include "RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterizerTagMCmerged.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

SiStripClusterizerTagMCmerged::
SiStripClusterizerTagMCmerged(const edm::ParameterSet& conf) 
  : SiStripClusterizer(conf),
    confClusterizer_(conf.getParameter<edm::ParameterSet>("Clusterizer")) {
}

void SiStripClusterizerTagMCmerged::
refiner_iniEvent(edm::Event& event, const edm::EventSetup& evtSetup) {
  associator_.reset(new TrackerHitAssociator(event, confClusterizer_));
}

void SiStripClusterizerTagMCmerged::
refineCluster(const edm::Handle< edm::DetSetVector<SiStripDigi> >& input,
	      std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >& output) {

  // Flag MC-truth merged clusters

  for (edmNew::DetSetVector<SiStripCluster>::const_iterator det=output->begin(); det!=output->end(); det++) {
    uint32_t detId = det->id();
    int ntk = 0;
    int NtkAll = 0;
    for (edmNew::DetSet<SiStripCluster>::iterator clust = det->begin(); clust != det->end(); clust++) {
	// float dEdx = siStripClusterTools::chargePerCM(ssdid, *clust, locDir);
      if (associator_ != 0) {
        std::vector<SimHitIdpr> simtrackid;
	  bool useAssociateHit = !confClusterizer_.getParameter<bool>("associateRecoTracks");
	  if (useAssociateHit) {
	    std::vector<PSimHit> simhit;
	    associator_->associateCluster(clust, DetId(detId), simtrackid, simhit);
	    NtkAll = simtrackid.size();
	    ntk = 0;
	    if (simtrackid.size() > 1) {
	      for (auto const& it : simtrackid) {
		int NintimeHit = 0;
		for (auto const& ih : simhit) {
		  // std::cout << "  hit(tk, evt) trk(tk, evt) bunch ("
		  // 	  << ih.trackId() << ", " << ih.eventId().rawId() << ") ("
		  // 	  << it.first << ", " << it.second.rawId() << ") "
		  // 	  << ih.eventId().bunchCrossing()
		  // 	  << std::endl;
		  if (ih.trackId() == it.first && ih.eventId() == it.second && ih.eventId().bunchCrossing() == 0) ++NintimeHit;
		}
		if (NintimeHit > 0) ++ntk;
	      }
	    }
	  } else {
	    associator_->associateSimpleRecHitCluster(clust, DetId(detId), simtrackid);
	    ntk = NtkAll = simtrackid.size();
	  }
	  if (ntk > 1) {
	    clust->setMerged(true);
	  } else {
	    clust->setMerged(false);
	  }
      }
      // std::cout << "t_m_w " << " " << NtkAll << " " << clust->isMerged()
      // 		<< " " << clust->amplitudes().size()
      // 		<< std::endl;
    } // traverse clusters
  }  // traverse sensors
}
